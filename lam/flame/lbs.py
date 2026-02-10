"""
Linear Blend Skinning (LBS) implementation for FLAME.

This module contains the core skeletal animation functions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..utils.math_utils import batch_rodrigues


def blend_shapes(
    betas: torch.Tensor,
    shape_disps: torch.Tensor
) -> torch.Tensor:
    """
    Calculate per-vertex displacement due to blend shapes.
    
    Args:
        betas: (B, num_betas) shape/expression coefficients
        shape_disps: (V, 3, num_betas) shape displacement basis
        
    Returns:
        (B, V, 3) per-vertex displacements
        
    Formula:
        displacement[b, v, :] = Σ_i (betas[b, i] * shape_disps[v, :, i])
    """
    # Einstein summation: batch x coefficients → batch x vertices x xyz
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def vertices_to_joints(
    J_regressor: torch.Tensor,
    vertices: torch.Tensor
) -> torch.Tensor:
    """
    Calculate 3D joint locations from vertices.
    
    Args:
        J_regressor: (J, V) joint regressor matrix
        vertices: (B, V, 3) mesh vertices
        
    Returns:
        (B, J, 3) joint locations
        
    Formula:
        joints[b, j, :] = Σ_v (J_regressor[j, v] * vertices[b, v, :])
    """
    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Create 4x4 transformation matrices from rotation and translation.
    
    Args:
        R: (B, 3, 3) rotation matrices
        t: (B, 3, 1) translation vectors
        
    Returns:
        (B, 4, 4) transformation matrices
    """
    # Pad rotation matrix and translation vector to 4x4
    return torch.cat([
        F.pad(R, [0, 0, 0, 1]),           # Add row of zeros below R
        F.pad(t, [0, 0, 0, 1], value=1)   # Add [t; 1] column
    ], dim=2)


def batch_rigid_transform(
    rot_mats: torch.Tensor,
    joints: torch.Tensor,
    parents: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rigid transformations through kinematic tree.
    
    Args:
        rot_mats: (B, J, 3, 3) rotation matrices for each joint
        joints: (B, J, 3) joint locations
        parents: (J,) parent indices for kinematic tree (-1 for root)
        dtype: Output tensor dtype
        
    Returns:
        posed_joints: (B, J, 3) transformed joint locations
        rel_transforms: (B, J, 4, 4) relative transformations
        
    Algorithm:
        1. Compute relative joint positions (distance from parent)
        2. Create local transformation matrices (rotation + translation)
        3. Propagate transforms through kinematic chain
        4. Compute relative transforms for skinning
    """
    joints = torch.unsqueeze(joints, dim=-1)  # (B, J, 3, 1)

    # Relative joint positions
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]  # Offset from parent

    # Create transformation matrices
    transforms_mat = transform_mat(
        rot_mats.view(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)
    ).reshape(-1, joints.shape[1], 4, 4)

    # Propagate through kinematic chain
    transform_chain = [transforms_mat[:, 0]]  # Start with root
    for i in range(1, parents.shape[0]):
        # Transform_{world,i} = Transform_{world,parent} @ Transform_{parent,i}
        curr_transform = torch.matmul(
            transform_chain[parents[i]],
            transforms_mat[:, i]
        )
        transform_chain.append(curr_transform)

    transforms = torch.stack(transform_chain, dim=1)  # (B, J, 4, 4)

    # Extract posed joint positions
    posed_joints = transforms[:, :, :3, 3]

    # Compute relative transforms for skinning
    # This accounts for the rest pose offset
    joints_homogen = F.pad(joints, [0, 0, 0, 1])  # (B, J, 4, 1)
    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen),
        [3, 0, 0, 0, 0, 0, 0, 0]
    )

    return posed_joints, rel_transforms


def linear_blend_skinning(
    pose: torch.Tensor,
    v_shaped: torch.Tensor,
    posedirs: torch.Tensor,
    J_regressor: torch.Tensor,
    parents: torch.Tensor,
    lbs_weights: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform Linear Blend Skinning with pose and shape parameters.
    
    Args:
        pose: (B, J*3) pose parameters in axis-angle format
        v_shaped: (B, V, 3) shaped vertices (template + shape blend)
        posedirs: (num_pose_basis, V*3) pose-dependent deformation basis
        J_regressor: (J, V) joint regressor matrix
        parents: (J,) parent indices for kinematic tree
        lbs_weights: (V, J) linear blend skinning weights
        dtype: Output tensor dtype
        
    Returns:
        vertices: (B, V, 3) final posed vertices
        joints: (B, J, 3) final joint positions
        rot_mats: (B, J, 3, 3) rotation matrices
        
    Algorithm:
        1. Regress joints from shaped vertices
        2. Convert axis-angle pose to rotation matrices
        3. Add pose-dependent corrective deformations
        4. Apply rigid transformations through kinematic chain
        5. Skin vertices using LBS weights
    """
    batch_size = v_shaped.shape[0]
    device = v_shaped.device
    num_joints = J_regressor.shape[0]

    # 1. Get joints from shaped vertices
    J = vertices_to_joints(J_regressor, v_shaped)  # (B, J, 3)

    # 2. Convert axis-angle to rotation matrices
    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mats = batch_rodrigues(
        pose.view(-1, 3), dtype=dtype
    ).view(batch_size, -1, 3, 3)  # (B, J, 3, 3)

    # 3. Pose-dependent deformations
    # Compute pose feature (deviation from identity pose)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch_size, -1)
    
    # Apply pose-dependent offsets
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = v_shaped + pose_offsets

    # 4. Rigid transformation through kinematic tree
    J_transformed, A = batch_rigid_transform(
        rot_mats, J, parents, dtype=dtype
    )  # A: (B, J, 4, 4)

    # 5. Linear blend skinning
    # Compute transformation matrix for each vertex
    W = lbs_weights.unsqueeze(dim=0).expand(batch_size, -1, -1)  # (B, V, J)
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    # Apply transformation to vertices
    v_posed_homo = torch.cat([
        v_posed,
        torch.ones(batch_size, v_posed.shape[1], 1, dtype=dtype, device=device)
    ], dim=2)  # (B, V, 4)
    
    v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))  # (B, V, 4, 1)
    vertices = v_homo[:, :, :3, 0]  # (B, V, 3)

    return vertices, J_transformed, rot_mats


def vertices_to_landmarks(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    lmk_faces_idx: torch.Tensor,
    lmk_bary_coords: torch.Tensor
) -> torch.Tensor:
    """
    Calculate landmarks by barycentric interpolation.
    
    Args:
        vertices: (B, V, 3) mesh vertices
        faces: (F, 3) face triangle indices
        lmk_faces_idx: (B, L) face indices for landmarks
        lmk_bary_coords: (B, L, 3) barycentric coordinates
        
    Returns:
        (B, L, 3) landmark positions
        
    Algorithm:
        For each landmark:
            1. Get the 3 vertices of the corresponding face
            2. Interpolate using barycentric coordinates
            landmark = w1*v1 + w2*v2 + w3*v3
    """
    batch_size, num_verts = vertices.shape[:2]
    device = vertices.device

    # Get face vertices for each landmark
    lmk_faces = torch.index_select(
        faces, 0, lmk_faces_idx.view(-1)
    ).view(batch_size, -1, 3)  # (B, L, 3)

    # Offset face indices by batch
    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=device
    ).view(-1, 1, 1) * num_verts
    lmk_faces = lmk_faces + batch_offset

    # Gather vertices
    lmk_vertices = vertices.view(-1, 3)[lmk_faces].view(
        batch_size, -1, 3, 3
    )  # (B, L, 3_vertices, 3_xyz)

    # Barycentric interpolation
    landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])

    return landmarks
