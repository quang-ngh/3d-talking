"""
FLAME Head Model - Main implementation.

This module contains the FlameHead class which implements the FLAME parametric
head model with a clean, testable interface.
"""

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path

from .lbs import (
    linear_blend_skinning,
    blend_shapes,
    vertices_to_landmarks,
    vertices_to_joints,
)
from ..utils.math_utils import to_tensor, to_numpy


@dataclass
class FlameConfig:
    """Configuration for FLAME model."""
    
    # Model paths
    flame_model_path: str
    flame_lmk_embedding_path: str
    
    # Parameter dimensions
    n_shape: int = 100
    n_expr: int = 50
    
    # Optional features
    add_teeth: bool = False
    add_shoulder: bool = False
    
    #  Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class FlameHead(nn.Module):
    """
    FLAME: Faces Learned with an Articulated Model and Expressions
    
    This class implements a clean version of the FLAME parametric head model.
    
    Attributes:
        n_shape: Number of shape parameters
        n_expr: Number of expression parameters
        v_template: Template mesh vertices
        shapedirs: Shape and expression basis
        posedirs: Pose-dependent deformation basis
        J_regressor: Joint regressor matrix
        parents: Kinematic tree parent indices
        lbs_weights: Linear blend skinning weights
        faces: Face triangle indices
    
    Example:
        >>> config = FlameConfig(
        ...     flame_model_path="assets/FLAME2023/flame2023.pkl",
        ...     flame_lmk_embedding_path="assets/landmark_embedding.npy",
        ...     n_shape=100,
        ...     n_expr=50
        ... )
        >>> flame = FlameHead(config)
        >>> output = flame.forward(
        ...     shape_params=torch.zeros(1, 100),
        ...     expr_params=torch.zeros(1, 50),
        ...     pose_params=torch.zeros(1, 3),
        ...     jaw_params=torch.zeros(1, 3),
        ... )
        >>> vertices = output["vertices"]  # (1, V, 3)
    """
    
    def __init__(self, config: FlameConfig):
        super().__init__()
        
        self.config = config
        self.n_shape = config.n_shape
        self.n_expr = config.n_expr
        self.dtype = config.dtype
        
        # Load FLAME model
        self._load_flame_model(config.flame_model_path)
        
        # Load landmark embeddings
        self._load_landmarks(config.flame_lmk_embedding_path)
        
    def _load_flame_model(self, model_path: str):
        """Load FLAME model parameters from pickle file."""
        with open(model_path, 'rb') as f:
            flame_model = pickle.load(f, encoding='latin1')
        
        # Template vertices
        self.register_buffer(
            'v_template',
            to_tensor(flame_model['v_template'], dtype=self.dtype)
        )
        
        # Shape and expression directions
        shapedirs = to_tensor(flame_model['shapedirs'], dtype=self.dtype)
        shapedirs = torch.cat([
            shapedirs[:, :, :self.n_shape],
            shapedirs[:, :, 300:300+self.n_expr]
        ], dim=2)
        self.register_buffer('shapedirs', shapedirs)
        
        # Pose blend shapes
        num_pose_basis = flame_model['posedirs'].shape[-1]
        posedirs = np.reshape(flame_model['posedirs'], [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(posedirs, dtype=self.dtype))
        
        # Joint regressor
        self.register_buffer(
            'J_regressor',
            to_tensor(flame_model['J_regressor'], dtype=self.dtype)
        )
        
        # Kinematic tree
        parents = to_tensor(flame_model['kintree_table'][0]).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        
        # LBS weights
        self.register_buffer(
            'lbs_weights',
            to_tensor(flame_model['weights'], dtype=self.dtype)
        )
        
        # Faces
        self.register_buffer(
            'faces',
            to_tensor(flame_model['f'], dtype=torch.long)
        )
        
        # Build neck kinematic chain
        neck_kin_chain = []
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
    
    def _load_landmarks(self, lmk_path: str):
        """Load landmark embeddings."""
        lmk_embeddings = np.load(lmk_path, allow_pickle=True, encoding='latin1')[()]
        
        # Full 68 landmarks
        self.register_buffer(
            'full_lmk_faces_idx',
            torch.tensor(lmk_embeddings['full_lmk_faces_idx'], dtype=torch.long)
        )
        self.register_buffer(
            'full_lmk_bary_coords',
            torch.tensor(lmk_embeddings['full_lmk_bary_coords'], dtype=self.dtype)
        )
    
    def forward(
        self,
        shape_params: torch.Tensor,
        expr_params: torch.Tensor,
        pose_params: torch.Tensor,
        jaw_params: torch.Tensor,
        neck_params: Optional[torch.Tensor] = None,
        eye_params: Optional[torch.Tensor] = None,
        translation: Optional[torch.Tensor] = None,
        return_landmarks: bool = True,
        return_joints: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of FLAME model.
        
        Args:
            shape_params: (B, n_shape) shape parameters
            expr_params: (B, n_expr) expression parameters
            pose_params: (B, 3) global head rotation (axis-angle)
            jaw_params: (B, 3) jaw rotation (axis-angle)
            neck_params: Optional (B, 3) neck rotation
            eye_params: Optional (B, 6) eye rotations
            translation: Optional (B, 3) global translation
            return_landmarks: Whether to compute landmarks
            return_joints: Whether to return joint positions
            
        Returns:
            Dictionary containing:
                - vertices: (B, V, 3) mesh vertices
                - landmarks: (B, 68, 3) facial landmarks (if return_landmarks=True)
                - joints: (B, J, 3) joint positions (if return_joints=True)
        """
        batch_size = shape_params.shape[0]
        device = shape_params.device
        
        # Default neck and eyes to zero if not provided
        if neck_params is None:
            neck_params = torch.zeros(batch_size, 3, device=device, dtype=self.dtype)
        if eye_params is None:
            eye_params = torch.zeros(batch_size, 6, device=device, dtype=self.dtype)
        if translation is None:
            translation = torch.zeros(batch_size, 3, device=device, dtype=self.dtype)
        
        # Pad parameters if needed
        if shape_params.shape[1] < self.n_shape:
            pad_size = self.n_shape - shape_params.shape[1]
            shape_params = torch.cat([
                shape_params,
                torch.zeros(batch_size, pad_size, device=device, dtype=self.dtype)
            ], dim=1)
        
        if expr_params.shape[1] < self.n_expr:
            pad_size = self.n_expr - expr_params.shape[1]
            expr_params = torch.cat([
                expr_params,
                torch.zeros(batch_size, pad_size, device=device, dtype=self.dtype)
            ], dim=1)
        
        # Combine shape and expression
        betas = torch.cat([shape_params, expr_params], dim=1)
        
        # Combine all pose parameters
        full_pose = torch.cat([pose_params, neck_params, jaw_params, eye_params], dim=1)
        
        # Apply shape blend
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        v_shaped = template_vertices + blend_shapes(betas, self.shapedirs)
        
        # Linear blend skinning
        vertices, joints, _ = linear_blend_skinning(
            full_pose,
            v_shaped,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
            dtype=self.dtype
        )
        
        # Apply translation
        vertices = vertices + translation.unsqueeze(1)
        joints = joints + translation.unsqueeze(1)
        
        # Prepare output
        output = {'vertices': vertices}
        
        # Compute landmarks if requested
        if return_landmarks:
            bz = vertices.shape[0]
            landmarks = vertices_to_landmarks(
                vertices,
                self.faces,
                self.full_lmk_faces_idx.repeat(bz, 1),
                self.full_lmk_bary_coords.repeat(bz, 1, 1)
            )
            output['landmarks'] = landmarks
        
        # Include joints if requested
        if return_joints:
            output['joints'] = joints
        
        return output
    
    def get_num_verts(self) -> int:
        """Return number of vertices in the mesh."""
        return self.v_template.shape[0]
    
    def get_num_faces(self) -> int:
        """Return number of faces in the mesh."""
        return self.faces.shape[0]
    
    def get_neutral_mesh(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get neutral mesh (no shape/expression/pose deformations).
        
        Args:
            batch_size: Number of meshes to generate
            
        Returns:
            (batch_size, V, 3) neutral mesh vertices
        """
        device = self.v_template.device
        return self.v_template.unsqueeze(0).expand(batch_size, -1, -1).clone()
