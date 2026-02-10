"""
Mathematical utility functions.
"""

import torch
import numpy as np
from typing import Union, Optional


def to_tensor(
    array: Union[np.ndarray, torch.Tensor, list],
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert array to PyTorch tensor.
    
    Args:
        array: Input array (numpy, torch, or list)
        dtype: Target tensor dtype
        device: Target device (None = CPU)
        
    Returns:
        PyTorch tensor
    """
    if isinstance(array, torch.Tensor):
        tensor = array.to(dtype=dtype)
    else:
        if hasattr(array, 'todense'):  # scipy sparse matrix
            array = array.todense()
        tensor = torch.tensor(np.array(array), dtype=dtype)
    
    if device is not None:
        tensor = tensor.to(device)
    
    return tensor


def to_numpy(
    tensor: Union[torch.Tensor, np.ndarray],
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Convert tensor to NumPy array.
    
    Args:
        tensor: Input tensor
        dtype: Target numpy dtype
        
    Returns:
        NumPy array
    """
    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.array(tensor)
    
    return array.astype(dtype)


def batch_rodrigues(
    rot_vecs: torch.Tensor,
    epsilon: float = 1e-8,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Convert axis-angle rotation vectors to rotation matrices using Rodrigues' formula.
    
    Args:
        rot_vecs: (N, 3) axis-angle rotation vectors
        epsilon: Small value for numerical stability
        dtype: Output tensor dtype
        
    Returns:
        (N, 3, 3) rotation matrices
        
    Formula:
        R = I + sin(θ)K + (1 - cos(θ))K²
        where θ = ||rot_vec||, K = skew_symmetric(rot_vec/θ)
    """
    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    # Compute rotation angle
    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)  # (N, 1)
    rot_dir = rot_vecs / angle  # Normalized rotation axis

    cos = torch.unsqueeze(torch.cos(angle), dim=1)  # (N, 1, 1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)  # (N, 1, 1)

    # Create skew-symmetric matrix K
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)  # Each (N, 1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1)
    K = K.view((batch_size, 3, 3))

    # Rodrigues' formula: R = I + sin(θ)K + (1 - cos(θ))K²
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    
    return rot_mat


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix.
    Alias for batch_rodrigues for clarity.
    
    Args:
        axis_angle: (N, 3) axis-angle vectors
        
    Returns:
        (N, 3, 3) rotation matrices
    """
    return batch_rodrigues(axis_angle)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        matrix: (N, 3, 3) rotation matrices
        
    Returns:
        (N, 3) axis-angle vectors
    """
    # Using PyTorch3D if available, otherwise implement manually
    try:
        from pytorch3d.transforms import matrix_to_axis_angle as pt3d_m2aa
        return pt3d_m2aa(matrix)
    except ImportError:
        # Simple implementation
        batch_size = matrix.shape[0]
        
        # Compute rotation angle from trace
        trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
        angle = torch.acos((trace - 1) / 2).unsqueeze(1)  # (N, 1)
        
        # Compute rotation axis from skew-symmetric part
        skew = (matrix - matrix.transpose(1, 2)) / 2
        axis = torch.stack([
            skew[:, 2, 1],
            skew[:, 0, 2],
            skew[:, 1, 0]
        ], dim=1)  # (N, 3)
        
        # Normalize axis
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)
        
        # Axis-angle = angle * axis
        return angle * axis


def safe_inverse(tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute safe matrix inverse with numerical stability.
    
    Args:
        tensor: (..., N, N) matrices to invert
        epsilon: Small value added to diagonal for stability
        
    Returns:
        Inverted matrices
    """
    # Add small value to diagonal for stability
    eye = torch.eye(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype)
    stabilized = tensor + epsilon * eye
    return torch.inverse(stabilized)
