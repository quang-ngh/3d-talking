"""
Camera-related utility functions.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def perspective_projection(
    fov_y: float,
    aspect_ratio: float,
    near: float = 0.1,
    far: float = 100.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create perspective projection matrix.
    
    Args:
        fov_y: Vertical field of view in radians
        aspect_ratio: Width / height
        near: Near clipping plane
        far: Far clipping plane
        device: Target device
        
    Returns:
        (4, 4) projection matrix
    """
    tan_half_fov = np.tan(fov_y / 2.0)
    
    proj = torch.zeros((4, 4), device=device)
    proj[0, 0] = 1.0 / (aspect_ratio * tan_half_fov)
    proj[1, 1] = 1.0 / tan_half_fov
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    
    return proj


def look_at_matrix(
    eye: torch.Tensor,
    target: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """
    Create look-at view matrix (camera-to-world).
    
    Args:
        eye: (3,) camera position
        target: (3,) look-at target position
        up: (3,) up vector
        
    Returns:
        (4, 4) camera-to-world matrix
    """
    # Compute camera coordinate frame
    z_axis = torch.nn.functional.normalize(eye - target, dim=0)  # Forward
    x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis), dim=0)  # Right
    y_axis = torch.cross(z_axis, x_axis)  # Up
    
    # Build rotation matrix
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (3, 3)
    
    # Build 4x4 transformation matrix
    matrix = torch.eye(4, device=eye.device, dtype=eye.dtype)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = eye
    
    return matrix


def get_camera_intrinsics(
    focal_length: float,
    image_size: Tuple[int, int],
    principal_point: Optional[Tuple[float, float]] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Create camera intrinsic matrix.
    
    Args:
        focal_length: Focal length in pixels
        image_size: (height, width)
        principal_point: (cx, cy), defaults to image center
        device: Target device
        
    Returns:
        (3, 3) intrinsic matrix
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]
    """
    h, w = image_size
    
    if principal_point is None:
        cx, cy = w / 2.0, h / 2.0
    else:
        cx, cy = principal_point
    
    K = torch.eye(3, device=device)
    K[0, 0] = focal_length  # fx
    K[1, 1] = focal_length  # fy
    K[0, 2] = cx
    K[1, 2] = cy
    
    return K


def get_camera_rays(
    height: int,
    width: int,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate camera rays for each pixel.
    
    Args:
        height: Image height
        width: Image width
        intrinsics: (3, 3) camera intrinsic matrix
        c2w: (4, 4) camera-to-world transformation
        
    Returns:
        rays_o: (H, W, 3) ray origins
        rays_d: (H, W, 3) ray directions (normalized)
    """
    device = intrinsics.device
    
    # Create pixel grid
    i, j = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing='ij'
    )
    
    # Pixel coordinates to camera coordinates
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    dirs = torch.stack([
        (j - cx) / fx,
        (i - cy) / fy,
        torch.ones_like(i)
    ], dim=-1)  # (H, W, 3)
    
    # Transform to world coordinates
    rays_d = torch.sum(
        dirs[..., None, :] * c2w[:3, :3],
        dim=-1
    )  # (H, W, 3)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    
    rays_o = c2w[:3, 3].expand(height, width, 3)  # (H, W, 3)
    
    return rays_o, rays_d


def project_points(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points: (..., 3) 3D points
        intrinsics: (3, 3) camera intrinsic matrix
        extrinsics: Optional (4, 4) world-to-camera matrix
        
    Returns:
        (..., 2) 2D image coordinates
    """
    # Transform to camera space if extrinsics provided
    if extrinsics is not None:
        # Convert to homogeneous coordinates
        ones = torch.ones_like(points[..., :1])
        points_h = torch.cat([points, ones], dim=-1)  # (..., 4)
        
        # Transform
        points_cam = torch.matmul(points_h, extrinsics.T)
        points = points_cam[..., :3]
    
    # Project to image plane
    points_2d = torch.matmul(points, intrinsics.T)  # (..., 3)
    points_2d = points_2d[..., :2] / points_2d[..., 2:3]  # Perspective divide
    
    return points_2d
