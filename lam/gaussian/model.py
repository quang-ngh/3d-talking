"""
3D Gaussian Model - Represents geometry as Gaussian primitives.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class GaussianConfig:
    """Configuration for Gaussian model."""
    num_gaussians: int = 20000
    sh_degree: int = 3  # Spherical harmonics degree
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32


class GaussianModel:
    """
    3D Gaussian Splatting representation.
    
    Each Gaussian is defined by:
        - Position (xyz): Center of the Gaussian
        - Rotation (quaternion): Orientation
        - Scaling: Size along each axis
        - Opacity: Transparency
        - SH coefficients: View-dependent color
    
    Example:
        >>> config = GaussianConfig(num_gaussians=10000)
        >>> gaussians = GaussianModel(config)
        >>> gaussians.initialize_random()
    """
    
    def __init__(self, config: GaussianConfig):
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        
        # Gaussian parameters (initialized to None, call initialize_*() to set)
        self._xyz = None            # (N, 3) positions
        self._rotation = None       # (N, 4) quaternions
        self._scaling = None        # (N, 3) scales
        self._opacity = None        # (N, 1) opacities
        self._features_dc = None    # (N, 3) base color (SH degree 0)
        self._features_rest = None  # (N, K, 3) higher-order SH
    
    def initialize_random(self, bounds: float = 1.0):
        """
        Initialize Gaussians with random parameters.
        
        Args:
            bounds: Spatial bounds for initialization
        """
        N = self.config.num_gaussians
        
        # Random positions
        self._xyz = (torch.rand(N, 3, device=self.device, dtype=self.dtype) - 0.5) * 2 * bounds
        
        # Identity rotations (quaternion [1, 0, 0, 0])
        self._rotation = torch.zeros(N, 4, device=self.device, dtype=self.dtype)
        self._rotation[:, 0] = 1.0
        
        # Small random scales
        self._scaling = torch.rand(N, 3, device=self.device, dtype=self.dtype) * 0.1
        
        # Medium opacity
        self._opacity = torch.ones(N, 1, device=self.device, dtype=self.dtype) * 0.5
        
        # Random colors
        self._features_dc = torch.rand(N, 3, device=self.device, dtype=self.dtype)
        
        # Zero higher-order SH
        num_sh_coef = (self.config.sh_degree + 1) ** 2 - 1
        self._features_rest = torch.zeros(N, num_sh_coef, 3, device=self.device, dtype=self.dtype)
    
    def initialize_from_points(self, points: torch.Tensor, colors: Optional[torch.Tensor] = None):
        """
        Initialize Gaussians from point cloud.
        
        Args:
            points: (N, 3) point positions
            colors: Optional (N, 3) RGB colors
        """
        N = points.shape[0]
        
        self._xyz = points.to(device=self.device, dtype=self.dtype)
        
        # Identity rotations
        self._rotation = torch.zeros(N, 4, device=self.device, dtype=self.dtype)
        self._rotation[:, 0] = 1.0
        
        # Uniform small scales
        self._scaling = torch.ones(N, 3, device=self.device, dtype=self.dtype) * 0.05
        
        # Full opacity
        self._opacity = torch.ones(N, 1, device=self.device, dtype=self.dtype)
        
        # Set colors
        if colors is not None:
            self._features_dc = colors.to(device=self.device, dtype=self.dtype)
        else:
            self._features_dc = torch.ones(N, 3, device=self.device, dtype=self.dtype) * 0.5
        
        # Zero higher-order SH
        num_sh_coef = (self.config.sh_degree + 1) ** 2 - 1
        self._features_rest = torch.zeros(N, num_sh_coef, 3, device=self.device, dtype=self.dtype)
    
    @property
    def xyz(self) -> torch.Tensor:
        """Get Gaussian positions."""
        return self._xyz
    
    @property
    def rotation(self) -> torch.Tensor:
        """Get Gaussian rotations (quaternions)."""
        return self._rotation
    
    @property
    def scaling(self) -> torch.Tensor:
        """Get Gaussian scales."""
        return self._scaling
    
    @property
    def opacity(self) -> torch.Tensor:
        """Get Gaussian opacities."""
        return self._opacity
    
    @property
    def features_dc(self) -> torch.Tensor:
        """Get base SH coefficients (degree 0)."""
        return self._features_dc
    
    @property
    def features_rest(self) -> torch.Tensor:
        """Get higher-order SH coefficients."""
        return self._features_rest
    
    def get_num_gaussians(self) -> int:
        """Return number of Gaussians."""
        return self._xyz.shape[0] if self._xyz is not None else 0
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Export as dictionary."""
        return {
            'xyz': self._xyz,
            'rotation': self._rotation,
            'scaling': self._scaling,
            'opacity': self._opacity,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
        }
    
    def from_dict(self, data: Dict[str, torch.Tensor]):
        """Load from dictionary."""
        self._xyz = data['xyz'].to(device=self.device, dtype=self.dtype)
        self._rotation = data['rotation'].to(device=self.device, dtype=self.dtype)
        self._scaling = data['scaling'].to(device=self.device, dtype=self.dtype)
        self._opacity = data['opacity'].to(device=self.device, dtype=self.dtype)
        self._features_dc = data['features_dc'].to(device=self.device, dtype=self.dtype)
        self._features_rest = data['features_rest'].to(device=self.device, dtype=self.dtype)
