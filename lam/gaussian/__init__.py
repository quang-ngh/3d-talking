"""
3D Gaussian Splatting Module

This module provides a simplified interface for 3D Gaussian representation and rendering.
"""

from .model import GaussianModel, GaussianConfig
from .renderer import GaussianRenderer, RenderConfig

__all__ = [
    "GaussianModel",
    "GaussianConfig",
    "GaussianRenderer",
    "RenderConfig",
]
