"""
FLAME: Faces Learned with an Articulated Model and Expressions

This module provides a clean, modular implementation of the FLAME parametric head model.
"""

from .model import FlameHead, FlameConfig
from .lbs import (
    linear_blend_skinning,
    blend_shapes,
    vertices_to_joints,
    vertices_to_landmarks,
    batch_rigid_transform,
)

__all__ = [
    # Main model
    "FlameHead",
    "FlameConfig",
    
    # LBS functions
    "linear_blend_skinning",
    "blend_shapes",
    "vertices_to_joints",
    "vertices_to_landmarks",
    "batch_rigid_transform",
]
