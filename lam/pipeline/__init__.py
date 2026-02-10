"""
High-level Pipeline Module

Provides end-to-end pipelines for common LAM workflows.
"""

from .animation import AnimationPipeline, AnimationConfig

__all__ = [
    "AnimationPipeline",
    "AnimationConfig",
]
