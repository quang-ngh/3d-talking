"""
Image Encoder Module

Provides image feature extraction using pre-trained models.
Uses real DINOv2 (Dinov2FusionWrapper) when available from LAM.
"""

from .base import ImageEncoder
from .dinov2_wrapper import DINOv2Encoder, DINOv2Config

__all__ = [
    "ImageEncoder",
    "DINOv2Encoder",
    "DINOv2Config",
]

try:
    from .dinov2_fusion_wrapper import Dinov2FusionWrapper, DPTHead
    __all__.extend(["Dinov2FusionWrapper", "DPTHead"])
except Exception:
    Dinov2FusionWrapper = DPTHead = None
