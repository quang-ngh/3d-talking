"""
DINOv2 Encoder Wrapper

Uses real Dinov2FusionWrapper from LAM when available (lam/encoder/dinov2/ + dinov2_fusion_wrapper.py).
Otherwise falls back to a placeholder conv encoder.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataclasses import dataclass

from .base import ImageEncoder

try:
    from .dinov2_fusion_wrapper import Dinov2FusionWrapper
    _HAS_DINOV2 = True
except Exception:
    Dinov2FusionWrapper = None
    _HAS_DINOV2 = False


# LAM fusion wrapper model names
DINOV2_MODEL_NAMES = ("dinov2_vits14_reg", "dinov2_vitb14_reg", "dinov2_vitl14_reg", "dinov2_vitg14_reg")


@dataclass
class DINOv2Config:
    """Configuration for DINOv2 encoder."""
    model_name: str = "dinov2_vitl14_reg"  # LAM uses _reg variants with fusion
    pretrained: bool = True
    frozen: bool = True
    encoder_feat_dim: int = 1024  # fusion head output dim (must match LAM config)


class DINOv2Encoder(ImageEncoder):
    """
    DINOv2 Vision Transformer Encoder with fusion head (LAM-style).
    Uses Dinov2FusionWrapper when available; otherwise placeholder.
    Output features shape: (B, C, H', W') for compatibility with LAM pipeline.
    """
    
    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config
        self._patch_size = 14  # DINOv2 ViT-14
        
        if _HAS_DINOV2 and Dinov2FusionWrapper is not None:
            self._real_encoder = Dinov2FusionWrapper(
                model_name=config.model_name,
                modulation_dim=None,
                freeze=config.frozen,
                encoder_feat_dim=config.encoder_feat_dim,
            )
            self.feature_dim = config.encoder_feat_dim
            self._placeholder = False
        else:
            self._real_encoder = None
            self.feature_dim = 1024
            self._placeholder = True
            self.dummy_encoder = nn.Sequential(
                nn.Conv2d(3, 256, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, self.feature_dim, 3, stride=2, padding=1),
                nn.ReLU(),
            )
            if config.frozen:
                for p in self.dummy_encoder.parameters():
                    p.requires_grad = False
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (B, 3, H, W) in [0, 1], will be resized to multiple of patch_size if needed.
        Returns:
            'features': (B, C, H', W') for LAM compatibility
        """
        if self._placeholder:
            features = self.dummy_encoder(images)
            return {"features": features, "feature_maps": [features]}
        
        B, C, H, W = images.shape
        p = self._patch_size
        if H % p != 0 or W % p != 0:
            H_, W_ = (H // p) * p, (W // p) * p
            images = torch.nn.functional.interpolate(images, size=(H_, W_), mode="bilinear", align_corners=False)
        # (B, L, D)
        feats = self._real_encoder(images)
        L, D = feats.shape[1], feats.shape[2]
        patch_h, patch_w = images.shape[2] // p, images.shape[3] // p
        # (B, L, D) -> (B, patch_h, patch_w, D) -> (B, D, patch_h, patch_w)
        features = feats.reshape(B, patch_h, patch_w, D).permute(0, 3, 1, 2)
        return {"features": features, "feature_maps": [features]}
    
    def get_feature_dims(self) -> Tuple[int, ...]:
        return (self.feature_dim,)
