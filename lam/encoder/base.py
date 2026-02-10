"""
Base Image Encoder Interface
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Tuple


class ImageEncoder(nn.Module, ABC):
    """
    Abstract base class for image encoders.
    
    All image encoders should inherit from this and implement the forward method.
    """
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encode images to feature maps.
        
        Args:
            images: (B, 3, H, W) input images
            
        Returns:
            Dictionary containing:
                - features: Multi-scale feature maps
                - (other encoder-specific outputs)
        """
        pass
    
    @abstractmethod
    def get_feature_dims(self) -> Tuple[int, ...]:
        """
        Get dimensions of output features.
        
        Returns:
            Tuple of feature dimensions
        """
        pass
