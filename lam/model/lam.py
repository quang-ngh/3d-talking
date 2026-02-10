"""
LAM Model - Large Avatar Model

Refactored version with optional full transformer (SD3/cogvideo-style)
from LAM. Uses real TransformerDecoder when diffusers is available.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..encoder import ImageEncoder, DINOv2Encoder, DINOv2Config
from ..gaussian import GaussianModel
from ..flame import FlameHead

# Optional: real LAM transformer (requires diffusers)
try:
    from .transformer import TransformerDecoder
    _HAS_TRANSFORMER = True
except Exception:
    TransformerDecoder = None
    _HAS_TRANSFORMER = False


@dataclass
class LAMConfig:
    """Configuration for LAM model."""
    
    # Encoder config
    encoder_type: str = "dinov2"
    encoder_freeze: bool = True
    encoder_feat_dim: int = 1024
    
    # Transformer config
    transformer_dim: int = 1024
    transformer_layers: int = 16
    transformer_heads: int = 16
    transformer_type: str = "sd3_cond"  # sd3_cond, cogvideo_cond, basic, cond
    
    # Gaussian config
    num_gaussians: int = 20000
    sh_degree: int = 3
    
    # FLAME config
    shape_param_dim: int = 100
    expr_param_dim: int = 50
    
    # Use real TransformerDecoder (from LAM) when True and available
    use_real_transformer: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LAMModel(nn.Module):
    """
    LAM: Large Avatar Model
    
    Architecture:
        Image → DINOv2 Encoder → Transformer → Gaussian Parameters
                                            ↓
                              FLAME (for structure) → Rendering
    
    This is a simplified interface. Full implementation requires:
    1. TransformerDecoder from lam.model.transformer
    2. GS3DRenderer from lam.rendering (same as LAM)
    3. Model weights from trained LAM
    
    Example:
        >>> config = LAMConfig()
        >>> lam = LAMModel(config)
        >>> gaussians = lam.encode_image(image)
        >>> video = lam.animate(gaussians, flame_params)
    
    Integration Guide:
        To use full LAM model from the original repo:
        
        1. Copy transformer:
           LAM/lam/models/transformer.py → lam/model/transformer.py
           
        2. Full renderer: lam.rendering.GS3DRenderer (already integrated).
        3. For full LAM pipeline use lam.model.full_lam.FullLAMModel.
           
        4. Load pretrained weights:
           lam.load_state_dict(torch.load("weights.safetensors"))
    """
    
    def __init__(self, config: LAMConfig):
        super().__init__()
        self.config = config
        
        # Image encoder (real DINOv2 fusion when available)
        print("Initializing image encoder...")
        if config.encoder_type == "dinov2":
            encoder_config = DINOv2Config(
                encoder_feat_dim=config.encoder_feat_dim,
                frozen=config.encoder_freeze,
            )
            self.encoder = DINOv2Encoder(encoder_config)
        else:
            raise ValueError(f"Unknown encoder type: {config.encoder_type}")
        
        # Transformer: use real TransformerDecoder when available else dummy
        use_real = config.use_real_transformer and _HAS_TRANSFORMER
        if use_real:
            self.transformer = TransformerDecoder(
                block_type=config.transformer_type,
                num_layers=config.transformer_layers,
                num_heads=config.transformer_heads,
                inner_dim=config.transformer_dim,
                cond_dim=config.encoder_feat_dim,
                mod_dim=None,
                gradient_checkpointing=False,
                eps=1e-6,
            )
            self._transformer_cond = True  # forward needs cond (image feats)
        else:
            if config.use_real_transformer:
                print("[WARNING] Real transformer not available (install diffusers); using dummy.")
            self.transformer = DummyTransformer(
                input_dim=config.encoder_feat_dim,
                hidden_dim=config.transformer_dim,
                num_layers=config.transformer_layers,
                num_heads=config.transformer_heads,
            )
            self._transformer_cond = False
        
        # Gaussian decoder (placeholder)
        print("[WARNING] Using dummy Gaussian decoder!")
        print("  For full LAM: copy LAM/lam/models/rendering/gs_renderer.py")
        self.gaussian_decoder = DummyGaussianDecoder(
            input_dim=config.transformer_dim,
            num_gaussians=config.num_gaussians,
        )
        
        print(f"\nLAM Model initialized (simplified version)")
        print(f"  Encoder: {config.encoder_type}")
        print(f"  Transformer: {config.transformer_layers} layers")
        print(f"  Output: {config.num_gaussians} Gaussians")
    
    def encode_image(
        self,
        image: torch.Tensor,
        flame_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> GaussianModel:
        """
        Encode image to 3D Gaussians.
        
        Args:
            image: (B, 3, H, W) input image
            flame_params: Optional FLAME parameters for structure
            
        Returns:
            GaussianModel representing the avatar
        """
        # 1. Encode image
        encoder_output = self.encoder(image)
        image_features = encoder_output['features']  # (B, C, H', W')
        
        # Flatten spatial dimensions for transformer
        B, C, H, W = image_features.shape
        image_features_flat = image_features.view(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        
        # 2. Transformer processing (query tokens vs image as condition)
        if getattr(self, "_transformer_cond", False):
            # Real LAM-style: query_points would come from FLAME; here we use a learned query
            # For encode_image we use image feats as both query and cond (simplified)
            transformer_output = self.transformer(
                image_features_flat, cond=image_features_flat, mod=None
            )  # (B, H*W, D)
            # Pool to fixed number of tokens for Gaussian decoder (e.g. first N or mean)
            N = self.config.num_gaussians
            if transformer_output.shape[1] >= N:
                transformer_output = transformer_output[:, :N]  # (B, N, D)
            else:
                padding = transformer_output[:, -1:].expand(-1, N - transformer_output.shape[1], -1)
                transformer_output = torch.cat([transformer_output, padding], dim=1)
        else:
            transformer_output = self.transformer(image_features_flat)  # (B, H*W, D)
            N = self.config.num_gaussians
            if transformer_output.shape[1] >= N:
                transformer_output = transformer_output[:, :N]
            else:
                padding = transformer_output[:, -1:].expand(-1, N - transformer_output.shape[1], -1)
                transformer_output = torch.cat([transformer_output, padding], dim=1)
        
        # 3. Decode to Gaussian parameters
        gaussian_params = self.gaussian_decoder(transformer_output)
        
        # 4. Create Gaussian model
        from ..gaussian import GaussianConfig
        gaussian_config = GaussianConfig(num_gaussians=self.config.num_gaussians)
        gaussians = GaussianModel(gaussian_config)
        gaussians.from_dict(gaussian_params)
        
        return gaussians
    
    def animate(
        self,
        gaussians: GaussianModel,
        flame_params: Dict[str, torch.Tensor],
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        """
        Animate Gaussians with FLAME parameters.
        
        Args:
            gaussians: GaussianModel to animate
            flame_params: FLAME parameters for each frame
            camera_poses: (N, 4, 4) camera poses
            camera_intrinsics: (N, 3, 3) camera intrinsics
            
        Returns:
            (N, H, W, 3) rendered video frames
        """
        # For full LAM rendering use lam.model.FullLAMModel (encoder + transformer + lam.rendering.GS3DRenderer).
        print("[WARNING] Placeholder rendering; use FullLAMModel for full LAM rendering.")
        
        N = camera_poses.shape[0]
        H, W = 512, 512
        
        # Placeholder: return dummy frames
        frames = torch.ones(N, H, W, 3, device=gaussians.xyz.device) * 0.5
        
        return frames
    
    def forward(
        self,
        image: torch.Tensor,
        flame_params: Dict[str, torch.Tensor],
        camera_poses: torch.Tensor,
        camera_intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: image → Gaussians → animated video.
        
        Args:
            image: (B, 3, H, W) input image
            flame_params: FLAME parameters
            camera_poses: (N, 4, 4) camera poses
            camera_intrinsics: (N, 3, 3) camera intrinsics
            
        Returns:
            Dictionary with rendered frames and intermediate outputs
        """
        # Encode
        gaussians = self.encode_image(image, flame_params)
        
        # Animate
        frames = self.animate(gaussians, flame_params, camera_poses, camera_intrinsics)
        
        return {
            'gaussians': gaussians,
            'frames': frames,
        }


class DummyTransformer(nn.Module):
    """
    Placeholder transformer.
    
    Replace with actual TransformerDecoder from LAM/lam/models/transformer.py
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return x


class DummyGaussianDecoder(nn.Module):
    """
    Placeholder Gaussian decoder.
    
    Replace with actual decoder from LAM/lam/models/rendering/gs_renderer.py
    """
    
    def __init__(self, input_dim, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # Simple MLPs for each Gaussian parameter
        self.xyz_head = nn.Linear(input_dim, 3)
        self.rotation_head = nn.Linear(input_dim, 4)
        self.scaling_head = nn.Linear(input_dim, 3)
        self.opacity_head = nn.Linear(input_dim, 1)
        self.color_head = nn.Linear(input_dim, 3)
    
    def forward(self, x):
        """
        Decode transformer output to Gaussian parameters.
        
        Args:
            x: (B, N, D) transformer output
            
        Returns:
            Dictionary of Gaussian parameters
        """
        B, N, D = x.shape
        
        # Decode parameters
        xyz = self.xyz_head(x)  # (B, N, 3)
        rotation = self.rotation_head(x)  # (B, N, 4)
        rotation = torch.nn.functional.normalize(rotation, dim=-1)  # Normalize quaternions
        scaling = torch.sigmoid(self.scaling_head(x)) * 0.1  # (B, N, 3)
        opacity = torch.sigmoid(self.opacity_head(x))  # (B, N, 1)
        color = torch.sigmoid(self.color_head(x))  # (B, N, 3)
        
        return {
            'xyz': xyz[0],  # Remove batch dim for GaussianModel
            'rotation': rotation[0],
            'scaling': scaling[0],
            'opacity': opacity[0],
            'features_dc': color[0],
            'features_rest': torch.zeros(N, 15, 3, device=x.device),  # Placeholder SH
        }
