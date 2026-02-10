"""
Animation Pipeline - High-level interface for animating avatars.

Supports:
- FLAME-only: drive FLAME mesh with motion sequence; renders via PyTorch3D mesh
  rasterization when available (otherwise placeholder).
- Optional LAM: when lam_config and reference_image are set, uses LAMModel to encode
  image to 3D Gaussians and animate with FLAME params (LAM rendering may be
  placeholder unless FullLAMModel/GS3DRenderer is used).
"""

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, Dict, TYPE_CHECKING

from ..flame import FlameHead, FlameConfig
from ..gaussian import GaussianModel, GaussianRenderer, GaussianConfig, RenderConfig
from ..io import load_image, save_video, load_flame_params, MotionSequence, flame_params_frame_first_to_sequence

if TYPE_CHECKING:
    from ..model import LAMModel, LAMConfig


@dataclass
class AnimationConfig:
    """Configuration for animation pipeline."""
    
    # FLAME configuration
    flame_config: FlameConfig
    
    # Gaussian configuration
    gaussian_config: Optional[GaussianConfig] = None
    render_config: Optional[RenderConfig] = None
    
    # Optional LAM model for image-driven animation (encode image -> Gaussians -> animate)
    lam_config: Optional["LAMConfig"] = None
    
    # Output settings
    output_fps: int = 30
    output_size: tuple = (512, 512)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AnimationPipeline:
    """
    High-level pipeline for avatar animation.
    
    This pipeline handles the complete workflow:
    1. Load reference image
    2. Initialize FLAME model
    3. Generate/load motion sequence
    4. Animate FLAME mesh with motion
    5. Render frames
    6. Export video
    
    Example:
        >>> config = AnimationConfig(
        ...     flame_config=FlameConfig(...),
        ... )
        >>> pipeline = AnimationPipeline(config)
        >>> video = pipeline.animate(
        ...     reference_image="path/to/image.jpg",
        ...     motion_sequence="path/to/motion.json",
        ...     output_path="output.mp4"
        ... )
    """
    
    def __init__(self, config: AnimationConfig):
        self.config = config
        self.device = config.device
        
        # Initialize FLAME model
        print("Initializing FLAME model...")
        self.flame = FlameHead(config.flame_config).to(self.device)
        
        # Optional LAM model (image -> Gaussians)
        self.lam_model = None
        if config.lam_config is not None:
            try:
                from ..model import LAMModel
                self.lam_model = LAMModel(config.lam_config).to(self.device)
                print("LAM model loaded for image-driven animation.")
            except Exception as e:
                print(f"[WARNING] Could not load LAM model: {e}")
        
        # Initialize Gaussian renderer if config provided
        if config.gaussian_config is not None:
            print("Initializing Gaussian renderer...")
            self.gaussian_config = config.gaussian_config
            self.render_config = config.render_config or RenderConfig()
            self.renderer = GaussianRenderer(self.render_config)
        else:
            self.gaussian_config = None
            self.renderer = None
    
    def animate(
        self,
        motion_sequence: Union[str, Path, MotionSequence, Dict[str, torch.Tensor]],
        output_path: Optional[Union[str, Path]] = None,
        reference_image: Optional[Union[str, Path]] = None,
    ) -> torch.Tensor:
        """
        Animate avatar with motion sequence.
        
        Args:
            motion_sequence: FLAME parameters (file path, MotionSequence, or dict)
            output_path: Optional path to save video
            reference_image: Optional reference image for texture/identity
            
        Returns:
            (N, H, W, 3) rendered video frames
        """
        # Load motion sequence (supports single .pt with frame-first [N, dim] params, no motion dir)
        cam_params_from_file = None  # [N, 3] = (scale, tx, ty) from .pt if present
        if isinstance(motion_sequence, (str, Path)):
            print(f"Loading motion sequence from {motion_sequence}...")
            motion_params = load_flame_params(motion_sequence, device=self.device)
            if isinstance(motion_params, dict):
                cam_params_from_file = motion_params.get("cam_params")
                if cam_params_from_file is not None:
                    cam_params_from_file = torch.as_tensor(cam_params_from_file, device=self.device)
                    if cam_params_from_file.dim() == 1:
                        cam_params_from_file = cam_params_from_file.unsqueeze(0)
                motion_seq = self._dict_to_motion_sequence(motion_params)
            else:
                motion_seq = motion_params
        elif isinstance(motion_sequence, dict):
            cam_params_from_file = motion_sequence.get("cam_params")
            if cam_params_from_file is not None:
                cam_params_from_file = torch.as_tensor(cam_params_from_file, device=self.device)
                if cam_params_from_file.dim() == 1:
                    cam_params_from_file = cam_params_from_file.unsqueeze(0)
            motion_seq = self._dict_to_motion_sequence(motion_sequence)
        else:
            motion_seq = motion_sequence
        
        num_frames = motion_seq.get_num_frames()
        print(f"Animating {num_frames} frames...")
        
        # If we have LAM and a reference image, encode to Gaussians then animate (optional path)
        if reference_image is not None and self.lam_model is not None:
            try:
                ref_tensor = load_image(reference_image, as_tensor=True, device=self.device)
                if ref_tensor.dim() == 3:
                    ref_tensor = ref_tensor.unsqueeze(0)
                ref_tensor = ref_tensor.permute(0, 3, 1, 2)
                gaussians = self.lam_model.encode_image(ref_tensor)
                # Build per-frame cameras for LAM.animate (frontal placeholder)
                c2w = torch.eye(4, device=self.device).unsqueeze(0).expand(num_frames, -1, -1)
                c2w[:, 2, 3] = 2.0
                intr = torch.eye(3, device=self.device).unsqueeze(0).expand(num_frames, -1, -1)
                intr[:, 0, 0] = intr[:, 1, 1] = 512.0
                intr[:, 0, 2] = intr[:, 1, 2] = 256.0
                flame_dict = {
                    "expr": motion_seq.expr_params,
                    "pose_params": motion_seq.pose_params,
                    "jaw_params": motion_seq.jaw_params,
                    "neck_params": motion_seq.neck_params if motion_seq.neck_params is not None else torch.zeros(1, 3, num_frames, device=self.device),
                    "eye_params": motion_seq.eye_params if motion_seq.eye_params is not None else torch.zeros(1, 6, num_frames, device=self.device),
                    "translation": motion_seq.translation if motion_seq.translation is not None else torch.zeros(1, 3, num_frames, device=self.device),
                }
                frames = self.lam_model.animate(gaussians, flame_dict, c2w, intr)
                if output_path is not None:
                    save_video(frames, output_path, fps=self.config.output_fps)
                return frames
            except Exception as e:
                print(f"[WARNING] LAM image-driven path failed: {e}; falling back to FLAME-only.")
        
        # FLAME-only: generate mesh per frame
        vertices_sequence = []
        with torch.no_grad():
            for i in range(num_frames):
                frame_params = motion_seq.get_frame(i)
                output = self.flame.forward(**frame_params, return_landmarks=False)
                vertices_sequence.append(output['vertices'])
        vertices_sequence = torch.stack(vertices_sequence, dim=0)
        print(f"Generated {num_frames} FLAME meshes")
        
        # Render: use cam_params [N, 3] from .pt if available, else default camera
        h, w = self.config.output_size
        frames = self._render_flame_sequence(vertices_sequence, h, w, cam_params=cam_params_from_file)
        
        # Save video if path provided
        if output_path is not None:
            print(f"Saving video to {output_path}...")
            save_video(frames, output_path, fps=self.config.output_fps)
            print("Video saved!")
        
        return frames
    
    def _render_flame_sequence(
        self,
        vertices_sequence: torch.Tensor,
        height: int,
        width: int,
        cam_params: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Render FLAME mesh sequence to (N, H, W, 3) frames in [0, 1].
        Uses PyTorch3D mesh rasterization when available; otherwise placeholder.
        cam_params: optional [N, 3] from .pt file (scale, tx, ty) per frame; same convention
                    as batch_orth_proj / SMIRK. If provided, vertices are transformed before
                    rendering and camera is fixed.
        """
        if vertices_sequence.shape[1] == 1:
            vertices_sequence = vertices_sequence.squeeze(1)
        N, V, _ = vertices_sequence.shape
        device = vertices_sequence.device
        h, w = int(height), int(width)
        bg_val = 0.95  # light gray background

        try:
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer import (
                PerspectiveCameras,
                RasterizationSettings,
                MeshRasterizer,
                MeshRendererWithFragments,
                SoftPhongShader,
                PointLights,
                Materials,
                TexturesVertex,
            )
        except ImportError:
            return torch.ones(N, h, w, 3, device=device) * 0.5

        try:
            faces = self.flame.faces.to(device)  # (F, 3)
            verts = vertices_sequence  # (N, V, 3)

            # Apply cam_params [N, 3] = (scale, tx, ty) from .pt if provided (same as batch_orth_proj)
            if cam_params is not None and cam_params.shape[0] == N and cam_params.shape[1] >= 3:
                cam_params = cam_params.to(device).to(verts.dtype)
                scale = cam_params[:, 0:1].view(N, 1, 1)  # (N, 1, 1)
                txty = cam_params[:, 1:3].view(N, 1, 2)   # (N, 1, 2)
                verts = torch.cat([verts[:, :, :2] + txty, verts[:, :, 2:3]], dim=-1)
                verts = scale * verts

            # Neutral gray vertex colors
            verts_rgb = torch.ones(N, V, 3, device=device, dtype=verts.dtype) * 0.7
            faces_batch = faces.unsqueeze(0).expand(N, -1, -1)  # (N, F, 3)
            meshes = Meshes(verts=verts, faces=faces_batch, textures=TexturesVertex(verts_rgb))

            # Single fixed camera: at (0, 0, 1) looking at origin (mesh after cam_params is centered).
            # With in_ndc=False, fx/fy and cx/cy are in PIXEL space (not normalized).
            R = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)
            T = torch.zeros(N, 3, device=device, dtype=torch.float32)
            T[:, 2] = -1.0  # camera at (0, 0, 1) in world
            # fx = fy = float(max(h, w))  # focal length in pixels
            fx, fy = 1.2
            cx, cy = w / 2.0, h / 2.0   # principal point at image center (pixels)
            focal_len = torch.tensor([[fx, fy]], device=device, dtype=torch.float32)
            principal = torch.tensor([[cx, cy]], device=device, dtype=torch.float32)
            cameras = PerspectiveCameras(
                # R=R,
                # T=T,
                focal_length=focal_len,
                principal_point=principal,
                device=device,
                in_ndc=False,
                image_size=torch.tensor([[h, w]], device=device),
            )
            raster_settings = RasterizationSettings(
                image_size=(h, w),
                blur_radius=0.0,
                faces_per_pixel=1,
            )
            rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
            lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])
            materials = Materials(device=device, specular_color=[[0.0, 0.0, 0.0]], shininess=0.0)
            shader = SoftPhongShader(device=device, cameras=cameras, lights=lights)
            renderer = MeshRendererWithFragments(rasterizer=rasterizer, shader=shader)

            with torch.no_grad():
                images, fragments = renderer(meshes, materials=materials)
            rgb = images[..., :3]
            alpha = (fragments.zbuf[..., 0] >= 0).float().unsqueeze(-1)
            frames = rgb * alpha + (1.0 - alpha) * bg_val
            return frames.clamp(0.0, 1.0)
        except Exception as e:
            print(f"[WARNING] Mesh rendering failed ({e}); using placeholder.")
            return torch.ones(N, h, w, 3, device=device) * 0.5

    def _dict_to_motion_sequence(self, params: Dict[str, torch.Tensor]) -> MotionSequence:
        """Convert parameter dictionary to MotionSequence. Handles frame-first [N, dim] from single .pt."""
        n_shape = getattr(self.config.flame_config, "n_shape", 100)
        n_expr = getattr(self.config.flame_config, "n_expr", 50)
        fps = getattr(self.config, "output_fps", 30)
        try:
            return flame_params_frame_first_to_sequence(
                params, device=self.device, n_shape=n_shape, n_expr=n_expr, fps=fps
            )
        except (KeyError, ValueError):
            pass
        # Fallback: already (1, dim, N) and pipeline key names
        return MotionSequence(
            shape_params=params.get("shape_params", params.get("betas")),
            expr_params=params["expr_params"],
            pose_params=params["pose_params"],
            jaw_params=params["jaw_params"],
            neck_params=params.get("neck_params"),
            eye_params=params.get("eye_params"),
            translation=params.get("translation"),
            fps=params.get("fps", 30),
        )
    
    def generate_neutral_animation(
        self,
        num_frames: int = 30,
        jaw_open_amount: float = 0.0
    ) -> MotionSequence:
        """
        Generate a neutral animation sequence (for testing).
        
        Args:
            num_frames: Number of frames to generate
            jaw_open_amount: Amount of jaw opening (0 = closed, 1 = open)
            
        Returns:
            MotionSequence with neutral animation
        """
        batch_size = 1
        
        # Create basic parameters
        shape_params = torch.zeros(batch_size, self.config.flame_config.n_shape, device=self.device)
        expr_params = torch.zeros(batch_size, self.config.flame_config.n_expr, num_frames, device=self.device)
        pose_params = torch.zeros(batch_size, 3, num_frames, device=self.device)
        jaw_params = torch.zeros(batch_size, 3, num_frames, device=self.device)
        
        # Animate jaw opening
        if jaw_open_amount > 0:
            jaw_params[:, 0, :] = jaw_open_amount
        
        return MotionSequence(
            shape_params=shape_params,
            expr_params=expr_params,
            pose_params=pose_params,
            jaw_params=jaw_params,
            fps=self.config.output_fps
        )
