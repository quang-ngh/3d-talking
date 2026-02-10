"""
Motion sequence (FLAME parameters) I/O utilities.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Union, Optional
from dataclasses import dataclass, asdict


@dataclass
class MotionSequence:
    """
    Container for FLAME parameter sequences.
    
    Attributes:
        shape_params: (B, n_shape) shape parameters (constant across frames)
        expr_params: (B, n_expr, N) expression parameters per frame
        pose_params: (B, 3, N) global head pose per frame
        neck_params: (B, 3, N) neck rotation per frame
        jaw_params: (B, 3, N) jaw rotation per frame
        eye_params: (B, 6, N) eye rotations per frame
        translation: (B, 3, N) translation per frame
        fps: Frames per second
    """
    shape_params: torch.Tensor
    expr_params: torch.Tensor
    pose_params: torch.Tensor
    jaw_params: torch.Tensor
    neck_params: Optional[torch.Tensor] = None
    eye_params: Optional[torch.Tensor] = None
    translation: Optional[torch.Tensor] = None
    fps: int = 30
    
    def get_num_frames(self) -> int:
        """Get number of frames in the sequence."""
        return self.expr_params.shape[-1]
    
    def get_frame(self, frame_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get FLAME parameters for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Dictionary of FLAME parameters for this frame
        """
        params = {
            'shape_params': self.shape_params,
            'expr_params': self.expr_params[..., frame_idx],
            'pose_params': self.pose_params[..., frame_idx],
            'jaw_params': self.jaw_params[..., frame_idx],
        }
        
        if self.neck_params is not None:
            params['neck_params'] = self.neck_params[..., frame_idx]
        if self.eye_params is not None:
            params['eye_params'] = self.eye_params[..., frame_idx]
        if self.translation is not None:
            params['translation'] = self.translation[..., frame_idx]
        
        return params


def load_flame_params(
    param_path: Union[str, Path],
    device: str = "cpu"
) -> Union[Dict[str, torch.Tensor], MotionSequence]:
    """
    Load FLAME parameters from file.
    
    Args:
        param_path: Path to parameter file (.json, .npz, or .pt)
        device: Device to load tensors to
        
    Returns:
        Dictionary of FLAME parameters or MotionSequence
    """
    param_path = Path(param_path)
    
    if not param_path.exists():
        raise FileNotFoundError(f"Parameter file not found: {param_path}")
    
    # Load based on extension
    if param_path.suffix == '.json':
        with open(param_path, 'r') as f:
            data = json.load(f)
        
        # Convert to tensors
        params = {}
        for key, value in data.items():
            if isinstance(value, (list, float, int)):
                params[key] = torch.tensor(value, device=device)
        
        return params
    
    elif param_path.suffix == '.npz':
        data = np.load(param_path)
        params = {key: torch.from_numpy(data[key]).to(device) for key in data.files}
        return params
    
    elif param_path.suffix in ['.pt', '.pth']:
        data = torch.load(param_path, map_location=device)
        return data
    
    else:
        raise ValueError(f"Unsupported file format: {param_path.suffix}")


# Optional key aliases: LAM / tracker style -> pipeline (MotionSequence) style
_FLAME_KEY_ALIASES = {
    "betas": "shape_params",
    "shape": "shape_params",
    "expr": "expr_params",
    "rotation": "pose_params",
    "root_pose": "pose_params",
    "jaw_pose": "jaw_params",
    "neck_pose": "neck_params",
    "eyes_pose": "eye_params",
    "leye_pose": "eye_params",  # will be merged with reye for eye_params (6,)
    "reye_pose": "eye_params",
    "trans": "translation",
    "translation": "translation",
}


def _normalize_flame_dict(
    params: Dict[str, Union[torch.Tensor, np.ndarray]],
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Apply key aliases and ensure tensors on device. Pipeline keys take precedence."""
    out = {}
    for k, v in params.items():
        if v is None or (isinstance(v, (torch.Tensor, np.ndarray)) and v.size == 0):
            continue
        key = _FLAME_KEY_ALIASES.get(k, k)
        if key in out:
            continue  # keep first (e.g. prefer shape_params over later betas)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        out[key] = v.to(device)
    return out


def flame_params_frame_first_to_sequence(
    params: Dict[str, torch.Tensor],
    device: str = "cpu",
    n_shape: int = 100,
    n_expr: int = 50,
    fps: int = 30,
) -> MotionSequence:
    """
    Convert a single .pt-style dict with frame-first shapes [N, dim] into a MotionSequence.
    Use when you have one file with e.g. expr [500, 50], pose [500, 3], etc. (no motion dir).
    
    Args:
        params: Dict of FLAME params. Keys can be pipeline style (shape_params, expr_params, ...)
                or LAM/tracker style (betas, expr, rotation, jaw_pose, ...).
                Time-varying tensors can be (N, dim) [frame-first] or (1, dim, N).
        device: Target device.
        n_shape: FLAME shape dimension (pad/truncate).
        n_expr: FLAME expression dimension (pad/truncate).
        fps: Frames per second for the sequence.
        
    Returns:
        MotionSequence with shapes (1, dim, N) for time-varying params.
    """
    params = _normalize_flame_dict(params, device)
    
    # Infer number of frames from first time-varying param
    N = None
    for key in ("expr_params", "pose_params", "jaw_params", "expr", "rotation", "jaw_pose"):
        if key in params:
            t = params[key]
            if t.dim() == 2:
                N = t.shape[0]
            elif t.dim() == 3:
                N = t.shape[-1]
            if N is not None:
                break
    if N is None:
        raise ValueError(
            "No time-varying FLAME params found. Need at least one of: expr_params, pose_params, "
            "jaw_params (or expr, rotation, jaw_pose) with shape [N, dim] or (1, dim, N)."
        )
    
    def ensure_sequence(t: torch.Tensor, target_dim: int, name: str) -> torch.Tensor:
        t = t.to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(0).expand(1, target_dim, N)
        elif t.dim() == 2:
            # [N, dim] -> (1, dim, N)
            if t.shape[0] == N:
                t = t.t().unsqueeze(0)  # (1, dim, N)
            else:
                # (dim, N) or (1, dim*N) - treat as (1, dim, N) if second dim matches
                t = t.unsqueeze(0)
                if t.shape[-1] == N and t.shape[1] != N:
                    pass  # (1, dim, N)
                elif t.shape[1] == N:
                    t = t.permute(0, 2, 1)  # (1, N, dim) -> (1, dim, N)
                else:
                    t = t.t().unsqueeze(0)
        elif t.dim() == 3 and t.shape[0] == 1 and t.shape[-1] == N:
            pass  # already (1, dim, N)
        if t.shape[-1] != N:
            t = t[..., :N] if t.shape[-1] > N else torch.nn.functional.pad(t, (0, N - t.shape[-1]))
        if t.shape[1] > target_dim:
            t = t[:, :target_dim]
        elif t.shape[1] < target_dim:
            t = torch.nn.functional.pad(t, (0, 0, 0, target_dim - t.shape[1]))
        return t
    
    # Shape: constant -> (1, n_shape)
    if "shape_params" in params:
        s = params["shape_params"].to(device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if s.shape[0] > 1:
            s = s[0:1]
        if s.shape[1] > n_shape:
            s = s[:, :n_shape]
        elif s.shape[1] < n_shape:
            s = torch.nn.functional.pad(s, (0, n_shape - s.shape[1]))
        shape_params = s
    else:
        shape_params = torch.zeros(1, n_shape, device=device)
    
    expr_params = ensure_sequence(
        params["expr_params"] if "expr_params" in params else torch.zeros(N, n_expr, device=device),
        n_expr, "expr",
    )
    pose_params = ensure_sequence(
        params["pose_params"] if "pose_params" in params else torch.zeros(N, 3, device=device),
        3, "pose",
    )
    jaw_params = ensure_sequence(
        params["jaw_params"] if "jaw_params" in params else torch.zeros(N, 3, device=device),
        3, "jaw",
    )
    neck_params = ensure_sequence(params["neck_params"], 3, "neck") if "neck_params" in params else None
    eye_params = ensure_sequence(params["eye_params"], 6, "eye") if "eye_params" in params else None
    translation = ensure_sequence(params["translation"], 3, "trans") if "translation" in params else None
    
    return MotionSequence(
        shape_params=shape_params,
        expr_params=expr_params,
        pose_params=pose_params,
        jaw_params=jaw_params,
        neck_params=neck_params,
        eye_params=eye_params,
        translation=translation,
        fps=int(params.get("fps", fps)),
    )


def save_flame_params(
    params: Union[Dict[str, torch.Tensor], MotionSequence],
    save_path: Union[str, Path],
    format: str = 'auto'
):
    """
    Save FLAME parameters to file.
    
    Args:
        params: FLAME parameters dictionary or MotionSequence
        save_path: Path to save file
        format: 'json', 'npz', 'pt', or 'auto' (infer from extension)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert MotionSequence to dict
    if isinstance(params, MotionSequence):
        params = asdict(params)
    
    # Infer format from extension
    if format == 'auto':
        format = save_path.suffix[1:]  # Remove leading dot
    
    # Save based on format
    if format == 'json':
        # Convert tensors to lists
        json_data = {}
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                json_data[key] = value.detach().cpu().tolist()
            else:
                json_data[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    elif format == 'npz':
        # Convert tensors to numpy
        numpy_data = {}
        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                numpy_data[key] = value.detach().cpu().numpy()
            else:
                numpy_data[key] = value
        
        np.savez(save_path, **numpy_data)
    
    elif format in ['pt', 'pth']:
        torch.save(params, save_path)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
