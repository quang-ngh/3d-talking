#!/usr/bin/env python3
"""
Run LAM animation: single image + driving FLAME params → output video.

Supports:
  - Single .pt file with all FLAME params for a video (no motion dir needed).
    Example: one .pt with expr [500, 50], pose [500, 3], jaw [500, 3], etc. (frame-first).
  - Key names: pipeline (shape_params, expr_params, pose_params, jaw_params, ...)
    or LAM/tracker (betas, expr, rotation, jaw_pose, neck_pose, eyes_pose, trans).

Usage:
  python run_animate_lam.py --image ref.png --motion flame.pt --output out.mp4
  python run_animate_lam.py --motion flame.pt --output out.mp4
"""

import argparse
import sys
from pathlib import Path

# Repo root: parent of scripts/
REPO_ROOT = Path(__file__).resolve().parent
LAM_ROOT = REPO_ROOT / "lam"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default FLAME assets (same as test_pipeline.py)
DEFAULT_FLAME_MODEL = REPO_ROOT / "assets" / "FLAME2023" / "flame2023.pkl"
DEFAULT_FLAME_LMK = REPO_ROOT / "assets" / "landmark_embedding.npy"


def _ensure_motion_sequence_shape(params: dict, device: str, n_shape: int = 100, n_expr: int = 50):
    """
    Convert track_flame_params-style dict [N, dim] to MotionSequence-style (1, dim, N).
    Truncate/pad shape and expr to n_shape, n_expr. Pad eye_params 2 -> 6 if needed.
    """
    import torch

    # Infer number of frames from first available sequence
    N = None
    for k in ("expr_params", "pose_params", "jaw_params", "neck_params", "eye_params"):
        if k in params and params[k] is not None:
            t = params[k]
            if hasattr(t, "shape"):
                N = t.shape[0] if t.dim() >= 1 else None
            if N is not None:
                break
    if N is None:
        raise ValueError(
            "No time-varying FLAME params found. Need at least one of: expr_params, pose_params, jaw_params (each shape [N, dim])."
        )

    out = {}

    # shape_params: use first frame, constant across time -> (1, n_shape)
    if "shape_params" in params and params["shape_params"] is not None:
        s = params["shape_params"]
        if isinstance(s, torch.Tensor):
            s = s.to(device)
        else:
            s = torch.as_tensor(s, device=device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if s.shape[0] > 1:
            s = s[0:1]
        if s.shape[1] > n_shape:
            s = s[:, :n_shape]
        elif s.shape[1] < n_shape:
            s = torch.nn.functional.pad(s, (0, n_shape - s.shape[1]))
        out["shape_params"] = s
    else:
        out["shape_params"] = torch.zeros(1, n_shape, device=device)

    for key, target_dim in [
        ("expr_params", n_expr),
        ("pose_params", 3),
        ("jaw_params", 3),
        ("neck_params", 3),
        ("eye_params", 6),
        ("translation", 3),
    ]:
        if key not in params or params[key] is None:
            if key in ("neck_params", "eye_params", "translation"):
                out[key] = torch.zeros(1, target_dim, N, device=device)
            else:
                raise ValueError(f"Missing required key: {key}")
            continue
        t = params[key]
        if isinstance(t, torch.Tensor):
            t = t.to(device)
        else:
            t = torch.as_tensor(t, device=device)
        if t.dim() == 2:
            # (N, dim) -> (1, dim, N)
            t = t.unsqueeze(0).permute(0, 2, 1)
        elif t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(0).expand(1, target_dim, N)
        if t.shape[-1] != N:
            t = t[..., :N] if t.shape[-1] > N else torch.nn.functional.pad(t, (0, N - t.shape[-1]))
        if t.shape[1] > target_dim:
            t = t[:, :target_dim]
        elif t.shape[1] < target_dim:
            t = torch.nn.functional.pad(t, (0, 0, 0, target_dim - t.shape[1]))
        out[key] = t

    out["fps"] = params.get("fps", 30)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Run LAM animation: image + driving FLAME params → video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", type=Path, default=None, help="Path to reference image (optional)")
    parser.add_argument(
        "--motion",
        type=Path,
        required=True,
        help="Path to FLAME params: single .pt/.npz with frame-first [N, dim] (e.g. expr [500,50], pose [500,3]) or pipeline-style keys",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/lam_animate_out.mp4"),
        help="Output video path",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output FPS")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="Output frame size (default: 512 512)",
    )
    parser.add_argument(
        "--flame-model",
        type=Path,
        default=None,
        help="Path to FLAME model pkl (default: repo assets/FLAME2023/flame2023.pkl)",
    )
    parser.add_argument(
        "--flame-lmk",
        type=Path,
        default=None,
        help="Path to FLAME landmark embedding npy (default: repo assets/landmark_embedding.npy)",
    )
    parser.add_argument("--n-shape", type=int, default=100, help="FLAME shape dimension (default: 100)")
    parser.add_argument("--n-expr", type=int, default=50, help="FLAME expression dimension (default: 50)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu); default: cuda if available",
    )
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    flame_model = args.flame_model or DEFAULT_FLAME_MODEL
    flame_lmk = args.flame_lmk or DEFAULT_FLAME_LMK
    if not flame_model.exists():
        print(f"FLAME model not found: {flame_model}", file=sys.stderr)
        print("Set --flame-model or place flame2023.pkl at repo assets/FLAME2023/", file=sys.stderr)
        sys.exit(1)
    if not flame_lmk.exists():
        print(f"FLAME landmark embedding not found: {flame_lmk}", file=sys.stderr)
        print("Set --flame-lmk or place landmark_embedding.npy at repo assets/", file=sys.stderr)
        sys.exit(1)

    if not args.motion.exists():
        print(f"Motion file not found: {args.motion}", file=sys.stderr)
        sys.exit(1)

    from lam.flame import FlameConfig
    from lam.pipeline import AnimationPipeline, AnimationConfig
    from lam.io import load_flame_params, MotionSequence

    flame_config = FlameConfig(
        flame_model_path=str(flame_model),
        flame_lmk_embedding_path=str(flame_lmk),
        n_shape=args.n_shape,
        n_expr=args.n_expr,
        device=device,
    )
    config = AnimationConfig(
        flame_config=flame_config,
        output_fps=args.fps,
        output_size=tuple(args.size),
        device=device,
    )
    pipeline = AnimationPipeline(config)

    # Load motion: single .pt with [N, dim] or (1, dim, N) both supported
    motion_params = load_flame_params(args.motion, device=device)
    if isinstance(motion_params, dict):
        motion_seq = pipeline._dict_to_motion_sequence(motion_params)
    else:
        motion_seq = motion_params

    ref_image = args.image if args.image and args.image.exists() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Running LAM animation (image + driving FLAME params)")
    if ref_image:
        print(f"  Reference image: {ref_image}")
    print(f"  Motion:          {args.motion} ({motion_seq.get_num_frames()} frames)")
    print(f"  Output:          {args.output}")

    frames = pipeline.animate(
        motion_sequence=motion_seq,
        output_path=args.output,
        reference_image=ref_image,
    )
    print(f"  Rendered:        {frames.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
