"""
Run full LAM inference using the LAM repo (image + motion → video).

Requires the LAM repo at LAM_DIR with model checkpoint and motion in LAM format
(transforms.json + per-frame FLAME .npz in motion_seqs_dir).
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def run_lam_repo_inference(
    lam_dir: str | Path,
    config_path: str | Path,
    model_name: str,
    image_input: str | Path,
    motion_seqs_dir: str | Path,
    output_path: Optional[str | Path] = None,
    export_video: bool = True,
    render_fps: int = 30,
    motion_video_read_fps: int = 30,
    device: int = 0,
    extra_env: Optional[dict] = None,
) -> int:
    """
    Run LAM inference via the LAM repo's infer.lam runner.

    Args:
        lam_dir: Path to LAM repo (must contain lam/launch.py, configs, etc.).
        config_path: Path to inference config yaml (e.g. configs/inference/lam-20k-8gpu.yaml).
        model_name: Model checkpoint path (e.g. model_zoo/lam_models/.../step_045500/).
        image_input: Path to single reference image (or dir of images).
        motion_seqs_dir: Dir with transforms.json and per-frame FLAME .npz (LAM format).
        output_path: If set, LAM writes videos under exps/; this is just for your reference.
        export_video: Passed to LAM.
        render_fps: Passed to LAM.
        motion_video_read_fps: Passed to LAM.
        device: CUDA device index.
        extra_env: Extra env vars for the subprocess.

    Returns:
        Return code of the LAM process (0 = success).
    """
    lam_dir = Path(lam_dir).resolve()
    if not (lam_dir / "lam" / "launch.py").exists():
        raise FileNotFoundError(f"LAM repo not found at {lam_dir} (expected lam/launch.py)")
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (lam_dir / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    model_name = str(model_name)
    image_input = str(Path(image_input).resolve())
    motion_seqs_dir = str(Path(motion_seqs_dir).resolve())

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(lam_dir), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable,
        "-m",
        "lam.launch",
        "infer.lam",
        "--config",
        str(config_path),
        f"model_name={model_name}",
        f"image_input={image_input}",
        f"motion_seqs_dir={motion_seqs_dir}",
        f"export_video={str(export_video).lower()}",
        f"render_fps={render_fps}",
        f"motion_video_read_fps={motion_video_read_fps}",
        f"rank={device}",
        "nodes=0",
    ]
    return subprocess.run(cmd, cwd=str(lam_dir), env=env).returncode
