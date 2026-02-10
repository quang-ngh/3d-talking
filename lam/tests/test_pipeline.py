import math
import os
import sys
import torch
import pytest
from pathlib import Path

# Repo root: parent of lam
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
print(f"REPO_ROOT: {REPO_ROOT}")
LAM_ROOT = REPO_ROOT / "lam"
print(f"LAM_ROOT: {LAM_ROOT}")
FLAME_MODEL_PATH = REPO_ROOT / "assets" / "FLAME2023" / "flame2023.pkl"
FLAME_LMK_PATH = REPO_ROOT / "assets" / "landmark_embedding.npy"
if not os.path.exists(FLAME_MODEL_PATH):
    assert False, f"FLAME model not found at {FLAME_MODEL_PATH}"
if not os.path.exists(FLAME_LMK_PATH):
    assert False, f"FLAME landmark embedding not found at {FLAME_LMK_PATH}"

def _flame_assets_exist():
    return FLAME_MODEL_PATH.exists() and FLAME_LMK_PATH.exists()


def _get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----- Fixtures -----


@pytest.fixture(scope="module")
def device():
    return _get_device()


@pytest.fixture(scope="module")
def flame_config(device):
    """FlameConfig if assets exist."""
    if not _flame_assets_exist():
        pytest.skip(
            f"FLAME assets not found. Place flame2023.pkl at {FLAME_MODEL_PATH} "
            f"and landmark_embedding.npy at {FLAME_LMK_PATH}"
        )
    from lam.flame import FlameConfig
    return FlameConfig(
        flame_model_path=str(FLAME_MODEL_PATH),
        flame_lmk_embedding_path=str(FLAME_LMK_PATH),
        n_shape=100,
        n_expr=50,
        device=device,
    )


@pytest.fixture(scope="module")
def driving_sequence(device):
    """Synthetic driving FLAME sequence: (B=1, N frames)."""
    from lam.io import MotionSequence
    batch_size = 1
    num_frames = 30
    n_shape, n_expr = 100, 50

    shape_params = torch.zeros(batch_size, n_shape, device=device)
    expr_params = torch.zeros(batch_size, n_expr, num_frames, device=device)
    pose_params = torch.zeros(batch_size, 3, num_frames, device=device)
    jaw_params = torch.zeros(batch_size, 3, num_frames, device=device)

    # Slight expression and jaw over time
    for i in range(num_frames):
        t = i / max(num_frames - 1, 1)
        expr_params[:, 0, i] = 0.5 * (1 - math.cos(t * math.pi))
        jaw_params[:, 0, i] = 0.2 * t

    return MotionSequence(
        shape_params=shape_params,
        expr_params=expr_params,
        pose_params=pose_params,
        jaw_params=jaw_params,
        fps=30,
    )


# ----- Tests -----


def test_imports():
    """Pipeline and FLAME modules import cleanly."""
    from lam.flame import FlameHead, FlameConfig
    from lam.pipeline import AnimationPipeline, AnimationConfig
    from lam.io import load_image, load_flame_params, MotionSequence
    assert True


def test_flame_animate_3d_head(flame_config, driving_sequence, device):
    """
    Drive FLAME with a sequence of parameters and produce 3D head vertices per frame.
    Single identity (from shape_params) + driving FLAME sequence → animated 3D meshes.
    """
    from lam.flame import FlameHead

    flame = FlameHead(flame_config).to(device)
    num_frames = driving_sequence.get_num_frames()

    vertices_list = []
    with torch.no_grad():
        for i in range(num_frames):
            frame_params = driving_sequence.get_frame(i)
            out = flame.forward(**frame_params, return_landmarks=False)
            vertices_list.append(out["vertices"])

    vertices_sequence = torch.stack(vertices_list, dim=0)  # (N, B, V, 3)

    assert vertices_sequence.ndim == 4
    assert vertices_sequence.shape[0] == num_frames
    assert vertices_sequence.shape[1] == 1
    assert vertices_sequence.shape[3] == 3
    assert vertices_sequence.isnan().logical_not().all(), "vertices must be finite"


def test_pipeline_with_synthetic_motion(flame_config, driving_sequence, device):
    """
    Run AnimationPipeline.animate() with synthetic motion.
    Pipeline builds FLAME meshes per frame and returns (placeholder) video frames.
    """
    from lam.pipeline import AnimationPipeline, AnimationConfig

    config = AnimationConfig(
        flame_config=flame_config,
        output_fps=30,
        output_size=(512, 512),
        device=device,
    )
    pipeline = AnimationPipeline(config)

    frames = pipeline.animate(
        motion_sequence=driving_sequence,
        output_path=None,
    )

    assert frames.ndim == 4
    assert frames.shape[0] == driving_sequence.get_num_frames()
    assert frames.shape[1] == 512 and frames.shape[2] == 512
    assert frames.shape[3] == 3


def test_pipeline_with_reference_image_and_motion(
    flame_config, driving_sequence, device, tmp_path
):
    """
    Load a single reference image and a driving FLAME sequence, then animate.
    If no image path is provided via env, uses a tiny dummy image.
    """
    from lam.pipeline import AnimationPipeline, AnimationConfig
    from lam.io import load_image, save_image

    # Create a minimal dummy image if no reference image in env
    ref_image_env = os.environ.get("LAM_TEST_REFERENCE_IMAGE")
    if ref_image_env and Path(ref_image_env).exists():
        ref_image_path = Path(ref_image_env)
    else:
        ref_image_path = tmp_path / "dummy_ref.png"
        dummy = torch.zeros(64, 64, 3)
        dummy[16:48, 16:48] = 0.8
        save_image(dummy, ref_image_path)

    load_image(ref_image_path, as_tensor=True, device=device)

    config = AnimationConfig(
        flame_config=flame_config,
        output_fps=30,
        output_size=(256, 256),
        device=device,
    )
    pipeline = AnimationPipeline(config)

    out_path = tmp_path / "pipeline_out.mp4"
    frames = pipeline.animate(
        motion_sequence=driving_sequence,
        output_path=out_path,
        reference_image=ref_image_path,
    )

    assert frames.shape[0] == driving_sequence.get_num_frames()
    assert frames.shape[1] == 256 and frames.shape[2] == 256
    if out_path.exists():
        assert out_path.stat().st_size > 0


def test_pipeline_with_loaded_motion_file(flame_config, device, tmp_path):
    """
    Save a driving sequence to file, load it, then run pipeline.animate().
    """
    from lam.pipeline import AnimationPipeline, AnimationConfig
    from lam.io import MotionSequence, save_flame_params, load_flame_params

    if not _flame_assets_exist():
        pytest.skip("FLAME assets not found")

    batch_size = 1
    num_frames = 15
    motion = MotionSequence(
        shape_params=torch.zeros(batch_size, 100, device=device),
        expr_params=torch.zeros(batch_size, 50, num_frames, device=device),
        pose_params=torch.zeros(batch_size, 3, num_frames, device=device),
        jaw_params=torch.zeros(batch_size, 3, num_frames, device=device),
        fps=30,
    )
    motion_path = tmp_path / "driving_motion.pt"
    save_flame_params(motion, motion_path)

    loaded = load_flame_params(motion_path, device=device)
    assert isinstance(loaded, dict)
    assert "expr_params" in loaded or "shape_params" in loaded

    config = AnimationConfig(
        flame_config=flame_config,
        output_fps=30,
        output_size=(128, 128),
        device=device,
    )
    pipeline = AnimationPipeline(config)
    frames = pipeline.animate(motion_sequence=str(motion_path), output_path=None)

    assert frames.shape[0] == num_frames


if __name__ == "__main__":
    # Run as script: optional --image, --motion, --output
    import argparse
    parser = argparse.ArgumentParser(
        description="Test pipeline: single image + driving FLAME sequence → animate 3D head"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to reference image (optional)",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=None,
        help="Path to driving FLAME params (.pt, .npz, .json). If not set, use synthetic.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pipeline_test_out.mp4"),
        help="Output video path",
    )
    parser.add_argument("--no-pytest", action="store_true", help="Run script path only (no pytest)")
    args = parser.parse_args()

    if args.no_pytest:
        if not _flame_assets_exist():
            print(f"FLAME assets not found. Expected:\n  {FLAME_MODEL_PATH}\n  {FLAME_LMK_PATH}")
            sys.exit(1)

        from lam.flame import FlameConfig, FlameHead
        from lam.pipeline import AnimationPipeline, AnimationConfig
        from lam.io import (
            load_image,
            load_flame_params,
            MotionSequence,
            save_flame_params,
        )

        device = _get_device()
        flame_config = FlameConfig(
            flame_model_path=str(FLAME_MODEL_PATH),
            flame_lmk_embedding_path=str(FLAME_LMK_PATH),
            n_shape=100,
            n_expr=50,
            device=device,
        )
        config = AnimationConfig(
            flame_config=flame_config,
            output_fps=30,
            output_size=(512, 512),
            device=device,
        )
        pipeline = AnimationPipeline(config)

        if args.motion and args.motion.exists():
            motion_seq = load_flame_params(args.motion, device=device)
            if isinstance(motion_seq, dict):
                motion_seq = pipeline._dict_to_motion_sequence(motion_seq)
        else:
            motion_seq = pipeline.generate_neutral_animation(num_frames=60, jaw_open_amount=0.0)

        ref_image = args.image if args.image and args.image.exists() else None
        args.output.parent.mkdir(parents=True, exist_ok=True)

        print("Running pipeline: single image + driving FLAME → animate 3D head")
        if ref_image:
            print(f"  Reference image: {ref_image}")
        print(f"  Motion frames:   {motion_seq.get_num_frames()}")
        print(f"  Output:          {args.output}")

        frames = pipeline.animate(
            motion_sequence=motion_seq,
            output_path=args.output,
            reference_image=ref_image,
        )
        print(f"  Output shape:    {frames.shape}")
        print("Done.")
    else:
        pytest.main([__file__, "-v"])
