"""
FLAME-only tracking pipeline - simplified version focusing only on face/head tracking and FLAME optimization.
No body (SMPL-X) or hand (MANO) tracking.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tyro
from dataclasses import dataclass
from typing_extensions import Annotated
from src.utils.rprint import rlog as log
from src.flame_only_pipeline import FlameOnlyPipeline
from src.configs.base_config import PrintableConfig


@dataclass(repr=False)
class FlameTrackingConfig(PrintableConfig):
    """Configuration for FLAME-only tracking."""
    in_root: Annotated[str, tyro.conf.arg(aliases=["-i"])] = 'assets/videos'
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = 'outputs/flame_tracking/'
    
    # Visualization options
    save_vis_video: bool = False
    save_images: bool = False
    save_mesh_render: bool = False
    
    # Frame sampling
    tracking_with_interval: bool = False
    default_frame_interval: int = 6
    max_frames: int = 75
    min_frames: int = 5
    
    # GPU
    device_id: int = 0
    
    # Optimization
    flame_optim_steps: int = 1001
    skip_optimization: bool = False  # If True, only run base tracking without refinement


def find_videos_in_path(path):
    """Find all video files in path (file or directory)."""
    videos = []
    if os.path.isfile(path):
        if path.split('.')[-1].lower() in ('mp4', 'avi', 'mkv'):
            videos.append(path)
    elif os.path.isdir(path):
        videos.extend(
            [os.path.join(path, f) for f in sorted(os.listdir(path)) 
             if f.split('.')[-1].lower() in ('mp4', 'avi', 'mkv')]
        )
    else:
        print(f"[Warning] Path '{path}' is not a valid file or directory. Skipping.")
    return videos


def main():
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(FlameTrackingConfig)
    
    # Find all videos
    all_videos = find_videos_in_path(args.in_root)
    if not all_videos:
        log(f"No videos found in {args.in_root}")
        return
    
    log(f"Found {len(all_videos)} video(s) to process")
    
    # Create pipeline
    pipeline = FlameOnlyPipeline(args)
    
    # Process videos
    pipeline.execute(all_videos)
    
    log("✓ All done!")


if __name__ == '__main__':
    main()
