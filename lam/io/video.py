"""
Video I/O utilities.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import subprocess


def images_to_video(
    images: Union[List[np.ndarray], List[torch.Tensor], np.ndarray, torch.Tensor],
    output_path: Union[str, Path],
    fps: int = 30,
    codec: str = 'libx264',
    quality: int = 10
):
    """
    Save sequence of images as video using ffmpeg.
    
    Args:
        images: List of images or stacked array (N, H, W, 3)
        output_path: Path to save video
        fps: Frames per second
        codec: Video codec
        quality: Video quality (lower = better for x264, 0-51)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy array
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    elif isinstance(images, list) and len(images) > 0:
        if isinstance(images[0], torch.Tensor):
            images = [img.detach().cpu().numpy() for img in images]
        images = np.stack(images)
    
    # Ensure uint8
    if images.dtype != np.uint8:
        if images.max() <= 1.0:
            images = (np.clip(images, 0, 1) * 255).astype(np.uint8)
        else:
            images = np.clip(images, 0, 255).astype(np.uint8)
    
    # Use imageio or moviepy if available
    try:
        import imageio
        imageio.mimwrite(output_path, images, fps=fps, codec=codec, quality=quality)
        return
    except ImportError:
        pass
    
    try:
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(list(images), fps=fps)
        clip.write_videofile(str(output_path), codec=codec, logger=None)
        return
    except ImportError:
        pass
    
    # Fallback: raise error
    raise ImportError(
        "No video library available. Install imageio or moviepy:\n"
        "  pip install imageio imageio-ffmpeg\n"
        "  or\n"
        "  pip install moviepy"
    )


def save_video(
    video_tensor: torch.Tensor,
    output_path: Union[str, Path],
    fps: int = 30,
    **kwargs
):
    """
    Save video tensor to file.
    
    Args:
        video_tensor: (N, H, W, 3) video frames
        output_path: Path to save video
        fps: Frames per second
        **kwargs: Additional arguments for images_to_video
    """
    images_to_video(video_tensor, output_path, fps, **kwargs)


def add_audio_to_video(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path]
):
    """
    Add audio track to video using ffmpeg.
    
    Args:
        video_path: Path to input video
        audio_path: Path to audio file
        output_path: Path to save output video with audio
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use moviepy if available
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        video_clip = VideoFileClip(str(video_path))
        audio_clip = AudioFileClip(str(audio_path))
        
        video_with_audio = video_clip.set_audio(audio_clip)
        video_with_audio.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            logger=None
        )
        return
    except ImportError:
        pass
    
    # Fallback to ffmpeg command
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Failed to add audio. Make sure ffmpeg is installed or install moviepy:\n"
            f"  pip install moviepy\n"
            f"Error: {e}"
        )
