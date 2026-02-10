"""
Image I/O utilities.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
from PIL import Image


def load_image(
    image_path: Union[str, Path],
    as_tensor: bool = True,
    device: str = "cpu"
) -> Union[torch.Tensor, np.ndarray]:
    """
    Load image from file.
    
    Args:
        image_path: Path to image file
        as_tensor: Return as torch tensor (else numpy array)
        device: Device for tensor
        
    Returns:
        Image as (H, W, 3) tensor or array, values in [0, 1]
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load with PIL
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img).astype(np.float32) / 255.0
    
    if as_tensor:
        img_tensor = torch.from_numpy(img_array).to(device)
        return img_tensor
    
    return img_array


def save_image(
    image: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    denormalize: bool = True
):
    """
    Save image to file.
    
    Args:
        image: (H, W, 3) image tensor or array
        save_path: Path to save image
        denormalize: If True, assumes image is in [0, 1] and scales to [0, 255]
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Denormalize if needed
    if denormalize:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Save
    img_pil = Image.fromarray(image)
    img_pil.save(save_path)


def preprocess_image(
    image: Union[torch.Tensor, np.ndarray, str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Preprocess image for model input.
    
    Args:
        image: Image as path, array, or tensor
        target_size: Optional (height, width) to resize to
        normalize: Normalize to [0, 1] range
        device: Target device
        
    Returns:
        Preprocessed image tensor (H, W, 3)
    """
    # Load if path
    if isinstance(image, (str, Path)):
        image = load_image(image, as_tensor=False)
    
    # Convert to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    
    # Resize if needed
    if target_size is not None:
        h, w = target_size
        img_pil = Image.fromarray((image * 255).astype(np.uint8) if normalize else image.astype(np.uint8))
        img_pil = img_pil.resize((w, h), Image.LANCZOS)
        image = np.array(img_pil).astype(np.float32)
        if normalize:
            image = image / 255.0
    
    # Ensure [0, 1] range
    if normalize and image.max() > 1.0:
        image = image / 255.0
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image).to(device)
    
    return image_tensor


def remove_background(
    image: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
    """
    Simple background removal (placeholder).
    
    Args:
        image: Input image
        threshold: Threshold for background detection
        
    Returns:
        image: Image with background removed
        mask: Foreground mask
        
    NOTE: For production, use proper segmentation models like U2Net or SAM.
    """
    # Placeholder: return original image and full mask
    is_tensor = isinstance(image, torch.Tensor)
    
    if is_tensor:
        mask = torch.ones_like(image[..., :1])
    else:
        mask = np.ones_like(image[..., :1])
    
    return image, mask
