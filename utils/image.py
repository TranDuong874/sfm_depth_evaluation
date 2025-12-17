"""
Image preprocessing utilities for SfM Depth Evaluation Pipeline.

Handles image resizing with aspect ratio preservation and alignment
to multiples of 16 (for ViT-based models).
"""

from pathlib import Path
from typing import Tuple, Optional, Union

import cv2
import numpy as np


def compute_resize_dimensions(
    height: int,
    width: int,
    max_edge: int = 512,
    multiple_of: int = 16,
) -> Tuple[int, int]:
    """
    Compute target dimensions for resizing.

    Args:
        height: Original image height
        width: Original image width
        max_edge: Maximum edge length
        multiple_of: Align dimensions to this multiple

    Returns:
        (new_height, new_width) tuple
    """
    # Compute scale to fit max_edge
    max_dim = max(height, width)
    if max_dim <= max_edge:
        scale = 1.0
    else:
        scale = max_edge / max_dim

    # Apply scale
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Align to multiple (round down)
    new_height = new_height - (new_height % multiple_of)
    new_width = new_width - (new_width % multiple_of)

    # Ensure minimum size
    new_height = max(new_height, multiple_of)
    new_width = max(new_width, multiple_of)

    return new_height, new_width


def resize_image(
    image: np.ndarray,
    max_edge: int = 512,
    multiple_of: int = 16,
    interpolation: int = cv2.INTER_AREA,
) -> Tuple[np.ndarray, dict]:
    """
    Resize image with aspect ratio preservation and alignment.

    Args:
        image: Input image (H, W, C) or (H, W)
        max_edge: Maximum edge length
        multiple_of: Align dimensions to this multiple
        interpolation: OpenCV interpolation method

    Returns:
        Tuple of (resized_image, info_dict)
        info_dict contains: original_size, new_size, scale
    """
    h, w = image.shape[:2]
    new_h, new_w = compute_resize_dimensions(h, w, max_edge, multiple_of)

    # Compute actual scale
    scale = new_w / w  # Use width scale (same as height scale due to aspect ratio)

    # Resize
    if (new_h, new_w) != (h, w):
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    else:
        resized = image.copy()

    info = {
        "original_size": (h, w),
        "new_size": (new_h, new_w),
        "scale": scale,
    }

    return resized, info


def resize_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize binary mask to target size using nearest neighbor interpolation.

    Args:
        mask: Input mask (H, W), values 0 or 255/1
        target_size: Target (height, width)

    Returns:
        Resized mask
    """
    target_h, target_w = target_size
    h, w = mask.shape[:2]

    if (h, w) == (target_h, target_w):
        return mask.copy()

    resized = cv2.resize(
        mask,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized


def resize_depth(
    depth: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_NEAREST,
) -> np.ndarray:
    """
    Resize depth map to target size.

    Args:
        depth: Input depth map (H, W)
        target_size: Target (height, width)
        interpolation: OpenCV interpolation method
            Use INTER_NEAREST for sparse depth, INTER_LINEAR for dense

    Returns:
        Resized depth map
    """
    target_h, target_w = target_size
    h, w = depth.shape[:2]

    if (h, w) == (target_h, target_w):
        return depth.copy()

    resized = cv2.resize(
        depth.astype(np.float32),
        (target_w, target_h),
        interpolation=interpolation,
    )

    return resized


def scale_intrinsics(
    K: np.ndarray,
    original_size: Tuple[int, int],
    new_size: Tuple[int, int],
) -> np.ndarray:
    """
    Scale intrinsic matrix for image resize.

    Args:
        K: 3x3 intrinsic matrix
        original_size: Original (height, width)
        new_size: New (height, width)

    Returns:
        Scaled 3x3 intrinsic matrix
    """
    orig_h, orig_w = original_size
    new_h, new_w = new_size

    scale_x = new_w / orig_w
    scale_y = new_h / orig_h

    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x  # fx
    K_scaled[1, 1] *= scale_y  # fy
    K_scaled[0, 2] *= scale_x  # cx
    K_scaled[1, 2] *= scale_y  # cy

    return K_scaled


def load_image(
    path: Union[str, Path],
    color: bool = True,
) -> np.ndarray:
    """
    Load image from file.

    Args:
        path: Image file path
        color: If True, load as RGB. If False, load as grayscale.

    Returns:
        Image array (H, W, 3) for color or (H, W) for grayscale
    """
    path = str(path)

    if color:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return img


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
) -> None:
    """
    Save image to file.

    Args:
        image: Image array (H, W, 3) RGB or (H, W) grayscale
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), image)


def load_and_resize_image(
    path: Union[str, Path],
    max_edge: int = 512,
    multiple_of: int = 16,
) -> Tuple[np.ndarray, dict]:
    """
    Load and resize image in one step.

    Args:
        path: Image file path
        max_edge: Maximum edge length
        multiple_of: Align dimensions to this multiple

    Returns:
        Tuple of (resized_image, info_dict)
    """
    image = load_image(path, color=True)
    return resize_image(image, max_edge, multiple_of)
