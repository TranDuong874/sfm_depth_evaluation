"""Depth map to point cloud conversion utilities."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pose: Optional[np.ndarray] = None,
    rgb: Optional[np.ndarray] = None,
    depth_threshold: float = 0.0,
    max_depth: float = 100.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert depth map to point cloud.

    Args:
        depth: (H, W) depth map in meters
        intrinsic: 3x3 camera intrinsic matrix
        pose: Optional 4x4 camera-to-world pose matrix
        rgb: Optional (H, W, 3) RGB image for colors
        depth_threshold: Minimum depth value to include
        max_depth: Maximum depth value to include

    Returns:
        (points, colors) where points is (N, 3) and colors is (N, 3) or None
    """
    h, w = depth.shape[:2]

    # Create pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Get intrinsics
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Unproject to camera coordinates
    z = depth.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to (H, W, 3)
    points_cam = np.stack([x, y, z], axis=-1)

    # Apply mask
    mask = (z > depth_threshold) & (z < max_depth)

    # Flatten and filter
    points_cam = points_cam[mask]

    # Transform to world coordinates if pose provided
    if pose is not None:
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = (R @ points_cam.T).T + t
    else:
        points_world = points_cam

    # Get colors
    colors = None
    if rgb is not None:
        colors = rgb[mask]
        if colors.max() > 1.0:
            colors = colors / 255.0

    return points_world, colors


def fuse_depth_maps(
    depth_maps: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    poses: Dict[str, np.ndarray],
    rgb_images: Optional[Dict[str, np.ndarray]] = None,
    voxel_size: Optional[float] = None,
    depth_threshold: float = 0.01,
    max_depth: float = 100.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fuse multiple depth maps into a single point cloud.

    Args:
        depth_maps: Dict mapping image name to (H, W) depth map
        intrinsics: Dict mapping image name to 3x3 K matrix
        poses: Dict mapping image name to 4x4 camera-to-world pose
        rgb_images: Optional dict mapping image name to (H, W, 3) RGB
        voxel_size: Optional voxel size for downsampling
        depth_threshold: Minimum depth value
        max_depth: Maximum depth value

    Returns:
        (points, colors) tuple
    """
    all_points = []
    all_colors = []

    for name in depth_maps:
        if name not in poses or name not in intrinsics:
            print(f"  Warning: Missing pose/intrinsic for {name}, skipping")
            continue

        depth = depth_maps[name]
        intrinsic = intrinsics[name]
        pose = poses[name]

        rgb = rgb_images.get(name) if rgb_images else None

        points, colors = depth_to_pointcloud(
            depth,
            intrinsic,
            pose,
            rgb,
            depth_threshold,
            max_depth,
        )

        if len(points) > 0:
            all_points.append(points)
            if colors is not None:
                all_colors.append(colors)

    if not all_points:
        return np.array([]).reshape(0, 3), None

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0) if all_colors else None

    # Voxel downsample if requested
    if voxel_size is not None and voxel_size > 0:
        points, colors = voxel_downsample(points, colors, voxel_size)

    return points, colors


def voxel_downsample(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Downsample point cloud using voxel grid.

    Args:
        points: (N, 3) points
        colors: Optional (N, 3) colors
        voxel_size: Voxel size

    Returns:
        Downsampled (points, colors)
    """
    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Use dictionary to aggregate points per voxel
    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = {'points': [], 'colors': []}
        voxel_dict[key]['points'].append(points[i])
        if colors is not None:
            voxel_dict[key]['colors'].append(colors[i])

    # Average points in each voxel
    new_points = []
    new_colors = []
    for key, data in voxel_dict.items():
        new_points.append(np.mean(data['points'], axis=0))
        if colors is not None and data['colors']:
            new_colors.append(np.mean(data['colors'], axis=0))

    new_points = np.array(new_points)
    new_colors = np.array(new_colors) if new_colors else None

    return new_points, new_colors


def load_depth_maps(
    depth_dir: str,
    image_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load depth maps from directory.

    Args:
        depth_dir: Directory containing *_depth.npy files
        image_names: List of image names (without extension)

    Returns:
        Dict mapping image name to depth array
    """
    depth_path = Path(depth_dir)
    depth_maps = {}

    for name in image_names:
        stem = Path(name).stem
        npy_path = depth_path / f"{stem}_depth.npy"

        if npy_path.exists():
            depth_maps[name] = np.load(str(npy_path))
        else:
            print(f"  Warning: Depth not found: {npy_path}")

    return depth_maps


def load_rgb_images(
    image_dir: str,
    image_names: List[str],
) -> Dict[str, np.ndarray]:
    """
    Load RGB images from directory.

    Args:
        image_dir: Directory containing images
        image_names: List of image filenames

    Returns:
        Dict mapping image name to RGB array
    """
    image_path = Path(image_dir)
    images = {}

    for name in image_names:
        img_path = image_path / name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                images[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return images
