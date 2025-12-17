"""
Point cloud reconstruction from depth maps.

Handles backprojection, mask filtering, and point cloud cleaning.

POSE CONVENTION:
    Camera-to-World (C2W): T_cw = [R | t]
    Point transformation:  X_w = R * X_c + t
    Inverse (W2C):         X_c = R^T * (X_w - t)
    
    In code (row vectors):
    Backproject: points_world = points_cam @ R.T + t  (Apply R)
    Filter:      points_cam   = (points_world - t) @ R  (Apply R^T)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def backproject_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    pose: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    depth_min: float = 0.01,
    depth_max: float = 100.0,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Backproject depth map to 3D point cloud.

    Args:
        depth: (H, W) depth map in meters
        intrinsic: 3x3 camera intrinsic matrix K
        pose: 4x4 camera-to-world transformation matrix (C2W)
        rgb: Optional (H, W, 3) RGB image for colors
        mask: Optional (H, W) binary mask (1 = keep, 0 = discard)
        depth_min: Minimum depth threshold
        depth_max: Maximum depth threshold
        stride: Pixel stride for subsampling

    Returns:
        Tuple of (points, colors)
        - points: (N, 3) world coordinates
        - colors: (N, 3) RGB colors or None
    """
    h, w = depth.shape[:2]

    # Create pixel grid
    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    u, v = np.meshgrid(u, v)
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    # Sample depth at stride
    z = depth[::stride, ::stride].astype(np.float32)

    # Sample mask if provided - RESIZE to match depth size first
    if mask is not None:
        # Resize mask to match depth map size if needed
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        mask_sampled = mask[::stride, ::stride]
    else:
        mask_sampled = np.ones_like(z, dtype=bool)

    # Get intrinsics
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Unproject to camera coordinates
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack to (H/s, W/s, 3)
    points_cam = np.stack([x, y, z], axis=-1)

    # Create valid mask
    valid = (z > depth_min) & (z < depth_max) & (mask_sampled > 0)

    # Flatten and filter
    points_cam = points_cam[valid]

    if len(points_cam) == 0:
        return np.zeros((0, 3)), None

    # Transform to world coordinates (C2W)
    # X_w = R * X_c + t
    # Row vectors: X_w_row = X_c_row @ R^T + t_row
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points_cam.T).T + t

    # Get colors - resize RGB to match depth size if needed
    colors = None
    if rgb is not None:
        if rgb.shape[0] != h or rgb.shape[1] != w:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb_sampled = rgb[::stride, ::stride]
        colors = rgb_sampled[valid]
        if len(colors) > 0 and colors.max() > 1.0:
            colors = colors / 255.0

    return points_world, colors


def fuse_depth_maps(
    depth_maps: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    poses: Dict[str, np.ndarray],
    rgb_images: Optional[Dict[str, np.ndarray]] = None,
    masks: Optional[Dict[str, np.ndarray]] = None,
    depth_min: float = 0.01,
    depth_max: float = 100.0,
    stride: int = 1,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fuse multiple depth maps into a single point cloud.

    Args:
        depth_maps: Dict mapping image name to depth map
        intrinsics: Dict mapping image name to 3x3 K matrix
        poses: Dict mapping image name to 4x4 c2w pose
        rgb_images: Optional dict of RGB images
        masks: Optional dict of binary masks
        depth_min: Minimum depth threshold
        depth_max: Maximum depth threshold
        stride: Pixel stride for subsampling

    Returns:
        Tuple of (points, colors)
    """
    all_points = []
    all_colors = []

    for name in depth_maps:
        if name not in poses or name not in intrinsics:
            continue

        depth = depth_maps[name]
        K = intrinsics[name]
        pose = poses[name]
        rgb = rgb_images.get(name) if rgb_images else None
        mask = masks.get(name) if masks else None

        points, colors = backproject_depth(
            depth, K, pose, rgb, mask,
            depth_min=depth_min,
            depth_max=depth_max,
            stride=stride,
        )

        if len(points) > 0:
            all_points.append(points)
            if colors is not None:
                all_colors.append(colors)

    if not all_points:
        return np.zeros((0, 3)), None

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0) if all_colors else None

    return points, colors


def clean_point_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    nb_neighbors: int = 50,
    std_ratio: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Clean point cloud using statistical outlier removal.

    Args:
        points: (N, 3) point cloud
        colors: Optional (N, 3) colors
        nb_neighbors: Number of neighbors for outlier detection
        std_ratio: Standard deviation ratio threshold

    Returns:
        Cleaned (points, colors)
    """
    if len(points) < nb_neighbors + 1:
        return points, colors

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio,
        )

        clean_points = np.asarray(cl.points)
        clean_colors = np.asarray(cl.colors) if cl.has_colors() else None

        return clean_points, clean_colors

    except ImportError:
        # Fallback without Open3D
        return _clean_point_cloud_fallback(points, colors, nb_neighbors, std_ratio)


def _clean_point_cloud_fallback(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    nb_neighbors: int,
    std_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Simple outlier removal without Open3D."""
    from scipy.spatial import KDTree

    if len(points) < nb_neighbors + 1:
        return points, colors

    tree = KDTree(points)
    distances, _ = tree.query(points, k=nb_neighbors + 1)
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self

    threshold = mean_distances.mean() + std_ratio * mean_distances.std()
    mask = mean_distances < threshold

    clean_points = points[mask]
    clean_colors = colors[mask] if colors is not None else None

    return clean_points, clean_colors


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
        voxel_size: Size of voxel grid

    Returns:
        Downsampled (points, colors)
    """
    if len(points) == 0 or voxel_size <= 0:
        return points, colors

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

        down_points = np.asarray(pcd_down.points)
        down_colors = np.asarray(pcd_down.colors) if pcd_down.has_colors() else None

        return down_points, down_colors

    except ImportError:
        # Fallback
        return _voxel_downsample_fallback(points, colors, voxel_size)


def _voxel_downsample_fallback(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Voxel downsampling without Open3D."""
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    voxel_dict = {}
    for i, idx in enumerate(voxel_indices):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = {'points': [], 'colors': []}
        voxel_dict[key]['points'].append(points[i])
        if colors is not None:
            voxel_dict[key]['colors'].append(colors[i])

    new_points = []
    new_colors = []
    for data in voxel_dict.values():
        new_points.append(np.mean(data['points'], axis=0))
        if colors is not None and data['colors']:
            new_colors.append(np.mean(data['colors'], axis=0))

    new_points = np.array(new_points)
    new_colors = np.array(new_colors) if new_colors else None

    return new_points, new_colors


def reconstruct_scene(
    depth_dir: str,
    sfm_output,
    image_dir: str,
    mask_dir: Optional[str] = None,
    depth_min: float = 0.01,
    depth_max: float = 100.0,
    stride: int = 1,
    clean: bool = True,
    nb_neighbors: int = 50,
    std_ratio: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Reconstruct scene and object point clouds.

    Args:
        depth_dir: Directory with depth maps (*_depth.npy)
        sfm_output: SfMOutput object with poses and intrinsics
        image_dir: Directory with RGB images
        mask_dir: Optional directory with masks
        depth_min: Minimum depth
        depth_max: Maximum depth
        stride: Pixel stride
        clean: Apply outlier removal
        nb_neighbors: Neighbors for cleaning
        std_ratio: Std ratio for cleaning

    Returns:
        Tuple of (scene_points, scene_colors, object_points, object_colors)
    """
    from core.sfm.base import SfMOutput

    depth_path = Path(depth_dir)

    # Load depth maps
    depth_maps = {}
    for name in sfm_output.image_names:
        stem = Path(name).stem
        npy_path = depth_path / f"{stem}_depth.npy"
        if npy_path.exists():
            depth_maps[name] = np.load(str(npy_path))

    # Load RGB images
    rgb_images = {}
    image_path = Path(image_dir)
    for name in sfm_output.image_names:
        img_path = image_path / name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                rgb_images[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load masks if provided
    masks = {}
    if mask_dir:
        mask_path = Path(mask_dir)
        for name in sfm_output.image_names:
            stem = Path(name).stem
            # Try different mask naming conventions
            for pattern in [f"{name}", f"frame_{stem.split('_')[-1]}.png", f"{stem}.png"]:
                m_path = mask_path / pattern
                if m_path.exists():
                    mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        masks[name] = (mask > 127).astype(np.uint8)
                    break

    # Fuse scene point cloud (no mask)
    scene_points, scene_colors = fuse_depth_maps(
        depth_maps,
        sfm_output.intrinsics,
        sfm_output.poses,
        rgb_images=rgb_images,
        masks=None,
        depth_min=depth_min,
        depth_max=depth_max,
        stride=stride,
    )

    # Fuse object point cloud (with mask)
    object_points, object_colors = fuse_depth_maps(
        depth_maps,
        sfm_output.intrinsics,
        sfm_output.poses,
        rgb_images=rgb_images,
        masks=masks if masks else None,
        depth_min=depth_min,
        depth_max=depth_max,
        stride=stride,
    )

    # Clean point clouds
    if clean:
        if len(scene_points) > 0:
            scene_points, scene_colors = clean_point_cloud(
                scene_points, scene_colors, nb_neighbors, std_ratio
            )
        if len(object_points) > 0:
            object_points, object_colors = clean_point_cloud(
                object_points, object_colors, nb_neighbors, std_ratio
            )

    return scene_points, scene_colors, object_points, object_colors


def filter_points_by_mask(
    points: np.ndarray,
    colors: Optional[np.ndarray],
    masks: Dict[str, np.ndarray],
    poses: Dict[str, np.ndarray],
    intrinsics: Dict[str, np.ndarray],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Filter point cloud to keep only points that project onto foreground in masks.

    Args:
        points: (N, 3) point cloud
        colors: Optional (N, 3) colors
        masks: Dict mapping image name to (H, W) binary mask
        poses: Dict mapping image name to 4x4 c2w pose
        intrinsics: Dict mapping image name to 3x3 K matrix
        threshold: Fraction of views point must be visible in foreground

    Returns:
        Filtered (points, colors)
    """
    if len(points) == 0:
        return points, colors

    n_points = len(points)
    fg_votes = np.zeros(n_points)
    total_votes = np.zeros(n_points)

    # Convert dicts to lists for iteration
    names = sorted(list(masks.keys()))
    
    for name in names:
        if name not in poses or name not in intrinsics:
            continue
            
        mask = masks[name]
        pose = poses[name]
        K = intrinsics[name]
        
        h, w = mask.shape[:2]

        # Transform points to camera space
        # pose is c2w (camera-to-world), so w2c = inv(c2w) = [R.T, -R.T @ t]
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # w2c: p_cam = R.T @ (p_world - t)
        # Using row vectors: p_cam.T = (p_world - t).T @ R
        points_cam = (points - t) @ R  # Nx3 @ 3x3 = Nx3

        # Project to image
        z = points_cam[:, 2]
        valid_z = z > 0.01

        # Avoid division by zero
        z_safe = np.where(valid_z, z, 1.0)
        x = (points_cam[:, 0] / z_safe) * K[0, 0] + K[0, 2]
        y = (points_cam[:, 1] / z_safe) * K[1, 1] + K[1, 2]

        # Check bounds
        valid_x = (x >= 0) & (x < w)
        valid_y = (y >= 0) & (y < h)
        valid = valid_z & valid_x & valid_y
        
        # Sample mask
        x_int = np.clip(x.astype(int), 0, w-1)
        y_int = np.clip(y.astype(int), 0, h-1)

        # Only count votes for points that project within image bounds
        valid_idx = np.where(valid)[0]
        if len(valid_idx) > 0:
            fg_votes[valid_idx] += mask[y_int[valid_idx], x_int[valid_idx]] > 0
            total_votes[valid_idx] += 1

    # Keep points visible in foreground in at least threshold fraction of views
    # Default to keeping points that aren't seen in ANY view (to be safe? No, remove them)
    keep = np.zeros(n_points, dtype=bool)
    
    # Avoid division by zero
    has_votes = total_votes > 0
    fg_ratio = np.zeros(n_points)
    fg_ratio[has_votes] = fg_votes[has_votes] / total_votes[has_votes]
    
    keep = fg_ratio >= threshold

    filtered_points = points[keep]
    filtered_colors = colors[keep] if colors is not None else None

    return filtered_points, filtered_colors


def align_point_clouds(
    source: np.ndarray,
    target: np.ndarray,
    use_scaling: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Align source point cloud to target using Sim3 (Rotation, Translation, Scale).
    
    Uses Procrustes analysis on centroids and then ICP refinement if Open3D is available.

    Args:
        source: (N, 3) source points
        target: (M, 3) target points
        use_scaling: Whether to estimate scale

    Returns:
        Tuple of (aligned_source, transformation_4x4, scale_factor)
        transformation_4x4 includes rotation and translation.
        scale_factor is separate applied BEFORE transformation.
    """
    if len(source) == 0 or len(target) == 0:
        return source, np.eye(4), 1.0

    # 1. Initial Scale Estimation (Ratio of std devs)
    src_centroid = np.mean(source, axis=0)
    tgt_centroid = np.mean(target, axis=0)
    
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    
    scale = 1.0
    if use_scaling:
        src_std = np.sqrt(np.mean(np.sum(src_centered**2, axis=1)))
        tgt_std = np.sqrt(np.mean(np.sum(tgt_centered**2, axis=1)))
        if src_std > 1e-6:
            scale = tgt_std / src_std
            
    source_scaled = (source - src_centroid) * scale + src_centroid

    # 2. ICP Alignment using Open3D (Rigid)
    try:
        import open3d as o3d
        
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_scaled)
        
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)
        
        # Initial alignment: translate centroids to match
        init_trans = np.eye(4)
        init_trans[:3, 3] = tgt_centroid - src_centroid
        
        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd,
            max_correspondence_distance=1.0, # Adjust based on scale?
            init=init_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        transformation = result.transformation
        aligned_source = np.asarray(source_pcd.transform(transformation).points)
        
        # Combine scale into the pipeline
        # Points transformed as: T @ ( (P - C_src) * s + C_src )
        # We return T, s. Caller must apply s first then T.
        # But wait, T acts on source_scaled.
        
        return aligned_source, transformation, scale

    except ImportError:
        # Fallback: Just centroid alignment
        translation = tgt_centroid - src_centroid
        T = np.eye(4)
        T[:3, 3] = translation
        
        aligned_source = source_scaled + translation
        return aligned_source, T, scale