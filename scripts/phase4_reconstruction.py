#!/usr/bin/env python3
"""
Phase 4: Point Cloud Reconstruction

Reconstructs point clouds from depth maps and SfM poses.
Outputs both scene and object (masked) point clouds.

Usage:
    python scripts/phase4_reconstruction.py \
        --sampled output/phase1_sampled \
        --sfm output/phase2_sfm \
        --depth output/phase3_depth \
        --output output/phase4_reconstruction

Output structure:
    output/phase4_reconstruction/{seq_id}/{n}_views/{method}/
    ├── scene_pointcloud.ply
    └── object_pointcloud.ply
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np

from config import ReconstructionConfig, EVALUATION_METHODS
from core.reconstruction import (
    fuse_depth_maps,
    clean_point_cloud,
    backproject_depth,
)
from core.co3d import save_ply
from utils.timer import PipelineTimer


def reconstruct_method(
    method: str,
    sampled_dir: Path,
    sfm_dir: Path,
    depth_dir: Path,
    output_dir: Path,
    config: ReconstructionConfig,
):
    """
    Reconstruct point clouds for a specific method.

    Args:
        method: Method name (e.g., 'colmap_murre', 'mast3r_sparse')
        sampled_dir: Directory with sampled images
        sfm_dir: Directory with SfM outputs
        depth_dir: Directory with depth outputs
        output_dir: Output directory
        config: Reconstruction configuration
    """
    from core.sfm.base import SfMOutput
    from core.reconstruction import filter_points_by_mask, align_point_clouds

    output_dir.mkdir(parents=True, exist_ok=True)

    scene_ply = output_dir / "scene_pointcloud.ply"
    object_ply = output_dir / "object_pointcloud.ply"

    if scene_ply.exists() and object_ply.exists():
        print(f"        Already exists, skipping")
        return

    # Parse method name
    parts = method.split('_')
    sfm_method = parts[0]  # colmap or mast3r

    is_baseline = method.endswith('_sparse')

    # Load SfM output
    sfm_path = sfm_dir / sfm_method
    if not sfm_path.exists():
        print(f"        SfM not found: {sfm_path}")
        return

    sfm_output = SfMOutput.load(str(sfm_path))

    # Load images and masks
    images_dir = sampled_dir / "images"
    
    # Check for undistorted images (COLMAP)
    if 'undistorted_images_path' in sfm_output.metadata:
        undistorted_path = Path(sfm_output.metadata['undistorted_images_path'])
        if undistorted_path.exists():
            print(f"        Using undistorted images from: {undistorted_path}")
            images_dir = undistorted_path
            
    masks_dir = sampled_dir / "masks"

    rgb_images = {}
    masks = {}

    for name in sfm_output.image_names:
        # Load RGB
        img_path = images_dir / name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                rgb_images[name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        mask_path = masks_dir / name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks[name] = (mask > 127).astype(np.uint8)

    # Scale intrinsics if SfM resolution differs from loaded RGB (e.g. COLMAP 1600 vs RGB 512)
    # We use 512 images/depth, so K must be scaled to 512.
    scaled_intrinsics = {}
    for name, K in sfm_output.intrinsics.items():
        if name in rgb_images:
            h_rgb, w_rgb = rgb_images[name].shape[:2]
            h_sfm, w_sfm = sfm_output.image_sizes.get(name, (h_rgb, w_rgb))
            
            if (h_sfm, w_sfm) != (h_rgb, w_rgb):
                scale_x = w_rgb / w_sfm
                scale_y = h_rgb / h_sfm
                K_new = K.copy()
                K_new[0, 0] *= scale_x
                K_new[0, 2] *= scale_x
                K_new[1, 1] *= scale_y
                K_new[1, 2] *= scale_y
                scaled_intrinsics[name] = K_new
            else:
                scaled_intrinsics[name] = K
        else:
            scaled_intrinsics[name] = K

    # -------------------------------------------------------------------------
    # 1. Prepare Depth Maps
    # -------------------------------------------------------------------------
    depth_maps = {}
    
    if is_baseline:
        # Baseline: Use SfM sparse depths (anchors) as the depth maps
        print(f"        Using SfM sparse depth for reconstruction")
        for name in sfm_output.image_names:
            depth = sfm_output.sparse_depths.get(name)
            if depth is not None:
                # Resize if needed (e.g. COLMAP 1600 -> 512)
                # We already handle scaling K below, but we need depth map to match RGB size
                # RGB is loaded from 'images' (512)
                h_rgb, w_rgb = rgb_images[name].shape[:2] if name in rgb_images else (0,0)
                if h_rgb > 0 and depth.shape[:2] != (h_rgb, w_rgb):
                     depth = cv2.resize(depth, (w_rgb, h_rgb), interpolation=cv2.INTER_NEAREST)
                depth_maps[name] = depth
    else:
        # Non-baseline: use learned depth maps
        depth_method = '_'.join(parts[1:])  # e.g., murre, metric3d
        depth_path = depth_dir / sfm_method / depth_method

        if not depth_path.exists():
            print(f"        Depth not found: {depth_path}")
            return

        # Mandatory Item 4: Enforce SfM scale as immutable
        if not (depth_path / "scale_factors.json").exists():
            # For Murre native, it might not have it if we didn't save it explicitly, 
            # but Phase 3 script saves it for all methods now.
            print(f"        [Error] Scale calibration missing. Run Phase 3.")
            return

        # Load depth maps
        for name in sfm_output.image_names:
            stem = Path(name).stem
            npy_path = depth_path / f"{stem}_depth.npy"
            if npy_path.exists():
                depth_maps[name] = np.load(str(npy_path))

    if not depth_maps:
        print(f"        No depth maps found")
        return

    # -------------------------------------------------------------------------
    # 2. Generate SCENE Point Cloud (Unaligned, No Mask)
    # -------------------------------------------------------------------------
    scene_points, scene_colors = fuse_depth_maps(
        depth_maps,
        scaled_intrinsics,
        sfm_output.poses,
        rgb_images=rgb_images,
        masks=None,
        depth_min=config.depth_min,
        depth_max=config.depth_max,
        stride=config.stride,
    )

    # Clean scene point cloud
    if len(scene_points) > 0:
        scene_points, scene_colors = clean_point_cloud(
            scene_points, scene_colors,
            nb_neighbors=config.statistical_outlier_nb_neighbors,
            std_ratio=config.statistical_outlier_std_ratio,
        )
    
    # -------------------------------------------------------------------------
    # 3. Generate OBJECT Point Cloud (Masking)
    # -------------------------------------------------------------------------
    # Apply masks per-view before fusion (Dense approach for all)
    object_points, object_colors = fuse_depth_maps(
        depth_maps,
        scaled_intrinsics,
        sfm_output.poses,
        rgb_images=rgb_images,
        masks=masks,  # Apply mask here
        depth_min=config.depth_min,
        depth_max=config.depth_max,
        stride=config.stride,
    )

    # Clean object point cloud
    if len(object_points) > 0:
        object_points, object_colors = clean_point_cloud(
            object_points, object_colors,
            nb_neighbors=config.statistical_outlier_nb_neighbors,
            std_ratio=config.statistical_outlier_std_ratio,
        )

    # Save
    save_ply(scene_ply, scene_points, scene_colors)
    save_ply(object_ply, object_points, object_colors)

    print(f"        Scene: {len(scene_points)} pts, Object: {len(object_points)} pts")


def main():
    parser = argparse.ArgumentParser(description='Phase 4: Point Cloud Reconstruction')
    parser.add_argument('--sampled', type=str, default='output/phase1_sampled',
                        help='Sampled images directory (phase1 output)')
    parser.add_argument('--sfm', type=str, default='output/phase2_sfm',
                        help='SfM output directory (phase2 output)')
    parser.add_argument('--depth', type=str, default='output/phase3_depth',
                        help='Depth output directory (phase3 output)')
    parser.add_argument('--output', type=str, default='output/phase4_reconstruction',
                        help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=EVALUATION_METHODS,
                        help='Methods to reconstruct')
    parser.add_argument('--n-views', type=int, nargs='+', default=None,
                        help='View counts to process (default: all)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Process specific sequence only')

    args = parser.parse_args()

    sampled_root = Path(args.sampled)
    sfm_root = Path(args.sfm)
    depth_root = Path(args.depth)
    output_root = Path(args.output)

    config = ReconstructionConfig()

    # Find sequences from SfM output
    if args.sequence:
        seq_dirs = [sfm_root / args.sequence]
    else:
        seq_dirs = sorted([d for d in sfm_root.iterdir() if d.is_dir()])

    print(f"Found {len(seq_dirs)} sequences")
    print(f"Methods: {args.methods}")
    print(f"Output: {output_root}")
    print()

    # Initialize Timer
    estimated_steps = len(seq_dirs) * 3 * len(args.methods)
    if args.n_views:
        estimated_steps = len(seq_dirs) * len(args.n_views) * len(args.methods)
    timer = PipelineTimer(total_steps=estimated_steps, name="Phase4")

    # Process sequences
    for i, seq_dir in enumerate(seq_dirs):
        seq_id = seq_dir.name
        print(f"[{i+1}/{len(seq_dirs)}] {seq_id}")

        # Find view directories
        view_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])

        for view_dir in view_dirs:
            n_views_str = view_dir.name

            if args.n_views:
                n = int(n_views_str.split('_')[0])
                if n not in args.n_views:
                    continue

            print(f"  {n_views_str}")

            sampled_dir = sampled_root / seq_id / n_views_str
            depth_dir = depth_root / seq_id / n_views_str

            # Process each method
            for method in args.methods:
                print(f"    {method}:")

                output_dir = output_root / seq_id / n_views_str / method

                try:
                    reconstruct_method(
                        method,
                        sampled_dir,
                        view_dir,  # sfm_dir for this view
                        depth_dir,
                        output_dir,
                        config,
                    )
                except Exception as e:
                    print(f"        Error: {e}")
                
                print(f"    {timer.step()}")

        print()

    timer.save_stats(str(output_root / "phase4_stats.json"))
    print("Phase 4 complete!")


if __name__ == '__main__':
    main()
