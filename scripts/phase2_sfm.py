#!/usr/bin/env python3
"""
Phase 2: Structure-from-Motion

Runs COLMAP and MASt3R on sampled images to get:
- Camera poses
- Camera intrinsics
- Sparse depth maps
- Sparse point clouds

Usage:
    python scripts/phase2_sfm.py --input output/phase1_sampled --output output/phase2_sfm

Output structure:
    output/phase2_sfm/{seq_id}/{n}_views/
    ├── colmap/
    │   ├── metadata.json
    │   ├── sparse_points.ply
    │   ├── intrinsics/
    │   ├── poses/
    │   └── sparse_depth/
    └── mast3r/
        └── (same structure)
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import SfMConfig, PathConfig, SFM_METHODS


def run_sfm_for_sequence(
    sampled_dir: Path,
    output_dir: Path,
    methods: list,
    config: SfMConfig,
    device: str = 'cuda',
):
    """
    Run SfM methods on a sampled sequence.

    Args:
        sampled_dir: Directory with sampled images
        output_dir: Output directory for SfM results
        methods: List of SfM methods to run
        config: SfM configuration
        device: Device for GPU methods
    """
    images_dir = sampled_dir / "images"

    if not images_dir.exists():
        print(f"    Warning: Images not found: {images_dir}")
        return

    # Check number of images
    n_images = len(list(images_dir.glob("*.png")))
    if n_images < 2:
        print(f"    Warning: Need at least 2 images, found {n_images}")
        return

    for method in methods:
        method_output = output_dir / method
        if method_output.exists() and (method_output / "metadata.json").exists():
            print(f"    {method}: Already exists, skipping")
            continue

        print(f"    Running {method}...")

        try:
            if method == 'colmap':
                from core.sfm.colmap import COLMAPSfM

                sfm = COLMAPSfM(device=device)
                sfm_output = sfm.reconstruct(
                    image_dir=str(images_dir),
                    output_dir=str(method_output),
                    camera_model=config.colmap_camera_model,
                    single_camera=config.colmap_single_camera,
                    exhaustive_matching=config.colmap_exhaustive_matching,
                )

            elif method == 'mast3r':
                from core.sfm.mast3r import MASt3RSfM

                sfm = MASt3RSfM(device=device, image_size=config.image_size)
                sfm_output = sfm.reconstruct(
                    image_dir=str(images_dir),
                    output_dir=str(method_output),
                    scene_graph=config.mast3r_scene_graph,
                    lr1=config.mast3r_lr1,
                    niter1=config.mast3r_niter1,
                    lr2=config.mast3r_lr2,
                    niter2=config.mast3r_niter2,
                    opt_depth=config.mast3r_opt_depth,
                )

            print(f"      {len(sfm_output.image_names)} images reconstructed")
            if sfm_output.points3d is not None:
                print(f"      {len(sfm_output.points3d)} 3D points")

        except Exception as e:
            print(f"      Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Structure-from-Motion')
    parser.add_argument('--input', type=str, default='output/phase1_sampled',
                        help='Input directory (phase1 output)')
    parser.add_argument('--output', type=str, default='output/phase2_sfm',
                        help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=SFM_METHODS,
                        choices=SFM_METHODS,
                        help='SfM methods to run')
    parser.add_argument('--n-views', type=int, nargs='+', default=None,
                        help='View counts to process (default: all)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Process specific sequence only')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for GPU methods')

    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        print(f"Error: Input not found: {input_root}")
        return

    config = SfMConfig(methods=args.methods)

    # Find sequences
    if args.sequence:
        seq_dirs = [input_root / args.sequence]
    else:
        seq_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])

    print(f"Found {len(seq_dirs)} sequences")
    print(f"Methods: {args.methods}")
    print(f"Output: {output_root}")
    print()

    # Process sequences
    for i, seq_dir in enumerate(seq_dirs):
        seq_id = seq_dir.name
        print(f"[{i+1}/{len(seq_dirs)}] {seq_id}")

        # Find view directories
        view_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])

        for view_dir in view_dirs:
            n_views_str = view_dir.name  # e.g., "5_views"

            # Filter by n_views if specified
            if args.n_views:
                n = int(n_views_str.split('_')[0])
                if n not in args.n_views:
                    continue

            print(f"  {n_views_str}")

            output_dir = output_root / seq_id / n_views_str
            run_sfm_for_sequence(
                view_dir,
                output_dir,
                args.methods,
                config,
                args.device,
            )

        print()

    print("Phase 2 complete!")


if __name__ == '__main__':
    main()
