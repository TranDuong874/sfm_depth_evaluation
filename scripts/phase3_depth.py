#!/usr/bin/env python3
"""
Phase 3: Depth Estimation & Sequence-Level Calibration

Runs depth estimation methods on sampled images using SfM outputs.
Implements orthodox sequence-level scale calibration using SfM depth anchors.

Usage:
    python scripts/phase3_depth.py \
        --sampled output/phase1_sampled \
        --sfm output/phase2_sfm \
        --output output/phase3_depth

Output structure:
    output/phase3_depth/{seq_id}/{n}_views/{sfm_method}/{depth_method}/
    ├── frame_XXXXXX_depth.npy
    ├── scale_factors.json
    └── (for Murre: ../murre_calibrated/...)
"""

import argparse
import sys
import json
import shutil
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch

from config import DepthConfig, SFM_METHODS, DEPTH_METHODS


def compute_sequence_scale(raw_depths, anchors, masks):
    """
    Compute global scale factor s for the sequence.
    D_calibrated = s * D_raw
    s = median(D_anchor / D_raw)
    
    Domain: (D_anchor > 0) & (D_raw > 0) & (Mask > 0)
    """
    ratios = []
    
    for depth, anchor, mask in zip(raw_depths, anchors, masks):
        # Ensure dimensions match (they should, but safety first)
        if depth.shape != anchor.shape:
            continue
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        valid = (anchor > 0) & (depth > 0) & (mask > 0)
        
        if valid.sum() > 10:
            # s = Anchor / Raw
            r = anchor[valid] / depth[valid]
            ratios.append(r)
    
    if not ratios:
        return 1.0
    
    all_ratios = np.concatenate(ratios)
    scale_factor = float(np.median(all_ratios))
    return scale_factor


def run_sequence_depth_estimation(
    images_dir: Path,
    masks_dir: Path,
    sfm_dir: Path,
    output_root: Path, # seq_id/n_views/sfm_method
    depth_method: str,
    sfm_method: str,
    config: DepthConfig,
    device: str = 'cuda',
):
    """
    Run depth estimation and global calibration for a sequence.
    """
    from core.sfm.base import SfMOutput

    # Load SfM output
    sfm_output = SfMOutput.load(str(sfm_dir))
    
    # Check for undistorted images (COLMAP)
    if 'undistorted_images_path' in sfm_output.metadata:
        undistorted_path = Path(sfm_output.metadata['undistorted_images_path'])
        if undistorted_path.exists():
            print(f"      Using undistorted images from: {undistorted_path}")
            images_dir = undistorted_path

    # Define output directory
    output_dir = output_root / depth_method
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize depth estimator
    if depth_method == 'murre':
        from core.depth.murre import MURREEstimator
        estimator = MURREEstimator(
            device=device,
            processing_res=config.murre_processing_res,
            denoise_steps=config.murre_denoise_steps,
            ensemble_size=config.murre_ensemble_size,
            max_depth=config.murre_max_depth,
        )
    elif depth_method == 'metric3d':
        from core.depth.metric3d import Metric3DEstimator
        estimator = Metric3DEstimator(device=device)
    elif depth_method == 'unidepth':
        from core.depth.unidepth import UniDepthEstimator
        estimator = UniDepthEstimator(device=device)
    elif depth_method == 'depth_anything':
        from core.depth.depth_anything import DepthAnythingEstimator
        estimator = DepthAnythingEstimator(device=device, max_depth=config.max_depth)
    else:
        raise ValueError(f"Unknown depth method: {depth_method}")

    # Data collection
    raw_depths_map = {} # name -> np.array
    anchors_map = {}    # name -> np.array
    masks_map = {}      # name -> np.array
    
    print(f"      Processing {len(sfm_output.image_names)} images...")

    # -------------------------------------------------------------------------
    # PASS 1: Prediction (Raw Depth)
    # -------------------------------------------------------------------------
    for name in sfm_output.image_names:
        # Load RGB
        img_path = images_dir / name
        if not img_path.exists():
            print(f"        Warning: Image not found: {img_path}")
            continue

        rgb = cv2.imread(str(img_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        mask_path = masks_dir / name
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.ones(rgb.shape[:2], dtype=np.uint8)

        # Get SfM depth anchor (sparse depth)
        intrinsic = sfm_output.intrinsics.get(name)
        sfm_depth_anchor = sfm_output.sparse_depths.get(name)
        
        if sfm_depth_anchor is None:
            sfm_depth_anchor = np.zeros(rgb.shape[:2], dtype=np.float32)

        # Run prediction
        # For Murre: Must provide sparse_depth as input guidance
        # For Others: sparse_depth is ignored/optional (we removed _scale_with_sparse)
        result = estimator.predict(
            rgb,
            intrinsic=intrinsic,
            sparse_depth=sfm_depth_anchor if depth_method == 'murre' else None,
        )
        
        raw_depth = result['depth'].astype(np.float32)
        
        # Store
        raw_depths_map[name] = raw_depth
        anchors_map[name] = sfm_depth_anchor
        masks_map[name] = mask

    # -------------------------------------------------------------------------
    # PASS 2: Calibration
    # -------------------------------------------------------------------------
    
    # Compute global scale factor
    scale_factor = compute_sequence_scale(
        list(raw_depths_map.values()),
        list(anchors_map.values()),
        list(masks_map.values())
    )
    
    print(f"      Sequence Scale Factor: {scale_factor:.4f}")
    
    # Save scale factor
    with open(output_dir / "scale_factors.json", 'w') as f:
        json.dump({"scale_factor": scale_factor, "method": depth_method}, f, indent=2)

    # -------------------------------------------------------------------------
    # PASS 3: Saving
    # -------------------------------------------------------------------------
    
    # Murre Special Handling: Save Native AND Calibrated
    if depth_method == 'murre':
        calibrated_dir = output_root / f"{depth_method}_calibrated"
        calibrated_dir.mkdir(parents=True, exist_ok=True)
        
        with open(calibrated_dir / "scale_factors.json", 'w') as f:
            json.dump({"scale_factor": scale_factor, "method": f"{depth_method}_calibrated"}, f, indent=2)
            
        print(f"      Saving Murre (Native) and Murre (Calibrated)...")
    else:
        calibrated_dir = None
        print(f"      Saving Aligned Depth...")

    for name, raw_depth in raw_depths_map.items():
        stem = Path(name).stem
        
        if depth_method == 'murre':
            # 1. Native (Uncalibrated / Implicitly Calibrated)
            # This corresponds to "Murre already operates in SfM scale"
            native_path = output_dir / f"{stem}_depth.npy"
            np.save(str(native_path), raw_depth)
            
            # 2. Explicitly Calibrated (Optional check)
            calibrated_depth = raw_depth * scale_factor
            calib_path = calibrated_dir / f"{stem}_depth.npy"
            np.save(str(calib_path), calibrated_depth)
            
        else:
            # Standard Method: Always apply calibration
            aligned_depth = raw_depth * scale_factor
            out_path = output_dir / f"{stem}_depth.npy"
            np.save(str(out_path), aligned_depth)

    # Clean GPU
    del estimator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Phase 3: Depth Estimation')
    parser.add_argument('--sampled', type=str, default='output/phase1_sampled',
                        help='Sampled images directory (phase1 output)')
    parser.add_argument('--sfm', type=str, default='output/phase2_sfm',
                        help='SfM output directory (phase2 output)')
    parser.add_argument('--output', type=str, default='output/phase3_depth',
                        help='Output directory')
    parser.add_argument('--sfm-methods', type=str, nargs='+', default=SFM_METHODS,
                        help='SfM methods to use')
    parser.add_argument('--depth-methods', type=str, nargs='+', default=DEPTH_METHODS,
                        help='Depth methods to run')
    parser.add_argument('--n-views', type=int, nargs='+', default=None,
                        help='View counts to process (default: all)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Process specific sequence only')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')

    args = parser.parse_args()

    sampled_root = Path(args.sampled)
    sfm_root = Path(args.sfm)
    output_root = Path(args.output)

    if not sampled_root.exists():
        print(f"Error: Sampled directory not found: {sampled_root}")
        return
    if not sfm_root.exists():
        print(f"Error: SfM directory not found: {sfm_root}")
        return

    config = DepthConfig(methods=args.depth_methods)

    # Find sequences
    if args.sequence:
        seq_dirs = [sfm_root / args.sequence]
    else:
        seq_dirs = sorted([d for d in sfm_root.iterdir() if d.is_dir()])

    print(f"Found {len(seq_dirs)} sequences")
    print(f"SfM methods: {args.sfm_methods}")
    print(f"Depth methods: {args.depth_methods}")
    print(f"Output: {output_root}")
    print()

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

            # Get sampled images/masks directory
            # Structure: output/phase1_sampled/seq_id/n_views/images
            sampled_view_dir = sampled_root / seq_id / n_views_str
            images_dir = sampled_view_dir / "images"
            masks_dir = sampled_view_dir / "masks"
            
            if not images_dir.exists():
                print(f"    Warning: Images not found: {images_dir}")
                continue

            # Process each SfM method
            for sfm_method in args.sfm_methods:
                sfm_dir = view_dir / sfm_method
                if not sfm_dir.exists():
                    print(f"    {sfm_method}: SfM output not found, skipping")
                    continue

                print(f"    {sfm_method}:")
                
                # Output root for this SfM method
                method_output_root = output_root / seq_id / n_views_str / sfm_method

                # Run each depth method
                for depth_method in args.depth_methods:
                    # Check if already done
                    target_dir = method_output_root / depth_method
                    if target_dir.exists() and (target_dir / "scale_factors.json").exists():
                         print(f"      {depth_method}: Already done, skipping")
                         continue

                    print(f"      {depth_method}...")

                    try:
                        run_sequence_depth_estimation(
                            images_dir,
                            masks_dir,
                            sfm_dir,
                            method_output_root,
                            depth_method,
                            sfm_method,
                            config,
                            args.device,
                        )
                    except Exception as e:
                        print(f"        Error: {e}")
                        import traceback
                        traceback.print_exc()

        print()

    print("Phase 3 complete!")


if __name__ == '__main__':
    main()