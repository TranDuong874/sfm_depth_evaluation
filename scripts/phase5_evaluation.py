#!/usr/bin/env python3
"""
Phase 5: Evaluation

Evaluates reconstructed point clouds against CO3D ground truth.
Includes geometric metrics (CD, F-Score) and photometric consistency checks.

Usage:
    python scripts/phase5_evaluation.py \
        --reconstructions output/phase4_reconstruction \
        --sampled output/phase1_sampled \
        --co3d data/co3d \
        --output output/phase5_evaluation

Output:
    output/phase5_evaluation/
    ├── results.csv
    ├── results_summary.json
    └── per_sequence/
        └── {seq_id}_{n}_views_{method}.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import cv2

from config import EvaluationConfig, EVALUATION_METHODS
from core.evaluation import (
    evaluate_reconstruction,
    aggregate_results,
    EvaluationResult,
)
from core.co3d import load_ply
from core.sfm.base import SfMOutput
from utils.timer import PipelineTimer


def load_gt_point_cloud(co3d_root: Path, seq_id: str, sampled_root: Path = None) -> np.ndarray:
    """
    Load ground truth point cloud from CO3D.
    """
    # Try to get original path from sampling_info.json
    if sampled_root is not None:
        sampling_info_path = sampled_root / seq_id
        for view_dir in sampling_info_path.iterdir() if sampling_info_path.exists() else []:
            info_file = view_dir / "sampling_info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                seq_path = Path(info.get('sequence_path', ''))
                if seq_path.exists():
                    # Try segmented point cloud first (object only)
                    for name in ['segmented_point_cloud.ply', 'point_cloud.ply', 'pointcloud.ply']:
                        gt_path = seq_path / name
                        if gt_path.exists():
                            points, _ = load_ply(gt_path)
                            return points
                break

    # Fallback: Search CO3D structure
    for category_dir in co3d_root.iterdir():
        if not category_dir.is_dir(): continue
        for object_dir in category_dir.iterdir():
            if not object_dir.is_dir(): continue
            for seq_dir in object_dir.iterdir():
                if not seq_dir.is_dir(): continue
                constructed_id = f"{category_dir.name}_{object_dir.name}_{seq_dir.name}"
                if constructed_id == seq_id:
                    for name in ['segmented_point_cloud.ply', 'point_cloud.ply', 'pointcloud.ply']:
                        gt_path = seq_dir / name
                        if gt_path.exists():
                            points, _ = load_ply(gt_path)
                            return points
    return None


def compute_reprojection_consistency(
    points: np.ndarray,
    colors: np.ndarray,
    sfm_dir: Path,
    images_dir: Path,
    max_points: int = 5000
) -> float:
    """
    Compute mean photometric reprojection error.
    Projects reconstructed points back to source images and compares color.
    
    Args:
        points: (N, 3) points
        colors: (N, 3) colors in [0, 1]
        sfm_dir: Path to SfM output directory
        images_dir: Path to images directory
        max_points: Max points to sample for speed
        
    Returns:
        Mean L1 error in RGB [0, 255]
    """
    try:
        sfm = SfMOutput.load(str(sfm_dir))
    except Exception:
        return -1.0

    if len(points) == 0:
        return 0.0

    # Subsample
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        pts = points[idx]
        cols = colors[idx]
    else:
        pts = points
        cols = colors

    total_error = 0.0
    total_count = 0

    for name in sfm.image_names:
        if name not in sfm.intrinsics or name not in sfm.poses:
            continue
            
        img_path = images_dir / name
        if not img_path.exists():
            continue
            
        # Load image (BGR) -> RGB
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        K = sfm.intrinsics[name]
        pose = sfm.poses[name] # C2W
        
        # Project points: P_cam = (P_world - t) @ R
        # pose = [R|t], X_w = R X_c + t => X_c = R^T (X_w - t)
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # (N, 3)
        pts_cam = (pts - t) @ R 
        
        # Filter Z > 0
        z = pts_cam[:, 2]
        valid_z = z > 0.01
        
        # Project
        x = (pts_cam[:, 0] / np.maximum(z, 1e-6)) * K[0, 0] + K[0, 2]
        y = (pts_cam[:, 1] / np.maximum(z, 1e-6)) * K[1, 1] + K[1, 2]
        
        # Filter bounds
        valid_x = (x >= 0) & (x < w-1)
        valid_y = (y >= 0) & (y < h-1)
        valid = valid_z & valid_x & valid_y
        
        if valid.sum() == 0:
            continue
            
        # Sample image
        x_valid = x[valid].astype(int)
        y_valid = y[valid].astype(int)
        
        sampled_colors = img[y_valid, x_valid] # (M, 3) in [0, 255]
        point_colors = cols[valid] * 255.0     # (M, 3) in [0, 255]
        
        # L1 Error
        diff = np.abs(sampled_colors - point_colors)
        total_error += diff.mean() * len(x_valid) # Weighted by count
        total_count += len(x_valid)

    if total_count == 0:
        return -1.0
        
    return total_error / total_count


def evaluate_method(
    reconstruction_dir: Path,
    gt_points: np.ndarray,
    method: str,
    seq_id: str,
    n_views: int,
    config: EvaluationConfig,
    sfm_dir: Path = None,
    images_dir: Path = None,
) -> EvaluationResult:
    """Evaluate a single method's reconstruction."""
    pred_path = reconstruction_dir / "object_pointcloud.ply"

    if not pred_path.exists():
        return EvaluationResult(method=method, sequence=seq_id, n_views=n_views)

    pred_points, pred_colors = load_ply(pred_path)

    if pred_points is None or len(pred_points) == 0:
        return EvaluationResult(method=method, sequence=seq_id, n_views=n_views)

    # Evaluation-only alignment (no feedback)
    result = evaluate_reconstruction(
        pred_points=pred_points,
        gt_points=gt_points,
        method=method,
        sequence=seq_id,
        n_views=n_views,
        normalize=config.normalize_scale,
        use_icp=config.use_icp,
        f_thresholds=config.f_score_thresholds,
        max_points=config.max_eval_points,
    )
    
    # Compute Reprojection Consistency (Optional)
    if sfm_dir and images_dir and pred_colors is not None:
        reproj_error = compute_reprojection_consistency(
            pred_points, pred_colors, sfm_dir, images_dir
        )
        if reproj_error >= 0:
            print(f"        Reproj Error: {reproj_error:.2f} px (RGB)")
            # Store in result metadata if supported, otherwise just print for now
            # (EvaluationResult class might need update to store this, skipping for mandatory scope)

    return result

def main():
    parser = argparse.ArgumentParser(description='Phase 5: Evaluation')
    parser.add_argument('--reconstructions', type=str, default='output/phase4_reconstruction',
                        help='Reconstructions directory (phase4 output)')
    parser.add_argument('--sampled', type=str, default='output/phase1_sampled',
                        help='Sampled images directory (phase1 output)')
    parser.add_argument('--co3d', type=str, default='data/co3d',
                        help='CO3D dataset root')
    parser.add_argument('--output', type=str, default='output/phase5_evaluation',
                        help='Output directory')
    parser.add_argument('--methods', type=str, nargs='+', default=EVALUATION_METHODS,
                        help='Methods to evaluate')
    parser.add_argument('--n-views', type=int, nargs='+', default=None,
                        help='View counts to evaluate (default: all)')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Evaluate specific sequence only')

    args = parser.parse_args()

    reconstructions_root = Path(args.reconstructions)
    sampled_root = Path(args.sampled)
    co3d_root = Path(args.co3d)
    output_root = Path(args.output)
    
    # Infer SfM root (Phase 2 output)
    sfm_root = reconstructions_root.parent / "phase2_sfm"

    if not reconstructions_root.exists():
        print(f"Error: Reconstructions not found: {reconstructions_root}")
        return

    config = EvaluationConfig()
    output_root.mkdir(parents=True, exist_ok=True)
    per_seq_dir = output_root / "per_sequence"
    per_seq_dir.mkdir(exist_ok=True)

    # Find sequences
    if args.sequence:
        seq_dirs = [reconstructions_root / args.sequence]
    else:
        seq_dirs = sorted([d for d in reconstructions_root.iterdir() if d.is_dir()])

    print(f"Found {len(seq_dirs)} sequences")
    
    # Initialize Timer
    estimated_steps = len(seq_dirs) * 3 * len(args.methods)
    if args.n_views:
        estimated_steps = len(seq_dirs) * len(args.n_views) * len(args.methods)
    timer = PipelineTimer(total_steps=estimated_steps, name="Phase5")

    # CSV file
    csv_path = output_root / "results.csv"
    csv_fields = [
        'sequence', 'n_views', 'method',
        'chamfer_distance', 'f_score_2', 'f_score_5', 'f_score_10',
        'precision_2', 'recall_2', 'precision_5', 'recall_5', 'precision_10', 'recall_10',
        'point_density', 'n_pred_points', 'n_gt_points'
    ]

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        for i, seq_dir in enumerate(seq_dirs):
            seq_id = seq_dir.name
            print(f"[{i+1}/{len(seq_dirs)}] {seq_id}")

            gt_points = load_gt_point_cloud(co3d_root, seq_id, sampled_root)
            if gt_points is None:
                print(f"  Warning: GT not found")
                continue

            view_dirs = sorted([d for d in seq_dir.iterdir() if d.is_dir()])

            for view_dir in view_dirs:
                n_views_str = view_dir.name
                if args.n_views:
                    if int(n_views_str.split('_')[0]) not in args.n_views:
                        continue
                
                print(f"  {n_views_str}")
                
                # Image dir for reprojection check
                images_dir = sampled_root / seq_id / n_views_str / "images"

                for method in args.methods:
                    reconstruction_dir = view_dir / method
                    if not reconstruction_dir.exists(): 
                        timer.step()
                        continue

                    print(f"    {method}:", end=" ")
                    
                    # Deduce SfM dir for this method
                    sfm_method = method.split('_')[0]
                    method_sfm_dir = sfm_root / seq_id / n_views_str / sfm_method

                    try:
                        result = evaluate_method(
                            reconstruction_dir, gt_points, method, seq_id,
                            int(n_views_str.split('_')[0]), config,
                            sfm_dir=method_sfm_dir,
                            images_dir=images_dir
                        )

                        row = result.to_dict()
                        writer.writerow(row)
                        csvfile.flush()
                        
                        result.save(per_seq_dir / f"{seq_id}_{n_views_str}_{method}.json")

                        if result.chamfer_distance > 0:
                            print(f"CD={result.chamfer_distance:.4f} | {timer.step()}")
                        else:
                            print(f"Failed | {timer.step()}")

                    except Exception as e:
                        print(f"Error: {e} | {timer.step()}")
            print()

    timer.save_stats(str(output_root / "phase5_stats.json"))
    print("\nPhase 5 complete!")


if __name__ == '__main__':
    main()