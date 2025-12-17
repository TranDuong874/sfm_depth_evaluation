"""
Evaluation metrics for 3D reconstruction.

Computes Chamfer distance, F-score, and point density metrics.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    method: str
    sequence: str
    n_views: int
    chamfer_distance: float = 0.0
    f_score_2: float = 0.0  # F-score @ 2%
    f_score_5: float = 0.0  # F-score @ 5%
    f_score_10: float = 0.0  # F-score @ 10%
    precision_2: float = 0.0
    recall_2: float = 0.0
    precision_5: float = 0.0
    recall_5: float = 0.0
    precision_10: float = 0.0
    recall_10: float = 0.0
    point_density: float = 0.0
    n_pred_points: int = 0
    n_gt_points: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'sequence': self.sequence,
            'n_views': self.n_views,
            'chamfer_distance': self.chamfer_distance,
            'f_score_2': self.f_score_2,
            'f_score_5': self.f_score_5,
            'f_score_10': self.f_score_10,
            'precision_2': self.precision_2,
            'recall_2': self.recall_2,
            'precision_5': self.precision_5,
            'recall_5': self.recall_5,
            'precision_10': self.precision_10,
            'recall_10': self.recall_10,
            'point_density': self.point_density,
            'n_pred_points': self.n_pred_points,
            'n_gt_points': self.n_gt_points,
        }

    def save(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EvaluationResult':
        with open(path) as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k != 'metadata'})


def normalize_point_cloud(
    points: np.ndarray,
    method: str = 'unit_sphere',
) -> Tuple[np.ndarray, Dict]:
    """
    Normalize point cloud.

    Args:
        points: (N, 3) point cloud
        method: 'unit_sphere' or 'unit_cube'

    Returns:
        Tuple of (normalized_points, normalization_params)
    """
    centroid = points.mean(axis=0)
    points_centered = points - centroid

    if method == 'unit_sphere':
        scale = np.linalg.norm(points_centered, axis=1).max()
    else:  # unit_cube
        scale = np.abs(points_centered).max()

    if scale > 0:
        points_normalized = points_centered / scale
    else:
        points_normalized = points_centered

    params = {'centroid': centroid, 'scale': scale, 'method': method}
    return points_normalized, params


def chamfer_distance(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    return_components: bool = False,
) -> Union[float, Tuple[float, float, float]]:
    """
    Compute Chamfer Distance between two point clouds.

    CD = mean(dist_pred_to_gt) + mean(dist_gt_to_pred)

    Args:
        pred_points: (N, 3) predicted points
        gt_points: (M, 3) ground truth points
        return_components: If True, return (CD, pred2gt, gt2pred)

    Returns:
        Chamfer distance (lower is better)
    """
    try:
        from scipy.spatial import cKDTree

        tree_gt = cKDTree(gt_points)
        tree_pred = cKDTree(pred_points)

        dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
        dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)

        mean_pred_to_gt = np.mean(dist_pred_to_gt)
        mean_gt_to_pred = np.mean(dist_gt_to_pred)
        cd = mean_pred_to_gt + mean_gt_to_pred

    except ImportError:
        # Brute force fallback
        cd, mean_pred_to_gt, mean_gt_to_pred = _chamfer_brute_force(
            pred_points, gt_points
        )

    if return_components:
        return cd, mean_pred_to_gt, mean_gt_to_pred
    return cd


def _chamfer_brute_force(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[float, float, float]:
    """Brute force Chamfer distance."""
    # pred → gt
    dist_p2g = []
    for p in pred:
        dist_p2g.append(np.linalg.norm(gt - p, axis=1).min())

    # gt → pred
    dist_g2p = []
    for p in gt:
        dist_g2p.append(np.linalg.norm(pred - p, axis=1).min())

    mean_p2g = np.mean(dist_p2g)
    mean_g2p = np.mean(dist_g2p)
    return mean_p2g + mean_g2p, mean_p2g, mean_g2p


def f_score(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    threshold: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Compute F-score at given threshold.

    F = 2 * P * R / (P + R)

    Args:
        pred_points: (N, 3) predicted points
        gt_points: (M, 3) ground truth points
        threshold: Distance threshold

    Returns:
        Tuple of (f_score, precision, recall)
    """
    try:
        from scipy.spatial import cKDTree

        tree_gt = cKDTree(gt_points)
        tree_pred = cKDTree(pred_points)

        dist_pred_to_gt, _ = tree_gt.query(pred_points, k=1)
        precision = np.mean(dist_pred_to_gt < threshold)

        dist_gt_to_pred, _ = tree_pred.query(gt_points, k=1)
        recall = np.mean(dist_gt_to_pred < threshold)

    except ImportError:
        precision = _precision_brute_force(pred_points, gt_points, threshold)
        recall = _precision_brute_force(gt_points, pred_points, threshold)

    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    else:
        f = 0.0

    return f, precision, recall


def _precision_brute_force(
    source: np.ndarray, target: np.ndarray, threshold: float
) -> float:
    """Brute force precision."""
    count = 0
    for p in source:
        if np.linalg.norm(target - p, axis=1).min() < threshold:
            count += 1
    return count / len(source)


def icp_alignment(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 200,
    threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align source to target using ICP.

    Args:
        source: (N, 3) source points to transform
        target: (M, 3) target points (reference)
        max_iterations: Maximum ICP iterations
        threshold: Convergence threshold

    Returns:
        Tuple of (aligned_source, transformation_4x4)
    """
    try:
        import open3d as o3d

        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source)

        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(target)

        # Run ICP
        result = o3d.pipelines.registration.registration_icp(
            source_pcd,
            target_pcd,
            threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iterations
            ),
        )

        transformation = result.transformation
        source_aligned = source_pcd.transform(transformation)

        return np.asarray(source_aligned.points), transformation

    except ImportError:
        # Return unchanged if Open3D not available
        return source, np.eye(4)


def compute_point_density(points: np.ndarray, k: int = 10) -> float:
    """
    Compute average point density (points per unit volume).

    Args:
        points: (N, 3) point cloud
        k: Number of neighbors for local density estimation

    Returns:
        Average local density
    """
    if len(points) < k + 1:
        return 0.0

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=k + 1)

        # Use k-th nearest neighbor distance as local density estimate
        knn_distances = distances[:, k]
        local_volumes = (4/3) * np.pi * (knn_distances ** 3)
        local_densities = k / local_volumes

        return np.median(local_densities)

    except ImportError:
        return len(points)


def evaluate_reconstruction(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    method: str,
    sequence: str,
    n_views: int,
    normalize: bool = True,
    use_icp: bool = True,
    f_thresholds: List[float] = [0.01, 0.02],
    max_points: int = 100000,
) -> EvaluationResult:
    """
    Evaluate reconstructed point cloud against ground truth.

    Args:
        pred_points: (N, 3) predicted points
        gt_points: (M, 3) ground truth points
        method: Method name
        sequence: Sequence name
        n_views: Number of input views
        normalize: Normalize to unit sphere
        use_icp: Apply ICP alignment
        f_thresholds: F-score thresholds
        max_points: Maximum points for evaluation

    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult(
        method=method,
        sequence=sequence,
        n_views=n_views,
        n_pred_points=len(pred_points),
        n_gt_points=len(gt_points),
    )

    if len(pred_points) == 0 or len(gt_points) == 0:
        return result

    # Subsample if too many points
    if len(pred_points) > max_points:
        idx = np.random.choice(len(pred_points), max_points, replace=False)
        pred_points = pred_points[idx]

    if len(gt_points) > max_points:
        idx = np.random.choice(len(gt_points), max_points, replace=False)
        gt_points = gt_points[idx]

    # Normalize EACH point cloud INDEPENDENTLY to unit sphere
    # This is critical because pred and GT are in different coordinate systems
    if normalize:
        pred_points, _ = normalize_point_cloud(pred_points)
        gt_points, _ = normalize_point_cloud(gt_points)

    # ICP alignment (align normalized prediction to normalized GT)
    if use_icp:
        pred_points, _ = icp_alignment(pred_points, gt_points)

    # Compute metrics
    result.chamfer_distance = chamfer_distance(pred_points, gt_points)

    # F-score @ 2%
    f2, p2, r2 = f_score(pred_points, gt_points, threshold=f_thresholds[0])
    result.f_score_2 = f2
    result.precision_2 = p2
    result.recall_2 = r2

    # F-score @ 5%
    if len(f_thresholds) > 1:
        f5, p5, r5 = f_score(pred_points, gt_points, threshold=f_thresholds[1])
        result.f_score_5 = f5
        result.precision_5 = p5
        result.recall_5 = r5

    # F-score @ 10%
    if len(f_thresholds) > 2:
        f10, p10, r10 = f_score(pred_points, gt_points, threshold=f_thresholds[2])
        result.f_score_10 = f10
        result.precision_10 = p10
        result.recall_10 = r10

    # Point density
    result.point_density = compute_point_density(pred_points)

    return result


def aggregate_results(results: List[EvaluationResult]) -> Dict:
    """
    Aggregate evaluation results across sequences.

    Args:
        results: List of EvaluationResult

    Returns:
        Dict with mean metrics per method and view count
    """
    from collections import defaultdict

    aggregated = defaultdict(lambda: defaultdict(list))

    for r in results:
        key = (r.method, r.n_views)
        aggregated[key]['chamfer'].append(r.chamfer_distance)
        aggregated[key]['f_score_2'].append(r.f_score_2)
        aggregated[key]['f_score_5'].append(r.f_score_5)
        aggregated[key]['f_score_10'].append(r.f_score_10)

    summary = {}
    for (method, n_views), metrics in aggregated.items():
        key = f"{method}_{n_views}views"
        summary[key] = {
            'method': method,
            'n_views': n_views,
            'chamfer_mean': np.mean(metrics['chamfer']),
            'chamfer_std': np.std(metrics['chamfer']),
            'f_score_2_mean': np.mean(metrics['f_score_2']),
            'f_score_2_std': np.std(metrics['f_score_2']),
            'f_score_5_mean': np.mean(metrics['f_score_5']),
            'f_score_5_std': np.std(metrics['f_score_5']),
            'f_score_10_mean': np.mean(metrics['f_score_10']),
            'f_score_10_std': np.std(metrics['f_score_10']),
            'n_sequences': len(metrics['chamfer']),
        }

    return summary
