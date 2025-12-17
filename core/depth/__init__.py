"""
Depth estimation methods for SfM Depth Evaluation Pipeline.

Provides MURRE, Metric3D, UniDepth, and Depth Anything V2 implementations.
"""

from .base import BaseDepthEstimator
from .murre import MURREEstimator
from .metric3d import Metric3DEstimator
from .unidepth import UniDepthEstimator
from .depth_anything import DepthAnythingEstimator

__all__ = [
    "BaseDepthEstimator",
    "MURREEstimator",
    "Metric3DEstimator",
    "UniDepthEstimator",
    "DepthAnythingEstimator",
]


def get_depth_estimator(
    method: str,
    device: str = 'cuda',
    **kwargs
) -> BaseDepthEstimator:
    """
    Factory function to create depth estimator.

    Args:
        method: One of 'murre', 'metric3d', 'unidepth', 'depth_anything'
        device: Device to run on
        **kwargs: Method-specific arguments

    Returns:
        Depth estimator instance
    """
    method = method.lower()

    if method == 'murre':
        return MURREEstimator(device=device, **kwargs)
    elif method == 'metric3d':
        return Metric3DEstimator(device=device, **kwargs)
    elif method == 'unidepth':
        return UniDepthEstimator(device=device, **kwargs)
    elif method in ['depth_anything', 'da2', 'depth_anything_v2']:
        return DepthAnythingEstimator(device=device, **kwargs)
    else:
        raise ValueError(f"Unknown depth method: {method}")
