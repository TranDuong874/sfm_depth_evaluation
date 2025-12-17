"""
Structure-from-Motion methods for SfM Depth Evaluation Pipeline.

Provides COLMAP and MASt3R implementations with standardized output format.
"""

from .base import BaseSfM, SfMOutput
from .colmap import COLMAPSfM
from .mast3r import MASt3RSfM

__all__ = [
    "BaseSfM",
    "SfMOutput",
    "COLMAPSfM",
    "MASt3RSfM",
]
