"""
Core modules for SfM Depth Evaluation Pipeline.

This package contains the main components:
- co3d: CO3D dataset handling
- sfm: Structure-from-Motion methods (COLMAP, MASt3R)
- depth: Depth estimation methods (MURRE, Metric3D, UniDepth, DA2)
- reconstruction: Point cloud reconstruction from depth maps
- evaluation: Metrics computation
"""

from .co3d import CO3DSequence, find_sequences, get_sequence_id

__all__ = [
    "CO3DSequence",
    "find_sequences",
    "get_sequence_id",
]
