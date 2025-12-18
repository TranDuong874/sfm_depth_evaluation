"""
Central configuration for SfM Depth Evaluation Pipeline.

This module contains all configuration dataclasses and default paths
used throughout the pipeline.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# Project root directory
ROOT_DIR = Path(__file__).parent.absolute()


@dataclass
class PathConfig:
    """Path configuration for the pipeline."""

    # Input data
    co3d_root: Path = ROOT_DIR / "data" / "co3d"

    # Output directories
    output_root: Path = ROOT_DIR / "output"
    phase1_output: Path = field(default=None)
    phase2_output: Path = field(default=None)
    phase3_output: Path = field(default=None)
    phase4_output: Path = field(default=None)
    phase5_output: Path = field(default=None)

    # Checkpoints
    checkpoints_dir: Path = ROOT_DIR / "checkpoints"
    mast3r_checkpoint: Path = field(default=None)
    murre_checkpoint: Path = field(default=None)

    # Dependencies (keep untouched)
    dependency_dir: Path = ROOT_DIR / "dependency"
    mast3r_dir: Path = field(default=None)
    murre_dir: Path = field(default=None)

    def __post_init__(self):
        # Set default output paths
        if self.phase1_output is None:
            self.phase1_output = self.output_root / "phase1_sampled"
        if self.phase2_output is None:
            self.phase2_output = self.output_root / "phase2_sfm"
        if self.phase3_output is None:
            self.phase3_output = self.output_root / "phase3_depth"
        if self.phase4_output is None:
            self.phase4_output = self.output_root / "phase4_reconstruction"
        if self.phase5_output is None:
            self.phase5_output = self.output_root / "phase5_evaluation"

        # Set checkpoint paths
        if self.mast3r_checkpoint is None:
            self.mast3r_checkpoint = self.checkpoints_dir / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if self.murre_checkpoint is None:
            self.murre_checkpoint = self.checkpoints_dir / "Murre"

        # Set dependency paths
        if self.mast3r_dir is None:
            self.mast3r_dir = self.dependency_dir / "mast3r"
        if self.murre_dir is None:
            self.murre_dir = self.dependency_dir / "Murre"


@dataclass
class SamplingConfig:
    """Configuration for Phase 1: Sampling."""

    # Number of views to sample
    n_views: List[int] = field(default_factory=lambda: [5, 10, 20])

    # Sampling strategy
    strategy: str = "uniform_time"  # uniform_time, uniform, random

    # Image preprocessing
    max_edge: int = 512  # Maximum edge length
    multiple_of: int = 16  # Align to multiple (for ViT models)

    # Minimum time gap for random sampling (seconds)
    min_time_gap: float = 0.5


@dataclass
class SfMConfig:
    """Configuration for Phase 2: Structure-from-Motion."""

    # Methods to run
    methods: List[str] = field(default_factory=lambda: ["colmap", "mast3r"])

    # COLMAP settings
    colmap_camera_model: str = "OPENCV"
    colmap_single_camera: bool = False
    colmap_exhaustive_matching: bool = True

    # MASt3R settings
    mast3r_scene_graph: str = "complete"  # complete, swin-N, oneref
    mast3r_lr1: float = 0.07
    mast3r_niter1: int = 300
    mast3r_lr2: float = 0.01
    mast3r_niter2: int = 300
    mast3r_opt_depth: bool = True

    # Processing
    image_size: int = 512


@dataclass
class DepthConfig:
    """Configuration for Phase 3: Depth Estimation."""

    # Methods to run
    methods: List[str] = field(default_factory=lambda: [
        "murre", "metric3d", "unidepth", "depth_anything"
    ])

    # MURRE settings
    murre_processing_res: int = 512
    murre_denoise_steps: int = 10
    murre_ensemble_size: int = 10
    murre_max_depth: float = 10.0

    # General depth settings
    max_depth: float = 10.0
    min_depth: float = 0.1


@dataclass
class ReconstructionConfig:
    """Configuration for Phase 4: Reconstruction."""

    # Depth filtering
    depth_min: float = 0.1
    depth_max: float = 10.0

    # Point cloud cleaning
    statistical_outlier_nb_neighbors: int = 50
    statistical_outlier_std_ratio: float = 1.0

    # Subsampling
    stride: int = 1  # Pixel stride for backprojection


@dataclass
class EvaluationConfig:
    """Configuration for Phase 5: Evaluation."""

    # Point cloud processing
    max_eval_points: int = 100000  # Maximum points for evaluation
    normalize_scale: bool = True  # Normalize to unit sphere
    use_icp: bool = True  # Apply ICP alignment (Required since reconstruction is in SfM frame)

    # F-score thresholds (matching object_recon: 2%, 5%, 10%)
    f_score_thresholds: List[float] = field(default_factory=lambda: [0.02, 0.05, 0.10])

    # ICP settings
    icp_max_iterations: int = 200
    icp_threshold: float = 0.1


@dataclass
class PipelineConfig:
    """Main configuration combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    sfm: SfMConfig = field(default_factory=SfMConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Hardware
    device: str = "cuda"

    # Processing limits
    max_sequences: Optional[int] = None


# Default configuration instance
default_config = PipelineConfig()


def get_config() -> PipelineConfig:
    """Get the default pipeline configuration."""
    return PipelineConfig()


# Evaluation methods (all 10 methods)
EVALUATION_METHODS = [
    # Baselines (SfM sparse only)
    "colmap_sparse",
    "mast3r_sparse",
    # SfM + Depth combinations
    "colmap_murre",
    "colmap_metric3d",
    "colmap_unidepth",
    "colmap_depth_anything",
    "mast3r_murre",
    "mast3r_metric3d",
    "mast3r_unidepth",
    "mast3r_depth_anything",
]

SFM_METHODS = ["colmap", "mast3r"]
DEPTH_METHODS = ["murre", "metric3d", "unidepth", "depth_anything"]
