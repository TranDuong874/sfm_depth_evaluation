"""
Base class and output format for Structure-from-Motion methods.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SfMOutput:
    """
    Standardized SfM output format.

    Compatible with both COLMAP and MASt3R outputs.
    Can be consumed by depth estimation methods.

    Directory structure when saved:
        sfm_output/
        ├── metadata.json
        ├── sparse_points.ply
        ├── intrinsics/
        │   └── {image_stem}.txt  # 3x3 K matrix
        ├── poses/
        │   └── {image_stem}.txt  # 4x4 c2w matrix
        └── sparse_depth/
            └── {image_stem}.npz  # depth, [error], [n_views]
    """

    # Image information
    image_names: List[str] = field(default_factory=list)
    image_sizes: Dict[str, tuple] = field(default_factory=dict)  # name -> (H, W)

    # Camera parameters (per image)
    intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)  # 3x3 K
    poses: Dict[str, np.ndarray] = field(default_factory=dict)  # 4x4 c2w

    # Sparse depth (per image)
    sparse_depths: Dict[str, np.ndarray] = field(default_factory=dict)  # (H, W)
    sparse_errors: Dict[str, np.ndarray] = field(default_factory=dict)  # (H, W)
    sparse_n_views: Dict[str, np.ndarray] = field(default_factory=dict)  # (H, W)

    # Global point cloud
    points3d: Optional[np.ndarray] = None  # (N, 3)
    points3d_colors: Optional[np.ndarray] = None  # (N, 3)

    # Metadata (status, errors, etc.)
    metadata: Dict = field(default_factory=dict)

    def save(self, output_dir: str) -> None:
        """Save SfM output to directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (output_path / "intrinsics").mkdir(exist_ok=True)
        (output_path / "poses").mkdir(exist_ok=True)
        (output_path / "sparse_depth").mkdir(exist_ok=True)

        # Save per-image data
        for name in self.image_names:
            stem = Path(name).stem

            # Intrinsics
            if name in self.intrinsics:
                K_path = output_path / "intrinsics" / f"{stem}.txt"
                np.savetxt(str(K_path), self.intrinsics[name], fmt='%.10f')

            # Poses
            if name in self.poses:
                pose_path = output_path / "poses" / f"{stem}.txt"
                np.savetxt(str(pose_path), self.poses[name], fmt='%.10f')

            # Sparse depth
            if name in self.sparse_depths:
                depth_path = output_path / "sparse_depth" / f"{stem}.npz"
                save_dict = {'depth': self.sparse_depths[name]}
                if name in self.sparse_errors:
                    save_dict['error'] = self.sparse_errors[name]
                if name in self.sparse_n_views:
                    save_dict['n_views'] = self.sparse_n_views[name]
                np.savez(str(depth_path), **save_dict)

        # Save metadata
        metadata_dict = {
            'image_names': self.image_names,
            'image_sizes': {k: list(v) for k, v in self.image_sizes.items()},
            'n_images': len(self.image_names),
            'n_points': len(self.points3d) if self.points3d is not None else 0,
        }
        # Include custom metadata (status, error, etc.)
        metadata_dict.update(self.metadata)
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        # Save point cloud
        if self.points3d is not None:
            self.save_pointcloud_ply(str(output_path / "sparse_points.ply"))

        print(f"Saved SfM output to {output_path}")

    def save_pointcloud_ply(self, output_path: str) -> None:
        """Save point cloud to PLY file."""
        if self.points3d is None:
            return

        from utils.ply_utils import save_pointcloud_ply
        save_pointcloud_ply(output_path, self.points3d, self.points3d_colors)

    @classmethod
    def load(cls, input_dir: str) -> 'SfMOutput':
        """Load SfM output from directory."""
        input_path = Path(input_dir)

        # Load metadata
        with open(input_path / "metadata.json") as f:
            metadata = json.load(f)

        output = cls()
        output.image_names = metadata['image_names']
        output.image_sizes = {k: tuple(v) for k, v in metadata.get('image_sizes', {}).items()}

        # Load per-image data
        for name in output.image_names:
            stem = Path(name).stem

            # Intrinsics
            K_path = input_path / "intrinsics" / f"{stem}.txt"
            if K_path.exists():
                output.intrinsics[name] = np.loadtxt(str(K_path))

            # Poses
            pose_path = input_path / "poses" / f"{stem}.txt"
            if pose_path.exists():
                output.poses[name] = np.loadtxt(str(pose_path))

            # Sparse depth
            depth_path = input_path / "sparse_depth" / f"{stem}.npz"
            if depth_path.exists():
                data = np.load(str(depth_path))
                output.sparse_depths[name] = data['depth']
                if 'error' in data:
                    output.sparse_errors[name] = data['error']
                if 'n_views' in data:
                    output.sparse_n_views[name] = data['n_views']

        # Load point cloud
        ply_path = input_path / "sparse_points.ply"
        if ply_path.exists():
            from utils.ply_utils import load_pointcloud_ply
            output.points3d, output.points3d_colors = load_pointcloud_ply(str(ply_path))

        return output

    def export_colmap_format(self, output_dir: str) -> None:
        """Export to COLMAP binary format."""
        from utils.colmap_format import export_to_colmap_format

        export_to_colmap_format(
            output_dir=output_dir,
            intrinsics=self.intrinsics,
            poses=self.poses,
            image_sizes=self.image_sizes,
            points3d=self.points3d,
            colors=self.points3d_colors,
        )


class BaseSfM(ABC):
    """Abstract base class for SfM methods."""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name."""
        pass

    @abstractmethod
    def reconstruct(
        self,
        image_dir: str,
        output_dir: str,
        **kwargs
    ) -> SfMOutput:
        """
        Run SfM reconstruction.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save outputs

        Returns:
            SfMOutput with reconstruction results
        """
        pass

    def get_image_paths(self, image_dir: str) -> List[str]:
        """Get sorted list of image paths."""
        image_path = Path(image_dir)
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        paths = []
        for ext in extensions:
            paths.extend(image_path.glob(ext))
        return sorted([str(p) for p in paths])
