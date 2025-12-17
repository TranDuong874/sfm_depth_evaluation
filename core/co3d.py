"""
CO3D Dataset Utilities for SfM Depth Evaluation Pipeline.

Handles CO3D video sequences: frame extraction, mask extraction,
and ground truth point cloud loading.

CO3D Dataset Structure:
    co3d/
    ├── <category>/
    │   └── <object_name>/
    │       └── <sequence_id>/
    │           ├── rgb_video.mp4
    │           ├── mask_video.mkv
    │           ├── depth_maps.h5
    │           ├── point_cloud.ply
    │           └── segmented_point_cloud.ply
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


class CO3DSequence:
    """
    Handler for a single CO3D video sequence.

    Provides methods to:
    - Extract RGB frames with various sampling strategies
    - Extract foreground masks
    - Load ground truth point clouds

    Example:
        >>> seq = CO3DSequence("data/co3d/apple/123_456_789")
        >>> frames, masks = seq.extract_frames(n_frames=10, strategy="uniform_time")
        >>> gt_points, gt_colors = seq.load_point_cloud()
    """

    def __init__(self, sequence_path: Union[str, Path]):
        """
        Initialize CO3D sequence handler.

        Args:
            sequence_path: Path to sequence directory containing rgb_video.mp4
        """
        self.seq_path = Path(sequence_path)

        if not self.seq_path.exists():
            raise FileNotFoundError(f"Sequence not found: {self.seq_path}")

        # Define file paths
        self.rgb_video = self.seq_path / "rgb_video.mp4"
        self.mask_video = self.seq_path / "mask_video.mkv"
        self.depth_h5 = self.seq_path / "depth_maps.h5"
        self.point_cloud_full = self.seq_path / "point_cloud.ply"
        self.point_cloud_segmented = self.seq_path / "segmented_point_cloud.ply"

        if not self.rgb_video.exists():
            raise FileNotFoundError(f"RGB video not found: {self.rgb_video}")

        # Load video properties
        self._load_video_properties()

    def _load_video_properties(self):
        """Load video metadata."""
        cap = cv2.VideoCapture(str(self.rgb_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.rgb_video}")

        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS)
        self._frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_size(self) -> Tuple[int, int]:
        """(width, height)"""
        return (self._frame_width, self._frame_height)

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._total_frames / self._fps if self._fps > 0 else 0

    @property
    def category(self) -> str:
        """CO3D category name."""
        return self.seq_path.parent.parent.name

    @property
    def object_name(self) -> str:
        """Object name within category."""
        return self.seq_path.parent.name

    @property
    def sequence_id(self) -> str:
        """Sequence ID."""
        return self.seq_path.name

    def get_frame_indices(
        self,
        n_frames: int,
        strategy: str = "uniform_time",
        min_time_gap: float = 0.5,
    ) -> np.ndarray:
        """
        Compute frame indices to sample.

        Args:
            n_frames: Number of frames to sample
            strategy: Sampling strategy
                - "uniform_time": Evenly spaced in time (recommended)
                - "uniform": Evenly spaced by frame index
                - "random": Random selection
            min_time_gap: Minimum time gap for random strategy (seconds)

        Returns:
            Array of frame indices
        """
        total = self._total_frames
        n_frames = min(n_frames, total)

        if strategy == "uniform_time":
            # Evenly spaced in time
            start_time = 0
            end_time = (total - 1) / self._fps
            sample_times = np.linspace(start_time, end_time, n_frames)
            indices = np.round(sample_times * self._fps).astype(int)
            indices = np.clip(indices, 0, total - 1)
            indices = np.unique(indices)

            # Fill in if duplicates removed
            while len(indices) < n_frames and len(indices) < total:
                gaps = np.diff(indices)
                if len(gaps) == 0:
                    break
                largest_gap_idx = np.argmax(gaps)
                new_frame = (indices[largest_gap_idx] + indices[largest_gap_idx + 1]) // 2
                if new_frame not in indices:
                    indices = np.sort(np.append(indices, new_frame))
                else:
                    break

        elif strategy == "uniform":
            # Evenly spaced by index
            indices = np.linspace(0, total - 1, n_frames, dtype=int)

        elif strategy == "random":
            np.random.seed(42)  # Reproducible
            indices = np.sort(np.random.choice(total, n_frames, replace=False))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return indices

    def extract_frames(
        self,
        n_frames: Optional[int] = None,
        frame_indices: Optional[np.ndarray] = None,
        strategy: str = "uniform_time",
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Extract RGB frames from video.

        Args:
            n_frames: Number of frames to extract
            frame_indices: Specific frame indices (overrides n_frames)
            strategy: Sampling strategy if using n_frames
            output_dir: Directory to save frames as PNG

        Returns:
            Tuple of (frames, frame_indices)
            - frames: List of RGB images (H, W, 3)
            - frame_indices: List of extracted frame indices
        """
        if frame_indices is None:
            if n_frames is None:
                raise ValueError("Provide n_frames or frame_indices")
            frame_indices = self.get_frame_indices(n_frames, strategy)

        cap = cv2.VideoCapture(str(self.rgb_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.rgb_video}")

        frames = []
        extracted_indices = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                print(f"[WARNING] Failed to read frame {frame_idx}")
                continue

            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            extracted_indices.append(int(frame_idx))

            if output_dir:
                # Save with frame index in filename
                frame_path = output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        cap.release()
        return frames, extracted_indices

    def extract_masks(
        self,
        frame_indices: np.ndarray,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[np.ndarray]:
        """
        Extract foreground masks for specific frames.

        Args:
            frame_indices: Frame indices to extract masks for
            output_dir: Directory to save masks as PNG

        Returns:
            List of binary masks (H, W) with values 0 or 1
        """
        if not self.mask_video.exists():
            raise FileNotFoundError(f"Mask video not found: {self.mask_video}")

        cap = cv2.VideoCapture(str(self.mask_video))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open mask video: {self.mask_video}")

        masks = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, frame = cap.read()

            if not ret:
                print(f"[WARNING] Failed to read mask frame {frame_idx}")
                # Create empty mask
                masks.append(np.ones((self._frame_height, self._frame_width), dtype=np.uint8))
                continue

            # Convert to binary mask
            mask = (frame[:, :, 0] > 127).astype(np.uint8)
            masks.append(mask)

            if output_dir:
                mask_path = output_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(mask_path), mask * 255)

        cap.release()
        return masks

    def load_point_cloud(
        self,
        cloud_type: str = "segmented",
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Load ground truth point cloud.

        Args:
            cloud_type: "segmented" (object only) or "full" (entire scene)

        Returns:
            Tuple of (points, colors)
            - points: Nx3 array of XYZ coordinates
            - colors: Nx3 array of RGB colors (0-255) or None
        """
        if cloud_type == "segmented":
            ply_path = self.point_cloud_segmented
        elif cloud_type == "full":
            ply_path = self.point_cloud_full
        else:
            raise ValueError(f"Unknown cloud_type: {cloud_type}")

        if not ply_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {ply_path}")

        return load_ply(ply_path)

    def get_info(self) -> Dict:
        """Get sequence information as dictionary."""
        return {
            "path": str(self.seq_path),
            "category": self.category,
            "object_name": self.object_name,
            "sequence_id": self.sequence_id,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "frame_size": self.frame_size,
            "has_masks": self.mask_video.exists(),
            "has_gt_pointcloud": self.point_cloud_segmented.exists(),
        }


def find_sequences(
    co3d_root: Union[str, Path],
    categories: Optional[List[str]] = None,
    max_sequences: Optional[int] = None,
) -> List[Path]:
    """
    Find all CO3D sequences in the dataset.

    Args:
        co3d_root: Root directory of CO3D dataset
        categories: Filter by categories (None = all)
        max_sequences: Maximum number of sequences to return

    Returns:
        List of sequence directory paths
    """
    co3d_root = Path(co3d_root)
    sequences = []

    # Find all rgb_video.mp4 files
    for video_path in co3d_root.rglob("rgb_video.mp4"):
        seq_path = video_path.parent

        # Filter by category if specified
        if categories is not None:
            category = seq_path.parent.parent.name
            if category not in categories:
                continue

        sequences.append(seq_path)

        if max_sequences and len(sequences) >= max_sequences:
            break

    return sorted(sequences)


def get_sequence_id(
    sequence_path: Union[str, Path],
    co3d_root: Optional[Union[str, Path]] = None,
) -> str:
    """
    Get a unique ID for a sequence.

    Args:
        sequence_path: Path to sequence directory
        co3d_root: CO3D root directory (for relative path computation)

    Returns:
        Sequence ID string (category_object_sequence)
    """
    sequence_path = Path(sequence_path)

    if co3d_root is not None:
        co3d_root = Path(co3d_root)
        try:
            rel_path = sequence_path.relative_to(co3d_root)
            return str(rel_path).replace("/", "_").replace("\\", "_")
        except ValueError:
            pass

    # Fallback: use last 3 parts of path
    parts = sequence_path.parts
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
    return sequence_path.name


def load_ply(ply_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load point cloud from PLY file.

    Args:
        ply_path: Path to PLY file

    Returns:
        Tuple of (points, colors)
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255 if pcd.has_colors() else None
        return points, colors
    except ImportError:
        # Fallback: manual parsing
        return _parse_ply_ascii(ply_path)


def _parse_ply_ascii(ply_path: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Manual PLY parser (ASCII format only)."""
    with open(ply_path, 'rb') as f:
        # Read header
        n_vertices = 0
        has_colors = False
        is_binary = False

        while True:
            line = f.readline().decode('ascii').strip()
            if line.startswith("element vertex"):
                n_vertices = int(line.split()[-1])
            elif "red" in line.lower():
                has_colors = True
            elif "binary" in line.lower():
                is_binary = True
            elif line == "end_header":
                break

        if is_binary:
            # Binary format
            points = []
            colors = [] if has_colors else None

            for _ in range(n_vertices):
                xyz = np.frombuffer(f.read(12), dtype=np.float32)
                points.append(xyz)
                if has_colors:
                    rgb = np.frombuffer(f.read(3), dtype=np.uint8)
                    colors.append(rgb)

            points = np.array(points)
            colors = np.array(colors) if colors else None
        else:
            # ASCII format
            points = []
            colors = [] if has_colors else None

            for _ in range(n_vertices):
                line = f.readline().decode('ascii').strip()
                values = line.split()
                points.append([float(values[0]), float(values[1]), float(values[2])])
                if has_colors and len(values) >= 6:
                    colors.append([int(float(values[3])), int(float(values[4])), int(float(values[5]))])

            points = np.array(points)
            colors = np.array(colors) if colors else None

    return points, colors


def save_ply(
    ply_path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Save point cloud to PLY file.

    Args:
        ply_path: Output path
        points: Nx3 array of XYZ coordinates
        colors: Nx3 array of RGB colors (0-255) or None
    """
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            if colors.max() > 1:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(str(ply_path), pcd)
    except ImportError:
        # Fallback: manual save
        _save_ply_ascii(ply_path, points, colors)


def _save_ply_ascii(
    ply_path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """Manual PLY writer (ASCII format)."""
    ply_path = Path(ply_path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    n_points = len(points)

    with open(ply_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # Data
        for i in range(n_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                c = colors[i]
                if c.max() <= 1:
                    c = c * 255
                line += f" {int(c[0])} {int(c[1])} {int(c[2])}"
            f.write(line + "\n")
