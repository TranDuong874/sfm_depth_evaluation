"""PLY file utilities for point cloud I/O."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def save_pointcloud_ply(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> None:
    """
    Save point cloud to PLY file.

    Args:
        filepath: Output PLY file path
        points: (N, 3) array of XYZ coordinates
        colors: Optional (N, 3) array of RGB values (0-255 or 0-1)
        normals: Optional (N, 3) array of normal vectors
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_points = len(points)

    # Normalize colors to 0-255 uint8
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if normals is not None:
        header_lines.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])

    if colors is not None:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header_lines.append("end_header\n")
    header = "\n".join(header_lines)

    # Write file
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))

        for i in range(n_points):
            # XYZ
            f.write(np.array(points[i], dtype=np.float32).tobytes())

            # Normals
            if normals is not None:
                f.write(np.array(normals[i], dtype=np.float32).tobytes())

            # Colors
            if colors is not None:
                f.write(np.array(colors[i], dtype=np.uint8).tobytes())


def save_pointcloud_ply_ascii(
    filepath: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> None:
    """
    Save point cloud to PLY file (ASCII format for debugging).

    Args:
        filepath: Output PLY file path
        points: (N, 3) array of XYZ coordinates
        colors: Optional (N, 3) array of RGB values
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_points = len(points)

    # Normalize colors
    if colors is not None:
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    with open(filepath, 'w') as f:
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
                line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
            f.write(line + "\n")


def load_pointcloud_ply(filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load point cloud from PLY file.

    Args:
        filepath: Input PLY file path

    Returns:
        (points, colors) tuple where colors may be None
    """
    filepath = Path(filepath)

    with open(filepath, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header
        n_points = 0
        has_colors = False
        is_binary = False

        for line in header_lines:
            if line.startswith("element vertex"):
                n_points = int(line.split()[-1])
            elif "red" in line or "green" in line or "blue" in line:
                has_colors = True
            elif "binary" in line:
                is_binary = True

        # Read data
        if is_binary:
            # Binary format
            points = []
            colors = [] if has_colors else None

            for _ in range(n_points):
                xyz = np.frombuffer(f.read(12), dtype=np.float32)
                points.append(xyz)
                if has_colors:
                    rgb = np.frombuffer(f.read(3), dtype=np.uint8)
                    colors.append(rgb)

            points = np.array(points)
            if colors:
                colors = np.array(colors)
        else:
            # ASCII format
            data = f.read().decode('ascii').strip().split('\n')
            points = []
            colors = [] if has_colors else None

            for line in data:
                parts = line.split()
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_colors and len(parts) >= 6:
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])

            points = np.array(points)
            if colors:
                colors = np.array(colors)

    return points, colors
