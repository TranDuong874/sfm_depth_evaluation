"""COLMAP format export utilities.

Exports SfM results to COLMAP binary format (cameras.bin, images.bin, points3D.bin).
This allows using the results with any tool that supports COLMAP format.
"""

import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx = R[0, 0], R[1, 0], R[2, 0]
    Rxy, Ryy, Rzy = R[0, 1], R[1, 1], R[2, 1]
    Rxz, Ryz, Rzz = R[0, 2], R[1, 2], R[2, 2]

    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0

    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[:, np.argmax(eigvals)]

    if qvec[3] < 0:
        qvec *= -1

    return np.array([qvec[3], qvec[0], qvec[1], qvec[2]])


def write_cameras_binary(
    filepath: str,
    intrinsics: Dict[str, np.ndarray],
    image_sizes: Dict[str, tuple],
    camera_model: str = "PINHOLE",
) -> Dict[str, int]:
    """
    Write cameras.bin file.

    Args:
        filepath: Output file path
        intrinsics: Dict mapping image name to 3x3 K matrix
        image_sizes: Dict mapping image name to (H, W)
        camera_model: Camera model type

    Returns:
        Dict mapping image name to camera_id
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Camera model IDs
    CAMERA_MODELS = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
    }
    model_id = CAMERA_MODELS.get(camera_model, 1)

    # Group images by intrinsics
    # For simplicity, we create one camera per image
    image_to_camera = {}

    with open(filepath, 'wb') as f:
        # Number of cameras
        f.write(struct.pack('Q', len(intrinsics)))

        for camera_id, (name, K) in enumerate(intrinsics.items(), 1):
            h, w = image_sizes.get(name, (480, 640))

            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            # Write camera
            f.write(struct.pack('I', camera_id))  # camera_id
            f.write(struct.pack('i', model_id))   # model_id
            f.write(struct.pack('Q', w))          # width
            f.write(struct.pack('Q', h))          # height

            # Parameters depend on model
            if model_id == 0:  # SIMPLE_PINHOLE
                f.write(struct.pack('ddd', fx, cx, cy))
            elif model_id == 1:  # PINHOLE
                f.write(struct.pack('dddd', fx, fy, cx, cy))
            elif model_id == 4:  # OPENCV
                f.write(struct.pack('dddddddd', fx, fy, cx, cy, 0, 0, 0, 0))
            else:
                f.write(struct.pack('dddd', fx, fy, cx, cy))

            image_to_camera[name] = camera_id

    return image_to_camera


def write_images_binary(
    filepath: str,
    poses: Dict[str, np.ndarray],
    image_to_camera: Dict[str, int],
    point2D_to_point3D: Optional[Dict[str, List]] = None,
) -> Dict[str, int]:
    """
    Write images.bin file.

    Args:
        filepath: Output file path
        poses: Dict mapping image name to 4x4 camera-to-world matrix
        image_to_camera: Dict mapping image name to camera_id
        point2D_to_point3D: Optional dict mapping image name to list of
                           (x, y, point3D_id) tuples

    Returns:
        Dict mapping image name to image_id
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    image_to_id = {}

    with open(filepath, 'wb') as f:
        # Number of images
        f.write(struct.pack('Q', len(poses)))

        for image_id, (name, pose) in enumerate(poses.items(), 1):
            # Convert camera-to-world to world-to-camera
            R = pose[:3, :3].T
            t = -R @ pose[:3, 3]

            # Convert to quaternion
            qvec = rotmat2qvec(R)

            # Write image data
            f.write(struct.pack('I', image_id))           # image_id
            f.write(struct.pack('dddd', *qvec))           # qvec (w, x, y, z)
            f.write(struct.pack('ddd', *t))               # tvec
            f.write(struct.pack('I', image_to_camera.get(name, 1)))  # camera_id

            # Image name (null-terminated)
            f.write(name.encode('utf-8'))
            f.write(b'\x00')

            # 2D points
            if point2D_to_point3D and name in point2D_to_point3D:
                points2D = point2D_to_point3D[name]
                f.write(struct.pack('Q', len(points2D)))
                for x, y, p3d_id in points2D:
                    f.write(struct.pack('dd', x, y))
                    f.write(struct.pack('q', p3d_id))
            else:
                f.write(struct.pack('Q', 0))

            image_to_id[name] = image_id

    return image_to_id


def write_points3D_binary(
    filepath: str,
    points3d: np.ndarray,
    colors: Optional[np.ndarray] = None,
    errors: Optional[np.ndarray] = None,
    image_ids_per_point: Optional[List[List[int]]] = None,
) -> None:
    """
    Write points3D.bin file.

    Args:
        filepath: Output file path
        points3d: (N, 3) array of 3D points
        colors: Optional (N, 3) array of RGB colors (0-255)
        errors: Optional (N,) array of reprojection errors
        image_ids_per_point: Optional list of image_id lists per point
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_points = len(points3d)

    if colors is None:
        colors = np.ones((n_points, 3), dtype=np.uint8) * 128

    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    if errors is None:
        errors = np.ones(n_points) * 0.5

    with open(filepath, 'wb') as f:
        # Number of points
        f.write(struct.pack('Q', n_points))

        for i in range(n_points):
            # Point3D_id
            f.write(struct.pack('Q', i + 1))

            # XYZ
            f.write(struct.pack('ddd', *points3d[i]))

            # RGB
            f.write(struct.pack('BBB', *colors[i]))

            # Error
            f.write(struct.pack('d', errors[i]))

            # Track (image_id, point2D_idx pairs)
            if image_ids_per_point and i < len(image_ids_per_point):
                track = image_ids_per_point[i]
                f.write(struct.pack('Q', len(track)))
                for img_id in track:
                    f.write(struct.pack('I', img_id))
                    f.write(struct.pack('I', 0))  # point2D_idx placeholder
            else:
                f.write(struct.pack('Q', 0))


def export_to_colmap_format(
    output_dir: str,
    intrinsics: Dict[str, np.ndarray],
    poses: Dict[str, np.ndarray],
    image_sizes: Dict[str, tuple],
    points3d: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
    camera_model: str = "PINHOLE",
) -> None:
    """
    Export SfM results to full COLMAP format.

    Args:
        output_dir: Output directory (will create sparse/0/ structure)
        intrinsics: Dict mapping image name to 3x3 K matrix
        poses: Dict mapping image name to 4x4 camera-to-world matrix
        image_sizes: Dict mapping image name to (H, W)
        points3d: Optional (N, 3) array of 3D points
        colors: Optional (N, 3) array of RGB colors
        camera_model: Camera model type
    """
    output_path = Path(output_dir) / "sparse" / "0"
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to COLMAP format: {output_path}")

    # Write cameras
    image_to_camera = write_cameras_binary(
        str(output_path / "cameras.bin"),
        intrinsics,
        image_sizes,
        camera_model,
    )
    print(f"  Wrote {len(image_to_camera)} cameras")

    # Write images
    image_to_id = write_images_binary(
        str(output_path / "images.bin"),
        poses,
        image_to_camera,
    )
    print(f"  Wrote {len(image_to_id)} images")

    # Write points3D
    if points3d is not None:
        write_points3D_binary(
            str(output_path / "points3D.bin"),
            points3d,
            colors,
        )
        print(f"  Wrote {len(points3d)} points")
    else:
        # Write empty points file
        with open(output_path / "points3D.bin", 'wb') as f:
            f.write(struct.pack('Q', 0))
        print("  Wrote 0 points (no point cloud)")

    print("  Done!")
