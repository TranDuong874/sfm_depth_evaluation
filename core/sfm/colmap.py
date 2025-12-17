"""
COLMAP Structure-from-Motion implementation.
"""

import os
import struct
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import BaseSfM, SfMOutput


class COLMAPSfM(BaseSfM):
    """COLMAP-based Structure-from-Motion."""

    def __init__(
        self,
        colmap_path: str = 'colmap',
        device: str = 'cuda',
    ):
        super().__init__(device)
        self.colmap_path = colmap_path
        self.use_gpu = device == 'cuda'

    @property
    def name(self) -> str:
        return "colmap"

    def _check_colmap(self) -> bool:
        """Check if COLMAP is available."""
        try:
            result = subprocess.run(
                [self.colmap_path, 'help'],
                capture_output=True,
                text=True,
                env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def reconstruct(
        self,
        image_dir: str,
        output_dir: str,
        camera_model: str = 'OPENCV',
        single_camera: bool = False,
        exhaustive_matching: bool = True,
        max_num_features: int = 8192,
        **kwargs
    ) -> SfMOutput:
        """
        Run COLMAP SfM reconstruction.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save outputs
            camera_model: Camera model (SIMPLE_PINHOLE, PINHOLE, OPENCV)
            single_camera: Use single camera for all images
            exhaustive_matching: Use exhaustive matching

        Returns:
            SfMOutput with reconstruction results
        """
        if not self._check_colmap():
            raise RuntimeError(
                "COLMAP not installed. Install from: https://colmap.github.io/"
            )

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        database_path = output_path / "database.db"
        sparse_path = output_path / "sparse"
        sparse_path.mkdir(exist_ok=True)

        print(f"[COLMAP] Processing images from {image_dir}")

        # Feature extraction
        print("  Extracting features...")
        self._run_colmap([
            'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', image_dir,
            '--ImageReader.camera_model', camera_model,
            '--ImageReader.single_camera', '1' if single_camera else '0',
            '--SiftExtraction.use_gpu', '1' if self.use_gpu else '0',
            '--SiftExtraction.max_num_features', str(max_num_features),
        ])

        # Feature matching
        print("  Matching features...")
        matcher = 'exhaustive_matcher' if exhaustive_matching else 'sequential_matcher'
        self._run_colmap([
            matcher,
            '--database_path', str(database_path),
            '--SiftMatching.use_gpu', '1' if self.use_gpu else '0',
        ])

        # Reconstruction
        print("  Running reconstruction...")
        try:
            self._run_colmap([
                'mapper',
                '--database_path', str(database_path),
                '--image_path', image_dir,
                '--output_path', str(sparse_path),
            ])
        except RuntimeError as e:
            # Mapper failed
            print(f"  Reconstruction failed: {e}")
            sfm_output = SfMOutput()
            sfm_output.metadata['status'] = 'failed'
            sfm_output.metadata['error'] = str(e)
            sfm_output.metadata['n_images'] = len(list(Path(image_dir).glob('*.png')))
            sfm_output.save(str(output_path))
            return sfm_output

        # Find reconstruction
        recon_dirs = list(sparse_path.glob('*'))
        if not recon_dirs:
            print("  Reconstruction failed - no sparse model created")
            sfm_output = SfMOutput()
            sfm_output.metadata['status'] = 'failed'
            sfm_output.metadata['error'] = 'No sparse model created'
            sfm_output.save(str(output_path))
            return sfm_output
            
        raw_recon_path = recon_dirs[0]

        # Undistort images (Optional but recommended for depth estimation)
        # This converts the camera model to PINHOLE and undistorts images
        print("  Undistorting images...")
        dense_path = output_path / "dense"
        try:
            self._run_colmap([
                'image_undistorter',
                '--image_path', image_dir,
                '--input_path', str(raw_recon_path),
                '--output_path', str(dense_path),
                '--output_type', 'COLMAP',
                '--max_image_size', str(max(cv2.imread(str(list(Path(image_dir).glob('*.png'))[0])).shape[:2])), # Keep size
            ])
            
            # Use the undistorted reconstruction (PINHOLE)
            final_recon_path = dense_path / "sparse"
            final_image_dir = dense_path / "images"
            print(f"  Using undistorted reconstruction from {final_recon_path}")
            
        except Exception as e:
            print(f"  Undistortion failed ({e}), using raw reconstruction")
            final_recon_path = raw_recon_path
            final_image_dir = Path(image_dir)

        # Parse output
        print("  Parsing results...")
        sfm_output = self._parse_colmap_output(final_recon_path, str(final_image_dir))
        sfm_output.metadata['status'] = 'success'
        
        # If we undistorted, note the new image location in metadata
        if dense_path.exists():
            sfm_output.metadata['undistorted_images_path'] = str(dense_path / "images")

        # Save standardized format
        sfm_output.save(str(output_path))

        print(f"[COLMAP] Done! {len(sfm_output.image_names)} images reconstructed")
        return sfm_output

    def _run_colmap(self, args: list) -> None:
        """Run COLMAP command."""
        cmd = [self.colmap_path] + args
        env = os.environ.copy()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(f"COLMAP failed: {result.stderr}")

    def _parse_colmap_output(
        self,
        recon_path: Path,
        image_dir: str,
    ) -> SfMOutput:
        """Parse COLMAP binary output files."""
        sfm_output = SfMOutput()

        # Parse cameras.bin
        cameras = self._read_cameras_binary(recon_path / "cameras.bin")

        # Parse images.bin
        images = self._read_images_binary(recon_path / "images.bin")

        # Parse points3D.bin
        points3d = self._read_points3D_binary(recon_path / "points3D.bin")

        # Build output
        for image_id, image_data in images.items():
            name = image_data['name']
            sfm_output.image_names.append(name)

            # Get image size
            img_path = Path(image_dir) / name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
            else:
                h, w = 480, 640
            sfm_output.image_sizes[name] = (h, w)

            # Intrinsics
            camera_id = image_data['camera_id']
            if camera_id in cameras:
                K = self._camera_to_intrinsic(cameras[camera_id], w, h)
                sfm_output.intrinsics[name] = K

            # Pose (world-to-camera -> camera-to-world)
            qvec = image_data['qvec']
            tvec = image_data['tvec']
            R = self._qvec2rotmat(qvec)
            pose = np.eye(4)
            pose[:3, :3] = R.T
            pose[:3, 3] = -R.T @ tvec
            sfm_output.poses[name] = pose

            # Sparse depth from correspondences
            xys = image_data['xys']
            point3d_ids = image_data['point3D_ids']

            depth_map = np.zeros((h, w), dtype=np.float32)
            error_map = np.zeros((h, w), dtype=np.float32)
            nviews_map = np.zeros((h, w), dtype=np.int32)

            for xy, p3d_id in zip(xys, point3d_ids):
                if p3d_id >= 0 and p3d_id in points3d:
                    x, y = int(round(xy[0])), int(round(xy[1]))
                    if 0 <= x < w and 0 <= y < h:
                        p3d = points3d[p3d_id]
                        p_cam = R @ p3d['xyz'] + tvec
                        depth = p_cam[2]
                        if depth > 0:
                            depth_map[y, x] = depth
                            error_map[y, x] = p3d['error']
                            nviews_map[y, x] = len(p3d['image_ids'])

            sfm_output.sparse_depths[name] = depth_map
            sfm_output.sparse_errors[name] = error_map
            sfm_output.sparse_n_views[name] = nviews_map

        # Global point cloud
        if points3d:
            pts = np.array([p['xyz'] for p in points3d.values()])
            colors = np.array([p['rgb'] for p in points3d.values()]) / 255.0
            sfm_output.points3d = pts
            sfm_output.points3d_colors = colors

        return sfm_output

    def _read_cameras_binary(self, path: Path) -> dict:
        """Read COLMAP cameras.bin."""
        cameras = {}
        if not path.exists():
            return cameras

        with open(path, 'rb') as f:
            num_cameras = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack('I', f.read(4))[0]
                model_id = struct.unpack('i', f.read(4))[0]
                width = struct.unpack('Q', f.read(8))[0]
                height = struct.unpack('Q', f.read(8))[0]
                num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
                params = struct.unpack('d' * num_params, f.read(8 * num_params))
                cameras[camera_id] = {
                    'model_id': model_id,
                    'width': width,
                    'height': height,
                    'params': params
                }
        return cameras

    def _read_images_binary(self, path: Path) -> dict:
        """Read COLMAP images.bin."""
        images = {}
        if not path.exists():
            return images

        with open(path, 'rb') as f:
            num_images = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_images):
                image_id = struct.unpack('I', f.read(4))[0]
                qvec = struct.unpack('dddd', f.read(32))
                tvec = struct.unpack('ddd', f.read(24))
                camera_id = struct.unpack('I', f.read(4))[0]

                name = b''
                while True:
                    char = f.read(1)
                    if char == b'\x00':
                        break
                    name += char
                name = name.decode('utf-8')

                num_points2D = struct.unpack('Q', f.read(8))[0]
                xys = []
                point3D_ids = []
                for _ in range(num_points2D):
                    x, y = struct.unpack('dd', f.read(16))
                    point3D_id = struct.unpack('q', f.read(8))[0]
                    xys.append([x, y])
                    point3D_ids.append(point3D_id)

                images[image_id] = {
                    'qvec': np.array(qvec),
                    'tvec': np.array(tvec),
                    'camera_id': camera_id,
                    'name': name,
                    'xys': np.array(xys),
                    'point3D_ids': np.array(point3D_ids)
                }
        return images

    def _read_points3D_binary(self, path: Path) -> dict:
        """Read COLMAP points3D.bin."""
        points3d = {}
        if not path.exists():
            return points3d

        with open(path, 'rb') as f:
            num_points = struct.unpack('Q', f.read(8))[0]
            for _ in range(num_points):
                point3D_id = struct.unpack('Q', f.read(8))[0]
                xyz = struct.unpack('ddd', f.read(24))
                rgb = struct.unpack('BBB', f.read(3))
                error = struct.unpack('d', f.read(8))[0]
                track_length = struct.unpack('Q', f.read(8))[0]
                image_ids = []
                for _ in range(track_length):
                    image_id = struct.unpack('I', f.read(4))[0]
                    struct.unpack('I', f.read(4))[0]  # point2D_idx
                    image_ids.append(image_id)
                points3d[point3D_id] = {
                    'xyz': np.array(xyz),
                    'rgb': np.array(rgb),
                    'error': error,
                    'image_ids': image_ids
                }
        return points3d

    def _camera_to_intrinsic(self, cam: dict, w: int, h: int) -> np.ndarray:
        """Convert COLMAP camera to intrinsic matrix."""
        params = cam['params']
        model_id = cam['model_id']
        scale_x = w / cam['width']
        scale_y = h / cam['height']

        if model_id == 0:  # SIMPLE_PINHOLE
            f, cx, cy = params
            K = np.array([[f * scale_x, 0, cx * scale_x],
                          [0, f * scale_y, cy * scale_y],
                          [0, 0, 1]])
        elif model_id == 1:  # PINHOLE
            fx, fy, cx, cy = params
            K = np.array([[fx * scale_x, 0, cx * scale_x],
                          [0, fy * scale_y, cy * scale_y],
                          [0, 0, 1]])
        else:  # OPENCV and others
            fx, fy, cx, cy = params[:4]
            K = np.array([[fx * scale_x, 0, cx * scale_x],
                          [0, fy * scale_y, cy * scale_y],
                          [0, 0, 1]])
        return K

    def _qvec2rotmat(self, qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
