"""
MASt3R Structure-from-Motion implementation.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from .base import BaseSfM, SfMOutput

# Add MASt3R to path
ROOT_DIR = Path(__file__).parent.parent.parent
MAST3R_DIR = ROOT_DIR / "dependency" / "mast3r"
DUST3R_DIR = MAST3R_DIR / "dust3r"

if str(MAST3R_DIR) not in sys.path:
    sys.path.insert(0, str(MAST3R_DIR))
if str(DUST3R_DIR) not in sys.path:
    sys.path.insert(0, str(DUST3R_DIR))


class MASt3RSfM(BaseSfM):
    """MASt3R-based Structure-from-Motion."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        image_size: int = 512,
    ):
        super().__init__(device)
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.model = None

    @property
    def name(self) -> str:
        return "mast3r"

    def load_model(self) -> None:
        """Load MASt3R model."""
        from mast3r.model import AsymmetricMASt3R

        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            print(f"Loading MASt3R from {self.checkpoint_path}")
            self.model = AsymmetricMASt3R.from_pretrained(self.checkpoint_path)
        else:
            print("Loading MASt3R from pretrained (naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric)")
            self.model = AsymmetricMASt3R.from_pretrained(
                "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
            )

        self.model = self.model.to(self.device)
        self.model.eval()

    def reconstruct(
        self,
        image_dir: str,
        output_dir: str,
        scene_graph: str = 'complete',
        lr1: float = 0.07,
        niter1: int = 300,
        lr2: float = 0.014,
        niter2: int = 300,
        opt_depth: bool = True,
        **kwargs
    ) -> SfMOutput:
        """
        Run MASt3R SfM reconstruction.

        Args:
            image_dir: Directory containing input images
            output_dir: Directory to save outputs
            scene_graph: Pairing strategy ('complete', 'swin-N', 'oneref')
            lr1: Coarse alignment learning rate
            niter1: Coarse alignment iterations
            lr2: Fine alignment learning rate
            niter2: Fine alignment iterations
            opt_depth: Optimize depth maps

        Returns:
            SfMOutput with reconstruction results
        """
        if self.model is None:
            self.load_model()

        from dust3r.inference import inference
        from dust3r.image_pairs import make_pairs
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        from dust3r.utils.image import load_images

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # cache_dir = str(output_path / "cache")  # Not used in dust3r alignment

        # Get image paths
        image_paths = self.get_image_paths(image_dir)
        if len(image_paths) < 2:
            raise ValueError(f"Need at least 2 images, found {len(image_paths)}")

        print(f"[MASt3R] Processing {len(image_paths)} images")

        # Load images
        print("  Loading images...")
        imgs = load_images(image_paths, size=self.image_size)

        # Create pairs
        print(f"  Creating pairs (scene_graph={scene_graph})...")
        pairs = make_pairs(imgs, scene_graph=scene_graph, symmetrize=True)

        # Run inference
        print("  Running inference...")
        with torch.autocast(device_type=self.device, dtype=torch.float16):
            output = inference(pairs, self.model, self.device, batch_size=1, verbose=True)

        # Run alignment
        print("  Running global alignment...")
        # Use default opt_lr=0.01 and opt_niter=300 if not provided
        opt_lr = kwargs.get('opt_lr', 0.01)
        opt_niter = kwargs.get('opt_niter', 300)
        
        scene = global_aligner(output, device=self.device, mode=GlobalAlignerMode.PointCloudOptimizer)
        scene.compute_global_alignment(init='mst', niter=opt_niter, schedule='linear', lr=opt_lr)

        # Extract results
        print("  Extracting results...")
        sfm_output = self._extract_sfm_output(scene, image_paths)

        # Save
        sfm_output.save(str(output_path))

        print(f"[MASt3R] Done! {len(sfm_output.image_names)} images reconstructed")
        return sfm_output

    def _extract_sfm_output(
        self,
        scene,
        image_paths: list,
    ) -> SfMOutput:
        """Extract SfMOutput from MASt3R scene."""
        sfm_output = SfMOutput()

        # Get poses and focals
        poses = scene.get_im_poses().detach().cpu().numpy()
        focals = scene.get_focals().detach().cpu().numpy()

        # Try to get principal points
        try:
            pps = scene.get_principal_points().detach().cpu().numpy()
        except:
            pps = None

        # Get depth maps
        depthmaps = scene.get_depthmaps()

        # Get confidence if available
        try:
            confs = scene.get_conf()
        except:
            confs = [None] * len(image_paths)

        for i, img_path in enumerate(image_paths):
            name = Path(img_path).name
            sfm_output.image_names.append(name)

            # Get original image size
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            sfm_output.image_sizes[name] = (h, w)

            # Build intrinsic matrix
            focal = float(focals[i])
            if pps is not None:
                cx, cy = float(pps[i, 0]), float(pps[i, 1])
            else:
                cx, cy = w / 2, h / 2

            # Scale focal from processing size to original
            scale = max(h, w) / self.image_size
            focal_scaled = focal * scale
            cx_scaled = cx * scale
            cy_scaled = cy * scale

            K = np.array([
                [focal_scaled, 0, cx_scaled],
                [0, focal_scaled, cy_scaled],
                [0, 0, 1]
            ], dtype=np.float64)
            sfm_output.intrinsics[name] = K

            # Pose (camera-to-world)
            sfm_output.poses[name] = poses[i]

            # Sparse depth
            if i < len(depthmaps) and depthmaps[i] is not None:
                depth = depthmaps[i]
                if isinstance(depth, torch.Tensor):
                    depth = depth.detach().cpu().numpy()

                # Clean invalid depth (match object_recon)
                depth[depth <= 0] = 0
                depth[~np.isfinite(depth)] = 0

                # Resize to original size
                if depth.shape[:2] != (h, w):
                    depth = cv2.resize(
                        depth.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR
                    )
                sfm_output.sparse_depths[name] = depth

                # Confidence as proxy for n_views
                if i < len(confs) and confs[i] is not None:
                    conf = confs[i]
                    if isinstance(conf, torch.Tensor):
                        conf = conf.detach().cpu().numpy()
                    if conf.shape[:2] != (h, w):
                        conf = cv2.resize(
                            conf.astype(np.float32),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST
                        )
                    sfm_output.sparse_n_views[name] = (conf > 1.0).astype(np.int32) + 1

        # Get global point cloud
        try:
            # Match object_recon approach: use get_pts3d() instead of get_sparse_pts3d()
            # and avoid aggressive filtering.
            pts3d_list = scene.get_pts3d()
            
            # Try to get colors from scene if available
            # Note: object_recon/utils/colmap_utils.py doesn't seem to extract colors from the scene object directly, 
            # but creates grey colors. If get_pts3d_colors is available, we should use it.
            if hasattr(scene, 'get_pts3d_colors'):
                colors_list = scene.get_pts3d_colors()
            else:
                colors_list = [None] * len(pts3d_list)

            all_pts = []
            all_colors = []

            for i, pts3d in enumerate(pts3d_list):
                if pts3d is None:
                    continue
                if isinstance(pts3d, torch.Tensor):
                    pts3d = pts3d.detach().cpu().numpy()

                pts_flat = pts3d.reshape(-1, 3)

                # Get colors
                if i < len(colors_list) and colors_list[i] is not None:
                    colors_data = colors_list[i]
                    if isinstance(colors_data, torch.Tensor):
                        colors_data = colors_data.detach().cpu().numpy()
                    colors_flat = colors_data.reshape(-1, 3)
                else:
                    # Fallback: Extract from image
                    try:
                        img = cv2.imread(image_paths[i])
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        if pts3d.ndim == 3:
                            h_map, w_map = pts3d.shape[:2]
                            img_resized = cv2.resize(img, (w_map, h_map))
                            colors_flat = img_resized.reshape(-1, 3) / 255.0
                        else:
                             colors_flat = np.ones_like(pts_flat) * 0.5
                    except Exception:
                        colors_flat = np.ones_like(pts_flat) * 0.5

                # Ensure dimensions match
                if len(colors_flat) != len(pts_flat):
                    colors_flat = np.ones_like(pts_flat) * 0.5

                # Minimal validity check to avoid NaNs or Infs
                valid = np.isfinite(pts_flat).all(axis=1)
                
                if np.any(valid):
                    all_pts.append(pts_flat[valid])
                    all_colors.append(colors_flat[valid])

            if all_pts:
                sfm_output.points3d = np.concatenate(all_pts, axis=0)
                sfm_output.points3d_colors = np.concatenate(all_colors, axis=0)

        except Exception as e:
            print(f"  Warning: Could not extract point cloud: {e}")

        return sfm_output
