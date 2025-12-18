"""
MURRE depth estimation with sparse depth guidance.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .base import BaseDepthEstimator

# Add MURRE to path
ROOT_DIR = Path(__file__).parent.parent.parent
MURRE_DIR = ROOT_DIR / "dependency" / "Murre"
if str(MURRE_DIR) not in sys.path:
    sys.path.insert(0, str(MURRE_DIR))


class MURREEstimator(BaseDepthEstimator):
    """
    MURRE depth estimation with SfM sparse depth guidance.

    Uses diffusion-based depth refinement conditioned on sparse depth.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = 'cuda',
        processing_res: int = 512,
        denoise_steps: int = 10,
        ensemble_size: int = 3,
        max_depth: float = 10.0,
    ):
        super().__init__(device)
        self.checkpoint_path = checkpoint_path or str(ROOT_DIR / "checkpoints" / "Murre")
        self.processing_res = processing_res
        self.denoise_steps = denoise_steps
        self.ensemble_size = ensemble_size
        self.max_depth = max_depth
        self.pipeline = None

    @property
    def name(self) -> str:
        return "murre"

    def load_model(self) -> None:
        """Load MURRE pipeline."""
        # Monkey patch importlib.metadata.version for numpy if it fails/returns None
        # This fixes a specific environment issue where accelerate fails to parse numpy version
        import importlib.metadata
        original_version = importlib.metadata.version

        def patched_version(package_name):
            try:
                v = original_version(package_name)
                if v is None and package_name == "numpy":
                    return "1.26.0"
                return v
            except Exception:
                if package_name == "numpy":
                    return "1.26.0"
                raise

        importlib.metadata.version = patched_version

        from murre.pipeline import MurrePipeline
        import torch

        print(f"Loading MURRE from {self.checkpoint_path}")
        self.pipeline = MurrePipeline.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.float16,
        ).to(self.device)

    def _resize_sparse_depth(
        self,
        sparse_depth: np.ndarray,
        rgb_shape: tuple,
    ) -> np.ndarray:
        """
        Resize sparse depth to match RGB resolution.
        """
        h, w = rgb_shape[:2]

        if sparse_depth.shape[:2] != (h, w):
            return cv2.resize(
                sparse_depth.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

        return sparse_depth

    def predict(
        self,
        rgb: np.ndarray,
        intrinsic: Optional[np.ndarray] = None,
        sparse_depth: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run MURRE depth prediction.

        Args:
            rgb: RGB image (H, W, 3) uint8
            intrinsic: Not used (kept for API compatibility)
            sparse_depth: Sparse depth from SfM (H, W)

        Returns:
            Dict with 'depth' key
        """
        if self.pipeline is None:
            self.load_model()

        # Convert to PIL
        pil_image = Image.fromarray(rgb)

        # Prepare sparse depth
        if sparse_depth is None:
            sparse_depth = np.zeros(rgb.shape[:2], dtype=np.float32)

        sparse_depth_resized = self._resize_sparse_depth(sparse_depth, rgb.shape)

        # Run inference
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            output = self.pipeline(
                input_image=pil_image,
                input_sparse_depth=sparse_depth_resized,
                max_depth=self.max_depth,
                denoising_steps=self.denoise_steps,
                ensemble_size=self.ensemble_size,
                processing_res=self.processing_res,
                match_input_res=True,
                batch_size=0,
                show_progress_bar=False,
            )

        # MURRE outputs metric depth directly (not normalized 0-1)
        # when guided by metric sparse depth from SfM
        depth = output.depth_np

        return {'depth': depth}
