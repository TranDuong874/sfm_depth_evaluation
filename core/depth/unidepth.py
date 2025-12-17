"""
UniDepth depth estimation.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .base import BaseDepthEstimator

# Add UniDepth to path if available
ROOT_DIR = Path(__file__).parent.parent.parent
UNIDEPTH_DIR = ROOT_DIR / "dependency" / "UniDepth"
if UNIDEPTH_DIR.exists() and str(UNIDEPTH_DIR) not in sys.path:
    sys.path.insert(0, str(UNIDEPTH_DIR))


class UniDepthEstimator(BaseDepthEstimator):
    """UniDepth depth estimation."""

    def __init__(
        self,
        version: str = 'v2',
        backbone: str = 'vits14',
        device: str = 'cuda',
    ):
        super().__init__(device)
        self.version = version
        self.backbone = backbone

    @property
    def name(self) -> str:
        return "unidepth"

    def load_model(self) -> None:
        """Load UniDepth model."""
        from unidepth.models import UniDepthV1, UniDepthV2

        model_name = f"lpiccinelli/unidepth-{self.version}-{self.backbone}"
        print(f"Loading UniDepth from {model_name}...")

        if self.version == 'v1':
            self.model = UniDepthV1.from_pretrained(model_name)
        else:
            self.model = UniDepthV2.from_pretrained(model_name)
            self.model.interpolation_mode = "bilinear"

        self.model = self.model.to(self.device).eval()

    def predict(
        self,
        rgb: np.ndarray,
        intrinsic: Optional[np.ndarray] = None,
        sparse_depth: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run UniDepth prediction."""
        if self.model is None:
            self.load_model()

        from unidepth.models import UniDepthV2
        from unidepth.utils.camera import Pinhole

        # Prepare input
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).to(self.device)

        # Prepare camera
        camera = None
        if intrinsic is not None:
            if intrinsic.shape == (3, 3):
                intrinsic_torch = torch.from_numpy(intrinsic).float().to(self.device)
            else:
                fx, fy, cx, cy = intrinsic[:4]
                intrinsic_torch = torch.tensor([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=torch.float32, device=self.device)

            if isinstance(self.model, UniDepthV2):
                camera = Pinhole(K=intrinsic_torch.unsqueeze(0))
            else:
                camera = intrinsic_torch

        # Inference
        with torch.no_grad():
            predictions = self.model.infer(rgb_torch, camera)
            
        depth = predictions['depth'].squeeze().cpu().numpy()

        return {'depth': depth}
