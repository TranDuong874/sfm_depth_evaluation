"""
Depth Anything V2 depth estimation.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .base import BaseDepthEstimator

# Add Depth-Anything-V2 to path if available
ROOT_DIR = Path(__file__).parent.parent.parent
DA_DIR = ROOT_DIR / "dependency" / "Depth-Anything-V2"
if DA_DIR.exists() and str(DA_DIR) not in sys.path:
    sys.path.insert(0, str(DA_DIR))


class DepthAnythingEstimator(BaseDepthEstimator):
    """
    Depth Anything V2 depth estimation.

    Outputs relative depth that needs scale alignment with SfM sparse depth.
    """

    MODELS = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    def __init__(
        self,
        model_name: str = 'vits',
        device: str = 'cuda',
        max_depth: float = 20.0,
    ):
        super().__init__(device)
        self.model_name = model_name
        self.model_config = self.MODELS.get(model_name, self.MODELS['vits'])
        self.max_depth = max_depth

    @property
    def name(self) -> str:
        return "depth_anything"

    def load_model(self) -> None:
        """Load Depth Anything V2 model."""
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError(
                f"depth_anything_v2 not found in {DA_DIR}. "
                "Please clone: git clone https://github.com/DepthAnything/Depth-Anything-V2 dependency/Depth-Anything-V2"
            )

        print(f"Loading Depth Anything V2 ({self.model_name})...")
        self.model = DepthAnythingV2(**self.model_config)

        # Load weights
        ckpt_name = f"depth_anything_v2_{self.model_name}.pth"
        ckpt_url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{self.model_name.upper()}/resolve/main/{ckpt_name}"

        state_dict = torch.hub.load_state_dict_from_url(
            ckpt_url, map_location='cpu', file_name=ckpt_name
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).half().eval()

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """Preprocess image."""
        h, w = rgb.shape[:2]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        if (new_h, new_w) != (h, w):
            rgb = cv2.resize(rgb, (new_w, new_h))

        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).half()

    def predict(
        self,
        rgb: np.ndarray,
        intrinsic: Optional[np.ndarray] = None,
        sparse_depth: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run Depth Anything V2 prediction.

        Returns relative depth scaled by max_depth.
        """
        if self.model is None:
            self.load_model()

        h, w = rgb.shape[:2]

        # Preprocess and predict
        rgb_tensor = self._preprocess(rgb).to(self.device)
        with torch.no_grad():
            relative_depth = self.model(rgb_tensor)

        relative_depth = relative_depth.squeeze().cpu().numpy().astype(np.float32)

        # Resize to original
        if relative_depth.shape != (h, w):
            relative_depth = cv2.resize(relative_depth, (w, h))

        # Normalize and scale by max_depth
        d_min, d_max = relative_depth.min(), relative_depth.max()
        if d_max - d_min > 1e-6:
            depth = (relative_depth - d_min) / (d_max - d_min) * self.max_depth
        else:
            depth = relative_depth

        return {'depth': np.clip(depth, 0, self.max_depth)}
