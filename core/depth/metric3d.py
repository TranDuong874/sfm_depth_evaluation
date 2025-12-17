"""
Metric3D depth estimation.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

from .base import BaseDepthEstimator

# Add Metric3D to path if available
ROOT_DIR = Path(__file__).parent.parent.parent
METRIC3D_DIR = ROOT_DIR / "dependency" / "Metric3D"
if METRIC3D_DIR.exists() and str(METRIC3D_DIR) not in sys.path:
    sys.path.insert(0, str(METRIC3D_DIR))


class Metric3DEstimator(BaseDepthEstimator):
    """Metric3D depth estimation."""

    def __init__(
        self,
        model_name: str = 'vit_small',
        device: str = 'cuda',
    ):
        super().__init__(device)
        self.model_name = model_name
        self.input_size = (616, 1064)

    @property
    def name(self) -> str:
        return "metric3d"

    def load_model(self) -> None:
        """Load Metric3D model."""
        try:
            from mmcv.utils import Config
        except ImportError:
            from mmengine import Config

        from mono.model.monodepth_model import get_configured_monodepth_model

        # Config and checkpoint paths
        configs = {
            'vit_small': {
                'cfg': METRIC3D_DIR / 'mono/configs/HourglassDecoder/vit.raft5.small.py',
                'ckpt': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth',
            },
            'vit_large': {
                'cfg': METRIC3D_DIR / 'mono/configs/HourglassDecoder/vit.raft5.large.py',
                'ckpt': 'https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth',
            },
        }

        config = configs.get(self.model_name, configs['vit_small'])

        print(f"Loading Metric3D ({self.model_name})...")
        cfg = Config.fromfile(str(config['cfg']))
        self.model = get_configured_monodepth_model(cfg)

        state_dict = torch.hub.load_state_dict_from_url(
            config['ckpt'], map_location='cpu'
        )
        self.model.load_state_dict(state_dict['model_state_dict'], strict=False)
        self.model = self.model.to(self.device).eval()

    def predict(
        self,
        rgb: np.ndarray,
        intrinsic: Optional[np.ndarray] = None,
        sparse_depth: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Run Metric3D prediction."""
        if self.model is None:
            self.load_model()

        h, w = rgb.shape[:2]

        # Get intrinsic
        if intrinsic is not None and intrinsic.shape == (3, 3):
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        else:
            fx = fy = max(h, w) * 0.8
            cx, cy = w / 2, h / 2

        # Preprocess
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        rgb_resized = cv2.resize(rgb, (new_w, new_h))

        # Scaled intrinsic
        fx_scaled = fx * scale

        # Pad to input size
        pad_h = self.input_size[0] - new_h
        pad_w = self.input_size[1] - new_w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2

        rgb_padded = cv2.copyMakeBorder(
            rgb_resized,
            pad_h_half, pad_h - pad_h_half,
            pad_w_half, pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=[123.675, 116.28, 103.53]
        )

        # Normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb_tensor = torch.from_numpy(rgb_padded.transpose((2, 0, 1))).float()
        rgb_tensor = ((rgb_tensor - mean) / std).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            pred_depth, _, _ = self.model.inference({'input': rgb_tensor})

        # Un-pad
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_h_half:pred_depth.shape[0] - (pad_h - pad_h_half) if (pad_h - pad_h_half) > 0 else None,
            pad_w_half:pred_depth.shape[1] - (pad_w - pad_w_half) if (pad_w - pad_w_half) > 0 else None
        ]

        # Resize to original
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], (h, w), mode='bilinear', align_corners=False
        ).squeeze()

        # De-canonical transform
        pred_depth = pred_depth * (fx_scaled / 1000.0)
        pred_depth = torch.clamp(pred_depth, 0, 300)
        
        depth_np = pred_depth.cpu().numpy()

        return {'depth': depth_np}
