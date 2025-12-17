"""
Base class for depth estimation methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


class BaseDepthEstimator(ABC):
    """Abstract base class for monocular depth estimation."""

    def __init__(self, device: str = 'cuda'):
        """
        Initialize depth estimator.

        Args:
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Method name."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def predict(
        self,
        rgb: np.ndarray,
        intrinsic: Optional[np.ndarray] = None,
        sparse_depth: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run depth prediction.

        Args:
            rgb: RGB image (H, W, 3) in uint8
            intrinsic: Camera intrinsic matrix (3, 3) or None
            sparse_depth: Sparse depth guidance (H, W) or None

        Returns:
            Dict with at minimum 'depth' key containing depth map in meters
        """
        pass

    def save_depth(
        self,
        depth: np.ndarray,
        output_path: str,
        save_visualization: bool = True,
    ) -> None:
        """
        Save depth map to file.

        Args:
            depth: Depth map in meters (H, W)
            output_path: Output file path
            save_visualization: Save colorized visualization
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as numpy array (primary format)
        npy_path = output_path.with_suffix('.npy')
        np.save(str(npy_path), depth.astype(np.float32))

        # Save as 16-bit PNG (depth in mm for visualization tools)
        depth_mm = np.clip(depth * 1000, 0, 65535).astype(np.uint16)
        raw_path = str(output_path).replace('.png', '_raw.png').replace('.npy', '_raw.png')
        cv2.imwrite(raw_path, depth_mm)

        # Save colorized visualization
        if save_visualization:
            vis_path = str(output_path).replace('.npy', '.png')
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            depth_colored = cv2.applyColorMap(
                (depth_normalized * 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA
            )
            cv2.imwrite(vis_path, depth_colored)

    def process_image(
        self,
        image_path: str,
        output_dir: str,
        intrinsic: Optional[np.ndarray] = None,
        sparse_depth: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Process a single image.

        Args:
            image_path: Path to input image
            output_dir: Directory to save output
            intrinsic: Camera intrinsics (optional)
            sparse_depth: Sparse depth guidance (optional)

        Returns:
            Prediction result dictionary
        """
        # Load image
        rgb = cv2.imread(image_path)
        if rgb is None:
            raise ValueError(f"Could not load image: {image_path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Predict
        result = self.predict(rgb, intrinsic=intrinsic, sparse_depth=sparse_depth)

        # Save
        stem = Path(image_path).stem
        output_path = Path(output_dir) / f"{stem}_depth.npy"
        self.save_depth(result['depth'], str(output_path))

        return result

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        intrinsic: Optional[np.ndarray] = None,
    ) -> None:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory with input images
            output_dir: Directory for output depth maps
            intrinsic: Camera intrinsics to use for all images
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find images
        extensions = ['*.png', '*.jpg', '*.jpeg']
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_path.glob(ext))
        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError(f"No images found in {input_dir}")

        print(f"[{self.name}] Processing {len(image_paths)} images...")

        for img_path in image_paths:
            print(f"  {img_path.name}")

            rgb = cv2.imread(str(img_path))
            if rgb is None:
                print(f"    Warning: Could not load")
                continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            result = self.predict(rgb, intrinsic=intrinsic)

            out_file = output_path / f"{img_path.stem}_depth.npy"
            self.save_depth(result['depth'], str(out_file))

        print(f"[{self.name}] Done!")

    def resize_depth_to_original(
        self,
        depth: np.ndarray,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Resize depth map to original image size.

        Args:
            depth: Depth map (H, W)
            original_size: Target (H, W)

        Returns:
            Resized depth map
        """
        h, w = original_size
        if depth.shape[:2] == (h, w):
            return depth

        return cv2.resize(
            depth.astype(np.float32),
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )
