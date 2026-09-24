#!/usr/bin/env python3
"""
Modular depth processing module for monocular depth estimation.

This module provides a clean interface for depth estimation that can be used
by detection and spatial audio pipelines. It handles:
- Depth estimation from frames
- ROI-based depth averaging
- Depth normalization and scaling
- Bounding box integration

Usage:
    from depth.depth_processor import DepthProcessor

    processor = DepthProcessor()
    depth_value = processor.get_depth_from_roi(frame, bbox)
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoImageProcessor, DPTForDepthEstimation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthProcessor:
    """
    Modular depth processor for monocular depth estimation.

    This class provides methods to:
    - Estimate depth from camera frames
    - Calculate average depth within bounding boxes
    - Normalize depth values for spatial audio
    """

    def __init__(
        self,
        model_id: str = "Intel/dpt-hybrid-midas",
        device: Optional[torch.device] = None,
        reference_depth: float = 1.0,
    ):
        """
        Initialize the depth processor.

        Args:
            model_id: HuggingFace model ID for depth estimation
            device: PyTorch device (auto-detected if None)
            reference_depth: Reference depth for normalization
        """
        self.model_id = model_id
        self.device = device or self._pick_device()
        self.reference_depth = reference_depth

        # Initialize model
        self._initialize_model()

        # Depth statistics for normalization
        self.depth_stats = {
            "min": float("inf"),
            "max": float("-inf"),
            "mean": 0.0,
            "std": 1.0,
        }

        logger.info(f"DepthProcessor initialized with device: {self.device}")

    def _pick_device(self) -> torch.device:
        """Auto-detect best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")  # Apple Silicon
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _initialize_model(self):
        """Initialize the depth estimation model."""
        try:
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = (
                DPTForDepthEstimation.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )

            # Warmup
            with torch.no_grad():
                dummy = np.zeros((384, 384, 3), dtype=np.uint8)
                inputs = self.processor(images=dummy, return_tensors="pt").to(
                    self.device
                )
                _ = self.model(**inputs)

            logger.info(f"Model {self.model_id} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    @torch.inference_mode()
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a single frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Depth map as float32 numpy array
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process with model
        inputs = self.processor(images=frame_rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Post-process to original frame size
        try:
            # Try the new method first
            post = self.processor.post_process_depth_estimation(
                outputs, target_sizes=[(frame_rgb.shape[0], frame_rgb.shape[1])]
            )
            depth = post[0]["predicted_depth"].detach().cpu().numpy().astype(np.float32)

            # Convert to meters (DPT outputs in different units)
            depth = depth / 1000.0  # Convert to meters
        except AttributeError:
            # Fallback for newer transformers versions
            depth = outputs.predicted_depth.detach().cpu().numpy().astype(np.float32)

            # Handle different depth map formats
            if len(depth.shape) == 3:
                # If 3D, take the first channel or average across channels
                if depth.shape[2] > 1:
                    depth = np.mean(depth, axis=2)
                else:
                    depth = depth[:, :, 0]

            # Resize to original frame size
            depth = cv2.resize(
                depth,
                (frame_rgb.shape[1], frame_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

            # Convert to meters (DPT outputs in different units)
            depth = depth / 1000.0  # Convert to meters

        # Update depth statistics
        self._update_depth_stats(depth)

        return depth

    def _update_depth_stats(self, depth: np.ndarray):
        """Update running statistics for depth normalization."""
        valid_depth = depth[np.isfinite(depth)]
        if valid_depth.size > 0:
            self.depth_stats["min"] = min(
                self.depth_stats["min"], float(np.min(valid_depth))
            )
            self.depth_stats["max"] = max(
                self.depth_stats["max"], float(np.max(valid_depth))
            )
            self.depth_stats["mean"] = float(np.mean(valid_depth))
            self.depth_stats["std"] = float(np.std(valid_depth))

    def get_depth_from_bbox(
        self,
        depth_map: np.ndarray,
        bbox: Union[List[int], Tuple[int, int, int, int]],
        method: str = "median",
    ) -> float:
        """
        Get depth value from a bounding box region.

        Args:
            depth_map: Depth map from estimate_depth()
            bbox: Bounding box as [x1, y1, x2, y2] or (x1, y1, x2, y2)
            method: Method for depth calculation ('mean', 'median', 'min', 'max')

        Returns:
            Depth value for the ROI
        """
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within bounds
        if len(depth_map.shape) == 3:
            h, w = depth_map.shape[:2]
        else:
            h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))

        # Extract ROI
        roi_depth = depth_map[y1:y2, x1:x2]
        valid_depth = roi_depth[np.isfinite(roi_depth)]

        if valid_depth.size == 0:
            logger.warning("No valid depth values in ROI")
            return float("nan")

        # Calculate depth based on method
        if method == "mean":
            return float(np.mean(valid_depth))
        elif method == "median":
            return float(np.median(valid_depth))
        elif method == "min":
            return float(np.min(valid_depth))
        elif method == "max":
            return float(np.max(valid_depth))
        else:
            raise ValueError(f"Unknown method: {method}")

    def normalize_depth(self, depth_value: float, method: str = "relative") -> float:
        """
        Normalize depth value for spatial audio.

        Args:
            depth_value: Raw depth value
            method: Normalization method ('relative', 'statistical', 'reference')

        Returns:
            Normalized depth value (0.0 to 1.0 range)
        """
        if not np.isfinite(depth_value):
            return 0.5  # Default middle value

        if method == "relative":
            # Normalize based on current depth statistics
            if self.depth_stats["max"] > self.depth_stats["min"]:
                normalized = (depth_value - self.depth_stats["min"]) / (
                    self.depth_stats["max"] - self.depth_stats["min"]
                )
                return float(np.clip(normalized, 0.0, 1.0))
            else:
                return 0.5

        elif method == "statistical":
            # Z-score normalization
            if self.depth_stats["std"] > 0:
                z_score = (depth_value - self.depth_stats["mean"]) / self.depth_stats[
                    "std"
                ]
                # Convert to 0-1 range (assuming normal distribution)
                normalized = 0.5 + 0.3 * np.tanh(z_score)  # Sigmoid-like function
                return float(np.clip(normalized, 0.0, 1.0))
            else:
                return 0.5

        elif method == "reference":
            # Normalize relative to reference depth
            if depth_value > 0:
                normalized = self.reference_depth / depth_value
                return float(np.clip(normalized, 0.0, 1.0))
            else:
                return 0.5

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def get_depth_for_spatial_audio(
        self,
        frame: np.ndarray,
        bbox: Union[List[int], Tuple[int, int, int, int]],
        method: str = "median",
        normalization: str = "relative",
    ) -> Dict[str, float]:
        """
        Get processed depth value for spatial audio pipeline.

        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box as [x1, y1, x2, y2]
            method: Depth calculation method ('mean', 'median', 'min', 'max')
            normalization: Normalization method ('relative', 'statistical', 'reference')

        Returns:
            Dictionary with depth information:
            {
                'raw_depth': float,
                'normalized_depth': float,
                'roi_stats': dict
            }
        """
        # Estimate depth
        depth_map = self.estimate_depth(frame)

        # Get depth from ROI
        raw_depth = self.get_depth_from_bbox(depth_map, bbox, method)

        # Normalize depth
        normalized_depth = self.normalize_depth(raw_depth, normalization)

        # Get ROI statistics
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]
        valid_depth = roi_depth[np.isfinite(roi_depth)]

        roi_stats = {
            "mean": (
                float(np.mean(valid_depth)) if valid_depth.size > 0 else float("nan")
            ),
            "std": float(np.std(valid_depth)) if valid_depth.size > 0 else float("nan"),
            "min": float(np.min(valid_depth)) if valid_depth.size > 0 else float("nan"),
            "max": float(np.max(valid_depth)) if valid_depth.size > 0 else float("nan"),
            "count": valid_depth.size,
        }

        return {
            "raw_depth": raw_depth,
            "normalized_depth": normalized_depth,
            "roi_stats": roi_stats,
            "method": method,
            "normalization": normalization,
        }

    def get_depth_stats(self) -> Dict[str, float]:
        """Get current depth statistics."""
        return self.depth_stats.copy()

    def reset_depth_stats(self):
        """Reset depth statistics."""
        self.depth_stats = {
            "min": float("inf"),
            "max": float("-inf"),
            "mean": 0.0,
            "std": 1.0,
        }

    def colorize_depth(
        self, depth: np.ndarray, clip_percentile: float = 0.05
    ) -> np.ndarray:
        """
        Colorize depth map for visualization.

        Args:
            depth: Depth map
            clip_percentile: Percentile for clipping

        Returns:
            Colorized depth map (BGR format)
        """
        d = np.nan_to_num(depth, nan=float(np.nanmedian(depth)))
        lo = np.percentile(d, 100 * clip_percentile)
        hi = np.percentile(d, 100 * (1.0 - clip_percentile))

        if hi - lo < 1e-6:
            hi = lo + 1e-6

        d_norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
        d_8u = (d_norm * 255.0).astype(np.uint8)

        return cv2.applyColorMap(d_8u, cv2.COLORMAP_MAGMA)


# Factory function
def create_depth_processor(model_id: str = "Intel/dpt-hybrid-midas") -> DepthProcessor:
    """Create a depth processor instance."""
    return DepthProcessor(model_id=model_id)


def process_frame_depth(
    frame: np.ndarray,
    bbox: Union[List[int], Tuple[int, int, int, int]],
    processor: Optional[DepthProcessor] = None,
) -> Dict[str, float]:
    """
    Process a single frame for depth information.

    Args:
        frame: Input frame (BGR format)
        bbox: Bounding box as [x1, y1, x2, y2]
        processor: Depth processor instance (created if None)

    Returns:
        Depth information dictionary
    """
    if processor is None:
        processor = create_depth_processor()

    return processor.get_depth_for_spatial_audio(frame, bbox)


# Example usage
if __name__ == "__main__":
    import time

    print("=== Depth Processor Test ===")

    # Create processor
    processor = create_depth_processor()

    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 200, 200]

    print("Testing depth processing...")
    start_time = time.time()

    result = processor.get_depth_for_spatial_audio(dummy_frame, dummy_bbox)

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.3f}s")
    print(f"Result: {result}")
    print(f"Depth stats: {processor.get_depth_stats()}")

    print("Test completed!")
