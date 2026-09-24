#!/usr/bin/env python3
"""
Live depth streaming module with modular depth processing.

This module provides live depth streaming capabilities while using the modular
DepthProcessor for depth estimation. It can integrate with object detection
to show depth for specific regions of interest.

Usage:
    from depth import DepthStream

    # Basic streaming
    stream = DepthStream()
    stream.start()

    # With object detection integration
    stream = DepthStream(show_bbox_depth=True)
    stream.start()
"""

import cv2
import numpy as np
import argparse
import time
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
import os

from .depth_processor import DepthProcessor, create_depth_processor


class DepthStream:
    """
    Live depth streaming with modular depth processing.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        model_id: str = "Intel/dpt-hybrid-midas",
        show_bbox_depth: bool = False,
        save_dir: Optional[str] = None,
    ):
        """
        Initialize depth stream.

        Args:
            camera_index: Camera index
            width: Frame width
            height: Frame height
            model_id: Depth model ID
            show_bbox_depth: Show depth for bounding boxes
            save_dir: Directory to save snapshots
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.show_bbox_depth = show_bbox_depth
        self.save_dir = save_dir

        # Initialize depth processor
        self.depth_processor = create_depth_processor(model_id)

        # Initialize camera
        self.cap = None
        self._initialize_camera()

        # Bounding boxes for depth display (can be set by external modules)
        self.bounding_boxes: List[Tuple[int, int, int, int]] = []

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()

    def _initialize_camera(self):
        """Initialize camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30.0)

        print(f"Camera initialized: {self.width}x{self.height}")

    def set_bounding_boxes(self, bboxes: List[Tuple[int, int, int, int]]):
        """
        Set bounding boxes for depth display.

        Args:
            bboxes: List of bounding boxes as (x1, y1, x2, y2)
        """
        self.bounding_boxes = bboxes

    def _draw_bounding_boxes(
        self, img: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
    ):
        """Draw bounding boxes on image."""
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            color = (
                (0, 255, 0) if i == 0 else (255, 0, 0)
            )  # Green for first, red for others
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Add label
            label = f"Object {i+1}"
            cv2.putText(
                img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    def _get_depth_for_bboxes(
        self, frame: np.ndarray, bboxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict[str, Any]]:
        """Get depth information for all bounding boxes."""
        depth_results = []

        for bbox in bboxes:
            try:
                result = self.depth_processor.get_depth_for_spatial_audio(frame, bbox)
                depth_results.append(result)
            except Exception as e:
                print(f"Error processing depth for bbox {bbox}: {e}")
                depth_results.append(
                    {
                        "raw_depth": float("nan"),
                        "normalized_depth": 0.5,
                        "roi_stats": {},
                        "method": "median",
                        "normalization": "relative",
                    }
                )

        return depth_results

    def _display_depth_info(self, img: np.ndarray, depth_results: List[Dict[str, Any]]):
        """Display depth information on image."""
        y_offset = 30

        for i, result in enumerate(depth_results):
            if i >= 3:  # Limit to 3 objects to avoid clutter
                break

            raw_depth = result["raw_depth"]
            normalized_depth = result["normalized_depth"]

            # Display depth info
            text = f"Object {i+1}: Raw={raw_depth:.3f}, Norm={normalized_depth:.3f}"
            cv2.putText(
                img,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += 25

    def _save_snapshot(
        self, frame: np.ndarray, depth_map: np.ndarray, depth_color: np.ndarray
    ):
        """Save current frame and depth data."""
        if not self.save_dir:
            return

        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Save images
        cv2.imwrite(os.path.join(self.save_dir, f"rgb_{timestamp}.png"), frame)
        cv2.imwrite(
            os.path.join(self.save_dir, f"depth_color_{timestamp}.png"), depth_color
        )

        # Save raw depth
        np.save(os.path.join(self.save_dir, f"depth_raw_{timestamp}.npy"), depth_map)

        print(f"Saved snapshot: {timestamp}")

    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def start(self):
        """Start the depth streaming loop."""
        print("Starting depth stream...")
        print("Controls:")
        print("  q - Quit")
        print("  s - Save snapshot")
        print("  r - Reset depth statistics")
        print("  b - Toggle bounding box display")

        cv2.namedWindow("Depth Stream", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break

                # Estimate depth
                depth_map = self.depth_processor.estimate_depth(frame)
                depth_color = self.depth_processor.colorize_depth(depth_map)

                # Process bounding boxes if available
                depth_results = []
                if self.bounding_boxes and self.show_bbox_depth:
                    depth_results = self._get_depth_for_bboxes(
                        frame, self.bounding_boxes
                    )

                # Create display frame
                display_frame = frame.copy()

                # Draw bounding boxes
                if self.bounding_boxes:
                    self._draw_bounding_boxes(display_frame, self.bounding_boxes)
                    self._draw_bounding_boxes(depth_color, self.bounding_boxes)

                # Display depth information
                if depth_results:
                    self._display_depth_info(display_frame, depth_results)

                # Add FPS and model info
                fps = self._calculate_fps()
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    "Intel/dpt-hybrid-midas",
                    (10, display_frame.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Stack frames side by side
                if display_frame.shape != depth_color.shape:
                    depth_color = cv2.resize(
                        depth_color, (display_frame.shape[1], display_frame.shape[0])
                    )

                stacked = np.hstack([display_frame, depth_color])
                cv2.imshow("Depth Stream", stacked)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and self.save_dir:
                    self._save_snapshot(frame, depth_map, depth_color)
                elif key == ord("r"):
                    self.depth_processor.reset_depth_stats()
                    print("Depth statistics reset")
                elif key == ord("b"):
                    self.show_bbox_depth = not self.show_bbox_depth
                    print(
                        f"Bounding box depth display: {'ON' if self.show_bbox_depth else 'OFF'}"
                    )

        except KeyboardInterrupt:
            print("Stream interrupted by user")

        finally:
            self.release()

    def release(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Depth stream released")


# Legacy HFDepthEstimator class for backward compatibility
class HFDepthEstimator:
    """
    Legacy depth estimator class for backward compatibility.
    """

    def __init__(self, model_id: str = "Intel/dpt-hybrid-midas", device=None):
        self.processor = create_depth_processor(model_id)

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Legacy method for depth inference."""
        return self.processor.estimate_depth(frame_bgr)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Live depth streaming with modular depth processing."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument(
        "--model", type=str, default="Intel/dpt-hybrid-midas", help="Depth model ID"
    )
    parser.add_argument(
        "--save-dir", type=str, default="", help="Directory to save snapshots"
    )
    parser.add_argument(
        "--show-bbox", action="store_true", help="Show bounding box depth"
    )
    return parser.parse_args()


def main():
    """Main function for standalone execution."""
    args = parse_args()

    stream = DepthStream(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        model_id=args.model,
        show_bbox_depth=args.show_bbox,
        save_dir=args.save_dir,
    )

    stream.start()


if __name__ == "__main__":
    main()
