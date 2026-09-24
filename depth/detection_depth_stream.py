#!/usr/bin/env python3
"""
Integrated Detection and Depth Processing Module

This module combines YOLO object detection with depth estimation to provide
real-time object detection with depth information overlaid on both RGB and depth maps.

Usage:
    from depth import DetectionDepthStream

    stream = DetectionDepthStream(target_classes=['person', 'bottle'])
    stream.start()
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Any
from ultralytics import YOLO

# Handle both relative imports (when used as a module) and absolute imports (when run as a script)
try:
    from .depth_processor import create_depth_processor
    from .enhanced_depth_processor_with_persistence import (
        create_enhanced_depth_processor_with_persistence,
        PersistenceConfig,
    )
except ImportError:
    # When run as a script, use absolute imports
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from depth.depth_processor import create_depth_processor
    from depth.enhanced_depth_processor_with_persistence import (
        create_enhanced_depth_processor_with_persistence,
        PersistenceConfig,
    )


class DetectionDepthStream:
    """
    Integrated stream combining YOLO object detection with depth estimation.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        model_path: str = "yolov8n.pt",
        target_classes: List[str] = None,
        depth_model_id: str = "Intel/dpt-hybrid-midas",
        confidence_threshold: float = 0.5,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        persistence_enabled: bool = True,
        persistence_duration: float = 2.0,
        max_missing_frames: int = 5,
    ):
        """
        Initialize detection and depth stream.

        Args:
            camera_index: Camera index
            width: Frame width
            height: Frame height
            model_path: Path to YOLO model
            target_classes: List of classes to detect (e.g., ['person', 'bottle', 'car'])
            depth_model_id: Depth model ID
            confidence_threshold: Minimum confidence for detections
            save_dir: Directory to save snapshots
            verbose: Enable verbose output
            persistence_enabled: Enable temporal persistence for objects
            persistence_duration: How long to keep objects after last detection (seconds)
            max_missing_frames: Maximum frames to keep object without detection
        """
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_classes = target_classes or [
            "person",
            "bottle",
            "cup",
            "wine glass",
            "car",
            "chair",
            "laptop",
        ]
        self.confidence_threshold = confidence_threshold
        self.save_dir = save_dir
        self.verbose = verbose
        self.persistence_enabled = persistence_enabled

        # Initialize YOLO model
        if self.verbose:
            print(f"Loading YOLO model from {model_path}")
        self.yolo_model = YOLO(model_path)
        if self.verbose:
            print(
                f"YOLO model loaded. Available classes: {list(self.yolo_model.names.values())}"
            )

        # Initialize enhanced depth processor with persistence
        if self.verbose:
            print(f"Loading enhanced depth model: {depth_model_id}")

        persistence_config = PersistenceConfig(
            enabled=persistence_enabled,
            persistence_duration=persistence_duration,
            max_missing_frames=max_missing_frames,
            confidence_decay_rate=0.1,
            bbox_expansion_rate=0.05,
            depth_prediction_enabled=True,
            motion_prediction_enabled=True,
        )

        self.depth_processor = create_enhanced_depth_processor_with_persistence(
            depth_model_id, persistence_config
        )

        # Initialize camera
        self.cap = None
        self._initialize_camera()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.detection_count = 0

        # Depth reference tracking
        self.background_depth_samples = []
        self.reference_depth = None
        self.depth_range = None

    def _initialize_camera(self):
        """Initialize camera."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30.0)

        if self.verbose:
            print(f"Camera initialized: {self.width}x{self.height}")

    def _estimate_background_depth(
        self, depth_map: np.ndarray, detections: List[Dict[str, Any]]
    ):
        """Estimate background depth by sampling areas not covered by detections."""
        h, w = depth_map.shape

        # Create mask for detected areas
        detection_mask = np.zeros((h, w), dtype=bool)
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            # Expand bounding box slightly to avoid edge effects
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            detection_mask[y1:y2, x1:x2] = True

        # Sample background areas (corners and edges)
        background_samples = []

        # Sample corners
        corner_size = min(50, w // 4, h // 4)
        corners = [
            depth_map[:corner_size, :corner_size],
            depth_map[:corner_size, -corner_size:],
            depth_map[-corner_size:, :corner_size],
            depth_map[-corner_size:, -corner_size:],
        ]

        for corner in corners:
            valid_depths = corner[~np.isnan(corner)]
            if len(valid_depths) > 0:
                background_samples.extend(valid_depths.tolist())

        # Sample center area if no detections
        if len(detections) == 0:
            center_h, center_w = h // 2, w // 2
            center_size = min(100, w // 3, h // 3)
            center_area = depth_map[
                center_h - center_size // 2 : center_h + center_size // 2,
                center_w - center_size // 2 : center_w + center_size // 2,
            ]
            valid_depths = center_area[~np.isnan(center_area)]
            if len(valid_depths) > 0:
                background_samples.extend(valid_depths.tolist())

        # Update background depth reference
        if background_samples:
            self.background_depth_samples.extend(background_samples)
            # Keep only recent samples (last 100 frames worth)
            if len(self.background_depth_samples) > 1000:
                self.background_depth_samples = self.background_depth_samples[-1000:]

            # Calculate reference depth (median of background samples)
            self.reference_depth = np.median(self.background_depth_samples)

            # Calculate depth range for normalization
            if len(self.background_depth_samples) > 10:
                depth_std = np.std(self.background_depth_samples)
                self.depth_range = (
                    self.reference_depth - 2 * depth_std,
                    self.reference_depth + 2 * depth_std,
                )

    def _normalize_depth_with_reference(self, depth_value: float) -> float:
        """Normalize depth value using background reference."""
        if not np.isfinite(depth_value) or self.reference_depth is None:
            return 0.5

        # Normalize relative to background depth
        if self.depth_range and self.depth_range[1] > self.depth_range[0]:
            # Closer objects (smaller depth) get higher values
            normalized = (self.depth_range[1] - depth_value) / (
                self.depth_range[1] - self.depth_range[0]
            )
            return float(np.clip(normalized, 0.0, 1.0))
        else:
            # Fallback: simple relative normalization
            if depth_value < self.reference_depth:
                return 0.8  # Closer than background
            else:
                return 0.2  # Farther than background

    def _detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame using YOLO.

        Args:
            frame: Input frame

        Returns:
            List of detection dictionaries with bbox, confidence, and class info
        """
        detections = []

        try:
            results = self.yolo_model(frame, stream=True)

            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    label = self.yolo_model.names[cls]
                    confidence = float(box.conf[0])

                    # Process detections of any target class above threshold
                    if (
                        label in self.target_classes
                        and confidence >= self.confidence_threshold
                    ):
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        detection = {
                            "bbox": (x1, y1, x2, y2),
                            "confidence": confidence,
                            "class": label,
                            "class_id": cls,
                        }
                        detections.append(detection)

        except Exception as e:
            if self.verbose:
                print(f"Error in object detection: {e}")

        return detections

    def _get_depth_for_detections(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get detailed depth information for all detections using enhanced processor."""
        depth_results = []

        for i, detection in enumerate(detections):
            try:
                bbox = detection["bbox"]
                class_name = detection["class"]

                # Generate unique object ID for tracking
                object_id = f"{class_name}_{i}_{int(time.time() * 1000) % 10000}"

                # Get enhanced depth information with persistence
                enhanced_result = (
                    self.depth_processor.get_enhanced_depth_for_spatial_audio(
                        frame, bbox, object_id=object_id, object_class=class_name
                    )
                )

                # Combine detection and enhanced depth information
                result = {
                    "raw_depth": enhanced_result["raw_depth"],
                    "normalized_depth": enhanced_result["normalized_depth"],
                    "depth_stats": enhanced_result.get("roi_stats", {}),
                    "quality_metrics": enhanced_result.get("quality_metrics", {}),
                    "spatial_audio": enhanced_result.get("spatial_audio", {}),
                    "depth_terminology": enhanced_result.get("depth_terminology", {}),
                    "persistence_info": enhanced_result.get("persistence_info", {}),
                    "detection": detection,
                    "bbox": bbox,
                    "class": class_name,
                    "confidence": detection["confidence"],
                    "object_id": object_id,
                }

                depth_results.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing depth for detection {detection}: {e}")
                depth_results.append(
                    {
                        "raw_depth": float("nan"),
                        "normalized_depth": 0.5,
                        "depth_stats": {},
                        "quality_metrics": {},
                        "spatial_audio": {},
                        "depth_terminology": {},
                        "persistence_info": {},
                        "detection": detection,
                        "bbox": detection["bbox"],
                        "class": detection["class"],
                        "confidence": detection["confidence"],
                        "object_id": f"error_{i}",
                    }
                )

        return depth_results

    def _draw_detections_with_depth(
        self,
        rgb_frame: np.ndarray,
        depth_color: np.ndarray,
        depth_results: List[Dict[str, Any]],
    ):
        """Draw detections with detailed depth information on both RGB and depth frames."""

        for i, result in enumerate(depth_results):
            bbox = result["bbox"]
            x1, y1, x2, y2 = bbox
            confidence = result["confidence"]
            class_name = result["class"]
            depth_stats = result["depth_stats"]

            # Choose color based on class
            class_colors = {
                "person": (0, 255, 0),  # Green
                "bottle": (255, 0, 0),  # Blue
                "cup": (0, 255, 255),  # Yellow
                "wine glass": (255, 165, 0),  # Orange
                "car": (0, 0, 255),  # Red
                "chair": (255, 255, 0),  # Cyan
                "laptop": (255, 0, 255),  # Magenta
            }
            color = class_colors.get(class_name, (128, 128, 128))  # Gray for unknown

            # Draw bounding box on RGB frame
            cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), color, 2)

            # Draw bounding box on depth frame
            cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)

            # Add detailed labels
            label_text = f"{class_name} {confidence:.2f}"

            # Clean depth display - just relative values
            raw_depth = result.get("raw_depth", float("nan"))

            if not np.isnan(raw_depth):
                depth_text = f"D: {raw_depth:.2f}"
            else:
                depth_text = "D: N/A"

            # RGB frame labels
            cv2.putText(
                rgb_frame,
                label_text,
                (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                rgb_frame,
                depth_text,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # Additional depth stats for RGB frame
            if not np.isnan(raw_depth):
                quality_metrics = result.get("quality_metrics", None)
                if quality_metrics and hasattr(quality_metrics, "overall_confidence"):
                    overall_confidence = quality_metrics.overall_confidence
                else:
                    overall_confidence = 0
                stats_text = f"Q: {overall_confidence:.2f}"

                # Add persistence info if available
                persistence_info = result.get("persistence_info", {})
                if persistence_info.get("is_tracked", False):
                    missing_frames = persistence_info.get("missing_frames", 0)
                    if missing_frames > 0:
                        stats_text += f" | P: {missing_frames}"

                cv2.putText(
                    rgb_frame,
                    stats_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

            # Depth frame labels (same as RGB frame)
            cv2.putText(
                depth_color,
                label_text,
                (x1, y1 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                depth_color,
                depth_text,
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            # Additional depth stats for depth frame
            if not np.isnan(depth_stats.get("min", float("nan"))):
                stats_text = (
                    f"Min: {depth_stats['min']:.2f} Max: {depth_stats['max']:.2f}"
                )
                cv2.putText(
                    depth_color,
                    stats_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

    def _display_stats(self, frame: np.ndarray, depth_results: List[Dict[str, Any]]):
        """Display detailed detection and depth statistics."""
        y_offset = 30

        # Display reference depth information
        if self.reference_depth is not None:
            cv2.putText(
                frame,
                f"Background: {self.reference_depth:.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            y_offset += 25

        # Detection count by class
        class_counts = {}
        for result in depth_results:
            class_name = result["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        # Display class counts
        for class_name, count in class_counts.items():
            cv2.putText(
                frame,
                f"{class_name}: {count}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            y_offset += 25

        # Individual detection details (limit to 3 to avoid clutter)
        for i, result in enumerate(depth_results[:3]):
            class_name = result["class"]
            confidence = result["confidence"]
            depth_stats = result["depth_stats"]

            if not np.isnan(depth_stats.get("median", float("nan"))):
                median_depth = depth_stats["median"]
                text = f"#{i+1} {class_name}: {confidence:.2f} conf, {median_depth:.2f}"
            else:
                text = f"#{i+1} {class_name}: {confidence:.2f} conf, depth N/A"

            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            y_offset += 20

    def _save_snapshot(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        depth_color: np.ndarray,
        detections: List[Dict[str, Any]],
    ):
        """Save current frame and detection data."""
        if not self.save_dir:
            return

        import os
        from datetime import datetime

        os.makedirs(self.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Save images
        cv2.imwrite(os.path.join(self.save_dir, f"rgb_{timestamp}.png"), frame)
        cv2.imwrite(
            os.path.join(self.save_dir, f"depth_color_{timestamp}.png"), depth_color
        )

        # Save raw depth
        np.save(os.path.join(self.save_dir, f"depth_raw_{timestamp}.npy"), depth_map)

        # Save detection data
        import json

        detection_data = {
            "timestamp": timestamp,
            "detections": [
                {
                    "bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "class": det["class"],
                    "raw_depth": result["raw_depth"],
                    "normalized_depth": result["normalized_depth"],
                }
                for det, result in zip(
                    detections, self._get_depth_for_detections(frame, detections)
                )
            ],
        }

        with open(
            os.path.join(self.save_dir, f"detections_{timestamp}.json"), "w"
        ) as f:
            json.dump(detection_data, f, indent=2)

        print(f"Saved snapshot with {len(detections)} detections: {timestamp}")

    def _output_spatial_audio_data(self, depth_results: List[Dict[str, Any]]):
        """Output spatial audio data for each detection."""
        for result in depth_results:
            class_name = result["class"]
            confidence = result["confidence"]
            raw_depth = result.get("raw_depth", float("nan"))
            normalized_depth = result.get("normalized_depth", 0.5)
            depth_stats = result.get("depth_stats", {})
            bbox = result["bbox"]

            # Use median depth for spatial audio (most robust metric)
            median_depth = depth_stats.get("median", raw_depth)

            # Calculate spatial position from bounding box center
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Calculate azimuth (left-right position) as percentage of frame width
            azimuth_percent = (center_x / self.width) * 100

            # Calculate elevation (up-down position) as percentage of frame height
            elevation_percent = (center_y / self.height) * 100

            # Output spatial audio data
            print(
                f"SPATIAL_AUDIO: {class_name} | "
                f"median_depth={median_depth:.2f} | "
                f"normalized={normalized_depth:.2f} | "
                f"azimuth={azimuth_percent:.1f}% | "
                f"elevation={elevation_percent:.1f}% | "
                f"confidence={confidence:.2f}"
            )

    def _calculate_fps(self) -> float:
        """Calculate current FPS."""
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0.0

    def start(self):
        """Start the integrated detection and depth streaming loop."""
        print("Starting Detection + Depth Stream with Persistence...")
        print(f"Target classes: {', '.join(self.target_classes)}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print(f"Persistence enabled: {self.persistence_enabled}")
        print(
            f"Persistence duration: {self.depth_processor.persistence_config.persistence_duration}s"
        )
        print("\nControls:")
        print("  q - Quit")
        print("  s - Save snapshot")
        print("  r - Reset depth statistics")
        print("  c - Cycle through target classes")
        print("  t - Toggle confidence threshold")
        print("  p - Toggle persistence")
        print("  + - Increase persistence duration")
        print("  - - Decrease persistence duration")

        cv2.namedWindow("Detection + Depth Stream", cv2.WINDOW_NORMAL)

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    if self.verbose:
                        print("Failed to read frame")
                    break

                # Detect objects
                detections = self._detect_objects(frame)

                # Estimate depth
                depth_map = self.depth_processor.estimate_depth(frame)
                depth_color = self.depth_processor.colorize_depth(depth_map)

                # Estimate background depth for reference
                self._estimate_background_depth(depth_map, detections)

                # Get depth for detections
                depth_results = []
                if detections:
                    depth_results = self._get_depth_for_detections(frame, detections)
                    self.detection_count += len(detections)

                    # Output spatial audio data for each detection
                    if self.verbose and depth_results:
                        self._output_spatial_audio_data(depth_results)

                # Update persistent objects (mark current detections as active)
                if self.persistence_enabled:
                    self.depth_processor.update_missing_objects()

                    # Get persistent objects that weren't detected in current frame
                    persistent_objects = self.depth_processor.get_persistent_objects()
                    persistent_results = [
                        obj for obj in persistent_objects if not obj["is_active"]
                    ]

                    # Add persistent objects to depth results
                    depth_results.extend(persistent_results)

                # Create display frames
                rgb_display = frame.copy()
                depth_display = depth_color.copy()

                # Draw detections with depth info
                if depth_results:
                    self._draw_detections_with_depth(
                        rgb_display, depth_display, depth_results
                    )

                # Display statistics
                self._display_stats(rgb_display, depth_results)

                # Add FPS and model info
                fps = self._calculate_fps()
                cv2.putText(
                    rgb_display,
                    f"FPS: {fps:.1f}",
                    (10, rgb_display.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    rgb_display,
                    f"Total detections: {self.detection_count}",
                    (10, rgb_display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Stack frames side by side
                if rgb_display.shape != depth_display.shape:
                    depth_display = cv2.resize(
                        depth_display, (rgb_display.shape[1], rgb_display.shape[0])
                    )

                stacked = np.hstack([rgb_display, depth_display])
                cv2.imshow("Detection + Depth Stream", stacked)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s") and self.save_dir:
                    self._save_snapshot(frame, depth_map, depth_color, detections)
                elif key == ord("r"):
                    self.depth_processor.reset_depth_stats()
                    if self.verbose:
                        print("Depth statistics reset")
                elif key == ord("c"):
                    # Cycle through common classes
                    common_classes = [
                        "person",
                        "bottle",
                        "car",
                        "chair",
                        "laptop",
                        "book",
                        "phone",
                    ]
                    current_classes = set(self.target_classes)

                    # Find next class not in current set
                    for cls in common_classes:
                        if cls not in current_classes:
                            self.target_classes = [cls]
                            break
                    else:
                        # If all classes are in use, reset to person only
                        self.target_classes = ["person"]

                    if self.verbose:
                        print(
                            f"Target classes changed to: {', '.join(self.target_classes)}"
                        )
                elif key == ord("t"):
                    # Toggle confidence threshold
                    if self.confidence_threshold == 0.5:
                        self.confidence_threshold = 0.3
                    else:
                        self.confidence_threshold = 0.5
                    if self.verbose:
                        print(f"Confidence threshold: {self.confidence_threshold}")
                elif key == ord("p"):
                    # Toggle persistence
                    self.persistence_enabled = not self.persistence_enabled
                    self.depth_processor.persistence_config.enabled = (
                        self.persistence_enabled
                    )
                    if self.verbose:
                        print(
                            f"Persistence: {'ON' if self.persistence_enabled else 'OFF'}"
                        )
                elif key == ord("+") or key == ord("="):
                    # Increase persistence duration
                    current_duration = (
                        self.depth_processor.persistence_config.persistence_duration
                    )
                    new_duration = min(10.0, current_duration + 0.5)
                    self.depth_processor.persistence_config.persistence_duration = (
                        new_duration
                    )
                    if self.verbose:
                        print(f"Persistence duration: {new_duration}s")
                elif key == ord("-"):
                    # Decrease persistence duration
                    current_duration = (
                        self.depth_processor.persistence_config.persistence_duration
                    )
                    new_duration = max(0.5, current_duration - 0.5)
                    self.depth_processor.persistence_config.persistence_duration = (
                        new_duration
                    )
                    if self.verbose:
                        print(f"Persistence duration: {new_duration}s")

        except KeyboardInterrupt:
            print("Stream interrupted by user")

        finally:
            self.release()

    def release(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Detection + Depth stream released")


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Detection + Depth Integration")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument(
        "--model", type=str, default="yolov8n.pt", help="YOLO model path"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["person", "bottle", "car"],
        help="Target classes to detect",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--depth-model",
        type=str,
        default="Intel/dpt-hybrid-midas",
        help="Depth model ID",
    )
    parser.add_argument(
        "--save-dir", type=str, default="", help="Directory to save snapshots"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable temporal persistence (default: enabled)",
    )
    parser.add_argument(
        "--persistence-duration",
        type=float,
        default=2.0,
        help="Persistence duration in seconds",
    )
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=5,
        help="Max frames to keep object without detection",
    )

    args = parser.parse_args()

    stream = DetectionDepthStream(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        model_path=args.model,
        target_classes=args.classes,
        depth_model_id=args.depth_model,
        confidence_threshold=args.confidence,
        save_dir=args.save_dir if args.save_dir else None,
        verbose=args.verbose,
        persistence_enabled=not args.no_persistence,
        persistence_duration=args.persistence_duration,
        max_missing_frames=args.max_missing_frames,
    )

    stream.start()


if __name__ == "__main__":
    main()
