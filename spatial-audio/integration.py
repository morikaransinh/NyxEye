#!/usr/bin/env python3
"""
Integration module for Detection + Depth + Spatial Audio

This module connects the DetectionDepthStream with the SpatialAudioEngine,
converting detected object positions and depths into 3D spatial audio.

Usage:
    python3 spatial-audio/integration.py
"""

import sys
import os
import time
from typing import List, Dict, Any

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Use the working simple spatial audio instead of broken OpenAL
from spatial_audio_simple import SimpleSpatialAudio

SpatialAudioEngine = SimpleSpatialAudio
PYOAL_AVAILABLE = True  # Set to True since we're using pygame instead

# Handle imports for depth module
try:
    from depth.detection_depth_stream import DetectionDepthStream
except ImportError:
    from depth.detection_depth_stream import DetectionDepthStream


class IntegratedSpatialAudioSystem:
    """
    Integrated system combining detection, depth, and spatial audio.

    This class bridges the DetectionDepthStream output to the SpatialAudioEngine,
    creating 3D spatial audio based on detected object positions and depths.
    """

    def __init__(
        self,
        target_classes: List[str] = None,
        confidence_threshold: float = 0.5,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        max_audio_sources: int = 10,
        master_volume: float = 0.2,
        verbose: bool = False,
    ):
        """
        Initialize the integrated spatial audio system.

        Args:
            target_classes: List of object classes to detect and create audio for
            confidence_threshold: Minimum confidence for detections
            camera_index: Camera index
            width: Frame width
            height: Frame height
            max_audio_sources: Maximum number of simultaneous audio sources
            master_volume: Master volume level (0.0 to 1.0)
            verbose: Enable verbose output
        """
        # PyGame audio is used instead of PyOpenAL
        if not PYOAL_AVAILABLE:
            raise RuntimeError("PyGame not available. Install with: pip install pygame")

        self.verbose = verbose
        self.frame_width = width
        self.frame_height = height

        # Initialize detection + depth stream
        self.detection_stream = DetectionDepthStream(
            camera_index=camera_index,
            width=width,
            height=height,
            target_classes=target_classes or ["person", "bottle", "cup", "car"],
            confidence_threshold=confidence_threshold,
            verbose=verbose,
            persistence_enabled=True,
            persistence_duration=2.0,
        )

        # Initialize spatial audio engine (SimpleSpatialAudio uses different API)
        self.audio_engine = SpatialAudioEngine(
            max_sources=max_audio_sources,
            master_volume=master_volume,
            min_volume=0.02,  # Very quiet when no objects detected
            max_volume=0.3,  # Louder for close objects
        )

        # Track objects for audio
        self.tracked_objects: Dict[str, dict] = {}

        print("Integrated Spatial Audio System initialized")

    def _convert_detection_to_audio(
        self, depth_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert detection + depth result to audio parameters.

        Args:
            depth_result: Detection + depth result dictionary

        Returns:
            Dictionary with audio parameters
        """
        bbox = depth_result.get("bbox", (0, 0, 0, 0))
        class_name = depth_result.get("class", "unknown")
        confidence = depth_result.get("confidence", 0.0)
        raw_depth = depth_result.get("raw_depth", 0.5)
        normalized_depth = depth_result.get("normalized_depth", 0.5)
        object_id = depth_result.get("object_id", "unknown")

        # Calculate center position of bounding box
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        # Normalize to [0, 1] range
        x_norm = center_x / self.frame_width
        y_norm = center_y / self.frame_height

        # Use normalized depth for spatial positioning
        # depth 0.0 = near, 1.0 = far
        depth_value = min(max(normalized_depth, 0.0), 1.0)

        # Calculate volume based on confidence and detection quality
        volume = confidence * 0.5  # Scale down confidence for volume
        volume = min(max(volume, 0.1), 1.0)  # Clamp volume

        return {
            "object_id": object_id,
            "class": class_name,
            "x": x_norm,
            "y": y_norm,
            "depth": depth_value,
            "volume": volume,
            "confidence": confidence,
            "raw_depth": raw_depth,
        }

    def _update_spatial_audio(self, depth_results: List[Dict[str, Any]]):
        """Update spatial audio based on detection results."""
        # Get current object IDs from detection results
        current_object_ids = set()

        for result in depth_results:
            audio_params = self._convert_detection_to_audio(result)
            object_id = audio_params["object_id"]
            current_object_ids.add(object_id)

            # Update or add audio source for this object
            self.audio_engine.update_object(
                object_id=object_id,
                x=audio_params["x"],
                y=audio_params["y"],
                depth=audio_params["depth"],
                volume=audio_params["volume"],
                active=True,
            )

            # Store for tracking
            self.tracked_objects[object_id] = audio_params

            if self.verbose:
                print(
                    f"AUDIO UPDATE: {audio_params['class']} ({object_id}) | "
                    f"pos=({audio_params['x']:.2f}, {audio_params['y']:.2f}) | "
                    f"depth={audio_params['depth']:.2f} | vol={audio_params['volume']:.2f}"
                )

        # Remove objects that are no longer detected (after timeout)
        objects_to_remove = []
        current_time = time.time()

        for object_id in self.tracked_objects:
            if object_id not in current_object_ids:
                # Check if object should be removed (keep tracked for 2 seconds)
                if not hasattr(self.tracked_objects[object_id], "last_seen"):
                    self.tracked_objects[object_id]["last_seen"] = current_time
                elif current_time - self.tracked_objects[object_id]["last_seen"] > 2.0:
                    objects_to_remove.append(object_id)
            else:
                # Update last seen time
                self.tracked_objects[object_id]["last_seen"] = current_time

        # Remove old objects
        for object_id in objects_to_remove:
            self.audio_engine.remove_object(object_id)
            del self.tracked_objects[object_id]
            if self.verbose:
                print(f"Removed object: {object_id}")

    def start(self):
        """Start the integrated system."""
        print("\n" + "=" * 60)
        print("Starting Integrated Detection + Depth + Spatial Audio System")
        print("=" * 60)
        print(f"Target classes: {', '.join(self.detection_stream.target_classes)}")
        print(f"Master volume: {self.audio_engine.master_volume}")
        print(f"Max audio sources: {self.audio_engine.max_sources}")
        print()
        print("Make sure you're wearing AirPods to experience 3D spatial audio!")
        print("Controls: q to quit, s to save snapshot")
        print("=" * 60 + "\n")

        # Start spatial audio engine
        self.audio_engine.start()

        # Modify detection stream to update audio
        original_start = self.detection_stream.start

        def wrapped_start():
            """Wrapped start function that updates audio."""
            cv2 = __import__("cv2")
            numpy = __import__("numpy")

            print("Starting detection + depth stream...")
            print("Controls:")
            print("  q - Quit")
            print("  s - Save snapshot")
            print("  r - Reset depth statistics")

            cv2.namedWindow("Detection + Depth + Audio", cv2.WINDOW_NORMAL)

            try:
                while True:
                    ret, frame = self.detection_stream.cap.read()
                    if not ret:
                        print("Failed to read frame")
                        break

                    # Detect objects
                    detections = self.detection_stream._detect_objects(frame)

                    # Estimate depth
                    depth_map = self.detection_stream.depth_processor.estimate_depth(
                        frame
                    )
                    depth_color = self.detection_stream.depth_processor.colorize_depth(
                        depth_map
                    )

                    # Get depth for detections
                    depth_results = []
                    if detections:
                        depth_results = self.detection_stream._get_depth_for_detections(
                            frame, detections
                        )
                        self._update_spatial_audio(depth_results)

                    # Create display frames
                    rgb_display = frame.copy()
                    depth_display = depth_color.copy()

                    # Draw detections
                    if depth_results:
                        self.detection_stream._draw_detections_with_depth(
                            rgb_display, depth_display, depth_results
                        )

                    # Display stats
                    self.detection_stream._display_stats(rgb_display, depth_results)

                    # Add audio stats
                    audio_stats = self.audio_engine.get_stats()
                    cv2.putText(
                        rgb_display,
                        f"Audio objects: {audio_stats['total_objects']}",
                        (10, rgb_display.shape[0] - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    # Add FPS
                    fps = self.detection_stream._calculate_fps()
                    cv2.putText(
                        rgb_display,
                        f"FPS: {fps:.1f}",
                        (10, rgb_display.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )

                    # Stack frames
                    if rgb_display.shape != depth_display.shape:
                        depth_display = cv2.resize(
                            depth_display, (rgb_display.shape[1], rgb_display.shape[0])
                        )

                    stacked = numpy.hstack([rgb_display, depth_display])
                    cv2.imshow("Detection + Depth + Audio", stacked)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("s") and self.detection_stream.save_dir:
                        self.detection_stream._save_snapshot(
                            frame, depth_map, depth_color, detections
                        )

            except KeyboardInterrupt:
                print("Stream interrupted by user")
            finally:
                self.cleanup()

        self.detection_stream.start = wrapped_start

        # Start the modified stream
        self.detection_stream.start()

    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        self.audio_engine.stop()
        self.detection_stream.release()
        print("Cleanup complete")

    def run(self):
        """Run the integrated system."""
        try:
            self.start()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detection + Depth + Spatial Audio Integration"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280, help="Frame width")
    parser.add_argument("--height", type=int, default=720, help="Frame height")
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["person", "bottle", "cup"],
        help="Target classes to detect",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Confidence threshold"
    )
    parser.add_argument(
        "--volume", type=float, default=0.2, help="Master volume (0.0-1.0)"
    )
    parser.add_argument("--max-sources", type=int, default=10, help="Max audio sources")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if not PYOAL_AVAILABLE:
        print("ERROR: PyOpenAL not available!")
        print("Install with:")
        print("  pip install PyOpenAL")
        print("  brew install openal-soft  # On macOS")
        sys.exit(1)

    system = IntegratedSpatialAudioSystem(
        target_classes=args.classes,
        confidence_threshold=args.confidence,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        max_audio_sources=args.max_sources,
        master_volume=args.volume,
        verbose=args.verbose,
    )

    system.run()


if __name__ == "__main__":
    main()
