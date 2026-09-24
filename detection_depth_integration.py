#!/usr/bin/env python3
"""
Integrated Detection and Depth Processing Module

This module now imports DetectionDepthStream from the depth module.

Usage:
    from depth import DetectionDepthStream

    stream = DetectionDepthStream(target_classes=['person', 'bottle'])
    stream.start()

Or run directly:
    python detection_depth_integration.py --camera 1 --verbose --classes person bottle --confidence 0.3
"""

# Import DetectionDepthStream from the depth module
from depth import DetectionDepthStream


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
