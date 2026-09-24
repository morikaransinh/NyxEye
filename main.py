#!/usr/bin/env python3
"""Project Daredevil - Main entry point (Detection + Depth + Spatial Audio)."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Project Daredevil - Full System Runner (Detection + Depth + Spatial Audio)"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=["person", "bottle", "cup"],
        help="Target classes to detect (default: person bottle cup)",
    )
    parser.add_argument(
        "--volume", type=float, default=0.15, help="Master volume (default: 0.15)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print("=" * 60)
    print("Project Daredevil - Main Runner")
    print("=" * 60)
    print(f"\nCamera: {args.camera}")
    print(f"Target classes: {', '.join(args.classes)}")
    print(f"Master volume: {args.volume}")
    print(f"Confidence threshold: {args.confidence}")
    print("\nThis will run:")
    print("  1. Camera capture")
    print("  2. Object detection (YOLO)")
    print("  3. Depth estimation")
    print("  4. Spatial audio positioning")
    print("\nPress Ctrl+C to stop")
    print("=" * 60)
    print()

    try:
        # Import and run integration
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "spatial-audio"))
        from integration import IntegratedSpatialAudioSystem

        system = IntegratedSpatialAudioSystem(
            target_classes=args.classes,
            confidence_threshold=args.confidence,
            camera_index=args.camera,
            master_volume=args.volume,
            verbose=args.verbose if args.verbose else True,
        )

        system.run()

    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


