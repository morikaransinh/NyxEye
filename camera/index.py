#!/usr/bin/env python3
"""
Camera module for streaming video from various sources including phone cameras.

Supports:
- Built-in laptop cameras
- External USB cameras
- Phone cameras via DroidCam or Continuity Camera
- IP camera streaming (for IP Webcam apps)

Usage example:
    from camera import CameraStream

    # Basic usage with auto-detection
    camera = CameraStream()
    frame = camera.get_frame()

    # Specific camera index
    camera = CameraStream(camera_index=1)

    # IP camera streaming
    camera = CameraStream(ip_camera_url="http://192.168.1.100:8080/video")
"""

import cv2
import numpy as np
import logging
from typing import Optional, Tuple, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraStream:
    """
    Unified camera streaming class that can handle various camera sources.
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        ip_camera_url: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        """
        Initialize camera stream.

        Args:
            camera_index: Camera index (0 for built-in, 1+ for external/phone cameras)
            ip_camera_url: URL for IP camera streaming (e.g., IP Webcam)
            width: Requested frame width
            height: Requested frame height
            fps: Requested frames per second
        """
        self.camera_index = camera_index
        self.ip_camera_url = ip_camera_url
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

        # Initialize camera
        self._initialize_camera()

    def _initialize_camera(self):
        """Initialize the camera source."""
        try:
            if self.ip_camera_url:
                # IP camera streaming
                self.cap = cv2.VideoCapture(self.ip_camera_url)
                logger.info(f"Attempting to connect to IP camera: {self.ip_camera_url}")
            else:
                # Local camera
                if self.camera_index is None:
                    # Auto-detect camera
                    self.camera_index = self._auto_detect_camera()

                self.cap = cv2.VideoCapture(self.camera_index)
                logger.info(
                    f"Attempting to connect to camera index: {self.camera_index}"
                )

                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Test if camera is working
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera source")

            # Test read
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read from camera")

            logger.info(f"Successfully connected to camera. Frame size: {frame.shape}")

        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            raise

    def _auto_detect_camera(self) -> int:
        """
        Auto-detect available cameras by testing different indices.

        Returns:
            First working camera index
        """
        logger.info("Auto-detecting cameras...")

        for i in range(10):  # Test first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    logger.info(f"Found camera at index {i}")
                    return i
                cap.release()

        # Default to 0 if no camera found
        logger.warning("No camera detected, defaulting to index 0")
        return 0

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Get the next frame from the camera.

        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            logger.warning("Failed to read frame from camera")
            return None

    def get_frame_rgb(self) -> Optional[np.ndarray]:
        """
        Get the next frame in RGB format.

        Returns:
            Frame as numpy array (RGB format) or None if failed
        """
        frame = self.get_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_frame_info(self) -> dict:
        """
        Get current camera frame information.

        Returns:
            Dictionary with frame dimensions and properties
        """
        if self.cap is None:
            return {}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "camera_index": self.camera_index,
            "ip_url": self.ip_camera_url,
        }

    def release(self):
        """Release the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def list_available_cameras() -> List[int]:
    """
    List all available camera indices.

    Returns:
        List of working camera indices
    """
    available_cameras = []

    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                logger.info(f"Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()

    return available_cameras


def create_phone_camera_stream() -> Optional[CameraStream]:
    """
    Create a camera stream for external cameras (including phones via Continuity Camera).

    This function tries to find external cameras:
    1. Continuity Camera (iPhone connected via USB/WiFi)
    2. External USB cameras
    3. Any camera that's not the built-in laptop camera

    Returns:
        CameraStream instance or None if no external camera found
    """
    logger.info("Looking for external camera (phone or USB)...")

    # Try external camera indices (built-in is usually 0)
    external_camera_indices = [1, 2, 3, 4]

    for idx in external_camera_indices:
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.release()
                    logger.info(f"Found external camera at index {idx}")
                    return CameraStream(camera_index=idx)
                cap.release()
        except Exception as e:
            logger.debug(f"Camera {idx} failed: {e}")

    logger.warning("No external camera detected")
    return None


def create_ip_camera_stream(
    ip_address: str, port: int = 8080, path: str = "video"
) -> CameraStream:
    """
    Create an IP camera stream (useful for IP Webcam apps).

    Args:
        ip_address: IP address of the phone
        port: Port number (default 8080 for IP Webcam)
        path: Video path (default "video")

    Returns:
        CameraStream instance
    """
    url = f"http://{ip_address}:{port}/{path}"
    return CameraStream(ip_camera_url=url)


# Example usage and testing
if __name__ == "__main__":
    import time

    print("=== Camera Module Test ===")

    # List available cameras
    print("\nAvailable cameras:")
    cameras = list_available_cameras()
    if cameras:
        for cam in cameras:
            print(f"  Camera {cam}")
    else:
        print("  No cameras detected")

    # Try to create phone camera stream
    print("\nTrying to connect to phone camera...")
    phone_camera = create_phone_camera_stream()

    if phone_camera:
        print("Phone camera connected successfully!")
        print(f"Frame info: {phone_camera.get_frame_info()}")

        # Test a few frames
        print("Testing frames...")
        for i in range(5):
            frame = phone_camera.get_frame()
            if frame is not None:
                print(f"  Frame {i+1}: {frame.shape}")
            else:
                print(f"  Frame {i+1}: Failed")
            time.sleep(0.1)

        phone_camera.release()
    else:
        print("No phone camera found, trying built-in camera...")

        # Fallback to built-in camera
        camera = CameraStream()
        print(f"Built-in camera info: {camera.get_frame_info()}")

        # Test a few frames
        for i in range(5):
            frame = camera.get_frame()
            if frame is not None:
                print(f"  Frame {i+1}: {frame.shape}")
            else:
                print(f"  Frame {i+1}: Failed")
            time.sleep(0.1)

        camera.release()

    print("\nTest completed!")
