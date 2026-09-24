#!/usr/bin/env python3
"""
Enhanced Depth Processor with Temporal Persistence and Clear Terminology

This module provides advanced depth processing with:
- Clear depth terminology based on reference points
- Temporal persistence to prevent objects from disappearing too quickly
- Configurable persistence settings for different use cases
- Better object tracking across frames

Usage:
    from depth.enhanced_depth_processor import EnhancedDepthProcessorWithPersistence

    processor = EnhancedDepthProcessorWithPersistence(
        persistence_enabled=True,
        persistence_duration=2.0,  # seconds
        max_missing_frames=5
    )
"""

import cv2
import numpy as np
import torch
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .enhanced_depth_processor import (
    EnhancedDepthProcessor,
    DepthQualityMetrics,
    SpatialAudioMetrics,
    ReferencePoint,
    ReferencePointManager,
    TemporalDepthTracker,
    DepthQualityAssessor,
    SpatialContextAnalyzer,
    DepthLayer,
    MovementDirection,
)
import logging

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ObjectState:
    """Represents the state of a tracked object."""

    object_id: str
    class_name: str
    bbox: Tuple[int, int, int, int]
    depth_info: Dict[str, Any]
    last_seen: float
    first_seen: float
    confidence_history: deque
    bbox_history: deque
    depth_history: deque
    is_active: bool
    missing_frames: int
    total_frames: int


@dataclass
class PersistenceConfig:
    """Configuration for temporal persistence."""

    enabled: bool = True
    persistence_duration: float = 2.0  # seconds to keep objects after last detection
    max_missing_frames: int = 5  # max frames to keep object without detection
    confidence_decay_rate: float = 0.1  # confidence decay per missing frame
    bbox_expansion_rate: float = 0.05  # bbox expansion per missing frame
    depth_prediction_enabled: bool = True
    motion_prediction_enabled: bool = True


class EnhancedDepthProcessorWithPersistence(EnhancedDepthProcessor):
    """
    Enhanced depth processor with temporal persistence and clear terminology.
    """

    def __init__(
        self,
        model_id: str = "Intel/dpt-hybrid-midas",
        device: Optional[torch.device] = None,
        persistence_config: Optional[PersistenceConfig] = None,
    ):
        super().__init__(model_id, device)

        # Persistence configuration
        self.persistence_config = persistence_config or PersistenceConfig()

        # Object tracking
        self.tracked_objects: Dict[str, ObjectState] = {}
        self.object_counter = 0

        # Reference point definitions for clear terminology
        self.reference_definitions = {
            "user_space": {
                "very_close": (0.0, 0.5),  # Personal space - immediate danger
                "close": (0.5, 1.5),  # Conversation distance
                "medium": (1.5, 3.0),  # Social interaction distance
                "far": (3.0, float("inf")),  # Background/ambient awareness
            },
            "background": {
                "foreground": (0.0, 1.0),  # Closer than background
                "midground": (1.0, 2.0),  # Similar to background
                "background": (2.0, float("inf")),  # Farther than background
            },
        }

        logger.info(f"EnhancedDepthProcessorWithPersistence initialized")
        logger.info(f"Persistence enabled: {self.persistence_config.enabled}")
        logger.info(
            f"Persistence duration: {self.persistence_config.persistence_duration}s"
        )

    def get_enhanced_depth_for_spatial_audio(
        self,
        frame: np.ndarray,
        bbox: Union[List[int], Tuple[int, int, int, int]],
        object_id: Optional[str] = None,
        object_class: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Get comprehensive depth information with temporal persistence.
        """
        # Generate object ID if not provided
        if object_id is None:
            object_id = f"obj_{self.object_counter}"
            self.object_counter += 1

        # Get basic depth information
        result = super().get_enhanced_depth_for_spatial_audio(
            frame, bbox, object_id, object_class
        )

        # Update or create object state
        self._update_object_state(object_id, object_class, bbox, result)

        # Add persistence information
        result["persistence_info"] = self._get_persistence_info(object_id)
        result["depth_terminology"] = self._get_depth_terminology(result["raw_depth"])

        return result

    def _update_object_state(
        self,
        object_id: str,
        object_class: str,
        bbox: Tuple[int, int, int, int],
        depth_result: Dict[str, Any],
    ):
        """Update object state with new detection."""
        current_time = time.time()

        if object_id not in self.tracked_objects:
            # Create new object state
            self.tracked_objects[object_id] = ObjectState(
                object_id=object_id,
                class_name=object_class,
                bbox=bbox,
                depth_info=depth_result,
                last_seen=current_time,
                first_seen=current_time,
                confidence_history=deque(maxlen=10),
                bbox_history=deque(maxlen=10),
                depth_history=deque(maxlen=10),
                is_active=True,
                missing_frames=0,
                total_frames=1,
            )
        else:
            # Update existing object state
            obj_state = self.tracked_objects[object_id]
            obj_state.bbox = bbox
            obj_state.depth_info = depth_result
            obj_state.last_seen = current_time
            obj_state.is_active = True
            obj_state.missing_frames = 0
            obj_state.total_frames += 1

            # Update histories
            obj_state.confidence_history.append(
                depth_result.get("quality_metrics", {}).overall_confidence
            )
            obj_state.bbox_history.append(bbox)
            obj_state.depth_history.append(depth_result["raw_depth"])

    def _get_persistence_info(self, object_id: str) -> Dict[str, Any]:
        """Get persistence information for an object."""
        if object_id not in self.tracked_objects:
            return {
                "is_tracked": False,
                "persistence_enabled": self.persistence_config.enabled,
                "missing_frames": 0,
                "time_since_last_seen": 0.0,
                "confidence_trend": 0.0,
                "predicted_position": None,
            }

        obj_state = self.tracked_objects[object_id]
        current_time = time.time()

        # Calculate persistence metrics
        time_since_last_seen = current_time - obj_state.last_seen
        confidence_trend = self._calculate_confidence_trend(obj_state)
        predicted_position = (
            self._predict_object_position(obj_state)
            if self.persistence_config.motion_prediction_enabled
            else None
        )

        return {
            "is_tracked": True,
            "persistence_enabled": self.persistence_config.enabled,
            "missing_frames": obj_state.missing_frames,
            "time_since_last_seen": time_since_last_seen,
            "confidence_trend": confidence_trend,
            "predicted_position": predicted_position,
            "total_frames_tracked": obj_state.total_frames,
            "tracking_duration": current_time - obj_state.first_seen,
        }

    def _get_depth_terminology(self, raw_depth: float) -> Dict[str, Any]:
        """Get clear depth terminology based on reference points."""
        terminology = {
            "raw_depth_meters": raw_depth,
            "user_space_category": self._classify_user_space_depth(raw_depth),
            "background_relationship": self._classify_background_relationship(
                raw_depth
            ),
            "safety_zone": self._classify_safety_zone(raw_depth),
            "social_distance": self._classify_social_distance(raw_depth),
        }

        return terminology

    def _classify_user_space_depth(self, depth: float) -> str:
        """Classify depth based on user space categories."""
        if depth < 0.5:
            return "very_close"  # Personal space - immediate attention needed
        elif depth < 1.5:
            return "close"  # Conversation distance
        elif depth < 3.0:
            return "medium"  # Social interaction distance
        else:
            return "far"  # Background/ambient awareness

    def _classify_background_relationship(self, depth: float) -> str:
        """Classify depth relative to background reference."""
        if self.reference_manager.background_reference is None:
            return "unknown"

        background_depth = self.reference_manager.background_reference.depth_value
        depth_diff = depth - background_depth

        if depth_diff < -0.5:
            return "foreground"  # Significantly closer than background
        elif depth_diff < 0.5:
            return "midground"  # Similar to background
        else:
            return "background"  # Farther than background

    def _classify_safety_zone(self, depth: float) -> str:
        """Classify depth based on safety considerations."""
        if depth < 0.3:
            return "danger_zone"  # Immediate collision risk
        elif depth < 0.8:
            return "caution_zone"  # High attention needed
        elif depth < 1.5:
            return "awareness_zone"  # Normal awareness
        else:
            return "safe_zone"  # Background awareness

    def _classify_social_distance(self, depth: float) -> str:
        """Classify depth based on social interaction distances."""
        if depth < 0.5:
            return "intimate"  # Intimate personal space
        elif depth < 1.2:
            return "personal"  # Personal space
        elif depth < 2.0:
            return "social"  # Social interaction distance
        elif depth < 3.5:
            return "public"  # Public space
        else:
            return "distant"  # Distant background

    def _calculate_confidence_trend(self, obj_state: ObjectState) -> float:
        """Calculate confidence trend over time."""
        if len(obj_state.confidence_history) < 2:
            return 0.0

        recent_confidences = list(obj_state.confidence_history)
        if len(recent_confidences) >= 3:
            # Calculate trend over last 3 frames
            trend = (recent_confidences[-1] - recent_confidences[-3]) / 2
        else:
            trend = recent_confidences[-1] - recent_confidences[0]

        return trend

    def _predict_object_position(
        self, obj_state: ObjectState
    ) -> Optional[Dict[str, Any]]:
        """Predict object position based on motion history."""
        if len(obj_state.bbox_history) < 2:
            return None

        # Get recent bounding boxes
        recent_bboxes = list(obj_state.bbox_history)[-3:]  # Last 3 frames

        if len(recent_bboxes) < 2:
            return None

        # Calculate motion vector
        current_bbox = recent_bboxes[-1]
        previous_bbox = recent_bboxes[-2]

        # Calculate center movement
        current_center = (
            (current_bbox[0] + current_bbox[2]) // 2,
            (current_bbox[1] + current_bbox[3]) // 2,
        )
        previous_center = (
            (previous_bbox[0] + previous_bbox[2]) // 2,
            (previous_bbox[1] + previous_bbox[3]) // 2,
        )

        motion_vector = (
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )

        # Predict next position (simple linear prediction)
        predicted_center = (
            current_center[0] + motion_vector[0],
            current_center[1] + motion_vector[1],
        )

        # Calculate predicted bounding box (assume same size)
        bbox_width = current_bbox[2] - current_bbox[0]
        bbox_height = current_bbox[3] - current_bbox[1]

        predicted_bbox = (
            predicted_center[0] - bbox_width // 2,
            predicted_center[1] - bbox_height // 2,
            predicted_center[0] + bbox_width // 2,
            predicted_center[1] + bbox_height // 2,
        )

        return {
            "predicted_bbox": predicted_bbox,
            "motion_vector": motion_vector,
            "prediction_confidence": min(
                1.0, len(recent_bboxes) / 5.0
            ),  # More history = higher confidence
        }

    def update_missing_objects(self):
        """Update objects that haven't been detected in recent frames."""
        if not self.persistence_config.enabled:
            return

        current_time = time.time()
        objects_to_remove = []

        for object_id, obj_state in self.tracked_objects.items():
            if obj_state.is_active:
                continue  # Object was detected in current frame

            obj_state.missing_frames += 1
            time_since_last_seen = current_time - obj_state.last_seen

            # Check if object should be removed
            should_remove = (
                time_since_last_seen > self.persistence_config.persistence_duration
                or obj_state.missing_frames > self.persistence_config.max_missing_frames
            )

            if should_remove:
                objects_to_remove.append(object_id)
            else:
                # Apply persistence effects
                self._apply_persistence_effects(obj_state)

        # Remove expired objects
        for object_id in objects_to_remove:
            del self.tracked_objects[object_id]
            if self.verbose:
                logger.info(f"Removed expired object: {object_id}")

    def _apply_persistence_effects(self, obj_state: ObjectState):
        """Apply persistence effects to missing objects."""
        # Decay confidence
        if len(obj_state.confidence_history) > 0:
            current_confidence = obj_state.confidence_history[-1]
            decayed_confidence = max(
                0.0, current_confidence - self.persistence_config.confidence_decay_rate
            )
            obj_state.confidence_history.append(decayed_confidence)

        # Expand bounding box (simulate uncertainty)
        if len(obj_state.bbox_history) > 0:
            current_bbox = obj_state.bbox_history[-1]
            expansion = int(
                self.persistence_config.bbox_expansion_rate * obj_state.missing_frames
            )

            expanded_bbox = (
                max(0, current_bbox[0] - expansion),
                max(0, current_bbox[1] - expansion),
                current_bbox[2] + expansion,
                current_bbox[3] + expansion,
            )
            obj_state.bbox_history.append(expanded_bbox)

        # Predict depth if enabled
        if (
            self.persistence_config.depth_prediction_enabled
            and len(obj_state.depth_history) > 0
        ):
            # Simple depth prediction (could be enhanced with motion model)
            recent_depths = list(obj_state.depth_history)[-3:]
            if len(recent_depths) >= 2:
                depth_trend = recent_depths[-1] - recent_depths[-2]
                predicted_depth = recent_depths[-1] + depth_trend
                obj_state.depth_history.append(predicted_depth)

    def get_persistent_objects(self) -> List[Dict[str, Any]]:
        """Get all currently tracked objects (including persistent ones)."""
        persistent_objects = []

        for object_id, obj_state in self.tracked_objects.items():
            # Skip if object is too old or missing too many frames
            if (
                not obj_state.is_active
                and obj_state.missing_frames
                > self.persistence_config.max_missing_frames
            ):
                continue

            # Get current or predicted bounding box
            if obj_state.is_active:
                current_bbox = obj_state.bbox
                confidence = obj_state.depth_info.get(
                    "quality_metrics", {}
                ).overall_confidence
            else:
                # Use predicted or last known position
                if len(obj_state.bbox_history) > 0:
                    current_bbox = obj_state.bbox_history[-1]
                else:
                    current_bbox = obj_state.bbox

                # Use decayed confidence
                if len(obj_state.confidence_history) > 0:
                    confidence = obj_state.confidence_history[-1]
                else:
                    confidence = 0.5

            persistent_objects.append(
                {
                    "object_id": object_id,
                    "class_name": obj_state.class_name,
                    "bbox": current_bbox,
                    "confidence": confidence,
                    "is_active": obj_state.is_active,
                    "missing_frames": obj_state.missing_frames,
                    "time_since_last_seen": time.time() - obj_state.last_seen,
                    "persistence_info": self._get_persistence_info(object_id),
                    "depth_info": obj_state.depth_info if obj_state.is_active else None,
                }
            )

        return persistent_objects

    def reset_persistence(self):
        """Reset all persistent object tracking."""
        self.tracked_objects.clear()
        self.object_counter = 0
        logger.info("Persistence tracking reset")

    def set_persistence_config(self, config: PersistenceConfig):
        """Update persistence configuration."""
        self.persistence_config = config
        logger.info(
            f"Persistence config updated: enabled={config.enabled}, duration={config.persistence_duration}s"
        )


# Factory function
def create_enhanced_depth_processor_with_persistence(
    model_id: str = "Intel/dpt-hybrid-midas",
    persistence_config: Optional[PersistenceConfig] = None,
) -> EnhancedDepthProcessorWithPersistence:
    """Create an enhanced depth processor with persistence."""
    return EnhancedDepthProcessorWithPersistence(
        model_id=model_id, persistence_config=persistence_config
    )


# Example usage
if __name__ == "__main__":
    import time

    print("=== Enhanced Depth Processor with Persistence Test ===")

    # Create processor with persistence
    config = PersistenceConfig(
        enabled=True,
        persistence_duration=3.0,
        max_missing_frames=10,
        confidence_decay_rate=0.05,
    )

    processor = create_enhanced_depth_processor_with_persistence(
        persistence_config=config
    )

    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 200, 200]

    print("Testing enhanced depth processing with persistence...")
    start_time = time.time()

    result = processor.get_enhanced_depth_for_spatial_audio(
        dummy_frame, dummy_bbox, object_id="test_obj", object_class="person"
    )

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.3f}s")
    print(f"Depth terminology: {result['depth_terminology']}")
    print(f"Persistence info: {result['persistence_info']}")

    print("Enhanced persistence test completed!")
