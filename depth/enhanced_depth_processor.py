#!/usr/bin/env python3
"""
Enhanced Depth Processor for Project Daredevil

This module provides advanced depth processing capabilities optimized for spatial audio
applications. It includes improved reference point selection, temporal tracking,
quality assessment, and spatial context analysis.

Key Features:
- Multi-layer reference point system
- Temporal depth tracking and prediction
- Depth quality assessment and validation
- Spatial context analysis for audio positioning
- Motion detection and compensation

Usage:
    from depth.enhanced_depth_processor import EnhancedDepthProcessor

    processor = EnhancedDepthProcessor()
    result = processor.get_enhanced_depth_for_spatial_audio(frame, bbox, object_id)
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import time
import logging
from dataclasses import dataclass
from enum import Enum

from .depth_processor import DepthProcessor, create_depth_processor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DepthLayer(Enum):
    """Depth layer classification."""

    VERY_CLOSE = 1  # 0-0.5m
    CLOSE = 2  # 0.5-1.5m
    MEDIUM = 3  # 1.5-3m
    FAR = 4  # 3m+


class MovementDirection(Enum):
    """Movement direction classification."""

    APPROACHING = "approaching"
    RECEDING = "receding"
    LATERAL = "lateral"
    STATIONARY = "stationary"


@dataclass
class DepthQualityMetrics:
    """Comprehensive depth quality metrics."""

    signal_to_noise_ratio: float
    edge_sharpness: float
    spatial_coherence: float
    temporal_consistency: float
    illumination_quality: float
    texture_richness: float
    overall_confidence: float


@dataclass
class SpatialAudioMetrics:
    """Metrics optimized for spatial audio positioning."""

    azimuth_angle: float  # Left-right position (-180° to +180°)
    elevation_angle: float  # Up-down position (-90° to +90°)
    distance_category: str  # 'very_close', 'close', 'medium', 'far'
    audio_priority: int  # Priority for audio rendering (1-10)
    spatial_uncertainty: float  # Uncertainty in spatial position
    proximity_warning: bool  # Should trigger proximity alert?
    movement_direction: str  # Movement classification
    movement_speed: str  # 'slow', 'medium', 'fast'
    audio_intensity: float  # Volume/intensity (0.0-1.0)
    audio_frequency: float  # Pitch/frequency modulation


@dataclass
class ReferencePoint:
    """Reference point with quality metrics."""

    depth_value: float
    position: Tuple[int, int]
    confidence: float
    stability_score: float
    temporal_consistency: float
    spatial_coverage: float
    last_updated: float


class ReferencePointManager:
    """Manages multiple reference points for depth normalization."""

    def __init__(self, max_history: int = 100):
        self.user_reference: Optional[ReferencePoint] = None
        self.background_reference: Optional[ReferencePoint] = None
        self.scene_references: Dict[str, ReferencePoint] = {}
        self.dynamic_references: List[ReferencePoint] = []
        self.max_history = max_history

        # Reference point categories
        self.reference_categories = {
            "user_space": {"very_close": 0.5, "close": 1.5, "medium": 3.0},
            "indoor": {"floor": None, "wall": None, "ceiling": None},
            "outdoor": {"ground": None, "horizon": None, "buildings": None},
        }

    def update_background_reference(
        self, depth_map: np.ndarray, detections: List[Dict]
    ) -> None:
        """Update background reference using corner and edge sampling."""
        if len(depth_map.shape) == 3:
            h, w = depth_map.shape[:2]
        else:
            h, w = depth_map.shape

        # Create mask for detected areas
        detection_mask = np.zeros((h, w), dtype=bool)
        for detection in detections:
            x1, y1, x2, y2 = detection.get("bbox", [0, 0, w, h])
            margin = 20
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            detection_mask[y1:y2, x1:x2] = True

        # Sample background areas (corners and edges)
        background_samples = []
        sample_size = min(50, w // 4, h // 4)

        # Corner sampling
        corners = [
            depth_map[:sample_size, :sample_size],
            depth_map[:sample_size, -sample_size:],
            depth_map[-sample_size:, :sample_size],
            depth_map[-sample_size:, -sample_size:],
        ]

        for corner in corners:
            # Handle 3D depth maps
            if len(corner.shape) == 3:
                corner_2d = corner[:, :, 0]  # Take first channel
            else:
                corner_2d = corner

            mask_slice = detection_mask[: corner_2d.shape[0], : corner_2d.shape[1]]
            valid_depths = corner_2d[~np.isnan(corner_2d) & ~mask_slice]
            if len(valid_depths) > 0:
                background_samples.extend(valid_depths.tolist())

        # Edge sampling if no detections
        if len(detections) == 0:
            edge_samples = [
                depth_map[:10, :],  # Top edge
                depth_map[-10:, :],  # Bottom edge
                depth_map[:, :10],  # Left edge
                depth_map[:, -10:],  # Right edge
            ]
            for edge in edge_samples:
                # Handle 3D depth maps
                if len(edge.shape) == 3:
                    edge_2d = edge[:, :, 0]  # Take first channel
                else:
                    edge_2d = edge

                valid_depths = edge_2d[~np.isnan(edge_2d)]
                if len(valid_depths) > 0:
                    background_samples.extend(valid_depths.tolist())

        # Update background reference
        if background_samples:
            median_depth = np.median(background_samples)
            std_depth = np.std(background_samples)

            # Calculate confidence based on sample size and consistency
            confidence = min(1.0, len(background_samples) / 100) * (
                1.0 - min(1.0, std_depth / median_depth)
            )

            self.background_reference = ReferencePoint(
                depth_value=median_depth,
                position=(w // 2, h // 2),  # Center position
                confidence=confidence,
                stability_score=self._calculate_stability_score(background_samples),
                temporal_consistency=0.8,  # Will be updated over time
                spatial_coverage=0.6,  # Estimated coverage
                last_updated=time.time(),
            )

    def _calculate_stability_score(self, samples: List[float]) -> float:
        """Calculate stability score based on sample variance."""
        if len(samples) < 2:
            return 0.0

        std_dev = np.std(samples)
        mean_val = np.mean(samples)

        if mean_val == 0:
            return 0.0

        # Lower coefficient of variation = higher stability
        cv = std_dev / mean_val
        stability = max(0.0, 1.0 - cv)
        return stability

    def get_best_reference_for_object(
        self, object_depth: float, object_class: str
    ) -> Optional[ReferencePoint]:
        """Get the most appropriate reference for an object."""
        references = []

        # Add background reference if available
        if self.background_reference:
            references.append(self.background_reference)

        # Add scene-specific references
        for ref_name, ref_point in self.scene_references.items():
            if ref_point.confidence > 0.5:
                references.append(ref_point)

        if not references:
            return None

        # Choose reference based on object class and depth
        if object_class in ["person", "hand"]:
            # For people, prefer user space references
            user_space_refs = [r for r in references if r.depth_value < 2.0]
            if user_space_refs:
                return max(user_space_refs, key=lambda x: x.confidence)

        # Default: highest confidence reference
        return max(references, key=lambda x: x.confidence)


class TemporalDepthTracker:
    """Tracks depth changes over time for temporal analysis."""

    def __init__(self, max_history: int = 30):
        self.depth_history: Dict[str, deque] = {}
        self.motion_tracker: Dict[str, Dict] = {}
        self.max_history = max_history

    def track_object_depth(
        self, object_id: str, depth_value: float, timestamp: float
    ) -> None:
        """Track depth changes over time for an object."""
        if object_id not in self.depth_history:
            self.depth_history[object_id] = deque(maxlen=self.max_history)
            self.motion_tracker[object_id] = {
                "velocity": 0.0,
                "acceleration": 0.0,
                "direction": MovementDirection.STATIONARY,
                "last_positions": deque(maxlen=5),
            }

        self.depth_history[object_id].append((timestamp, depth_value))

        # Update motion tracking
        if len(self.depth_history[object_id]) >= 2:
            self._update_motion_tracking(object_id, timestamp)

    def _update_motion_tracking(self, object_id: str, timestamp: float) -> None:
        """Update motion tracking for an object."""
        history = self.depth_history[object_id]
        motion_data = self.motion_tracker[object_id]

        if len(history) < 2:
            return

        # Calculate velocity (depth change per second)
        current_time, current_depth = history[-1]
        prev_time, prev_depth = history[-2]

        dt = current_time - prev_time
        if dt > 0:
            velocity = (current_depth - prev_depth) / dt
            motion_data["velocity"] = velocity

            # Calculate acceleration
            if len(history) >= 3:
                prev_prev_time, prev_prev_depth = history[-3]
                prev_velocity = (prev_depth - prev_prev_depth) / (
                    prev_time - prev_prev_time
                )
                acceleration = (velocity - prev_velocity) / dt
                motion_data["acceleration"] = acceleration

            # Determine movement direction
            if abs(velocity) < 0.1:  # Threshold for stationary
                motion_data["direction"] = MovementDirection.STATIONARY
            elif velocity < -0.1:  # Getting closer (depth decreasing)
                motion_data["direction"] = MovementDirection.APPROACHING
            elif velocity > 0.1:  # Getting farther (depth increasing)
                motion_data["direction"] = MovementDirection.RECEDING
            else:
                motion_data["direction"] = MovementDirection.LATERAL

    def get_depth_stability(self, object_id: str) -> float:
        """Get temporal stability score for an object."""
        if (
            object_id not in self.depth_history
            or len(self.depth_history[object_id]) < 3
        ):
            return 0.0

        depths = [d for _, d in self.depth_history[object_id]]
        std_dev = np.std(depths)
        mean_depth = np.mean(depths)

        if mean_depth == 0:
            return 0.0

        # Lower coefficient of variation = higher stability
        cv = std_dev / mean_depth
        stability = max(0.0, 1.0 - cv)
        return stability

    def predict_future_depth(
        self, object_id: str, time_horizon: float = 0.5
    ) -> Optional[float]:
        """Predict future depth for motion compensation."""
        if (
            object_id not in self.depth_history
            or len(self.depth_history[object_id]) < 3
        ):
            return None

        motion_data = self.motion_tracker[object_id]
        velocity = motion_data["velocity"]
        acceleration = motion_data["acceleration"]

        # Simple linear prediction with acceleration
        predicted_depth = (
            self.depth_history[object_id][-1][1]
            + velocity * time_horizon
            + 0.5 * acceleration * time_horizon**2
        )

        return predicted_depth


class DepthQualityAssessor:
    """Assesses depth quality and reliability."""

    def assess_depth_quality(
        self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> DepthQualityMetrics:
        """Assess comprehensive depth quality metrics."""
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]
        valid_depths = roi_depth[~np.isnan(roi_depth)]

        if len(valid_depths) == 0:
            return DepthQualityMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Signal-to-noise ratio
        signal_to_noise = self._calculate_snr(valid_depths)

        # Edge sharpness
        edge_sharpness = self._calculate_edge_sharpness(roi_depth)

        # Spatial coherence
        spatial_coherence = self._calculate_spatial_coherence(roi_depth)

        # Temporal consistency (placeholder - would need frame history)
        temporal_consistency = 0.8

        # Illumination quality (based on depth map characteristics)
        illumination_quality = self._assess_illumination_quality(depth_map, bbox)

        # Texture richness
        texture_richness = self._assess_texture_richness(roi_depth)

        # Overall confidence
        overall_confidence = np.mean(
            [
                signal_to_noise,
                edge_sharpness,
                spatial_coherence,
                temporal_consistency,
                illumination_quality,
                texture_richness,
            ]
        )

        return DepthQualityMetrics(
            signal_to_noise_ratio=signal_to_noise,
            edge_sharpness=edge_sharpness,
            spatial_coherence=spatial_coherence,
            temporal_consistency=temporal_consistency,
            illumination_quality=illumination_quality,
            texture_richness=texture_richness,
            overall_confidence=overall_confidence,
        )

    def _calculate_snr(self, depths: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        if len(depths) < 2:
            return 0.0

        signal = np.mean(depths)
        noise = np.std(depths)

        if noise == 0:
            return 1.0

        snr = signal / noise
        return min(1.0, snr / 10.0)  # Normalize to 0-1 range

    def _calculate_edge_sharpness(self, roi_depth: np.ndarray) -> float:
        """Calculate edge sharpness using gradient magnitude."""
        if roi_depth.size == 0:
            return 0.0

        # Calculate gradients
        grad_x = cv2.Sobel(roi_depth, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi_depth, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Average gradient magnitude as sharpness measure
        sharpness = np.mean(gradient_magnitude[~np.isnan(gradient_magnitude)])

        # Normalize to 0-1 range
        return min(1.0, sharpness / 5.0)

    def _calculate_spatial_coherence(self, roi_depth: np.ndarray) -> float:
        """Calculate spatial coherence within ROI."""
        if roi_depth.size == 0:
            return 0.0

        valid_depths = roi_depth[~np.isnan(roi_depth)]
        if len(valid_depths) < 2:
            return 0.0

        # Calculate local variance
        kernel = np.ones((3, 3), np.float32) / 9
        local_mean = cv2.filter2D(roi_depth, -1, kernel)
        local_variance = cv2.filter2D((roi_depth - local_mean) ** 2, -1, kernel)

        # Coherence is inverse of local variance
        avg_local_variance = np.mean(local_variance[~np.isnan(local_variance)])
        coherence = 1.0 / (1.0 + avg_local_variance)

        return min(1.0, coherence)

    def _assess_illumination_quality(
        self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> float:
        """Assess illumination quality based on depth characteristics."""
        # This is a simplified assessment - in practice, you'd analyze the RGB image
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]
        valid_depths = roi_depth[~np.isnan(roi_depth)]

        if len(valid_depths) == 0:
            return 0.0

        # Good illumination typically results in more consistent depth values
        depth_consistency = 1.0 - (np.std(valid_depths) / np.mean(valid_depths))

        return max(0.0, min(1.0, depth_consistency))

    def _assess_texture_richness(self, roi_depth: np.ndarray) -> float:
        """Assess texture richness for depth estimation reliability."""
        if roi_depth.size == 0:
            return 0.0

        # Calculate local standard deviation as texture measure
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(roi_depth, -1, kernel)
        local_std = np.sqrt(cv2.filter2D((roi_depth - local_mean) ** 2, -1, kernel))

        avg_local_std = np.mean(local_std[~np.isnan(local_std)])

        # Normalize to 0-1 range
        return min(1.0, avg_local_std / 2.0)


class SpatialContextAnalyzer:
    """Analyzes spatial context for audio positioning."""

    def analyze_spatial_context(
        self, detections: List[Dict], depth_map: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze spatial context for audio positioning."""
        context = {
            "scene_type": self._classify_scene_type(depth_map),
            "object_relationships": self._analyze_object_relationships(detections),
            "depth_layers": self._identify_depth_layers(detections),
            "spatial_density": self._calculate_spatial_density(detections, depth_map),
            "movement_patterns": self._analyze_movement_patterns(detections),
        }

        return context

    def _classify_scene_type(self, depth_map: np.ndarray) -> str:
        """Classify scene type based on depth characteristics."""
        valid_depths = depth_map[~np.isnan(depth_map)]

        if len(valid_depths) == 0:
            return "unknown"

        mean_depth = np.mean(valid_depths)
        depth_range = np.max(valid_depths) - np.min(valid_depths)

        if mean_depth < 2.0 and depth_range < 3.0:
            return "indoor_close"
        elif mean_depth < 5.0 and depth_range < 8.0:
            return "indoor_medium"
        elif mean_depth > 10.0 and depth_range > 15.0:
            return "outdoor_wide"
        else:
            return "mixed"

    def _analyze_object_relationships(self, detections: List[Dict]) -> Dict[str, Any]:
        """Analyze relationships between detected objects."""
        relationships = {
            "closest_pair": None,
            "depth_hierarchy": [],
            "spatial_clusters": [],
        }

        if len(detections) < 2:
            return relationships

        # Find closest pair
        min_distance = float("inf")
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i + 1 :], i + 1):
                # Calculate 3D distance (simplified)
                depth1 = det1.get("raw_depth", 0)
                depth2 = det2.get("raw_depth", 0)

                # 2D distance in image space
                bbox1 = det1.get("bbox", [0, 0, 0, 0])
                bbox2 = det2.get("bbox", [0, 0, 0, 0])

                center1 = ((bbox1[0] + bbox1[2]) // 2, (bbox1[1] + bbox1[3]) // 2)
                center2 = ((bbox2[0] + bbox2[2]) // 2, (bbox2[1] + bbox2[3]) // 2)

                distance_2d = np.sqrt(
                    (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
                )

                # Approximate 3D distance
                distance_3d = np.sqrt(distance_2d**2 + (depth1 - depth2) ** 2)

                if distance_3d < min_distance:
                    min_distance = distance_3d
                    relationships["closest_pair"] = (i, j, distance_3d)

        # Create depth hierarchy
        sorted_detections = sorted(
            detections, key=lambda x: x.get("raw_depth", float("inf"))
        )
        relationships["depth_hierarchy"] = [
            (i, det.get("class", "unknown"), det.get("raw_depth", 0))
            for i, det in enumerate(sorted_detections)
        ]

        return relationships

    def _identify_depth_layers(self, detections: List[Dict]) -> Dict[str, List[int]]:
        """Identify objects in different depth layers."""
        layers = {"very_close": [], "close": [], "medium": [], "far": []}

        for i, detection in enumerate(detections):
            depth = detection.get("raw_depth", float("inf"))

            if depth < 0.5:
                layers["very_close"].append(i)
            elif depth < 1.5:
                layers["close"].append(i)
            elif depth < 3.0:
                layers["medium"].append(i)
            else:
                layers["far"].append(i)

        return layers

    def _calculate_spatial_density(
        self, detections: List[Dict], depth_map: np.ndarray
    ) -> float:
        """Calculate spatial density of objects."""
        if len(detections) == 0:
            return 0.0

        if len(depth_map.shape) == 3:
            h, w = depth_map.shape[:2]
        else:
            h, w = depth_map.shape
        total_area = h * w

        # Calculate total area covered by detections
        covered_area = 0
        for detection in detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            covered_area += area

        density = covered_area / total_area
        return min(1.0, density)

    def _analyze_movement_patterns(self, detections: List[Dict]) -> Dict[str, Any]:
        """Analyze movement patterns of detected objects."""
        patterns = {
            "approaching_objects": [],
            "receding_objects": [],
            "stationary_objects": [],
            "lateral_movement": [],
        }

        for i, detection in enumerate(detections):
            movement = detection.get("movement_direction", MovementDirection.STATIONARY)

            if movement == MovementDirection.APPROACHING:
                patterns["approaching_objects"].append(i)
            elif movement == MovementDirection.RECEDING:
                patterns["receding_objects"].append(i)
            elif movement == MovementDirection.LATERAL:
                patterns["lateral_movement"].append(i)
            else:
                patterns["stationary_objects"].append(i)

        return patterns


class EnhancedDepthProcessor(DepthProcessor):
    """
    Enhanced depth processor with advanced metrics and reference point management.
    """

    def __init__(
        self,
        model_id: str = "Intel/dpt-hybrid-midas",
        device: Optional[torch.device] = None,
    ):
        super().__init__(model_id, device)

        # Initialize enhanced components
        self.reference_manager = ReferencePointManager()
        self.temporal_tracker = TemporalDepthTracker()
        self.quality_assessor = DepthQualityAssessor()
        self.spatial_analyzer = SpatialContextAnalyzer()

        # Object tracking
        self.object_counter = 0
        self.object_registry: Dict[str, Dict] = {}

        logger.info("EnhancedDepthProcessor initialized with advanced metrics")

    def get_enhanced_depth_for_spatial_audio(
        self,
        frame: np.ndarray,
        bbox: Union[List[int], Tuple[int, int, int, int]],
        object_id: Optional[str] = None,
        object_class: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Get comprehensive depth information optimized for spatial audio.

        Args:
            frame: Input frame (BGR format)
            bbox: Bounding box as [x1, y1, x2, y2]
            object_id: Unique identifier for object tracking
            object_class: Object class for context-aware processing

        Returns:
            Comprehensive depth information dictionary
        """
        # Generate object ID if not provided
        if object_id is None:
            object_id = f"obj_{self.object_counter}"
            self.object_counter += 1

        # Estimate depth map
        depth_map = self.estimate_depth(frame)

        # Get basic depth information
        raw_depth = self.get_depth_from_bbox(depth_map, bbox, method="median")

        # Update reference points
        detections = [{"bbox": bbox, "class": object_class, "raw_depth": raw_depth}]
        self.reference_manager.update_background_reference(depth_map, detections)

        # Track temporal changes
        timestamp = time.time()
        self.temporal_tracker.track_object_depth(object_id, raw_depth, timestamp)

        # Assess depth quality
        quality_metrics = self.quality_assessor.assess_depth_quality(depth_map, bbox)

        # Get best reference for normalization
        best_reference = self.reference_manager.get_best_reference_for_object(
            raw_depth, object_class
        )

        # Calculate normalized depth
        if best_reference:
            normalized_depth = self._normalize_with_reference(raw_depth, best_reference)
            relative_depth = raw_depth - best_reference.depth_value
        else:
            normalized_depth = self.normalize_depth(raw_depth, method="relative")
            relative_depth = 0.0

        # Determine depth layer
        depth_layer = self._classify_depth_layer(raw_depth)

        # Get temporal metrics
        depth_stability = self.temporal_tracker.get_depth_stability(object_id)
        motion_data = self.temporal_tracker.motion_tracker.get(object_id, {})

        # Calculate spatial audio metrics
        spatial_audio_metrics = self._calculate_spatial_audio_metrics(
            frame, bbox, raw_depth, normalized_depth, object_class
        )

        # Update object registry
        self.object_registry[object_id] = {
            "class": object_class,
            "bbox": bbox,
            "last_seen": timestamp,
            "depth_history": list(
                self.temporal_tracker.depth_history.get(object_id, [])
            ),
            "motion_data": motion_data,
        }

        # Compile comprehensive result
        result = {
            # Basic depth information
            "raw_depth": raw_depth,
            "normalized_depth": normalized_depth,
            "relative_depth": relative_depth,
            "depth_layer": depth_layer,
            # Quality metrics
            "quality_metrics": quality_metrics,
            # Temporal metrics
            "depth_stability": depth_stability,
            "temporal_consistency": quality_metrics.temporal_consistency,
            # Motion metrics
            "movement_direction": motion_data.get(
                "direction", MovementDirection.STATIONARY
            ).value,
            "movement_velocity": motion_data.get("velocity", 0.0),
            "movement_acceleration": motion_data.get("acceleration", 0.0),
            # Spatial audio metrics
            "spatial_audio": spatial_audio_metrics,
            # Reference information
            "reference_depth": best_reference.depth_value if best_reference else None,
            "reference_confidence": (
                best_reference.confidence if best_reference else 0.0
            ),
            # Object tracking
            "object_id": object_id,
            "object_class": object_class,
            "timestamp": timestamp,
            # Additional context
            "bbox": bbox,
            "roi_stats": self._calculate_roi_stats(depth_map, bbox),
        }

        return result

    def _normalize_with_reference(
        self, depth_value: float, reference: ReferencePoint
    ) -> float:
        """Normalize depth value using a specific reference point."""
        if not np.isfinite(depth_value) or reference is None:
            return 0.5

        # Use reference depth as baseline
        relative_depth = depth_value - reference.depth_value

        # Normalize based on typical depth ranges
        if relative_depth < -1.0:  # Much closer than reference
            return 0.9
        elif relative_depth < -0.5:  # Closer than reference
            return 0.8
        elif relative_depth < 0.5:  # Similar to reference
            return 0.5
        elif relative_depth < 1.0:  # Farther than reference
            return 0.3
        else:  # Much farther than reference
            return 0.1

    def _classify_depth_layer(self, depth_value: float) -> str:
        """Classify object into depth layer."""
        if depth_value < 0.5:
            return "very_close"
        elif depth_value < 1.5:
            return "close"
        elif depth_value < 3.0:
            return "medium"
        else:
            return "far"

    def _calculate_spatial_audio_metrics(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        raw_depth: float,
        normalized_depth: float,
        object_class: str,
    ) -> SpatialAudioMetrics:
        """Calculate metrics optimized for spatial audio positioning."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Calculate center position
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Calculate azimuth angle (-180° to +180°)
        azimuth_angle = ((center_x / w) - 0.5) * 360  # Convert to degrees

        # Calculate elevation angle (-90° to +90°)
        elevation_angle = ((center_y / h) - 0.5) * 180  # Convert to degrees

        # Determine distance category
        distance_category = self._classify_depth_layer(raw_depth)

        # Calculate audio priority (1-10)
        audio_priority = self._calculate_audio_priority(
            object_class, distance_category, raw_depth
        )

        # Calculate spatial uncertainty
        spatial_uncertainty = self._calculate_spatial_uncertainty(raw_depth, bbox)

        # Determine proximity warning
        proximity_warning = distance_category in ["very_close", "close"]

        # Get movement information
        movement_direction = "stationary"  # Would be updated from temporal tracker
        movement_speed = "slow"  # Would be calculated from velocity

        # Calculate audio parameters
        audio_intensity = self._calculate_audio_intensity(
            distance_category, audio_priority
        )
        audio_frequency = self._calculate_audio_frequency(
            distance_category, elevation_angle
        )

        return SpatialAudioMetrics(
            azimuth_angle=azimuth_angle,
            elevation_angle=elevation_angle,
            distance_category=distance_category,
            audio_priority=audio_priority,
            spatial_uncertainty=spatial_uncertainty,
            proximity_warning=proximity_warning,
            movement_direction=movement_direction,
            movement_speed=movement_speed,
            audio_intensity=audio_intensity,
            audio_frequency=audio_frequency,
        )

    def _calculate_audio_priority(
        self, object_class: str, distance_category: str, depth: float
    ) -> int:
        """Calculate audio priority (1-10) based on object importance."""
        priority = 5  # Base priority

        # Adjust based on object class
        class_priorities = {
            "person": 3,
            "hand": 2,
            "bottle": 1,
            "car": 2,
            "chair": 0,
            "laptop": 1,
        }
        priority += class_priorities.get(object_class, 0)

        # Adjust based on distance
        distance_priorities = {"very_close": 3, "close": 2, "medium": 1, "far": 0}
        priority += distance_priorities.get(distance_category, 0)

        # Adjust based on depth (closer = higher priority)
        if depth < 1.0:
            priority += 2
        elif depth < 2.0:
            priority += 1

        return max(1, min(10, priority))

    def _calculate_spatial_uncertainty(
        self, depth: float, bbox: Tuple[int, int, int, int]
    ) -> float:
        """Calculate spatial uncertainty based on depth and bbox size."""
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)

        # Larger objects at similar depth have lower uncertainty
        # Closer objects have lower uncertainty
        uncertainty = 1.0 / (1.0 + bbox_area / 10000)  # Normalize bbox area
        uncertainty *= 1.0 + depth / 10.0  # Increase with distance

        return min(1.0, uncertainty)

    def _calculate_audio_intensity(
        self, distance_category: str, audio_priority: int
    ) -> float:
        """Calculate audio intensity (0.0-1.0) for spatial audio."""
        base_intensity = {
            "very_close": 0.9,
            "close": 0.7,
            "medium": 0.5,
            "far": 0.3,
        }.get(distance_category, 0.5)

        # Adjust based on priority
        priority_factor = audio_priority / 10.0

        return min(1.0, base_intensity * (0.5 + 0.5 * priority_factor))

    def _calculate_audio_frequency(
        self, distance_category: str, elevation_angle: float
    ) -> float:
        """Calculate audio frequency modulation for spatial audio."""
        # Base frequency based on distance
        base_freq = {"very_close": 1000, "close": 800, "medium": 600, "far": 400}.get(
            distance_category, 600
        )

        # Modulate based on elevation
        elevation_factor = 1.0 + (elevation_angle / 90.0) * 0.5

        return base_freq * elevation_factor

    def _calculate_roi_stats(
        self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Dict[str, float]:
        """Calculate comprehensive ROI statistics."""
        x1, y1, x2, y2 = bbox
        roi_depth = depth_map[y1:y2, x1:x2]
        valid_depths = roi_depth[~np.isnan(roi_depth)]

        if len(valid_depths) == 0:
            return {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "valid_pixels": 0,
                "total_pixels": roi_depth.size,
            }

        return {
            "mean": float(np.mean(valid_depths)),
            "std": float(np.std(valid_depths)),
            "min": float(np.min(valid_depths)),
            "max": float(np.max(valid_depths)),
            "median": float(np.median(valid_depths)),
            "valid_pixels": len(valid_depths),
            "total_pixels": roi_depth.size,
        }

    def get_spatial_context(
        self, frame: np.ndarray, detections: List[Dict]
    ) -> Dict[str, Any]:
        """Get comprehensive spatial context analysis."""
        depth_map = self.estimate_depth(frame)
        return self.spatial_analyzer.analyze_spatial_context(detections, depth_map)

    def reset_tracking(self):
        """Reset all tracking data."""
        self.temporal_tracker = TemporalDepthTracker()
        self.object_registry.clear()
        self.object_counter = 0
        logger.info("Enhanced depth tracking reset")


# Factory function
def create_enhanced_depth_processor(
    model_id: str = "Intel/dpt-hybrid-midas",
) -> EnhancedDepthProcessor:
    """Create an enhanced depth processor instance."""
    return EnhancedDepthProcessor(model_id=model_id)


# Example usage
if __name__ == "__main__":
    import time

    print("=== Enhanced Depth Processor Test ===")

    # Create enhanced processor
    processor = create_enhanced_depth_processor()

    # Test with dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_bbox = [100, 100, 200, 200]

    print("Testing enhanced depth processing...")
    start_time = time.time()

    result = processor.get_enhanced_depth_for_spatial_audio(
        dummy_frame, dummy_bbox, object_id="test_obj", object_class="bottle"
    )

    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.3f}s")
    print(f"Result keys: {list(result.keys())}")
    print(f"Raw depth: {result['raw_depth']:.3f}")
    print(f"Normalized depth: {result['normalized_depth']:.3f}")
    print(f"Depth layer: {result['depth_layer']}")
    print(f"Audio priority: {result['spatial_audio'].audio_priority}")
    print(f"Quality confidence: {result['quality_metrics'].overall_confidence:.3f}")

    print("Enhanced test completed!")
