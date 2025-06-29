"""
SLIVS (Segment Layer Integrated Vision System) - Core Implementation
Static Scene Understanding Foundation with Depth-Informed SAM Point Selection

This module implements the basic data structures and pipeline for SLIVS
with static scene assumptions as the foundation.
"""

import numpy as np
import cv2
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from uuid import UUID, uuid4
import json
import pickle
from pathlib import Path
import time
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import DBSCAN


# Core Data Structures
@dataclass
class SLIVSObject:
    """
    Core object representation in SLIVS memory
    Basic version for static scene understanding
    """
    object_id: UUID = field(default_factory=uuid4)
    top_segment: np.ndarray = field(default=None)  # SAM 2 mask
    depth_range: Tuple[int, int] = (0, 255)  # [min_depth, max_depth] (0-255)
    confidence: float = 1.0  # 0.0-1.0
    last_seen_frame: int = 0
    motion_vector: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # [x, y, depth]
    physics_constraints: List[str] = field(default_factory=list)
    observation_history: List[Dict] = field(default_factory=list)
    is_movable: bool = False  # Start with static assumption
    occlusion_state: str = "visible"  # "visible", "predicted", "occluded", "lost"
    category: str = "unknown"
    creation_frame: int = 0
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # [x1, y1, x2, y2]
    
    def __post_init__(self):
        """Initialize computed properties after creation"""
        if self.top_segment is not None:
            self.bounding_box = self._calculate_bounding_box()
    
    def _calculate_bounding_box(self) -> Tuple[int, int, int, int]:
        """Calculate bounding box from segment mask"""
        if self.top_segment is None:
            return (0, 0, 0, 0)
        
        coords = np.where(self.top_segment)
        if len(coords[0]) == 0:
            return (0, 0, 0, 0)
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        return (int(x_min), int(y_min), int(x_max), int(y_max))
    
    def get_centroid(self) -> Tuple[float, float]:
        """Calculate centroid of the segment"""
        if self.top_segment is None:
            return (0.0, 0.0)
        
        coords = np.where(self.top_segment)
        if len(coords[0]) == 0:
            return (0.0, 0.0)
        
        y_center = float(np.mean(coords[0]))
        x_center = float(np.mean(coords[1]))
        return (x_center, y_center)
    
    def get_area(self) -> int:
        """Get segment area in pixels"""
        if self.top_segment is None:
            return 0
        return int(np.sum(self.top_segment))
    
    def update_confidence(self, validation_score: float):
        """Update confidence based on validation score"""
        if validation_score > 0.8:
            self.confidence = min(1.0, self.confidence + 0.02)
        elif validation_score < 0.3:
            self.confidence = max(0.0, self.confidence - 0.05)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary (for JSON export)"""
        return {
            'object_id': str(self.object_id),
            'depth_range': self.depth_range,
            'confidence': self.confidence,
            'last_seen_frame': self.last_seen_frame,
            'motion_vector': self.motion_vector,
            'physics_constraints': self.physics_constraints,
            'observation_history': self.observation_history,
            'is_movable': self.is_movable,
            'occlusion_state': self.occlusion_state,
            'category': self.category,
            'creation_frame': self.creation_frame,
            'bounding_box': self.bounding_box,
            'centroid': self.get_centroid(),
            'area': self.get_area()
        }


@dataclass
class DepthFrame:
    """
    Single frame with depth and segmentation information
    """
    frame_number: int
    midas_output: np.ndarray = field(default=None)  # Raw MiDAS depth
    ordinal_depth: np.ndarray = field(default=None)  # 0-255 ordinal ranking
    segments: Dict[UUID, np.ndarray] = field(default_factory=dict)  # object_id -> mask
    timestamp: float = field(default_factory=time.time)
    rgb_frame: np.ndarray = field(default=None)  # Original RGB frame
    depth_layers: Dict[int, np.ndarray] = field(default_factory=dict)  # layer_id -> depth mask
    
    def get_depth_at_point(self, x: int, y: int) -> int:
        """Get ordinal depth value at specific point"""
        if self.ordinal_depth is None:
            return 0
        
        h, w = self.ordinal_depth.shape
        if 0 <= y < h and 0 <= x < w:
            return int(self.ordinal_depth[y, x])
        return 0
    
    def get_depth_range_for_segment(self, segment: np.ndarray) -> Tuple[int, int]:
        """Get depth range covered by a segment"""
        if self.ordinal_depth is None:
            return (0, 255)
        
        masked_depth = self.ordinal_depth[segment > 0]
        if len(masked_depth) == 0:
            return (0, 255)
        
        return (int(masked_depth.min()), int(masked_depth.max()))


class SLIVSMemoryBank:
    """
    Memory management for tracked objects
    Minimal version for static scene understanding
    """
    
    def __init__(self, max_objects: int = 100, confidence_threshold: float = 0.1):
        self.objects: Dict[UUID, SLIVSObject] = {}
        self.max_objects = max_objects
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        
    def add_object(self, obj: SLIVSObject) -> UUID:
        """Add new object to memory"""
        # Clean memory if needed
        if len(self.objects) >= self.max_objects:
            self.consolidate_memory()
        
        self.objects[obj.object_id] = obj
        return obj.object_id
    
    def get_object(self, obj_id: UUID) -> Optional[SLIVSObject]:
        """Get object by ID"""
        return self.objects.get(obj_id)
    
    def update_object(self, obj_id: UUID, updates: Dict) -> bool:
        """Update object properties"""
        if obj_id not in self.objects:
            return False
        
        obj = self.objects[obj_id]
        for key, value in updates.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        
        return True
    
    def remove_object(self, obj_id: UUID) -> bool:
        """Remove object from memory"""
        if obj_id in self.objects:
            del self.objects[obj_id]
            return True
        return False
    
    def consolidate_memory(self):
        """Remove low-confidence objects"""
        to_remove = []
        for obj_id, obj in self.objects.items():
            if obj.confidence < self.confidence_threshold:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            self.remove_object(obj_id)
    
    def get_all_objects(self) -> List[SLIVSObject]:
        """Get all objects as list"""
        return list(self.objects.values())
    
    def get_objects_by_confidence(self, min_confidence: float = 0.5) -> List[SLIVSObject]:
        """Get objects above confidence threshold"""
        return [obj for obj in self.objects.values() if obj.confidence >= min_confidence]
    
    def to_dict(self) -> Dict:
        """Serialize memory bank to dictionary"""
        return {
            'frame_count': self.frame_count,
            'object_count': len(self.objects),
            'objects': {str(obj_id): obj.to_dict() for obj_id, obj in self.objects.items()}
        }


# Core Processing Functions

class SLIVSProcessor:
    """
    Main processing pipeline for SLIVS
    Static scene version with depth-informed SAM point selection
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.memory_bank = SLIVSMemoryBank(
            max_objects=self.config['max_objects'],
            confidence_threshold=self.config['confidence_threshold']
        )
        
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'max_objects': 100,
            'confidence_threshold': 0.1,
            'iou_threshold': 0.5,
            'depth_bins': 256,
            'static_assumption': True,
            'depth_layers': 6,  # 5 layers of 50 + 1 layer of 55
            'min_blob_area': 500,  # Minimum pixels for a valid blob
            'max_centroids_per_layer': 20  # Limit centroids per depth layer
        }

    def visualize_memory_state(self, rgb_frame: np.ndarray) -> np.ndarray:
        """Create visualization of current SLIVS understanding"""
        vis_frame = rgb_frame.copy()
        
        for obj in self.memory_bank.get_all_objects():
            if obj.top_segment is not None:
                # Draw object boundary
                contours, _ = cv2.findContours(
                    obj.top_segment.astype(np.uint8), 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Color based on confidence
                confidence_color = int(255 * obj.confidence)
                color = (0, confidence_color, 255 - confidence_color)
                
                cv2.drawContours(vis_frame, contours, -1, color, 2)
                
                # Draw object info
                centroid = obj.get_centroid()
                text = f"ID:{str(obj.object_id)[:4]} C:{obj.confidence:.2f}"
                cv2.putText(vis_frame, text, 
                        (int(centroid[0]), int(centroid[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return vis_frame

    def visualize_depth_layers(self, depth_layers: Dict[int, np.ndarray], 
                            sam_points: List[Tuple[int, int]]) -> np.ndarray:
        """Create visualization of depth layers and SAM points"""
        if not depth_layers:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Get dimensions from first layer
        first_layer = list(depth_layers.values())[0]
        h, w = first_layer.shape
        
        # Create RGB visualization
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Color each depth layer differently
        colors = [
            (255, 0, 0),    # Red - closest
            (255, 127, 0),  # Orange
            (255, 255, 0),  # Yellow
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (127, 0, 255),  # Purple - farthest
        ]
        
        for layer_id, layer_mask in depth_layers.items():
            if layer_id < len(colors):
                color = colors[layer_id]
                # For corrected binary masks: white pixels (255) get colored
                vis[layer_mask == 255] = color
        
        # Draw SAM points
        for x, y in sam_points:
            cv2.circle(vis, (x, y), 3, (255, 255, 255), -1)  # White circles
            cv2.circle(vis, (x, y), 5, (0, 0, 0), 1)        # Black outline
        
        return vis

    def export_memory_state(self, filepath: str) -> bool:
        """Save current memory state to disk"""
        try:
            # Export as JSON for readability
            if filepath.endswith('.json'):
                with open(filepath, 'w') as f:
                    json.dump(self.memory_bank.to_dict(), f, indent=2)
            else:
                # Export as pickle for full functionality
                with open(filepath, 'wb') as f:
                    pickle.dump(self.memory_bank, f)
            
            print(f"Memory state exported to {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to export memory state: {e}")
            return False

    def import_memory_state(self, filepath: str) -> bool:
        """Load memory state from disk"""
        try:
            if filepath.endswith('.json'):
                # Limited JSON import (metadata only)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                print(f"Loaded memory metadata: {data['object_count']} objects, "
                    f"frame {data['frame_count']}")
            else:
                # Full pickle import
                with open(filepath, 'rb') as f:
                    self.memory_bank = pickle.load(f)
                print(f"Memory state imported: {len(self.memory_bank.objects)} objects")
            
            return True
            
        except Exception as e:
            print(f"Failed to import memory state: {e}")
            return False

    def convert_to_ordinal_depth(self, midas_output: np.ndarray) -> np.ndarray:
        """Convert MiDAS relative depth to ordinal ranking (0=closest, 255=farthest)"""
        # Normalize to 0-255 range
        depth_normalized = cv2.normalize(midas_output, None, 0, 255, 
                                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Invert so that 0 = closest, 255 = farthest
        # MiDAS outputs larger values for closer objects, we want the opposite
        ordinal_depth = 255 - depth_normalized
        
        return ordinal_depth

    def calculate_segment_iou(self, seg1: np.ndarray, seg2: np.ndarray) -> float:
        """Calculate Intersection over Union between two segments"""
        if seg1.shape != seg2.shape:
            return 0.0
        
        intersection = np.sum((seg1 > 0) & (seg2 > 0))
        union = np.sum((seg1 > 0) | (seg2 > 0))
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)

    def create_depth_stack(self, segment: np.ndarray, 
                        depth_range: Tuple[int, int]) -> List[np.ndarray]:
        """
        Create "deck of cards" representation for solid object across depth layers
        For static scenes, this creates consistent depth layers
        """
        min_depth, max_depth = depth_range
        depth_layers = []
        
        # Create one layer for each depth value in range
        for depth in range(min_depth, max_depth + 1):
            depth_layers.append(segment.copy())
        
        return depth_layers


    def bootstrap_from_first_frame(self, rgb_frame: np.ndarray, 
                                    segments: Dict[int, np.ndarray],
                                    midas_depth: np.ndarray) -> DepthFrame:
            """
            Process the very first frame to establish initial objects
            
            Args:
                rgb_frame: RGB input frame
                segments: Dictionary of segment_id -> mask from SAM2
                midas_depth: Raw MiDAS depth estimation
                
            Returns:
                DepthFrame with initial objects
            """
            print("Bootstrapping SLIVS from first frame...")
            
            # Convert depth to ordinal ranking
            ordinal_depth = self.convert_to_ordinal_depth(midas_depth)
            
            # Create depth layers for analysis
            depth_layers, sam_points = self.create_depth_layers(ordinal_depth)
            
            # Create depth frame
            depth_frame = DepthFrame(
                frame_number=0,
                midas_output=midas_depth,
                ordinal_depth=ordinal_depth,
                rgb_frame=rgb_frame,
                timestamp=time.time(),
                depth_layers=depth_layers
            )
            
            # Create initial objects from segments
            for seg_id, segment_mask in segments.items():
                if np.sum(segment_mask) < 100:  # Skip tiny segments
                    continue
                    
                # Get depth range for this segment
                depth_range = depth_frame.get_depth_range_for_segment(segment_mask)
                
                # Create SLIVS object
                slivs_obj = SLIVSObject(
                    top_segment=segment_mask.astype(bool),
                    depth_range=depth_range,
                    confidence=1.0,  # High initial confidence
                    last_seen_frame=0,
                    creation_frame=0,
                    is_movable=False,  # Static assumption
                    occlusion_state="visible",
                    category="unknown"
                )
                
                # Add to memory
                obj_id = self.memory_bank.add_object(slivs_obj)
                depth_frame.segments[obj_id] = segment_mask
                
                print(f"Created object {str(obj_id)[:8]} with {slivs_obj.get_area()} pixels, "
                    f"depth range {depth_range}")
            
            self.memory_bank.frame_count = 1
            print(f"Bootstrap complete: {len(self.memory_bank.objects)} objects created")
            
            return depth_frame

    def create_depth_layers(self, ordinal_depth: np.ndarray) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int]]]:
        """
        Break ordinal depth into 6 layers and find blob centroids for SAM points
        
        Each depth layer is a FULL-SIZE binary image where:
        - Black pixels (0) = not in this depth range
        - White pixels (255) = in this depth range
        
        Args:
            ordinal_depth: 0-255 depth array (0=closest, 255=farthest)
            
        Returns:
            Tuple of (depth_layers_dict, sam_points_list)
            depth_layers_dict: {layer_id: full_size_binary_mask}
            sam_points_list: [(x, y), ...] points for SAM
        """
        print("Creating depth layers for SAM point selection...")
        
        # Get full image dimensions
        height, width = ordinal_depth.shape
        print(f"Creating depth layers for {width}x{height} image")
        
        depth_layers = {}
        all_centroids = []
        
        # Create 6 depth layers: 5 layers of 50 range + 1 layer of 55 range
        layer_ranges = [
            (0, 49),    # Layer 0: 0-49 (closest)
            (50, 99),   # Layer 1: 50-99
            (100, 149), # Layer 2: 100-149
            (150, 199), # Layer 3: 150-199
            (200, 249), # Layer 4: 200-249
            (250, 255)  # Layer 5: 250-255 (farthest)
        ]
        
        for layer_id, (min_depth, max_depth) in enumerate(layer_ranges):
            # Create FULL-SIZE binary mask for this depth range
            # All pixels start as black (0)
            layer_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Set pixels in depth range to white (255)
            depth_condition = ((ordinal_depth >= min_depth) & (ordinal_depth <= max_depth))
            layer_mask[depth_condition] = 255
            
            # Store the full-size binary layer
            depth_layers[layer_id] = layer_mask
            
            # Count pixels in this layer
            pixel_count = np.sum(layer_mask == 255)
            print(f"Layer {layer_id} (depth {min_depth}-{max_depth}): {pixel_count} pixels")
            
            # Skip if layer is too small
            if pixel_count < self.config['min_blob_area']:
                print(f"  Skipping layer {layer_id} - too few pixels ({pixel_count} < {self.config['min_blob_area']})")
                continue
            
            # Find connected components (blobs) in this layer
            # Convert to binary for connected components (0 and 1, not 0 and 255)
            binary_mask = (layer_mask == 255).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary_mask)
            
            layer_centroids = []
            for label in range(1, num_labels):  # Skip background (label 0)
                # Create mask for this blob
                blob_mask = (labels == label)
                blob_area = np.sum(blob_mask)
                
                # Skip tiny blobs
                if blob_area < self.config['min_blob_area']:
                    continue
                
                # Calculate centroid
                coords = np.where(blob_mask)
                if len(coords[0]) > 0:
                    y_center = int(np.mean(coords[0]))
                    x_center = int(np.mean(coords[1]))
                    layer_centroids.append((x_center, y_center))
            
            # Limit centroids per layer to avoid overwhelming SAM
            if len(layer_centroids) > self.config['max_centroids_per_layer']:
                # Keep the largest blobs by sorting by area
                blob_areas = []
                for x, y in layer_centroids:
                    # Find which blob this centroid belongs to
                    blob_label = labels[y, x]
                    blob_mask = (labels == blob_label)
                    blob_areas.append(np.sum(blob_mask))
                
                # Sort by area and keep top N
                sorted_indices = np.argsort(blob_areas)[::-1]
                layer_centroids = [layer_centroids[i] for i in sorted_indices[:self.config['max_centroids_per_layer']]]
            
            all_centroids.extend(layer_centroids)
            print(f"  Found {len(layer_centroids)} centroids from {num_labels-1} blobs")
        
        print(f"Total SAM points from depth analysis: {len(all_centroids)}")
        
        # Verify all depth layers are full-size
        for layer_id, layer_mask in depth_layers.items():
            assert layer_mask.shape == (height, width), f"Layer {layer_id} is not full size! Got {layer_mask.shape}, expected {(height, width)}"
            unique_values = np.unique(layer_mask)
            print(f"Layer {layer_id} values: {unique_values} (should be [0, 255] or [0] if empty)")
        
        return depth_layers, all_centroids


class ImprovedSAMPointSelector:
    """
    Enhanced SAM point selection that avoids noisy depth boundaries
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
    
    def _default_config(self) -> Dict:
        return {
            'depth_layers': 6,
            'min_blob_area': 800,  # Increased from 500
            'max_centroids_per_layer': 15,  # Reduced from 20
            'interior_bias_factor': 0.3,  # How far into blob interior to place points
            'rgb_homogeneity_threshold': 0.15,  # RGB variance threshold
            'boundary_avoidance_radius': 8,  # Pixels to avoid near boundaries
            'multi_scale_levels': 3,  # Number of erosion levels to try
            'min_interior_distance': 5,  # Minimum distance from blob edge
            'quality_threshold': 0.6,  # Minimum point quality score
        }
    
    def create_improved_depth_layers(self, ordinal_depth: np.ndarray, 
                                   rgb_frame: np.ndarray) -> Tuple[Dict[int, np.ndarray], List[Tuple[int, int]]]:
        """
        Create depth layers with improved SAM point selection
        """
        print("Creating improved depth layers with boundary-aware SAM points...")
        
        depth_layers = {}
        all_quality_points = []
        
        # Create 6 depth layers
        layer_ranges = [
            (0, 49), (50, 99), (100, 149), 
            (150, 199), (200, 249), (250, 255)
        ]
        
        for layer_id, (min_depth, max_depth) in enumerate(layer_ranges):
            # Create binary mask for this depth range
            layer_mask = ((ordinal_depth >= min_depth) & (ordinal_depth <= max_depth)).astype(np.uint8)
            depth_layers[layer_id] = layer_mask
            
            # Get high-quality points for this layer
            layer_points = self._extract_quality_points_from_layer(
                layer_mask, rgb_frame, layer_id
            )
            
            all_quality_points.extend(layer_points)
            print(f"Layer {layer_id} (depth {min_depth}-{max_depth}): {len(layer_points)} quality points")
        
        # Apply global point filtering
        final_points = self._apply_global_point_filtering(all_quality_points, rgb_frame)
        
        print(f"Final SAM points after quality filtering: {len(final_points)}")
        return depth_layers, final_points
    
    def _extract_quality_points_from_layer(self, layer_mask: np.ndarray, 
                                         rgb_frame: np.ndarray, 
                                         layer_id: int) -> List[Tuple[int, int]]:
        """
        Extract high-quality points from a single depth layer
        """
        if np.sum(layer_mask) < self.config['min_blob_area']:
            return []
        
        # Find connected components
        num_labels, labels = cv2.connectedComponents(layer_mask)
        quality_points = []
        
        for label_id in range(1, num_labels):
            blob_mask = (labels == label_id).astype(np.uint8)
            blob_area = np.sum(blob_mask)
            
            if blob_area < self.config['min_blob_area']:
                continue
            
            # Extract multiple quality points per blob using different strategies
            blob_points = self._extract_points_from_blob(blob_mask, rgb_frame)
            quality_points.extend(blob_points)
        
        # Limit points per layer
        if len(quality_points) > self.config['max_centroids_per_layer']:
            # Sort by quality score and keep the best
            scored_points = [(self._score_point_quality(p, rgb_frame), p) for p in quality_points]
            scored_points.sort(reverse=True, key=lambda x: x[0])
            quality_points = [p[1] for p in scored_points[:self.config['max_centroids_per_layer']]]
        
        return quality_points
    
    def _extract_points_from_blob(self, blob_mask: np.ndarray, 
                                rgb_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract multiple high-quality points from a single blob
        """
        points = []
        
        # Strategy 1: Interior-biased centroid
        interior_point = self._get_interior_biased_centroid(blob_mask)
        if interior_point and self._is_point_high_quality(interior_point, blob_mask, rgb_frame):
            points.append(interior_point)
        
        # Strategy 2: Multi-scale erosion points
        erosion_points = self._get_multi_scale_erosion_points(blob_mask, rgb_frame)
        points.extend(erosion_points)
        
        # Strategy 3: RGB homogeneity-based points
        homogeneity_points = self._get_rgb_homogeneity_points(blob_mask, rgb_frame)
        points.extend(homogeneity_points)
        
        # Remove duplicates and nearby points
        points = self._remove_nearby_points(points, min_distance=10)
        
        return points[:3]  # Max 3 points per blob
    
    def _get_interior_biased_centroid(self, blob_mask: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Get centroid biased towards blob interior using distance transform
        """
        # Compute distance transform to find interior
        distance_transform = distance_transform_edt(blob_mask)
        
        if np.max(distance_transform) < self.config['min_interior_distance']:
            return None
        
        # Find the point that's furthest from edges (most interior)
        max_dist_coords = np.unravel_index(np.argmax(distance_transform), distance_transform.shape)
        y, x = max_dist_coords
        
        return (int(x), int(y))
    
    def _get_multi_scale_erosion_points(self, blob_mask: np.ndarray, 
                                      rgb_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Use multi-scale erosion to find stable interior points
        """
        points = []
        
        for erosion_level in range(1, self.config['multi_scale_levels'] + 1):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_level*2+1, erosion_level*2+1))
            eroded = cv2.erode(blob_mask, kernel, iterations=1)
            
            if np.sum(eroded) < 50:  # Too small after erosion
                break
            
            # Find centroid of eroded blob
            coords = np.where(eroded > 0)
            if len(coords[0]) > 0:
                y_center = int(np.mean(coords[0]))
                x_center = int(np.mean(coords[1]))
                point = (x_center, y_center)
                
                if self._is_point_high_quality(point, blob_mask, rgb_frame):
                    points.append(point)
        
        return points
    
    def _get_rgb_homogeneity_points(self, blob_mask: np.ndarray, 
                                  rgb_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find points in regions with high RGB homogeneity
        """
        points = []
        
        # Extract RGB values within the blob
        blob_coords = np.where(blob_mask > 0)
        if len(blob_coords[0]) < 20:
            return points
        
        # Sample points within the blob
        sample_indices = np.random.choice(len(blob_coords[0]), 
                                        size=min(50, len(blob_coords[0])), 
                                        replace=False)
        
        for idx in sample_indices:
            y, x = blob_coords[0][idx], blob_coords[1][idx]
            
            # Check RGB homogeneity in local neighborhood
            if self._check_rgb_homogeneity((x, y), rgb_frame, radius=5):
                # Check distance from blob boundary
                if self._get_distance_from_boundary((x, y), blob_mask) >= self.config['min_interior_distance']:
                    points.append((x, y))
        
        return points[:2]  # Max 2 homogeneity points per blob
    
    def _check_rgb_homogeneity(self, point: Tuple[int, int], 
                             rgb_frame: np.ndarray, 
                             radius: int = 5) -> bool:
        """
        Check if local region around point has homogeneous RGB values
        """
        x, y = point
        h, w = rgb_frame.shape[:2]
        
        # Define neighborhood
        y1, y2 = max(0, y-radius), min(h, y+radius+1)
        x1, x2 = max(0, x-radius), min(w, x+radius+1)
        
        if y2 - y1 < 3 or x2 - x1 < 3:
            return False
        
        # Extract neighborhood RGB values
        neighborhood = rgb_frame[y1:y2, x1:x2]
        
        # Calculate RGB standard deviation
        rgb_std = np.std(neighborhood.reshape(-1, 3), axis=0)
        mean_std = np.mean(rgb_std) / 255.0  # Normalize
        
        return mean_std < self.config['rgb_homogeneity_threshold']
    
    def _get_distance_from_boundary(self, point: Tuple[int, int], 
                                  blob_mask: np.ndarray) -> float:
        """
        Get distance from point to blob boundary
        """
        x, y = point
        if not blob_mask[y, x]:
            return 0.0
        
        # Compute distance transform
        distance_transform = distance_transform_edt(blob_mask)
        return distance_transform[y, x]
    
    def _is_point_high_quality(self, point: Tuple[int, int], 
                             blob_mask: np.ndarray, 
                             rgb_frame: np.ndarray) -> bool:
        """
        Comprehensive quality check for a point
        """
        x, y = point
        h, w = rgb_frame.shape[:2]
        
        # Basic bounds check
        if not (0 <= x < w and 0 <= y < h):
            return False
        
        # Must be inside the blob
        if not blob_mask[y, x]:
            return False
        
        # Check distance from boundary
        boundary_distance = self._get_distance_from_boundary(point, blob_mask)
        if boundary_distance < self.config['min_interior_distance']:
            return False
        
        # Check RGB homogeneity
        if not self._check_rgb_homogeneity(point, rgb_frame):
            return False
        
        return True
    
    def _score_point_quality(self, point: Tuple[int, int], 
                           rgb_frame: np.ndarray) -> float:
        """
        Calculate quality score for a point (0.0 to 1.0)
        """
        x, y = point
        score = 0.0
        
        # RGB homogeneity score (0.0 to 0.4)
        homogeneity_score = self._calculate_homogeneity_score(point, rgb_frame)
        score += homogeneity_score * 0.4
        
        # Gradient magnitude score (lower is better for SAM)
        gradient_score = self._calculate_gradient_score(point, rgb_frame)
        score += (1.0 - gradient_score) * 0.3
        
        # Edge distance score (further from edges is better)
        edge_score = self._calculate_edge_distance_score(point, rgb_frame)
        score += edge_score * 0.3
        
        return score
    
    def _calculate_homogeneity_score(self, point: Tuple[int, int], 
                                   rgb_frame: np.ndarray) -> float:
        """Calculate RGB homogeneity score (1.0 = very homogeneous)"""
        x, y = point
        h, w = rgb_frame.shape[:2]
        
        radius = 7
        y1, y2 = max(0, y-radius), min(h, y+radius+1)
        x1, x2 = max(0, x-radius), min(w, x+radius+1)
        
        if y2 - y1 < 3 or x2 - x1 < 3:
            return 0.0
        
        neighborhood = rgb_frame[y1:y2, x1:x2]
        rgb_std = np.std(neighborhood.reshape(-1, 3), axis=0)
        mean_std = np.mean(rgb_std) / 255.0
        
        return max(0.0, 1.0 - mean_std * 5.0)  # Scale for scoring
    
    def _calculate_gradient_score(self, point: Tuple[int, int], 
                                rgb_frame: np.ndarray) -> float:
        """Calculate gradient magnitude score (0.0 = low gradient, good for SAM)"""
        x, y = point
        
        # Convert to grayscale for gradient calculation
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x[y, x]**2 + grad_y[y, x]**2)
        
        # Normalize to 0-1 range
        return min(1.0, gradient_magnitude / 100.0)
    
    def _calculate_edge_distance_score(self, point: Tuple[int, int], 
                                     rgb_frame: np.ndarray) -> float:
        """Calculate distance from image edges score"""
        x, y = point
        h, w = rgb_frame.shape[:2]
        
        # Distance to nearest edge
        edge_distances = [x, y, w-x-1, h-y-1]
        min_edge_distance = min(edge_distances)
        
        # Normalize by image size
        max_possible_distance = min(w, h) / 2
        return min(1.0, min_edge_distance / max_possible_distance)
    
    def _apply_global_point_filtering(self, points: List[Tuple[int, int]], 
                                    rgb_frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Apply global filtering to remove low-quality points
        """
        if not points:
            return points
        
        # Score all points
        scored_points = []
        for point in points:
            quality_score = self._score_point_quality(point, rgb_frame)
            if quality_score >= self.config['quality_threshold']:
                scored_points.append((quality_score, point))
        
        # Sort by quality
        scored_points.sort(reverse=True, key=lambda x: x[0])
        
        # Apply spatial clustering to avoid clustering
        filtered_points = self._apply_spatial_clustering([p[1] for p in scored_points])
        
        print(f"Global filtering: {len(points)} -> {len(filtered_points)} points")
        return filtered_points
    
    def _apply_spatial_clustering(self, points: List[Tuple[int, int]], 
                                min_distance: int = 15) -> List[Tuple[int, int]]:
        """
        Use DBSCAN clustering to avoid point clustering
        """
        if len(points) <= 1:
            return points
        
        # Convert to numpy array
        points_array = np.array(points)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=min_distance, min_samples=1).fit(points_array)
        
        # Keep one point per cluster (the first one)
        unique_points = []
        for cluster_id in set(clustering.labels_):
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            # Take the first point in each cluster
            unique_points.append(tuple(points_array[cluster_indices[0]]))
        
        return unique_points
    
    def _remove_nearby_points(self, points: List[Tuple[int, int]], 
                            min_distance: int = 10) -> List[Tuple[int, int]]:
        """
        Remove points that are too close to each other
        """
        if len(points) <= 1:
            return points
        
        filtered_points = [points[0]]  # Keep first point
        
        for point in points[1:]:
            too_close = False
            for existing_point in filtered_points:
                distance = np.sqrt((point[0] - existing_point[0])**2 + 
                                 (point[1] - existing_point[1])**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                filtered_points.append(point)
        
        return filtered_points
    
    def visualize_point_quality(self, rgb_frame: np.ndarray, 
                              points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Create visualization showing point quality scores
        """
        vis = rgb_frame.copy()
        
        for point in points:
            x, y = point
            quality_score = self._score_point_quality(point, rgb_frame)
            
            # Color based on quality (green = high, red = low)
            color_intensity = int(255 * quality_score)
            color = (255 - color_intensity, color_intensity, 0)
            
            # Draw point
            cv2.circle(vis, (x, y), 4, color, -1)
            cv2.circle(vis, (x, y), 6, (255, 255, 255), 1)
            
            # Add quality score text
            cv2.putText(vis, f"{quality_score:.2f}", 
                       (x + 8, y - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, 
                       (255, 255, 255), 1)
        
        return vis



# Utility Functions

def calculate_centroid(segment: np.ndarray) -> Tuple[float, float]:
    """Calculate centroid of a segment"""
    coords = np.where(segment)
    if len(coords[0]) == 0:
        return (0.0, 0.0)
    
    y_center = float(np.mean(coords[0]))
    x_center = float(np.mean(coords[1]))
    return (x_center, y_center)


def load_slivs_config(config_path: str) -> Dict:
    """Load system configuration from file"""
    try:
        with open(config_path, 'r') as f:
            if config_path.endswith('.json'):
                return json.load(f)
            else:
                return pickle.load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return {}


# Test and Validation Functions

def test_data_structures():
    """Test basic data structure functionality"""
    print("Testing SLIVS data structures...")
    
    # Test SLIVSObject
    test_segment = np.zeros((100, 100), dtype=bool)
    test_segment[20:80, 30:70] = True
    
    obj = SLIVSObject(
        top_segment=test_segment,
        depth_range=(50, 150),
        confidence=0.85,
        category="test_object"
    )
    
    print(f"Object ID: {obj.object_id}")
    print(f"Centroid: {obj.get_centroid()}")
    print(f"Area: {obj.get_area()}")
    print(f"Bounding box: {obj.bounding_box}")
    
    # Test DepthFrame
    test_depth = np.random.rand(100, 100) * 255
    depth_frame = DepthFrame(
        frame_number=1,
        midas_output=test_depth,
        ordinal_depth=test_depth.astype(np.uint8)
    )
    
    depth_range = depth_frame.get_depth_range_for_segment(test_segment)
    print(f"Depth range for segment: {depth_range}")
    
    # Test SLIVSMemoryBank
    memory = SLIVSMemoryBank()
    obj_id = memory.add_object(obj)
    print(f"Added object to memory: {obj_id}")
    print(f"Memory contains {len(memory.objects)} objects")
    
    # Test serialization
    obj_dict = obj.to_dict()
    memory_dict = memory.to_dict()
    print("Serialization successful")
    
    print("All data structure tests passed!")

def test_depth_layer_creation():
    """
    Test depth layer creation with corrected full-size binary masks
    """
    print("\nTesting CORRECTED depth layer creation...")
    
    # Create test depth data with clear depth regions
    height, width = 200, 300
    test_depth = np.zeros((height, width), dtype=np.uint8)
    
    # Fill background with far depth (230)
    test_depth.fill(230)
    
    # Add distinct objects at different depth ranges
    # Object 1: depth range 20-35 (should be in Layer 0: 0-49)
    test_depth[50:70, 50:100] = 25
    
    # Object 2: depth range 70-85 (should be in Layer 1: 50-99)  
    test_depth[100:130, 150:200] = 75
    
    # Object 3: depth range 120-135 (should be in Layer 2: 100-149)
    test_depth[80:120, 200:250] = 125
    
    # Object 4: depth range 170-185 (should be in Layer 3: 150-199)
    test_depth[30:60, 180:220] = 175
    
    # Object 5: depth range 220-235 (should be in Layer 4: 200-249)
    test_depth[140:180, 100:150] = 225
    
    # Object 6: depth range 252-255 (should be in Layer 5: 250-255)
    test_depth[160:190, 250:290] = 253
    
    print(f"Test depth range: {test_depth.min()} - {test_depth.max()}")
    print(f"Test image size: {height}x{width}")
    
    # Test with processor
    processor = SLIVSProcessor()
    depth_layers, sam_points = processor.create_depth_layers(test_depth)
    
    print(f"\nCreated {len(depth_layers)} depth layers")
    print(f"Generated {len(sam_points)} SAM points")
    
    # Verify each layer is full-size and binary
    for layer_id, layer_mask in depth_layers.items():
        print(f"\nLayer {layer_id}:")
        print(f"  Shape: {layer_mask.shape} (should be {(height, width)})")
        unique_vals = np.unique(layer_mask)
        print(f"  Unique values: {unique_vals} (should be [0, 255] or [0])")
        white_pixels = np.sum(layer_mask == 255)
        print(f"  White pixels: {white_pixels}")
        
        # Verify it's full size
        assert layer_mask.shape == (height, width), f"Layer {layer_id} wrong size!"
        
        # Verify it's binary (only 0 and 255)
        assert set(unique_vals).issubset({0, 255}), f"Layer {layer_id} not binary! Has: {unique_vals}"
    
    # Save individual layers to verify they're correct
    print(f"\nSaving individual layer masks...")
    for layer_id, layer_mask in depth_layers.items():
        filename = f"corrected_layer_{layer_id}_mask.png"
        cv2.imwrite(filename, layer_mask)  # layer_mask is already 0-255
        print(f"  Saved {filename}")
    
    # Save original depth for reference
    cv2.imwrite("corrected_original_depth.png", test_depth)
    print(f"  Saved corrected_original_depth.png")
    
    # Create composite visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Corrected Depth Layers - Full Size Binary Masks', fontsize=14)
    
    # Original depth
    axes[0, 0].imshow(test_depth, cmap='viridis')
    axes[0, 0].set_title('Original Depth')
    axes[0, 0].axis('off')
    
    # Individual layers
    for i, (layer_id, layer_mask) in enumerate(depth_layers.items()):
        if i < 6:  # We have 6 layers
            row = i // 3
            col = (i % 3) + 1
            if row < 2 and col < 4:
                axes[row, col].imshow(layer_mask, cmap='gray', vmin=0, vmax=255)
                axes[row, col].set_title(f'Layer {layer_id}')
                axes[row, col].axis('off')
    
    # Combined visualization  
    if len(depth_layers) >= 6:
        axes[1, 3].text(0.5, 0.5, 'All layers are\nfull-size binary\nmasks with\n0=black, 255=white', 
                       ha='center', va='center', fontsize=10)
        axes[1, 3].set_title('Layer Properties')
        axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('corrected_depth_layers_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved corrected_depth_layers_test.png")
    print("✓ Corrected depth layer test passed!")
    print("\nKey corrections made:")
    print("- Each layer is now full image size")
    print("- Black pixels (0) = not in depth range")  
    print("- White pixels (255) = in depth range")
    print("- No cropping or rectangular regions")
    print("- Binary masks suitable for connected components analysis")



def visualize_depth_layers(self, depth_layers: Dict[int, np.ndarray], 
                          sam_points: List[Tuple[int, int]]) -> np.ndarray:
    """
    Create visualization of depth layers and SAM points
    Now properly handles full-size binary masks
    """
    if not depth_layers:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Get dimensions from first layer (all should be same size)
    first_layer = list(depth_layers.values())[0]
    h, w = first_layer.shape
    
    # Create RGB visualization
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color each depth layer differently
    colors = [
        (255, 0, 0),    # Red - closest (Layer 0)
        (255, 127, 0),  # Orange (Layer 1)
        (255, 255, 0),  # Yellow (Layer 2)
        (0, 255, 0),    # Green (Layer 3)
        (0, 0, 255),    # Blue (Layer 4)
        (127, 0, 255),  # Purple - farthest (Layer 5)
    ]
    
    # Apply colors to each layer
    for layer_id, layer_mask in depth_layers.items():
        if layer_id < len(colors):
            color = colors[layer_id]
            # layer_mask is full-size binary: 0=black, 255=white
            # Apply color where mask is white (255)
            color_mask = (layer_mask == 255)
            vis[color_mask] = color
    
    # Draw SAM points
    for x, y in sam_points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 3, (255, 255, 255), -1)  # White circles
            cv2.circle(vis, (x, y), 5, (0, 0, 0), 1)        # Black outline
    
    return vis

def save_depth_analysis(original_depth, depth_layers):
    """Create depth analysis visualization showing histogram and layer distribution"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original depth map
    im1 = axes[0, 0].imshow(original_depth, cmap='viridis')
    axes[0, 0].set_title('Original Depth Map')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Depth histogram
    axes[0, 1].hist(original_depth.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_title('Depth Histogram')
    axes[0, 1].set_xlabel('Depth Value')
    axes[0, 1].set_ylabel('Pixel Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add layer boundaries to histogram
    layer_ranges = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 249), (250, 255)]
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i, (min_d, max_d) in enumerate(layer_ranges):
        if i < len(colors):
            axes[0, 1].axvspan(min_d, max_d, alpha=0.2, color=colors[i], 
                              label=f'Layer {i}')
    axes[0, 1].legend()
    
    # Layer pixel counts
    layer_counts = []
    layer_ids = []
    for layer_id, layer_mask in depth_layers.items():
        layer_counts.append(np.sum(layer_mask))
        layer_ids.append(f'Layer {layer_id}')
    
    bars = axes[1, 0].bar(layer_ids, layer_counts, color=colors[:len(layer_counts)])
    axes[1, 0].set_title('Pixels per Depth Layer')
    axes[1, 0].set_ylabel('Pixel Count')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, layer_counts):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(layer_counts)*0.01,
                       f'{count}', ha='center', va='bottom')
    
    # All layers combined
    combined_mask = np.zeros_like(original_depth)
    for layer_id, layer_mask in depth_layers.items():
        combined_mask[layer_mask > 0] = layer_id + 1  # +1 so layer 0 isn't black
    
    im2 = axes[1, 1].imshow(combined_mask, cmap='tab10', vmin=0, vmax=6)
    axes[1, 1].set_title('All Layers Combined')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], ticks=range(7), 
                label='Layer ID (0=background)')
    
    plt.tight_layout()
    plt.savefig('depth_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved depth_analysis.png")

def save_depth_layers_composite(original_depth, depth_layers, sam_points):
    """Create and save a composite visualization of all depth layers"""
    import matplotlib.pyplot as plt
    
    # Calculate grid dimensions
    num_layers = len(depth_layers)
    cols = 3
    rows = (num_layers + 2 + cols - 1) // cols  # +2 for original and combined
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    # Plot original depth
    axes[0].imshow(original_depth, cmap='viridis')
    axes[0].set_title('Original Test Depth')
    axes[0].axis('off')
    
    # Plot each depth layer
    for i, (layer_id, layer_mask) in enumerate(depth_layers.items()):
        ax_idx = i + 1
        if ax_idx < len(axes):
            axes[ax_idx].imshow(layer_mask, cmap='gray')
            axes[ax_idx].set_title(f'Layer {layer_id}')
            axes[ax_idx].axis('off')
    
    # Plot combined visualization with SAM points
    if len(axes) > num_layers + 1:
        combined_vis = create_colored_depth_visualization(depth_layers, sam_points)
        axes[num_layers + 1].imshow(combined_vis)
        axes[num_layers + 1].set_title(f'Combined Layers + SAM Points ({len(sam_points)})')
        axes[num_layers + 1].axis('off')
    
    # Hide unused subplots
    for i in range(num_layers + 2, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_layers_composite.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved depth_layers_composite.png")


def save_colored_depth_layers(depth_layers, sam_points):
    """Create and save a colored overlay visualization of depth layers"""
    if not depth_layers:
        return
    
    # Get dimensions from first layer
    first_layer = list(depth_layers.values())[0]
    h, w = first_layer.shape
    
    # Create RGB visualization
    colored_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color each depth layer differently
    colors = [
        (255, 0, 0),    # Red - closest (Layer 0)
        (255, 127, 0),  # Orange (Layer 1)
        (255, 255, 0),  # Yellow (Layer 2)
        (0, 255, 0),    # Green (Layer 3)
        (0, 0, 255),    # Blue (Layer 4)
        (127, 0, 255),  # Purple - farthest (Layer 5)
    ]
    
    for layer_id, layer_mask in depth_layers.items():
        if layer_id < len(colors):
            color = colors[layer_id]
            colored_vis[layer_mask > 0] = color
    
    # Draw SAM points
    for x, y in sam_points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(colored_vis, (x, y), 3, (255, 255, 255), -1)  # White circles
            cv2.circle(colored_vis, (x, y), 5, (0, 0, 0), 1)        # Black outline
    
    # Save the colored visualization
    cv2.imwrite('depth_layers_colored.png', cv2.cvtColor(colored_vis, cv2.COLOR_RGB2BGR))
    print("Saved depth_layers_colored.png")


def create_colored_depth_visualization(depth_layers, sam_points):
    """Create colored visualization for matplotlib display"""
    if not depth_layers:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Get dimensions from first layer
    first_layer = list(depth_layers.values())[0]
    h, w = first_layer.shape
    
    # Create RGB visualization
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color each depth layer differently
    colors = [
        (255, 0, 0),    # Red - closest
        (255, 127, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (127, 0, 255),  # Purple - farthest
    ]
    
    for layer_id, layer_mask in depth_layers.items():
        if layer_id < len(colors):
            color = colors[layer_id]
            vis[layer_mask > 0] = color
    
    # Draw SAM points
    for x, y in sam_points:
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 3, (255, 255, 255), -1)  # White circles
            cv2.circle(vis, (x, y), 5, (0, 0, 0), 1)        # Black outline
    
    return vis

if __name__ == "__main__":
    print("SLIVS Core Data Structures - Static Scene Foundation")
    print("=" * 60)
    
    # Run basic tests
    test_data_structures()
    test_depth_layer_creation()
    
    print("\nSLIVS pipeline ready for integration with:")
    print("- SAM2 segmentation with depth-informed point selection")
    print("- MiDAS depth estimation")
    print("- Static scene understanding")
    print("- Depth layer analysis for intelligent SAM points")