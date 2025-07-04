#!/usr/bin/env python3
"""
Enhanced SLIVS Segmentation with Depth-Aware Pixel Masking - CORRECTED VERSION
This module enhances the existing segmentation by creating depth-constrained images
for SAM2 processing, ensuring objects are only segmented within compatible depth layers.

CORRECTED: Uses LayerResult from slivs_depth_core instead of DepthLayer
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# Import existing SLIVS modules with correct data structures
from slivs_depth_core import SLIVSDepthProcessor, DepthProcessingResult, LayerResult
from slivs_segment_core import SLIVSSam2Processor


class SLIVSDepthVisualizer:
    """
    Depth visualization component extracted from core demo for integration
    """
    
    def __init__(self, panel_width: int = 320, panel_height: int = 240):
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.point_color = (0, 255, 0)  # Green
        self.point_size = 3
        self.font_scale = 0.6
        self.font_thickness = 1
    
    def draw_points_on_layer(self, layer_mask: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw points on a layer mask."""
        layer_colored = cv2.cvtColor(layer_mask, cv2.COLOR_GRAY2BGR)
        for point in points:
            cv2.circle(layer_colored, point, self.point_size, self.point_color, -1)
            # Add small black outline for visibility
            cv2.circle(layer_colored, point, self.point_size + 1, (0, 0, 0), 1)
        return layer_colored
    
    def create_depth_composite_display(self, result: DepthProcessingResult) -> np.ndarray:
        """Create a composite display of all depth layers with points."""
        total_panels = len(result.layers) + 1  # +1 for full depth map
        
        # Calculate optimal layout
        best_aspect_ratio = float('inf')
        best_rows, best_cols = 1, total_panels
        
        for rows in range(1, total_panels + 1):
            cols = math.ceil(total_panels / rows)
            aspect_ratio = abs((cols * self.panel_width) / (rows * self.panel_height) - 16/9)
            if aspect_ratio < best_aspect_ratio:
                best_aspect_ratio = aspect_ratio
                best_rows, best_cols = rows, cols
        
        composite_height = self.panel_height * best_rows
        composite_width = self.panel_width * best_cols
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Place full depth map in first panel
        depth_resized = cv2.resize(result.full_depth_map, (self.panel_width, self.panel_height))
        depth_colored = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
        composite[0:self.panel_height, 0:self.panel_width] = depth_colored
        
        # Add all points from all layers to full depth view
        total_points = len(result.all_points)
        scale_x = self.panel_width / result.full_depth_map.shape[1]
        scale_y = self.panel_height / result.full_depth_map.shape[0]
        
        for point in result.all_points:
            scaled_point = (int(point[0] * scale_x), int(point[1] * scale_y))
            cv2.circle(composite, scaled_point, self.point_size, self.point_color, -1)
            cv2.circle(composite, scaled_point, self.point_size + 1, (0, 0, 0), 1)
        
        cv2.putText(composite, f"Full Depth ({total_points}pts)", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Place individual layers
        panel_idx = 1
        for layer in result.layers:
            row = panel_idx // best_cols
            col = panel_idx % best_cols
            y = row * self.panel_height
            x = col * self.panel_width
            
            # Draw layer with points
            layer_with_points = self.draw_points_on_layer(layer.layer_mask, layer.points)
            layer_resized = cv2.resize(layer_with_points, (self.panel_width, self.panel_height))
            
            composite[y:y+self.panel_height, x:x+self.panel_width] = layer_resized
            
            # Add labels with point count
            label = f"{layer.config.label} ({len(layer.points)}pts)"
            cv2.putText(composite, label, (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness)
            
            panel_idx += 1
        
        return composite


class DepthAwareSAM2Processor:
    """
    Enhanced SAM2 processor that uses depth layer information to create
    depth-constrained images for more accurate segmentation.
    
    CORRECTED: Uses LayerResult from slivs_depth_core
    """
    
    def __init__(self, sam_processor: SLIVSSam2Processor, num_depth_layers: int = 7):
        """
        Initialize the depth-aware SAM2 processor.
        
        Args:
            sam_processor: Existing SAM2 processor instance
            num_depth_layers: Number of depth layers in the system
        """
        self.sam_processor = sam_processor
        self.num_depth_layers = num_depth_layers
        self.current_original_image = None
        self.current_depth_layers = None
        
    def set_frame_data(self, original_image: np.ndarray, depth_result: DepthProcessingResult):
        """
        Set the current frame data for processing.
        
        Args:
            original_image: Original RGB frame
            depth_result: Depth processing result with all layers (contains LayerResult objects)
        """
        self.current_original_image = original_image.copy()
        self.current_depth_layers = depth_result.layers  # List[LayerResult]
        
    def create_depth_constrained_image(self, target_layer_index: int) -> np.ndarray:
        """
        Create a depth-constrained version of the original image for a specific layer.
        
        For layer N, we include pixels from layers:
        - Layer 0 (closest): layers 0, 1
        - Layer 1: layers 0, 1, 2  
        - Layer 2: layers 1, 2, 3
        - Layer N: layers N-1, N, N+1 (clamped to valid range)
        - Last layer: layers N-1, N
        
        Args:
            target_layer_index: Index of the target depth layer (0-based)
            
        Returns:
            Depth-constrained image where incompatible pixels are blacked out
        """
        if self.current_original_image is None or self.current_depth_layers is None:
            raise ValueError("Frame data not set. Call set_frame_data() first.")
        
        # Create a copy of the original image
        constrained_image = self.current_original_image.copy()
        
        # Determine which depth layers to include
        if target_layer_index == 0:
            # First layer: include layers 0 and 1
            included_layers = [0, 1]
        elif target_layer_index == self.num_depth_layers - 1:
            # Last layer: include layers N-1 and N
            included_layers = [target_layer_index - 1, target_layer_index]
        else:
            # Middle layers: include previous, current, and next
            included_layers = [target_layer_index - 1, target_layer_index, target_layer_index + 1]
        
        # Clamp to valid layer indices
        included_layers = [i for i in included_layers if 0 <= i < len(self.current_depth_layers)]
        
        print(f"    Creating depth-constrained image for layer {target_layer_index}")
        print(f"    Including depth layers: {included_layers}")
        
        # Create combined mask of all included layers
        combined_mask = np.zeros(constrained_image.shape[:2], dtype=np.uint8)
        
        for layer_idx in included_layers:
            if layer_idx < len(self.current_depth_layers):
                # Use LayerResult.layer_mask
                layer_result = self.current_depth_layers[layer_idx]
                layer_mask = layer_result.layer_mask
                combined_mask = cv2.bitwise_or(combined_mask, layer_mask)
        
        # Black out pixels that are not in any included layer
        # Create 3-channel mask for RGB image
        mask_3channel = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask: keep pixels where mask is white (255), black out where mask is black (0)
        constrained_image = cv2.bitwise_and(constrained_image, mask_3channel)
        
        # Count how many pixels were masked out
        total_pixels = constrained_image.shape[0] * constrained_image.shape[1]
        included_pixels = np.sum(combined_mask > 0)
        masked_pixels = total_pixels - included_pixels
        
        print(f"    Masked out {masked_pixels}/{total_pixels} pixels ({masked_pixels/total_pixels*100:.1f}%)")
        
        return constrained_image
    
    def segment_points_with_depth_constraint(self, points: List[Tuple[int, int]], 
                                           target_layer_index: int) -> Optional[object]:
        """
        Segment points using depth-constrained image.
        
        Args:
            points: List of points to segment
            target_layer_index: Index of the depth layer these points belong to
            
        Returns:
            SAM2 segmentation result or None if failed
        """
        if not points:
            return None
        
        try:
            # Create depth-constrained image
            constrained_image = self.create_depth_constrained_image(target_layer_index)
            
            # Set the constrained image in SAM2
            self.sam_processor.set_image(constrained_image)
            
            # Perform segmentation
            if len(points) == 1:
                result = self.sam_processor.segment_from_point(points[0])
            else:
                result = self.sam_processor.segment_from_points(points)
            
            return result
            
        except Exception as e:
            print(f"    Error in depth-constrained segmentation: {e}")
            return None
    
    def visualize_depth_constraint(self, target_layer_index: int, 
                                 points: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        Create a visualization showing the depth constraint for a specific layer.
        
        Args:
            target_layer_index: Index of the target depth layer
            points: Optional points to highlight on the visualization
            
        Returns:
            Visualization image showing original, constrained, and difference
        """
        if self.current_original_image is None:
            raise ValueError("Frame data not set. Call set_frame_data() first.")
        
        # Get constrained image
        constrained_image = self.create_depth_constrained_image(target_layer_index)
        
        # Create difference image (what was masked out)
        difference = cv2.absdiff(self.current_original_image, constrained_image)
        
        # Create side-by-side visualization
        height, width = self.current_original_image.shape[:2]
        viz_width = width * 3
        visualization = np.zeros((height, viz_width, 3), dtype=np.uint8)
        
        # Place images side by side
        visualization[:, 0:width] = self.current_original_image
        visualization[:, width:2*width] = constrained_image
        visualization[:, 2*width:3*width] = difference
        
        # Add points if provided
        if points:
            point_color = (0, 255, 255)  # Yellow
            point_size = 4
            
            for point in points:
                # Draw on original image
                cv2.circle(visualization, point, point_size, point_color, -1)
                cv2.circle(visualization, point, point_size + 1, (0, 0, 0), 1)
                
                # Draw on constrained image
                constrained_point = (point[0] + width, point[1])
                cv2.circle(visualization, constrained_point, point_size, point_color, -1)
                cv2.circle(visualization, constrained_point, point_size + 1, (0, 0, 0), 1)
        
        # Add labels
        cv2.putText(visualization, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(visualization, f"Depth Layer {target_layer_index}", (width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(visualization, "Masked Pixels", (2*width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return visualization


# Integration function to update existing demo
def update_existing_demo_with_depth_awareness(existing_demo):
    """
    Helper function to update an existing SLIVSSingleFrameDemo instance
    with depth-aware segmentation capabilities.
    """
    # Create depth-aware SAM processor
    depth_aware_sam = DepthAwareSAM2Processor(
        sam_processor=existing_demo.sam_processor,
        num_depth_layers=7  # Adjust based on your configuration
    )
    
    # Replace the existing segment processing method
    existing_demo.depth_aware_sam = depth_aware_sam
    
    print("✓ Updated existing demo with depth-aware segmentation capabilities")
    
    return existing_demo


def add_depth_constraint_method_to_existing_demo():
    """
    Add this method to your existing SLIVSSingleFrameDemo class:
    """
    def process_depth_layer_with_constraints(self, layer_points, layer_index, layer_name, frame, depth_result):
        """Enhanced processing method to add to existing demo - CORRECTED VERSION"""
        print(f"  Processing layer '{layer_name}' with depth constraints...")
        
        if not hasattr(self, 'depth_aware_sam'):
            # Initialize if not already done
            self.depth_aware_sam = DepthAwareSAM2Processor(self.sam_processor, 7)
            self.depth_aware_sam.set_frame_data(frame, depth_result)
        
        objects = []
        processed_points = set()
        
        for i, point in enumerate(layer_points):
            if point in processed_points:
                continue
                
            # Use depth-constrained segmentation
            segment_result = self.depth_aware_sam.segment_points_with_depth_constraint([point], layer_index)
            
            if segment_result and hasattr(segment_result, 'confidence'):
                if segment_result.confidence >= self.debug_confidence_threshold:
                    # Continue with existing object creation logic
                    largest_blob = self.get_largest_blob(segment_result.mask)
                    if np.any(largest_blob):
                        # Create object using existing logic...
                        object_points = [point]
                        processed_points.add(point)
                        
                        # Group nearby points (existing logic)
                        for other_point in layer_points:
                            if (other_point not in processed_points and 
                                self.point_in_mask(other_point, largest_blob)):
                                object_points.append(other_point)
                                processed_points.add(other_point)
                        
                        # Create SLIVSObject (existing logic)
                        from dataclasses import dataclass
                        
                        @dataclass
                        class SLIVSObject:
                            object_id: str
                            points: List[Tuple[int, int]]
                            segment: np.ndarray
                            depth_layer: str
                            confidence: float
                            area: int
                            centroid: Tuple[float, float]
                        
                        obj = SLIVSObject(
                            object_id=f"{layer_name}_{len(objects)}",
                            points=object_points,
                            segment=largest_blob,
                            depth_layer=layer_name,
                            confidence=segment_result.confidence,
                            area=int(np.sum(largest_blob)),
                            centroid=self.calculate_centroid(largest_blob)
                        )
                        objects.append(obj)
            
            processed_points.add(point)
        
        return objects

    print("See function above for integration into existing demo")


if __name__ == "__main__":
    print("This is the corrected depth-aware segmentation module.")
    print("Use this instead of the previous version to work with LayerResult.")
    print("Import this module in your main demo file.")