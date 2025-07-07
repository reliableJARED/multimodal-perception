#!/usr/bin/env python3
"""
Cross-Depth Layer Segmentation Helper Functions
Handles merging segments that span multiple depth layers and grouping points appropriately.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass


@dataclass
class CrossDepthSegment:
    """Represents a segment that may span multiple depth layers"""
    segment_mask: np.ndarray
    confidence: float
    depth_layers: List[str]  # Which depth layers this segment spans
    all_points: List[Tuple[int, int]]  # All points from all layers
    points_by_layer: Dict[str, List[Tuple[int, int]]]  # Points organized by layer
    primary_layer: str  # The layer where segmentation was initiated
    bounding_box: Tuple[int, int, int, int]
    area: int


class CrossDepthSegmentationProcessor:
    """Handles cross-depth layer segmentation logic"""
    
    def __init__(self, sam_processor, neon_masked_frames):
        self.sam_processor = sam_processor
        self.neon_masked_frames = neon_masked_frames
        self.processed_points = set()  # Track points that have been used
        
    def check_points_in_segment(self, segment_mask: np.ndarray, points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Check which points fall inside a given segment mask"""
        points_inside = []
        for point in points:
            x, y = point
            if segment_mask[y, x]:  # Point is inside segment
                points_inside.append(point)
        return points_inside
    
    def get_adjacent_layers(self, current_layer_idx: int, total_layers: int, look_ahead: int = 2) -> List[int]:
        """Get indices of adjacent layers to check for cross-layer segments"""
        adjacent = []
        # Look ahead to next layers (objects can extend into deeper layers)
        for i in range(1, look_ahead + 1):
            next_idx = current_layer_idx + i
            if next_idx < total_layers:
                adjacent.append(next_idx)
        return adjacent
    
    def merge_cross_layer_segments(self, depth_result, initial_segments: Dict[str, List]) -> List[CrossDepthSegment]:
        """
        Main function to process all layers and merge cross-layer segments
        
        Args:
            depth_result: The depth processing result with layers
            initial_segments: Dict mapping layer names to lists of segments from that layer
            
        Returns:
            List of CrossDepthSegment objects representing final merged segments
        """
        layers = depth_result.layers
        final_segments = []
        self.processed_points = set()
        
        print(f"\n=== Processing {len(layers)} depth layers for cross-layer merging ===")
        
        for layer_idx, current_layer in enumerate(layers):
            if not current_layer.points:
                print(f"Layer '{current_layer.config.label}' has no points, skipping")
                continue
                
            # Filter out already processed points
            available_points = [p for p in current_layer.points 
                             if (p[0], p[1]) not in self.processed_points]
            
            if not available_points:
                print(f"Layer '{current_layer.config.label}' has no unprocessed points, skipping")
                continue
                
            print(f"\nProcessing layer '{current_layer.config.label}' ({layer_idx}/{len(layers)-1})")
            print(f"  Available points: {len(available_points)} (filtered from {len(current_layer.points)})")
            
            # Get the neon-masked RGB for this layer
            masked_rgb = self.neon_masked_frames[current_layer.config.label]
            self.sam_processor.set_image(masked_rgb)
            
            # Process remaining points in this layer
            remaining_points = available_points.copy()
            
            while remaining_points:
                # Start new segment with first remaining point
                initial_point = remaining_points.pop(0)
                
                print(f"  Starting new cross-layer segment with point {initial_point}")
                
                # Create initial segment
                try:
                    segment_result = self.sam_processor.segment_from_point([initial_point])
                    
                    if segment_result.confidence < self.sam_processor.config.min_mask_confidence:
                        print(f"    Low confidence segment ({segment_result.confidence:.3f}), skipping")
                        continue
                    
                    # Initialize cross-depth segment
                    cross_segment = CrossDepthSegment(
                        segment_mask=segment_result.mask,
                        confidence=segment_result.confidence,
                        depth_layers=[current_layer.config.label],
                        all_points=[initial_point],
                        points_by_layer={current_layer.config.label: [initial_point]},
                        primary_layer=current_layer.config.label,
                        bounding_box=segment_result.bounding_box,
                        area=segment_result.area
                    )
                    
                    # Check for points from current layer that fall inside segment
                    current_layer_points_inside = self.check_points_in_segment(
                        segment_result.mask, remaining_points
                    )
                    
                    if current_layer_points_inside:
                        print(f"    Found {len(current_layer_points_inside)} additional points from current layer")
                        cross_segment.all_points.extend(current_layer_points_inside)
                        cross_segment.points_by_layer[current_layer.config.label].extend(current_layer_points_inside)
                        # Remove these points from remaining
                        remaining_points = [p for p in remaining_points if p not in current_layer_points_inside]
                    
                    # Check adjacent layers for cross-layer segments
                    adjacent_layer_indices = self.get_adjacent_layers(layer_idx, len(layers))
                    
                    for adj_idx in adjacent_layer_indices:
                        adj_layer = layers[adj_idx]
                        
                        # Filter out already processed points from adjacent layer
                        adj_available_points = [p for p in adj_layer.points 
                                              if (p[0], p[1]) not in self.processed_points]
                        
                        if not adj_available_points:
                            continue
                            
                        # Check which points from adjacent layer fall inside current segment
                        adj_points_inside = self.check_points_in_segment(
                            segment_result.mask, adj_available_points
                        )
                        
                        if adj_points_inside:
                            print(f"    Found {len(adj_points_inside)} points from adjacent layer '{adj_layer.config.label}'")
                            
                            # Add this layer to the cross-segment
                            if adj_layer.config.label not in cross_segment.depth_layers:
                                cross_segment.depth_layers.append(adj_layer.config.label)
                                cross_segment.points_by_layer[adj_layer.config.label] = []
                            
                            cross_segment.all_points.extend(adj_points_inside)
                            cross_segment.points_by_layer[adj_layer.config.label].extend(adj_points_inside)
                    
                    # If we found points from multiple layers, re-segment with all points
                    if len(cross_segment.depth_layers) > 1 or len(cross_segment.all_points) > 1:
                        print(f"    Re-segmenting with {len(cross_segment.all_points)} total points across {len(cross_segment.depth_layers)} layers")
                        
                        try:
                            final_segment_result = self.sam_processor.segment_from_point(cross_segment.all_points)
                            
                            # Update cross-segment with final segmentation
                            cross_segment.segment_mask = final_segment_result.mask
                            cross_segment.confidence = final_segment_result.confidence
                            cross_segment.bounding_box = final_segment_result.bounding_box
                            cross_segment.area = final_segment_result.area
                            
                        except Exception as e:
                            print(f"    Error in re-segmentation: {e}, using initial segment")
                    
                    # Mark all points as processed
                    for point in cross_segment.all_points:
                        self.processed_points.add((point[0], point[1]))
                    
                    final_segments.append(cross_segment)
                    
                    print(f"    Created cross-layer segment spanning {cross_segment.depth_layers}")
                    print(f"    Total points: {len(cross_segment.all_points)}, Confidence: {cross_segment.confidence:.3f}")
                    
                except Exception as e:
                    print(f"    Error processing point {initial_point}: {e}")
                    continue
        
        print(f"\n=== Cross-layer merging complete ===")
        print(f"Total cross-layer segments created: {len(final_segments)}")
        
        # Print summary
        for i, seg in enumerate(final_segments):
            layers_str = " + ".join(seg.depth_layers)
            print(f"  Segment {i+1}: {layers_str} ({len(seg.all_points)} points, conf={seg.confidence:.3f})")
        
        return final_segments
    
    def convert_to_slivs_objects(self, cross_segments: List[CrossDepthSegment], object_counter_start: int = 1) -> List:
        """Convert CrossDepthSegment objects to SLIVSObject format"""
        slivs_objects = []
        object_counter = object_counter_start
        
        for cross_segment in cross_segments:
            # Calculate combined depth range across all layers
            # For now, we'll use the primary layer's depth range, but this could be expanded
            primary_layer_name = cross_segment.primary_layer
            
            # Create depth range that spans all involved layers
            # This is a simplified approach - you might want to be more sophisticated
            min_depth = float('inf')
            max_depth = float('-inf')
            
            # If we have access to the layer configs, we can get proper depth ranges
            # For now, using a placeholder approach
            if len(cross_segment.depth_layers) == 1:
                # Single layer - use its range (would need layer config access)
                depth_range = (110, 145)  # Placeholder
            else:
                # Multi-layer - span across layers (would need layer config access)
                depth_range = (110, 181)  # Placeholder for spanning multiple layers
            
            # Create SLIVSObject (assuming you have this class imported)
            # You'll need to adjust this based on your actual SLIVSObject implementation
            slivs_object = {
                'object_id': f"object_{object_counter:03d}",
                'depth_layers': cross_segment.depth_layers,  # New field for multi-layer objects
                'primary_depth_layer': cross_segment.primary_layer,
                'segment_mask': cross_segment.segment_mask,
                'confidence': cross_segment.confidence,
                'positive_points': cross_segment.all_points,
                'points_by_layer': cross_segment.points_by_layer,  # New field
                'depth_range': depth_range,
                'bounding_box': cross_segment.bounding_box,
                'area': cross_segment.area,
                'is_cross_layer': len(cross_segment.depth_layers) > 1
            }
            
            slivs_objects.append(slivs_object)
            object_counter += 1
        
        return slivs_objects


# Example usage with modified SLIVSObject class to support cross-layer attributes
@dataclass
class EnhancedSLIVSObject:
    """Enhanced SLIVSObject with cross-layer support"""
    object_id: str
    depth_layer: str  # Primary layer or combined layer names
    segment_mask: np.ndarray
    confidence: float
    positive_points: List[Tuple[int, int]]
    depth_range: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    area: int
    
    # New cross-layer attributes
    depth_layers: List[str] = None  # All layers this object spans
    primary_depth_layer: str = None  # The layer where segmentation started
    points_by_layer: Dict[str, List[Tuple[int, int]]] = None  # Points organized by layer
    is_cross_layer: bool = False  # Whether object spans multiple layers


def integrate_cross_depth_segmentation(demo_instance):
    """
    Integration function to modify the existing demo to use cross-depth segmentation
    
    This replaces step3_segment_objects_by_layer() and modifies step4_final_segmentation_pass()
    """
    
    def new_step3_cross_layer_segmentation(self):
        """Step 3: Process objects across depth layers with cross-layer merging"""
        print("\n=== Step 3: Cross-layer segmentation ===")
        
        # Initialize cross-depth processor
        cross_processor = CrossDepthSegmentationProcessor(
            self.sam_processor, 
            self.neon_masked_frames
        )
        
        # Process all layers and merge cross-layer segments
        cross_segments = cross_processor.merge_cross_layer_segments(
            self.depth_result, 
            {}  # We're not using initial_segments in this approach
        )
        
        # Convert to SLIVSObject format
        self.objects = []
        object_counter = 1
        
        for cross_segment in cross_segments:
            # Calculate proper depth range based on involved layers
            involved_layers = [layer for layer in self.depth_result.layers 
                             if layer.config.label in cross_segment.depth_layers]
            
            if involved_layers:
                min_depth = min(layer.config.min_depth for layer in involved_layers)
                max_depth = max(layer.config.max_depth for layer in involved_layers)
            else:
                min_depth, max_depth = 110, 145  # fallback
            
            # Use your existing SLIVSObject class but add custom attributes
            slivs_object = SLIVSObject(
                object_id=f"object_{object_counter:03d}",
                depth_layer=" + ".join(cross_segment.depth_layers),  # Combined layer names
                segment_mask=cross_segment.segment_mask,
                confidence=cross_segment.confidence,
                positive_points=cross_segment.all_points,
                depth_range=(min_depth, max_depth),
                bounding_box=cross_segment.bounding_box,
                area=cross_segment.area
            )
            
            # Add custom attributes for cross-layer tracking
            slivs_object.depth_layers = cross_segment.depth_layers
            slivs_object.primary_depth_layer = cross_segment.primary_layer
            slivs_object.points_by_layer = cross_segment.points_by_layer
            slivs_object.is_cross_layer = len(cross_segment.depth_layers) > 1
            
            self.objects.append(slivs_object)
            object_counter += 1
            
            print(f"Created {slivs_object.object_id}: {' + '.join(cross_segment.depth_layers)} "
                  f"({len(cross_segment.all_points)} points, conf={cross_segment.confidence:.3f})")
        
        print(f"\nCross-layer segmentation complete: {len(self.objects)} objects")
        for obj in self.objects:
            if hasattr(obj, 'is_cross_layer') and obj.is_cross_layer:
                print(f"  {obj.object_id}: Multi-layer object spanning {obj.depth_layers}")
            else:
                print(f"  {obj.object_id}: Single-layer object in {getattr(obj, 'primary_depth_layer', obj.depth_layer)}")
    
    def new_step4_final_cross_layer_segmentation(self):
        """Step 4: Final segmentation using primary layer for each object"""
        print("\n=== Step 4: Final cross-layer segmentation pass ===")
        
        for obj in self.objects:
            print(f"Final segmentation for {obj.object_id}")
            
            # Use the primary layer's neon-masked frame for final segmentation
            primary_layer = getattr(obj, 'primary_depth_layer', obj.depth_layer.split(' + ')[0])
            masked_rgb = self.neon_masked_frames[primary_layer]
            self.sam_processor.set_image(masked_rgb)
            
            try:
                # Segment with all positive points from all layers
                final_segment = self.sam_processor.segment_from_point(obj.positive_points)
                
                # Update object with final segmentation
                obj.segment_mask = final_segment.mask
                obj.confidence = final_segment.confidence
                obj.bounding_box = final_segment.bounding_box
                obj.area = final_segment.area
                
                print(f"  Final confidence: {final_segment.confidence:.3f}, area: {final_segment.area}")
                
                if hasattr(obj, 'is_cross_layer') and obj.is_cross_layer:
                    print(f"  Cross-layer object used {len(obj.positive_points)} points from layers: {obj.depth_layers}")
                
            except Exception as e:
                print(f"  Error in final segmentation: {e}")
    
    # Replace the methods in the demo instance
    demo_instance.step3_segment_objects_by_layer = new_step3_cross_layer_segmentation.__get__(demo_instance)
    demo_instance.step4_final_segmentation_pass = new_step4_final_cross_layer_segmentation.__get__(demo_instance)
    
    return demo_instance


# Example of how to use this in your main function:
def main_with_cross_depth():
    """Modified main function using cross-depth segmentation"""
    try:
        demo = SLIVSDepthSegmentationDemo()
        
        # Apply cross-depth segmentation modifications
        demo = integrate_cross_depth_segmentation(demo)
        
        # Run the demo with enhanced cross-layer capabilities
        demo.run_demo()
        
    except Exception as e:
        print(f"Failed to run demo: {e}")
        import traceback
        traceback.print_exc()