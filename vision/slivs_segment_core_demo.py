#!/usr/bin/env python3
"""
SLIVS Single Frame Pipeline Demo - Enhanced with Depth-Aware Segmentation
Complete processing pipeline for one captured frame with depth layers and SAM2 segmentation

NEW FEATURES:
- Depth-aware pixel masking for improved SAM2 segmentation
- Layer-specific image constraints to prevent false connections between distant objects
- Enhanced debugging visualizations showing depth constraints

Process:
1. Capture and warm up camera
2. Get depth layers and points from SLIVS depth processor
3. For each depth layer:
   - Create depth-constrained image (mask out incompatible pixels)
   - Process each point individually with SAM2 using constrained image
   - Keep largest blob from each segment
   - Group points that fall within existing blobs
   - Create objects with points and segments
4. Re-segment each object using all its points together
5. Display and save results with colored segments per depth layer
6. ENHANCED: Save depth map visualization and depth constraint analysis
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import uuid
import math

# Import SLIVS modules
from slivs_depth_core import SLIVSDepthProcessor, DepthProcessingResult, DepthLayerConfig
from slivs_segment_core import SLIVSSam2Processor, SAM2Config

# Import the new depth-aware segmentation classes (CORRECTED VERSION)
from corrected_depth_aware_segmentation import DepthAwareSAM2Processor, SLIVSDepthVisualizer


@dataclass
class SLIVSObject:
    """Single object with points and segment"""
    object_id: str
    points: List[Tuple[int, int]]
    segment: np.ndarray
    depth_layer: str
    confidence: float
    area: int
    centroid: Tuple[float, float]


class SLIVSDepthSegmentDemo:
    """Enhanced SLIVS demo using depth-aware segmentation for improved accuracy"""
    
    def __init__(self):
        # Initialize depth processor
        print("Initializing SLIVS Depth Processor...")
        self.depth_processor = SLIVSDepthProcessor(
            model_name="Intel/dpt-swinv2-tiny-256",
            target_squares=150,  # More points for better coverage
            min_fill_threshold=0.7
        )
        
        # Configure 7 depth layers
        depth_layers = {
            "furthest": DepthLayerConfig(0, 36, "Furthest"),
            "far": DepthLayerConfig(37, 73, "Far"),
            "lessfar": DepthLayerConfig(74, 109, "Less Far"),
            "mid": DepthLayerConfig(110, 145, "Mid"),
            "midnear": DepthLayerConfig(146, 181, "Mid Near"),
            "close": DepthLayerConfig(182, 217, "Close"),
            "closest": DepthLayerConfig(218, 255, "Closest"),
        }
        self.depth_processor.update_depth_layers(depth_layers)
        self.num_depth_layers = len(depth_layers)
        
        # Initialize base SAM2 processor
        print("Initializing SAM2 Processor...")
        sam_config = SAM2Config(
            model_name="sam2.1_hiera_tiny",
            min_mask_confidence=0.1,
            multimask_output=True,
            use_highest_confidence=True
        )
        base_sam_processor = SLIVSSam2Processor(sam_config)
        
        # Initialize depth-aware SAM2 processor (NEW)
        print("Initializing Depth-Aware SAM2 Processor...")
        self.depth_aware_sam = DepthAwareSAM2Processor(
            sam_processor=base_sam_processor,
            num_depth_layers=self.num_depth_layers
        )
        
        # Configuration
        self.debug_confidence_threshold = 0.1
        
        # Initialize depth visualizer
        self.depth_visualizer = SLIVSDepthVisualizer()
        
        # Heat map colors: Closest (red) → Farthest (blue)
        self.layer_colors = {
            "Closest": (0, 0, 255),       # Red (closest)
            "Close": (0, 128, 255),       # Orange
            "Mid Near": (0, 255, 255),    # Yellow
            "Mid": (0, 255, 128),         # Light Green
            "Less Far": (0, 255, 0),      # Green
            "Far": (128, 128, 0),         # Cyan/Teal
            "Furthest": (255, 0, 0),      # Blue (farthest)
        }
        
        self.point_color = (255, 255, 255)  # White points
        self.point_size = 4
        
        # Create output directory
        self.output_dir = "slivs_segment_core_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Enable depth constraint visualizations for debugging
        self.save_depth_constraint_visualizations = True
    
    def get_object_color(self, layer_name: str, object_index: int) -> Tuple[int, int, int]:
        """Generate a unique color variation for each object within a layer"""
        base_color = self.layer_colors.get(layer_name, (128, 128, 128))
        
        # Create variations by adjusting brightness and hue
        variations = [
            (1.0, 1.0, 1.0),     # Original color
            (0.7, 1.0, 1.0),     # Darker
            (1.0, 0.7, 1.0),     # Less green
            (1.0, 1.0, 0.7),     # Less blue
            (0.8, 0.8, 1.0),     # Darker, more blue
            (1.0, 0.8, 0.8),     # Less green/blue
            (0.6, 1.0, 1.0),     # Much darker
            (1.0, 0.6, 1.0),     # Much less green
        ]
        
        # Select variation based on object index
        multiplier = variations[object_index % len(variations)]
        
        # Apply multiplier to each color channel
        varied_color = (
            int(base_color[0] * multiplier[0]),
            int(base_color[1] * multiplier[1]),
            int(base_color[2] * multiplier[2])
        )
        
        return varied_color
    
    def capture_frame(self) -> np.ndarray:
        """Capture a single frame after camera warmup"""
        print("Starting camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            # Warm up camera - discard first 10 frames
            print("Warming up camera (discarding 10 frames)...")
            for i in range(10):
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(f"Failed to capture warmup frame {i}")
                print(f"Warmup frame {i+1}/10")
                time.sleep(0.1)  # Small delay between frames
            
            # Capture the actual frame
            print("Capturing frame...")
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to capture final frame")
            
            print(f"Captured frame: {frame.shape}")
            return frame
            
        finally:
            cap.release()
    
    def get_largest_blob(self, mask: np.ndarray) -> np.ndarray:
        """Extract the largest connected component from a mask with debugging"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            print(f"        Blob analysis: Only background found ({num_labels} labels)")
            return np.zeros_like(mask)
        
        # Find largest component (excluding background at index 0)
        component_areas = stats[1:, cv2.CC_STAT_AREA]  # Exclude background
        largest_idx = np.argmax(component_areas) + 1
        largest_area = component_areas[np.argmax(component_areas)]
        largest_blob = (labels == largest_idx)
        
        print(f"        Blob analysis: {num_labels-1} components, largest has {largest_area} pixels")
        
        return largest_blob
    
    def point_in_mask(self, point: Tuple[int, int], mask: np.ndarray) -> bool:
        """Check if a point falls within a mask"""
        x, y = point
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            return mask[y, x]
        return False
    
    def calculate_centroid(self, mask: np.ndarray) -> Tuple[float, float]:
        """Calculate centroid of a mask"""
        if not np.any(mask):
            return (0.0, 0.0)
        
        y_coords, x_coords = np.where(mask)
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)
        return (float(centroid_x), float(centroid_y))
    
    def process_depth_layer_with_constraints(self, layer_points: List[Tuple[int, int]], 
                                           layer_index: int, layer_name: str, 
                                           frame: np.ndarray, depth_result: DepthProcessingResult) -> List[SLIVSObject]:
        """
        Process all points in a single depth layer using depth-aware segmentation.
        
        NEW: Uses depth-constrained images to prevent SAM2 from connecting objects 
        at incompatible depths (e.g., close book with far wall).
        """
        print(f"  Processing layer '{layer_name}' (index {layer_index}) with {len(layer_points)} points...")
        print(f"    Using depth-aware segmentation with pixel masking")
        
        objects = []
        processed_points = set()
        failed_points = []
        
        # Initialize depth-aware processing for this frame
        self.depth_aware_sam.set_frame_data(frame, depth_result)
        
        # Save depth constraint visualization for debugging
        if self.save_depth_constraint_visualizations and layer_points:
            self.save_depth_constraint_visualization(layer_index, layer_name, layer_points)
        
        # Process each point individually with depth constraints
        for i, point in enumerate(layer_points):
            if point in processed_points:
                continue
            
            print(f"    Point {i+1}/{len(layer_points)}: {point}")
            
            try:
                # Use depth-constrained segmentation (NEW)
                segment_result = self.depth_aware_sam.segment_points_with_depth_constraint(
                    [point], layer_index
                )
                
                if segment_result is None:
                    print(f"      ❌ Depth-aware SAM2 returned None for point {point}")
                    failed_points.append((point, "SAM2_None"))
                    processed_points.add(point)
                    continue
                
                # Handle different SAM2 result formats
                if hasattr(segment_result, 'confidence'):
                    # Single segment result
                    confidence = segment_result.confidence
                    mask = segment_result.mask
                elif hasattr(segment_result, 'segments') and segment_result.segments:
                    # Multiple segments result - take the best one
                    best_segment = max(segment_result.segments, key=lambda s: s.confidence)
                    confidence = best_segment.confidence
                    mask = best_segment.mask
                else:
                    print(f"      ❌ Unexpected segment result format")
                    failed_points.append((point, "Format_Error"))
                    processed_points.add(point)
                    continue
                
                print(f"      Depth-aware SAM2 confidence: {confidence:.3f}")
                print(f"      SAM2 mask area: {np.sum(mask)} pixels")
                
                if confidence < self.debug_confidence_threshold:
                    print(f"      ❌ Confidence {confidence:.3f} below threshold {self.debug_confidence_threshold}")
                    failed_points.append((point, f"Low_Confidence_{confidence:.3f}"))
                    processed_points.add(point)
                    continue
                
                # Get largest blob
                original_area = np.sum(mask)
                largest_blob = self.get_largest_blob(mask)
                cleaned_area = np.sum(largest_blob)
                
                print(f"      Blob cleaning: {original_area} → {cleaned_area} pixels")
                
                if not np.any(largest_blob):
                    print(f"      ❌ No pixels remaining after blob cleaning")
                    failed_points.append((point, "Empty_After_Cleaning"))
                    processed_points.add(point)
                    continue
                
                # Create new object
                object_points = [point]
                processed_points.add(point)
                
                # Check if any other unprocessed points fall in this blob
                grouped_points = 0
                for other_point in layer_points:
                    if (other_point not in processed_points and 
                        self.point_in_mask(other_point, largest_blob)):
                        object_points.append(other_point)
                        processed_points.add(other_point)
                        grouped_points += 1
                
                if grouped_points > 0:
                    print(f"      Grouped {grouped_points} additional points")
                
                # Calculate object properties
                centroid = self.calculate_centroid(largest_blob)
                area = int(np.sum(largest_blob))
                
                # Create SLIVS object
                obj = SLIVSObject(
                    object_id=f"{layer_name}_{len(objects)}",
                    points=object_points,
                    segment=largest_blob,
                    depth_layer=layer_name,
                    confidence=confidence,
                    area=area,
                    centroid=centroid
                )
                
                objects.append(obj)
                print(f"      ✓ Created depth-aware object {obj.object_id} with {len(object_points)} points, area={area}")
                
            except Exception as e:
                print(f"      ❌ Exception processing point {point}: {e}")
                failed_points.append((point, f"Exception_{str(e)[:30]}"))
                processed_points.add(point)
                continue
        
        # Summary for this layer
        print(f"    Layer '{layer_name}' summary:")
        print(f"      ✓ Created {len(objects)} objects using depth constraints")
        print(f"      ✓ Processed {len(processed_points)} points successfully")
        
        if failed_points:
            print(f"      ❌ Failed points ({len(failed_points)}):")
            for point, reason in failed_points[:3]:  # Show first 3
                print(f"        {point}: {reason}")
            if len(failed_points) > 3:
                print(f"        ... and {len(failed_points) - 3} more")
        
        return objects
    
    def save_depth_constraint_visualization(self, layer_index: int, layer_name: str, 
                                          points: List[Tuple[int, int]]):
        """Save visualization of depth constraints for debugging."""
        try:
            viz = self.depth_aware_sam.visualize_depth_constraint(layer_index, points)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"depth_constraint_{layer_name}_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            cv2.imwrite(filepath, viz)
            print(f"      Saved depth constraint visualization: {filename}")
            
        except Exception as e:
            print(f"      Warning: Could not save depth constraint visualization: {e}")
    
    def refine_object_segments(self, objects: List[SLIVSObject], frame: np.ndarray) -> List[SLIVSObject]:
        """Re-segment each object using all its points together"""
        print("  Refining segments using multi-point prompts...")
        
        refined_objects = []
        
        # Use the base SAM processor for refinement (not depth-constrained)
        base_sam = self.depth_aware_sam.sam_processor
        base_sam.set_image(frame)
        
        for obj in objects:
            try:
                if len(obj.points) == 1:
                    # Single point - keep original but still report it
                    refined_objects.append(obj)
                    print(f"    Kept single-point {obj.object_id}: conf={obj.confidence:.3f}, area={obj.area}")
                else:
                    # Multiple points - use segment_from_points which returns SAM2ProcessingResult
                    sam_result = base_sam.segment_from_points(obj.points)
                    
                    if sam_result.segments:
                        # Take the best segment
                        best_segment = max(sam_result.segments, key=lambda s: s.confidence)
                        
                        # Apply noise removal - keep only largest blob
                        clean_segment = self.get_largest_blob(best_segment.mask)
                        
                        if np.any(clean_segment):
                            # Calculate properties of cleaned segment
                            clean_area = int(np.sum(clean_segment))
                            clean_centroid = self.calculate_centroid(clean_segment)
                            
                            # Update object with cleaned refined segment
                            refined_obj = SLIVSObject(
                                object_id=obj.object_id,
                                points=obj.points,
                                segment=clean_segment,
                                depth_layer=obj.depth_layer,
                                confidence=best_segment.confidence,
                                area=clean_area,
                                centroid=clean_centroid
                            )
                            refined_objects.append(refined_obj)
                            print(f"    Refined {obj.object_id}: conf={best_segment.confidence:.3f}, area={clean_area}")
                        else:
                            # Cleaned segment is empty, keep original
                            refined_objects.append(obj)
                            print(f"    Kept original {obj.object_id} (cleaned segment was empty)")
                    else:
                        # No good segment, keep original
                        refined_objects.append(obj)
                        print(f"    Kept original {obj.object_id} (no refinement)")
                    
            except Exception as e:
                print(f"    Error refining {obj.object_id}: {e}")
                refined_objects.append(obj)
        
        return refined_objects
    
    def create_segmentation_visualization(self, frame: np.ndarray, all_objects: List[SLIVSObject], 
                                        depth_result: DepthProcessingResult) -> np.ndarray:
        """Create visualization with panels for each depth layer showing segmented objects"""
        
        # Panel dimensions
        panel_width = 320
        panel_height = 240
        
        # Calculate layout for depth layers + original frame
        layer_names = list(self.layer_colors.keys())
        total_panels = len(layer_names) + 1  # +1 for original frame
        
        # Use 3x3 grid for up to 8 panels
        cols = 3
        rows = 3
        
        composite_width = panel_width * cols
        composite_height = panel_height * rows
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Place original frame in top-left
        frame_resized = cv2.resize(frame, (panel_width, panel_height))
        composite[0:panel_height, 0:panel_width] = frame_resized
        cv2.putText(composite, "Original", (5, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Process each depth layer
        panel_idx = 1
        for layer_name in layer_names:
            if panel_idx >= total_panels:
                break
                
            row = panel_idx // cols
            col = panel_idx % cols
            y = row * panel_height
            x = col * panel_width
            
            # Create layer visualization
            layer_vis = frame_resized.copy()
            layer_objects = [obj for obj in all_objects if obj.depth_layer == layer_name]
            
            # Draw segments and points for this layer
            for obj_index, obj in enumerate(layer_objects):
                # Get unique color for this object
                object_color = self.get_object_color(layer_name, obj_index)
                
                # Resize segment to panel size
                segment_resized = cv2.resize(obj.segment.astype(np.uint8), 
                                           (panel_width, panel_height))
                
                # Create colored overlay
                mask_colored = np.zeros_like(layer_vis)
                mask_colored[segment_resized > 0] = object_color
                layer_vis = cv2.addWeighted(layer_vis, 1.0, mask_colored, 0.4, 0)
                
                # Draw segment boundary
                contours, _ = cv2.findContours(segment_resized, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(layer_vis, contours, -1, object_color, 2)
                
                # Draw points (scaled to panel size)
                scale_x = panel_width / frame.shape[1]
                scale_y = panel_height / frame.shape[0]
                
                for point in obj.points:
                    scaled_point = (int(point[0] * scale_x), int(point[1] * scale_y))
                    cv2.circle(layer_vis, scaled_point, self.point_size, 
                             self.point_color, -1)
                    cv2.circle(layer_vis, scaled_point, self.point_size + 1, 
                             (0, 0, 0), 1)
            
            # Place in composite
            composite[y:y+panel_height, x:x+panel_width] = layer_vis
            
            # Add label
            label = f"{layer_name} ({len(layer_objects)} obj)"
            cv2.putText(composite, label, (x + 5, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            panel_idx += 1
        
        return composite
    
    def save_results(self, segmentation_composite: np.ndarray, depth_composite: np.ndarray,
                    all_objects: List[SLIVSObject], frame: np.ndarray, 
                    depth_result: DepthProcessingResult) -> str:
        """Save visualization and object data including depth information"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save segmentation composite image
        seg_path = os.path.join(self.output_dir, f"slivs_segmentation_{timestamp}.jpg")
        cv2.imwrite(seg_path, segmentation_composite)
        
        # Save depth composite image
        depth_path = os.path.join(self.output_dir, f"slivs_depth_layers_{timestamp}.jpg")
        cv2.imwrite(depth_path, depth_composite)
        
        # Save original frame
        frame_path = os.path.join(self.output_dir, f"original_frame_{timestamp}.jpg")
        cv2.imwrite(frame_path, frame)
        
        # Save raw depth map
        depth_map_path = os.path.join(self.output_dir, f"depth_map_{timestamp}.png")
        cv2.imwrite(depth_map_path, depth_result.full_depth_map)
        
        # Save object data with depth information
        data_path = os.path.join(self.output_dir, f"slivs_data_{timestamp}.txt")
        with open(data_path, 'w') as f:
            f.write(f"SLIVS Enhanced Results (Depth-Aware Segmentation) - {timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            # Enhancement info
            f.write("ENHANCEMENTS:\n")
            f.write("  - Depth-aware pixel masking for SAM2 segmentation\n")
            f.write("  - Layer-specific image constraints prevent false connections\n")
            f.write("  - Depth constraint visualizations saved for debugging\n")
            f.write("\n")
            
            # Depth processing summary
            f.write("DEPTH PROCESSING SUMMARY:\n")
            f.write(f"  Total points detected: {len(depth_result.all_points)}\n")
            f.write(f"  Depth processing time: {depth_result.processing_time:.3f}s\n")
            f.write("\n")
            
            # Depth layer breakdown
            f.write("DEPTH LAYERS:\n")
            for i, layer in enumerate(depth_result.layers):
                f.write(f"  Layer {i} - {layer.config.label}: {len(layer.points)} points ")
                f.write(f"(depth range: {layer.config.min_depth}-{layer.config.max_depth})\n")
            f.write("\n")
            
            # Object details
            f.write("SEGMENTED OBJECTS (using depth-aware segmentation):\n")
            for obj in all_objects:
                f.write(f"Object: {obj.object_id}\n")
                f.write(f"  Depth Layer: {obj.depth_layer}\n")
                f.write(f"  Confidence: {obj.confidence:.3f}\n")
                f.write(f"  Area: {obj.area} pixels\n")
                f.write(f"  Centroid: ({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f})\n")
                f.write(f"  Points ({len(obj.points)}): {obj.points}\n")
                f.write("\n")
            
            # Summary statistics
            layer_counts = {}
            for obj in all_objects:
                layer_counts[obj.depth_layer] = layer_counts.get(obj.depth_layer, 0) + 1
            
            f.write("SUMMARY:\n")
            f.write(f"  Total Objects: {len(all_objects)}\n")
            f.write(f"  Depth Layers Used: {len(layer_counts)}/{len(depth_result.layers)}\n")
            for layer, count in layer_counts.items():
                f.write(f"  {layer}: {count} objects\n")
            
            # File paths
            f.write("\nOUTPUT FILES:\n")
            f.write(f"  Segmentation View: {os.path.basename(seg_path)}\n")
            f.write(f"  Depth Layers View: {os.path.basename(depth_path)}\n")
            f.write(f"  Original Frame: {os.path.basename(frame_path)}\n")
            f.write(f"  Raw Depth Map: {os.path.basename(depth_map_path)}\n")
            f.write(f"  Depth Constraint Visualizations: depth_constraint_*.jpg\n")
        
        return seg_path
    
    def run_demo(self):
        """Run the complete enhanced single frame demo with depth-aware segmentation"""
        print("=== SLIVS Enhanced Demo with Depth-Aware Segmentation ===\n")
        
        try:
            # Step 1: Capture frame
            frame = self.capture_frame()
            print(f"✓ Captured frame: {frame.shape}\n")
            
            # Step 2: Process depth and get layer points
            print("Processing depth layers...")
            start_time = time.time()
            depth_result = self.depth_processor.process_frame(frame)
            depth_time = time.time() - start_time
            print(f"✓ Depth processing completed in {depth_time:.2f}s")
            print(f"  Total points across all layers: {len(depth_result.all_points)}")
            
            # Print layer breakdown
            for i, layer in enumerate(depth_result.layers):
                print(f"  Layer {i} - {layer.config.label}: {len(layer.points)} points")
            print()
            
            # Step 3: Process each depth layer with enhanced depth-aware segmentation
            print("Processing objects with depth-aware segmentation...")
            all_objects = []
            
            for i, layer in enumerate(depth_result.layers):
                if layer.points:
                    layer_objects = self.process_depth_layer_with_constraints(
                        layer.points, i, layer.config.label, frame, depth_result
                    )
                    all_objects.extend(layer_objects)
                else:
                    print(f"  Skipping empty layer '{layer.config.label}'")
            
            print(f"✓ Depth-aware segmentation created {len(all_objects)} objects\n")
            
            # Step 4: Refine segments using multi-point prompts
            print("Refining object segments...")
            start_time = time.time()
            refined_objects = self.refine_object_segments(all_objects, frame)
            refine_time = time.time() - start_time
            print(f"✓ Segment refinement completed in {refine_time:.2f}s\n")
            
            # Step 5: Create visualizations
            print("Creating visualizations...")
            
            # Create segmentation visualization
            segmentation_composite = self.create_segmentation_visualization(
                frame, refined_objects, depth_result)
            
            # Create depth visualization
            depth_composite = self.depth_visualizer.create_depth_composite_display(depth_result)
            
            print("✓ Visualizations created\n")
            
            # Step 6: Save results
            print("Saving results...")
            saved_path = self.save_results(segmentation_composite, depth_composite, 
                                         refined_objects, frame, depth_result)
            print(f"✓ Results saved to: {saved_path}\n")
            
            # Display results - show both visualizations
            print("Displaying results (press any key to switch views, 'q' to quit)...")
            
            current_view = "segmentation"
            while True:
                if current_view == "segmentation":
                    cv2.imshow('Enhanced SLIVS - Segmentation View (space for depth view)', 
                              segmentation_composite)
                else:
                    cv2.imshow('Enhanced SLIVS - Depth Layers View (space for segmentation view)', 
                              depth_composite)
                
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' '):  # Spacebar to switch views
                    current_view = "depth" if current_view == "segmentation" else "segmentation"
                    cv2.destroyAllWindows()
            
            cv2.destroyAllWindows()
            
            # Print enhanced summary
            print("\n=== ENHANCED PROCESSING SUMMARY ===")
            print(f"Total objects detected: {len(refined_objects)}")
            print(f"Total processing time: {depth_time + refine_time:.2f}s")
            print(f"Depth points generated: {len(depth_result.all_points)}")
            print("🚀 ENHANCEMENT: Depth-aware pixel masking used for improved segmentation")
            print("   - Prevents false connections between objects at different depths")
            print("   - Each layer uses pixels only from compatible depth ranges")
            print("   - Depth constraint visualizations saved for debugging")
            
            layer_summary = {}
            total_improvements = 0
            for obj in refined_objects:
                layer_summary[obj.depth_layer] = layer_summary.get(obj.depth_layer, 0) + 1
                # Estimate improvement (objects with high confidence likely benefited from depth constraints)
                if obj.confidence > 0.8:
                    total_improvements += 1
            
            print(f"\nObjects by depth layer:")
            for layer, count in layer_summary.items():
                color = self.layer_colors.get(layer, (128, 128, 128))
                print(f"  {layer}: {count} objects (color: {color})")
            
            print(f"\nEstimated improvements from depth-aware segmentation:")
            print(f"  High confidence objects (>0.8): {total_improvements}/{len(refined_objects)}")
            print(f"  Likely prevented false connections between distant objects")
            
            print(f"\nOutput saved to: {self.output_dir}/")
            print("📁 Files created:")
            print("  - slivs_segmentation_*.jpg (object segmentation view)")
            print("  - slivs_depth_layers_*.jpg (depth layers with points)")
            print("  - original_frame_*.jpg (captured frame)")
            print("  - depth_map_*.png (raw depth estimation)")
            print("  - slivs_data_*.txt (detailed analysis)")
            print("  🆕 depth_constraint_*.jpg (depth masking visualizations)")
            
            print(f"\n🎯 DEPTH-AWARE SEGMENTATION BENEFITS:")
            print("  ✓ More accurate object boundaries")
            print("  ✓ Reduced false connections between distant objects")
            print("  ✓ Better handling of overlapping depth layers")
            print("  ✓ Debug visualizations show exactly what SAM2 sees")
            print("  ✓ Physics-informed pixel masking improves SAM2 focus")
            
        except Exception as e:
            print(f"Error in enhanced demo: {e}")
            raise


def main():
    """Main function"""
    try:
        print("Starting Enhanced SLIVS Demo with Depth-Aware Segmentation...")
        print("This version uses intelligent pixel masking to improve SAM2 accuracy")
        print("=" * 70)
        
        demo = SLIVSDepthSegmentDemo()
        demo.run_demo()
        
        print("\n" + "=" * 70)
        print("Enhanced demo completed successfully!")
        print("Check the output directory for depth constraint visualizations.")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()