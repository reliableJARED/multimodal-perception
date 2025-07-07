#!/usr/bin/env python3
"""
SLIVS Depth-Guided Segmentation Demo
Combines depth estimation with SAM2 segmentation for object detection and depth understanding.

Process:
1. Generate depth map and depth layers with points
2. For each depth layer, create neon-green masked RGB frames
3. Use depth-guided points for SAM2 segmentation
4. Generate final object list with relative depth
"""

import cv2
import numpy as np
import os
import time
import math
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dataclasses import dataclass

# Import SLIVS modules
from slivs_depth_core import SLIVSDepthProcessor, DepthProcessingResult, DepthLayerConfig
from slivs_segment_core import SLIVSSam2Processor, SAM2Config, SegmentResult


@dataclass
class SLIVSObject:
    """Combined object with depth and segmentation information"""
    object_id: str
    depth_layer: str
    segment_mask: np.ndarray
    confidence: float
    positive_points: List[Tuple[int, int]]
    depth_range: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]
    area: int


class SLIVSDepthSegmentationDemo:
    """Demo combining depth estimation with SAM2 segmentation"""
    
    def __init__(self, output_dir: str = "slivs_demo_output"):
        """Initialize the demo with processors and output directory"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Colors
        self.neon_green = (0, 255, 0)  # BGR format for OpenCV
        self.point_color = (255, 255, 255)  # White points
        self.point_outline_color = (0, 0, 0)  # Black outline for points
        self.point_size = 6
        self.point_outline_size = 8
        
        # Initialize processors
        self._setup_processors()
        
        # Storage for results
        self.depth_result = None
        self.original_rgb = None
        self.neon_masked_frames = {}
        self.objects = []
        
    def _setup_processors(self):
        """Initialize depth and segmentation processors"""
        print("Initializing SLIVS processors...")
        
        # Custom depth layers (7 layers as in your demo)
        depth_layers = {
            "closest": DepthLayerConfig(218, 255, "Closest"),
            "close": DepthLayerConfig(182, 217, "Close"),
            "midnear": DepthLayerConfig(146, 181, "Mid Near"),
            "mid": DepthLayerConfig(110, 145, "Mid"),
            "lessfar": DepthLayerConfig(74, 109, "Less Far"),
            "far": DepthLayerConfig(37, 73, "Far"),
            "furthest": DepthLayerConfig(0, 36, "Furthest"),
        }
        
        # Initialize depth processor
        self.depth_processor = SLIVSDepthProcessor(
            model_name="Intel/dpt-swinv2-tiny-256",
            target_squares=200,
            min_fill_threshold=0.7,
            depth_layer_config=depth_layers
        )
        
        # Initialize SAM2 processor  
        sam_config = SAM2Config(
            model_name="sam2.1_hiera_tiny",
            device=None,  # Auto-detect
            multimask_output=True,
            min_mask_confidence=0.5,
            use_highest_confidence=True
        )
        self.sam_processor = SLIVSSam2Processor(sam_config)
        
        print("Processors initialized successfully!")
    
    def capture_single_frame(self) -> np.ndarray:
        """Capture a single frame from camera after warmup"""
        print("Starting camera for single frame capture...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Warm up camera - discard first 10 frames
        print("Warming up camera...")
        for i in range(10):
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture warmup frame {i}")
            print(f"Warmup frame {i+1}/10")
        
        # Capture the actual frame
        print("Capturing frame...")
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame")
        
        cap.release()
        print("Frame captured successfully!")
        return frame
    
    def step1_generate_depth_and_points(self, frame: np.ndarray):
        """Step 1: Generate depth map and depth layers with points"""
        print("\n=== Step 1: Generating depth map and points ===")
        
        self.original_rgb = frame.copy()
        
        # Process frame with depth processor
        self.depth_result = self.depth_processor.process_frame(frame)
        
        # Save original RGB and depth map
        cv2.imwrite(os.path.join(self.output_dir, "01_original_rgb.jpg"), frame)
        cv2.imwrite(os.path.join(self.output_dir, "02_depth_map.jpg"), self.depth_result.full_depth_map)
        
        # Create depth layers visualization
        self._visualize_depth_layers()
        
        # NEW: Create original points visualization for each layer
        self._visualize_original_points_on_layers()
        
        print(f"Generated depth map with {len(self.depth_result.layers)} layers")
        for layer in self.depth_result.layers:
            print(f"  Layer '{layer.config.label}': {len(layer.points)} points")
    
    def _visualize_depth_layers(self):
        """Create visualization of depth layers with points"""
        layers = self.depth_result.layers
        cols = 3
        rows = math.ceil(len(layers) / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, layer in enumerate(layers):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Show layer mask
            ax.imshow(layer.layer_mask, cmap='viridis')
            
            # Draw points
            for point in layer.points:
                ax.plot(point[0], point[1], 'ro', markersize=3)
            
            ax.set_title(f"{layer.config.label}\n({len(layer.points)} points)")
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(len(layers), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "03_depth_layers_with_points.jpg"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_original_points_on_layers(self):
        """NEW: Create individual visualizations of original points on each depth layer"""
        print("Creating original points visualizations for each layer...")
        
        for layer in self.depth_result.layers:
            # Start with original RGB
            visualization = self.original_rgb.copy()
            
            # Draw all original points for this layer
            for point in layer.points:
                cv2.circle(visualization, point, 6, (0, 0, 0), -1)  # Black outline
                cv2.circle(visualization, point, 4, (255, 255, 255), -1)  # White point
            
            # Add simple text overlay
            cv2.putText(visualization, f"{layer.config.label}: {len(layer.points)} points", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save the visualization
            filename = f"03b_original_points_{layer.config.label.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(os.path.join(self.output_dir, filename), visualization)
        
        print(f"Saved original points visualizations for {len(self.depth_result.layers)} layers")
    
    def step2_create_neon_masked_frames(self):
        """Step 2: Create neon-green masked RGB frames for each depth layer"""
        print("\n=== Step 2: Creating neon-masked RGB frames ===")
        
        layers = self.depth_result.layers
        layer_names = [layer.config.label for layer in layers]
        
        for i, current_layer in enumerate(layers):
            print(f"Processing layer {i+1}/{len(layers)}: {current_layer.config.label}")
            print(f"  Keeping layers {max(0, i-1)} to {min(len(layers)-1, i+1)} as RGB")
            print(f"  Masking layers 0 to {i-2} and {i+2} to {len(layers)-1} as neon green")
            
            # Start with original RGB
            masked_rgb = self.original_rgb.copy()
            
            # Mask objects from layers that are too far (i+2 and beyond)
            if i + 2 < len(layers):
                print(f"  Masking far layers: {i+2} to {len(layers)-1}")
                for j in range(i + 2, len(layers)):
                    far_layer = layers[j]
                    # Find non-black pixels in far layer
                    non_black_mask = far_layer.layer_mask > 0
                    if np.any(non_black_mask):
                        masked_rgb[non_black_mask] = self.neon_green
                        print(f"    Masked {np.sum(non_black_mask)} pixels from layer {j} ({far_layer.config.label})")
            
            # Mask objects from layers that are too close (0 to i-2)
            if i >= 2:  # Only mask if we have at least 2 layers before current
                print(f"  Masking close layers: 0 to {i-2}")
                for j in range(0, i - 1):  # Fixed: should be i-1, not i-2
                    close_layer = layers[j]
                    # Find non-black pixels in close layer
                    non_black_mask = close_layer.layer_mask > 0
                    if np.any(non_black_mask):
                        masked_rgb[non_black_mask] = self.neon_green
                        print(f"    Masked {np.sum(non_black_mask)} pixels from layer {j} ({close_layer.config.label})")
            
            # Store the masked frame
            self.neon_masked_frames[current_layer.config.label] = masked_rgb.copy()
            
            # Save the masked frame
            filename = f"04_neon_masked_{current_layer.config.label.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(os.path.join(self.output_dir, filename), masked_rgb)
        
        print(f"Created {len(self.neon_masked_frames)} neon-masked frames")
    
    def step3_segment_objects_by_layer(self):
        """Step 3: Process points in each depth layer for segmentation"""
        print("\n=== Step 3: Segmenting objects by layer ===")
        
        object_counter = 1
        
        for layer_idx, layer in enumerate(self.depth_result.layers):
            if not layer.points:
                print(f"Skipping layer '{layer.config.label}' - no points")
                continue
                
            print(f"\nProcessing layer '{layer.config.label}' with {len(layer.points)} points")
            
            # Get the neon-masked RGB for this layer
            masked_rgb = self.neon_masked_frames[layer.config.label]
            
            # Set the masked RGB in SAM processor
            self.sam_processor.set_image(masked_rgb)
            
            # Track which points have been processed
            remaining_points = layer.points.copy()
            layer_objects = []
            
            while remaining_points:
                # Take the first remaining point as initial positive prompt
                initial_point = remaining_points.pop(0)
                positive_points = [initial_point]
                
                print(f"  Starting new object with point {initial_point}")
                
                # Get initial segment
                try:
                    segment_result = self.sam_processor.segment_from_point(positive_points)
                    
                    if segment_result.confidence < self.sam_processor.config.min_mask_confidence:
                        print(f"    Low confidence segment ({segment_result.confidence:.3f}), skipping")
                        continue
                    
                    # Check which remaining points fall inside this segment
                    points_to_add = []
                    points_to_keep = []
                    
                    for point in remaining_points:
                        x, y = point
                        if segment_result.mask[y, x]:  # Point is inside segment
                            points_to_add.append(point)
                            print(f"    Adding point {point} to object (inside segment)")
                        else:
                            points_to_keep.append(point)
                    
                    # Update positive points and remaining points
                    positive_points.extend(points_to_add)
                    remaining_points = points_to_keep
                    
                    # If we added points, re-segment with all positive points
                    if points_to_add:
                        print(f"    Re-segmenting with {len(positive_points)} total points")
                        segment_result = self.sam_processor.segment_from_point(positive_points)
                    
                    # Create SLIVS object
                    slivs_object = SLIVSObject(
                        object_id=f"object_{object_counter:03d}",
                        depth_layer=layer.config.label,
                        segment_mask=segment_result.mask,
                        confidence=segment_result.confidence,
                        positive_points=positive_points,
                        depth_range=(layer.config.min_depth, layer.config.max_depth),
                        bounding_box=segment_result.bounding_box,
                        area=segment_result.area
                    )
                    
                    layer_objects.append(slivs_object)
                    self.objects.append(slivs_object)
                    
                    print(f"    Created {slivs_object.object_id} with {len(positive_points)} points, "
                          f"confidence={segment_result.confidence:.3f}, area={segment_result.area}")
                    
                    object_counter += 1
                    
                except Exception as e:
                    print(f"    Error segmenting point {initial_point}: {e}")
                    continue
            
            print(f"  Layer '{layer.config.label}' complete: {len(layer_objects)} objects")
    
    def step3b_cross_integrate_cross_depth_segmentation(self):
        """
        Step 3b: Cross-integrate segments across depth layers
        
        Process objects from step3 to merge cross-layer segments and consolidate points.
        Objects that lose all their points to other layers are deleted.
        """
        print("\n=== Step 3b: Cross-depth integration ===")
        
        if not self.objects:
            print("No objects to process")
            return
        
        print(f"Starting with {len(self.objects)} objects from step 3")
        
        # Track which points have been assigned to final objects
        assigned_points = set()
        
        # Group objects by their depth layer for easier processing
        objects_by_layer = {}
        for obj in self.objects:
            layer = obj.depth_layer
            if layer not in objects_by_layer:
                objects_by_layer[layer] = []
            objects_by_layer[layer].append(obj)
        
        print(f"Objects grouped across {len(objects_by_layer)} layers")
        for layer, objs in objects_by_layer.items():
            print(f"  {layer}: {len(objs)} objects")
        
        # Process layers in order (closest to furthest)
        layer_order = [layer.config.label for layer in self.depth_result.layers]
        updated_objects = []
        
        for current_layer_name in layer_order:
            if current_layer_name not in objects_by_layer:
                print(f"\nLayer '{current_layer_name}': No objects to process")
                continue
                
            current_objects = objects_by_layer[current_layer_name]
            print(f"\nProcessing layer '{current_layer_name}' with {len(current_objects)} objects")
            
            for obj_idx, obj in enumerate(current_objects):
                print(f"  Processing {obj.object_id}")
                
                # Filter out points that have already been assigned
                available_points = [p for p in obj.positive_points 
                                if (p[0], p[1]) not in assigned_points]
                
                if not available_points:
                    print(f"    {obj.object_id}: All points already assigned, marking for deletion")
                    continue
                
                # Set up segmentation with the object's neon-masked frame
                masked_rgb = self.neon_masked_frames[obj.depth_layer]
                self.sam_processor.set_image(masked_rgb)
                
                # Start with available points from current object
                cross_layer_points = available_points.copy()
                involved_layers = [obj.depth_layer]
                
                print(f"    Starting with {len(cross_layer_points)} available points")
                
                # Create initial segment to test for cross-layer expansion
                try:
                    test_segment = self.sam_processor.segment_from_point(cross_layer_points)
                    
                    # Check subsequent (deeper) layers for points that fall inside this segment
                    current_layer_idx = layer_order.index(current_layer_name)
                    
                    # Look ahead to next 2 layers (adjustable)
                    look_ahead_layers = 2
                    for look_ahead in range(1, look_ahead_layers + 1):
                        next_layer_idx = current_layer_idx + look_ahead
                        
                        if next_layer_idx >= len(layer_order):
                            break
                            
                        next_layer_name = layer_order[next_layer_idx]
                        
                        if next_layer_name not in objects_by_layer:
                            continue
                        
                        # Check all objects in the next layer
                        for next_obj in objects_by_layer[next_layer_name]:
                            # Find unassigned points from next layer object
                            next_available_points = [p for p in next_obj.positive_points 
                                                if (p[0], p[1]) not in assigned_points]
                            
                            if not next_available_points:
                                continue
                            
                            # Check which points fall inside current segment
                            points_inside = []
                            for point in next_available_points:
                                x, y = point
                                if test_segment.mask[y, x]:  # Point inside segment
                                    points_inside.append(point)
                            
                            if points_inside:
                                print(f"    Found {len(points_inside)} points from {next_obj.object_id} in {next_layer_name}")
                                cross_layer_points.extend(points_inside)
                                if next_layer_name not in involved_layers:
                                    involved_layers.append(next_layer_name)
                    
                    # If we found cross-layer points, re-segment with all points
                    if len(involved_layers) > 1 or len(cross_layer_points) > len(available_points):
                        print(f"    Re-segmenting with {len(cross_layer_points)} points across {len(involved_layers)} layers")
                        
                        try:
                            final_segment = self.sam_processor.segment_from_point(cross_layer_points)
                            
                            # Update object with cross-layer information
                            obj.segment_mask = final_segment.mask
                            obj.confidence = final_segment.confidence
                            obj.positive_points = cross_layer_points
                            obj.bounding_box = final_segment.bounding_box
                            obj.area = final_segment.area
                            
                            # Add cross-layer metadata
                            obj.depth_layers = involved_layers
                            obj.primary_depth_layer = obj.depth_layer
                            obj.is_cross_layer = len(involved_layers) > 1
                            
                            # Update depth range to span all involved layers
                            if len(involved_layers) > 1:
                                involved_layer_configs = [layer for layer in self.depth_result.layers 
                                                        if layer.config.label in involved_layers]
                                min_depth = min(layer.config.min_depth for layer in involved_layer_configs)
                                max_depth = max(layer.config.max_depth for layer in involved_layer_configs)
                                obj.depth_range = (min_depth, max_depth)
                                obj.depth_layer = " + ".join(involved_layers)
                            
                            print(f"    Updated {obj.object_id}: {obj.depth_layer} ({len(cross_layer_points)} points)")
                            
                        except Exception as e:
                            print(f"    Error re-segmenting: {e}, keeping original")
                            cross_layer_points = available_points  # Fallback to original points
                    
                    # Mark all used points as assigned
                    for point in cross_layer_points:
                        assigned_points.add((point[0], point[1]))
                    
                    # Update object with final points
                    obj.positive_points = cross_layer_points
                    
                    # Add to updated objects list
                    updated_objects.append(obj)
                    print(f"    Finalized {obj.object_id} with {len(cross_layer_points)} points")
                    
                except Exception as e:
                    print(f"    Error processing {obj.object_id}: {e}")
                    continue
        
        # Replace objects list with updated objects (this automatically removes deleted objects)
        deleted_count = len(self.objects) - len(updated_objects)
        self.objects = updated_objects
        
        print(f"\n=== Cross-depth integration complete ===")
        print(f"Final objects: {len(self.objects)} (deleted {deleted_count} objects)")
        
        # Print summary of final objects
        cross_layer_count = 0
        for obj in self.objects:
            if hasattr(obj, 'is_cross_layer') and obj.is_cross_layer:
                cross_layer_count += 1
                print(f"  {obj.object_id}: Cross-layer spanning {obj.depth_layers} ({len(obj.positive_points)} points)")
            else:
                print(f"  {obj.object_id}: Single-layer in {obj.depth_layer} ({len(obj.positive_points)} points)")
        
        print(f"Cross-layer objects: {cross_layer_count}/{len(self.objects)}")


    def step4_final_segmentation_pass(self):
        """Step 4: Final segmentation pass for each object using all positive points"""
        print("\n=== Step 4: Final segmentation pass ===")
        
        for obj in self.objects:
            print(f"Final segmentation for {obj.object_id}")
            
            # Use the primary layer's neon-masked RGB for final segmentation
            primary_layer = getattr(obj, 'primary_depth_layer', obj.depth_layer.split(' + ')[0])
            masked_rgb = self.neon_masked_frames[primary_layer]
            self.sam_processor.set_image(masked_rgb)
            
            try:
                # Segment with all positive points (potentially from multiple layers)
                final_segment = self.sam_processor.segment_from_point(obj.positive_points)
                
                # Update object with final segmentation
                obj.segment_mask = final_segment.mask
                obj.confidence = final_segment.confidence
                obj.bounding_box = final_segment.bounding_box
                obj.area = final_segment.area
                
                print(f"  Final confidence: {final_segment.confidence:.3f}, area: {final_segment.area}")
                
                if hasattr(obj, 'is_cross_layer') and obj.is_cross_layer:
                    print(f"  Cross-layer object used {len(obj.positive_points)} points from: {obj.depth_layers}")
                
            except Exception as e:
                print(f"  Error in final segmentation: {e}")

    def create_final_visualization(self):
        """Create the final visualization panel"""
        print("\n=== Creating final visualization ===")
        
        # Create color map for objects
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.objects)))
        
        # Create segmentation overlay on original RGB
        segmented_rgb = self.original_rgb.copy()
        overlay = np.zeros_like(segmented_rgb)
        
        for i, obj in enumerate(self.objects):
            color = (colors[i][:3] * 255).astype(np.uint8)
            overlay[obj.segment_mask] = color
        
        # Blend overlay with original
        alpha = 0.6
        cv2.addWeighted(segmented_rgb, 1-alpha, overlay, alpha, 0, segmented_rgb)
        
        # Add labels and points
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        for i, obj in enumerate(self.objects):
            # Draw bounding box
            x1, y1, x2, y2 = obj.bounding_box
            color = (colors[i][:3] * 255).astype(np.uint8).tolist()
            cv2.rectangle(segmented_rgb, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{obj.object_id}\n{obj.depth_layer}\nC:{obj.confidence:.2f}"
            label_lines = label.split('\n')
            
            for j, line in enumerate(label_lines):
                y_offset = y1 - 10 - (len(label_lines) - 1 - j) * 15
                cv2.putText(segmented_rgb, line, (x1, y_offset), 
                           font, font_scale, (255, 255, 255), thickness + 1)
                cv2.putText(segmented_rgb, line, (x1, y_offset), 
                           font, font_scale, color, thickness)
            
            # Draw positive points
            for point in obj.positive_points:
                cv2.circle(segmented_rgb, point, self.point_size, (255, 255, 255), -1)
                cv2.circle(segmented_rgb, point, self.point_size - 1, color, -1)
        
        # Save final segmented image
        cv2.imwrite(os.path.join(self.output_dir, "05_final_segmented.jpg"), segmented_rgb)
        
        # Create comprehensive panel
        self._create_comprehensive_panel(segmented_rgb)
        
        # Save individual object segments
        self._save_individual_objects()
    
    def _create_comprehensive_panel(self, segmented_rgb):
        """Create comprehensive visualization panel"""
        # Calculate panel layout
        panel_width = 400
        panel_height = 300
        
        # Images to include: original RGB, depth map, segmented result, and neon masks
        total_images = 3 + len(self.neon_masked_frames)
        cols = 4
        rows = math.ceil(total_images / cols)
        
        panel_full_width = panel_width * cols
        panel_full_height = panel_height * rows
        
        # Create large panel
        panel = np.zeros((panel_full_height, panel_full_width, 3), dtype=np.uint8)
        
        # Panel 0: Original RGB
        rgb_resized = cv2.resize(self.original_rgb, (panel_width, panel_height))
        panel[0:panel_height, 0:panel_width] = rgb_resized
        cv2.putText(panel, "Original RGB", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Panel 1: Depth map
        depth_colored = cv2.applyColorMap(self.depth_result.full_depth_map, cv2.COLORMAP_VIRIDIS)
        depth_resized = cv2.resize(depth_colored, (panel_width, panel_height))
        panel[0:panel_height, panel_width:panel_width*2] = depth_resized
        cv2.putText(panel, "Depth Map", (panel_width + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Panel 2: Final segmentation
        seg_resized = cv2.resize(segmented_rgb, (panel_width, panel_height))
        panel[0:panel_height, panel_width*2:panel_width*3] = seg_resized
        cv2.putText(panel, f"Final Segments ({len(self.objects)} objects)", 
                   (panel_width*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Remaining panels: Neon-masked frames
        panel_idx = 3
        for layer_name, masked_frame in self.neon_masked_frames.items():
            row = panel_idx // cols
            col = panel_idx % cols
            
            if row < rows:  # Make sure we don't exceed panel bounds
                y_start = row * panel_height
                x_start = col * panel_width
                
                masked_resized = cv2.resize(masked_frame, (panel_width, panel_height))
                panel[y_start:y_start+panel_height, x_start:x_start+panel_width] = masked_resized
                
                cv2.putText(panel, f"Neon Masked: {layer_name}", 
                           (x_start + 10, y_start + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            panel_idx += 1
        
        # Save comprehensive panel
        cv2.imwrite(os.path.join(self.output_dir, "06_comprehensive_panel.jpg"), panel)
        print(f"Saved comprehensive panel: {panel_full_width}x{panel_full_height}")
    
    def _save_individual_objects(self):
        """Save individual object segments"""
        print("\nSaving individual object segments...")
        
        for obj in self.objects:
            # Create individual object image
            obj_image = self.original_rgb.copy()
            
            # Create mask overlay
            overlay = np.zeros_like(obj_image)
            overlay[obj.segment_mask] = (0, 255, 255)  # Yellow overlay
            
            # Blend
            cv2.addWeighted(obj_image, 0.7, overlay, 0.3, 0, obj_image)
            
            # Draw bounding box and points
            x1, y1, x2, y2 = obj.bounding_box
            cv2.rectangle(obj_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            for point in obj.positive_points:
                cv2.circle(obj_image, point, 4, (255, 255, 255), -1)
                cv2.circle(obj_image, point, 3, (0, 0, 255), -1)
            
            # Add info text
            info_text = [
                f"ID: {obj.object_id}",
                f"Layer: {obj.depth_layer}",
                f"Depth: {obj.depth_range[0]}-{obj.depth_range[1]}",
                f"Confidence: {obj.confidence:.3f}",
                f"Area: {obj.area}",
                f"Points: {len(obj.positive_points)}"
            ]
            
            for i, text in enumerate(info_text):
                y_pos = 30 + i * 20
                cv2.putText(obj_image, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(obj_image, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Save individual object
            filename = f"07_{obj.object_id}_{obj.depth_layer.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(os.path.join(self.output_dir, filename), obj_image)
        
        print(f"Saved {len(self.objects)} individual object images")
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("SLIVS DEPTH-GUIDED SEGMENTATION SUMMARY")
        print("="*60)
        print(f"Total objects detected: {len(self.objects)}")
        print(f"Output directory: {self.output_dir}")
        print()
        
        # Group by depth layer
        by_layer = {}
        for obj in self.objects:
            if obj.depth_layer not in by_layer:
                by_layer[obj.depth_layer] = []
            by_layer[obj.depth_layer].append(obj)
        
        for layer_name, layer_objects in by_layer.items():
            print(f"Layer '{layer_name}': {len(layer_objects)} objects")
            for obj in layer_objects:
                print(f"  {obj.object_id}: confidence={obj.confidence:.3f}, "
                      f"area={obj.area}, points={len(obj.positive_points)}")
        
        print(f"\nFiles saved in: {os.path.abspath(self.output_dir)}")
        print("\nNew files added:")
        print("  03b_original_points_[layer_name].jpg - Original points overlaid on each depth layer")
    
    def run_demo(self):
        """Run the complete demo pipeline"""
        try:
            print("=== SLIVS DEPTH-GUIDED SEGMENTATION DEMO ===")
            
            # Start timing the entire process
            demo_start_time = time.time()
            
            # Capture frame
            capture_start = time.time()
            frame = self.capture_single_frame()
            capture_time = time.time() - capture_start
            print(f"Frame capture time: {capture_time:.2f}s")
            
            # Step 1: Generate depth and points
            step1_start = time.time()
            self.step1_generate_depth_and_points(frame)
            step1_time = time.time() - step1_start
            print(f"Step 1 (depth processing) time: {step1_time:.2f}s")
            
            # Step 2: Create neon-masked frames
            step2_start = time.time()
            self.step2_create_neon_masked_frames()
            step2_time = time.time() - step2_start
            print(f"Step 2 (neon masking) time: {step2_time:.2f}s")
            
            # Step 3: Segment objects by layer (original)
            step3_start = time.time()
            self.step3_segment_objects_by_layer()
            step3_time = time.time() - step3_start
            print(f"Step 3 (segmentation) time: {step3_time:.2f}s")
            
            # Step 3b: Cross-integrate across depth layers (NEW)
            step3b_start = time.time()
            self.step3b_cross_integrate_cross_depth_segmentation()
            step3b_time = time.time() - step3b_start
            print(f"Step 3b (cross-depth integration) time: {step3b_time:.2f}s")
            
            # Step 4: Final segmentation pass
            step4_start = time.time()
            self.step4_final_segmentation_pass()
            step4_time = time.time() - step4_start
            print(f"Step 4 (final segmentation) time: {step4_time:.2f}s")
            
            # Create visualizations
            viz_start = time.time()
            self.create_final_visualization()
            viz_time = time.time() - viz_start
            print(f"Visualization creation time: {viz_time:.2f}s")
            
            # Calculate total time
            total_time = time.time() - demo_start_time
            
            # Print summary
            self.print_summary()
            
            # Print timing summary
            print("\n" + "="*60)
            print("TIMING SUMMARY")
            print("="*60)
            print(f"Frame capture:        {capture_time:6.2f}s")
            print(f"Depth processing:     {step1_time:6.2f}s")
            print(f"Neon masking:         {step2_time:6.2f}s")
            print(f"Segmentation:         {step3_time:6.2f}s")
            print(f"Cross-depth integration: {step3b_time:6.2f}s")  # NEW
            print(f"Final segmentation:   {step4_time:6.2f}s")
            print(f"Visualization:        {viz_time:6.2f}s")
            print("-" * 60)
            print(f"TOTAL PROCESSING TIME: {total_time:6.2f}s")
            print("="*60)
            
            print(f"\nDemo completed successfully in {total_time:.2f} seconds!")
            
        except Exception as e:
            print(f"Demo failed: {e}")
            raise

def main():
    """Main function to run the demo"""
    try:
        demo = SLIVSDepthSegmentationDemo()
        demo.run_demo()
    except Exception as e:
        print(f"Failed to run demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()