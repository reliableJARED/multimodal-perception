"""
SLIVS Depth Layer Processor - Modular Pipeline Component
Provides depth estimation, layer creation, and point detection for SAM integration

@article{DBLP:journals/corr/abs-2103-13413,
  author    = {Ren{\'{e}} Reiner Birkl, Diana Wofk, Matthias Muller},
  title     = {MiDaS v3.1 - A Model Zoo for Robust Monocular Relative Depth Estimation},
  journal   = {CoRR},
  volume    = {abs/2307.14460},
  year      = {2021},
  url       = {https://arxiv.org/abs/2307.14460},
  eprinttype = {arXiv},
  eprint    = {2307.14460},
  timestamp = {Wed, 26 Jul 2023},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2307-14460.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time
import math
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from slivs_depth_core import SLIVSDepthProcessor, DepthProcessingResult, DepthLayerConfig



# Visualization utilities (separate from core processing)
class SLIVSVisualizer:
    """
    Separate visualization class for displaying SLIVS results.
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
        return layer_colored
    
    def create_composite_display(self, result: DepthProcessingResult) -> np.ndarray:
        """Create a composite display of all layers."""
        total_panels = len(result.layers) + 1  # +1 for full depth

        
        # Calculate layout
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
        
        # Place full depth map
        depth_resized = cv2.resize(result.full_depth_map, (self.panel_width, self.panel_height))
        depth_colored = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
        composite[0:self.panel_height, 0:self.panel_width] = depth_colored
        cv2.putText(composite, "Full Depth", (5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness)
        
        # Place layers
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
            
            # Add labels
            label = f"{layer.config.label} ({len(layer.points)}pts)"
            cv2.putText(composite, label, (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255), self.font_thickness)
            
            panel_idx += 1
        
        return composite


# Demo/testing class
class SLIVSDemo:
    """
    Demo class for testing the modular SLIVS processor.
    """
    
    def __init__(self, processor: SLIVSDepthProcessor, visualizer: SLIVSVisualizer):
        self.processor = processor
        self.visualizer = visualizer
        self.cap = None
    
    def run_camera_demo(self):
        """Run live camera demo."""
        print("Starting SLIVS camera demo...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0.0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                #uncomment when using the synchronized dual-camera
                height, width = frame.shape[:2] #get frame shape
                frame = frame[:, :width//2] # Keep all rows, crop columns 0-1279, so we only use the 'right' camera on the synchronized dual-cam rig

                # Process frame
                result = self.processor.process_frame(frame)
                
                # Create visualization
                composite = self.visualizer.create_composite_display(result)
                
                # Calculate FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    current_fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    
                    total_points = len(result.all_points)
                    print(f"FPS: {current_fps:.1f}, Processing: {result.processing_time*1000:.1f}ms, "
                          f"Total Points: {total_points}")
                
                # Add FPS to display
                cv2.putText(composite, f"FPS: {current_fps:.1f}", 
                           (composite.shape[1] - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('SLIVS Modular Demo', composite)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("Demo interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Demo cleanup complete")


def main():
    """Main demo function."""
    print("=== SLIVS Modular Depth Processor Demo ===")
    
    # Initialize processor
    processor = SLIVSDepthProcessor(
        model_name="Intel/dpt-swinv2-tiny-256",
        target_squares=100,
        min_fill_threshold=0.7
    )

    #Overide default 5 depth layers
    # Default depth layer configuration
    depth_layers = {
            "furthest": DepthLayerConfig(0, 36, "Furthest"),
            "far": DepthLayerConfig(37, 73, "Far"),
            "lessfar": DepthLayerConfig(74, 109, "Less Far"),
            "mid": DepthLayerConfig(110, 145, "Mid"),
            "midnear": DepthLayerConfig(146, 181, "Mid Near"),
            "close": DepthLayerConfig(182, 217, "Close"),
            "closest": DepthLayerConfig(218, 255, "Closest"),
        }
    processor.update_depth_layers(depth_layers)
    # Initialize visualizer
    visualizer = SLIVSVisualizer()
    
    # Run demo
    demo = SLIVSDemo(processor, visualizer)
    demo.run_camera_demo()


if __name__ == "__main__":
    main()