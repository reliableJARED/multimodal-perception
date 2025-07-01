#!/usr/bin/env python3
"""
SLIVS Real-Time Testing Script
Integrates depth and segmentation cores for live frame analysis

This script provides:
- Real-time camera processing
- Depth estimation with layer visualization
- SAM2 segmentation using depth-derived points
- Combined visualization display
- Performance monitoring
- Interactive controls
"""

import cv2
import numpy as np
import time
import argparse
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os

# Import SLIVS modules
from slivs_sam2_core import SLIVSSam2Processor, SAM2Config
from slivs_depth_core import SLIVSDepthProcessor, DepthLayerConfig


@dataclass
class SLIVSTestConfig:
    """Configuration for SLIVS testing"""
    # Camera settings
    camera_id: int = 0
    frame_width: int = 640
    frame_height: int = 480
    
    # Processing settings
    target_fps: float = 30.0
    max_points_per_frame: int = 50
    
    # SAM2 settings
    sam2_model: str = "sam2.1_hiera_tiny"  # tiny, small, base_plus, large
    sam2_confidence_threshold: float = 0.6
    use_multimask: bool = True
    
    # Depth settings
    depth_model: str = "Intel/dpt-swinv2-tiny-256"
    target_squares: int = 100
    min_fill_threshold: float = 0.7
    
    # Visualization settings
    show_depth_layers: bool = True
    show_segments: bool = True
    show_points: bool = True
    overlay_alpha: float = 0.6
    
    # Performance settings
    device: Optional[str] = None  # None for auto-detection
    enable_performance_logging: bool = True


class SLIVSVisualizer:
    """Handles visualization of SLIVS processing results"""
    
    def __init__(self, config: SLIVSTestConfig):
        self.config = config
        
        # Color palettes
        self.layer_colors = {
            "furthest": (64, 64, 128),      # Dark blue
            "far": (64, 128, 128),          # Teal
            "mid": (128, 128, 64),          # Olive
            "near": (128, 64, 64),          # Brown
            "closest": (128, 64, 128),      # Purple
        }
        
        self.point_color = (0, 255, 255)    # Yellow
        self.segment_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 255, 0), (255, 128, 0), (128, 0, 255),
            (0, 128, 255), (255, 0, 128), (0, 255, 128)
        ]
    
    def create_depth_visualization(self, depth_result) -> np.ndarray:
        """Create visualization of depth layers and points"""
        height, width = depth_result.full_depth_map.shape
        vis = np.zeros((height, width, 3), dtype=np.uint8)
        
        if self.config.show_depth_layers:
            # Visualize depth map as heatmap
            depth_colored = cv2.applyColorMap(depth_result.full_depth_map, cv2.COLORMAP_VIRIDIS)
            vis = cv2.addWeighted(vis, 0.3, depth_colored, 0.7, 0)
        
        if self.config.show_points:
            # Draw points from each layer
            for i, layer in enumerate(depth_result.layers):
                color = list(self.layer_colors.values())[i % len(self.layer_colors)]
                for point in layer.points:
                    cv2.circle(vis, point, 3, color, -1)
                    cv2.circle(vis, point, 4, (255, 255, 255), 1)
        
        return vis
    
    def create_segment_visualization(self, frame: np.ndarray, sam2_result) -> np.ndarray:
        """Create visualization of SAM2 segments"""
        vis = frame.copy()
        
        if not self.config.show_segments or not sam2_result.segments:
            return vis
        
        # Create overlay for segments
        overlay = vis.copy()
        
        for i, segment in enumerate(sam2_result.segments):
            color = self.segment_colors[i % len(self.segment_colors)]
            
            # Fill segment with color
            mask_colored = np.zeros_like(overlay)
            mask_colored[segment.mask] = color
            overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.3, 0)
            
            # Draw segment boundary
            contours, _ = cv2.findContours(
                segment.mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, color, 2)
            
            # Add segment info
            x, y, w, h = segment.bounding_box
            info_text = f"ID:{segment.segment_id[:8]}..."
            conf_text = f"Conf:{segment.confidence:.2f}"
            area_text = f"Area:{segment.area}"
            
            cv2.putText(overlay, info_text, (x, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(overlay, conf_text, (x, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            cv2.putText(overlay, area_text, (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw point prompt
            cv2.circle(overlay, segment.point_prompt, 5, self.point_color, -1)
            cv2.circle(overlay, segment.point_prompt, 6, (0, 0, 0), 1)
        
        return overlay
    
    def create_composite_display(self, frame: np.ndarray, depth_result, 
                               sam2_result, stats: Dict) -> np.ndarray:
        """Create comprehensive display with all visualizations"""
        
        # Create individual visualizations
        depth_vis = self.create_depth_visualization(depth_result)
        segment_vis = self.create_segment_visualization(frame, sam2_result)
        
        # Resize for display
        height = frame.shape[0]
        width = frame.shape[1]
        
        # Create composite layout (2x2 grid)
        composite_height = height * 2
        composite_width = width * 2
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Place visualizations
        composite[0:height, 0:width] = frame  # Top-left: original
        composite[0:height, width:2*width] = depth_vis  # Top-right: depth
        composite[height:2*height, 0:width] = segment_vis  # Bottom-left: segments
        
        # Bottom-right: statistics panel
        stats_panel = self.create_stats_panel(width, height, stats, depth_result, sam2_result)
        composite[height:2*height, width:2*width] = stats_panel
        
        return composite
    
    def create_stats_panel(self, width: int, height: int, stats: Dict, 
                          depth_result, sam2_result) -> np.ndarray:
        """Create statistics display panel"""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Statistics text
        y_offset = 30
        line_height = 25
        
        texts = [
            f"SLIVS Real-Time Analysis",
            f"Frame: {stats.get('frame_count', 0)}",
            f"FPS: {stats.get('fps', 0.0):.1f}",
            f"",
            f"Depth Processing:",
            f"  Time: {depth_result.processing_time*1000:.1f}ms",
            f"  Total Points: {len(depth_result.all_points)}",
            f"  Layers: {len(depth_result.layers)}",
            f"",
            f"SAM2 Segmentation:",
            f"  Time: {sam2_result.processing_time*1000:.1f}ms" if hasattr(sam2_result, 'processing_time') else "  Time: N/A",
            f"  Segments: {len(sam2_result.segments)}",
            f"  Points Used: {len(sam2_result.point_prompts)}",
            f"",
            f"Layer Breakdown:",
        ]
        
        # Add layer statistics
        for layer in depth_result.layers:
            texts.append(f"  {layer.config.label}: {len(layer.points)} pts")
        
        # Add controls
        texts.extend([
            f"",
            f"Controls:",
            f"  'q' - Quit",
            f"  'p' - Toggle points",
            f"  'd' - Toggle depth",
            f"  's' - Toggle segments",
            f"  'r' - Reset view",
        ])
        
        # Draw text
        for i, text in enumerate(texts):
            y_pos = y_offset + i * line_height
            if y_pos < height - 10:
                color = (255, 255, 255) if not text.startswith("  ") else (200, 200, 200)
                font_scale = 0.6 if not text.startswith("  ") else 0.5
                cv2.putText(panel, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        return panel


class SLIVSRealTimeTest:
    """Main testing class for SLIVS real-time analysis"""
    
    def __init__(self, config: SLIVSTestConfig):
        self.config = config
        self.cap = None
        
        print("Initializing SLIVS Real-Time Test...")
        
        # Initialize depth processor
        print(f"Loading depth model: {config.depth_model}")
        self.depth_processor = SLIVSDepthProcessor(
            model_name=config.depth_model,
            target_squares=config.target_squares,
            min_fill_threshold=config.min_fill_threshold,
            device=config.device
        )
        
        # Initialize SAM2 processor
        print(f"Loading SAM2 model: {config.sam2_model}")
        sam2_config = SAM2Config(
            model_name=config.sam2_model,
            device=config.device,
            multimask_output=config.use_multimask,
            min_mask_confidence=config.sam2_confidence_threshold,
            use_highest_confidence=True
        )
        self.sam2_processor = SLIVSSam2Processor(sam2_config)
        
        # Initialize visualizer
        self.visualizer = SLIVSVisualizer(config)
        
        # Statistics
        self.stats = {
            'frame_count': 0,
            'fps': 0.0,
            'total_processing_time': 0.0,
            'depth_processing_time': 0.0,
            'sam2_processing_time': 0.0
        }
        
        print("SLIVS initialization complete!")
    
    def process_frame(self, frame: np.ndarray) -> Tuple:
        """Process a single frame through the SLIVS pipeline"""
        start_time = time.time()
        
        # Step 1: Depth processing
        depth_start = time.time()
        depth_result = self.depth_processor.process_frame(frame)
        depth_time = time.time() - depth_start
        
        # Step 2: Extract points for SAM2 (limit to max_points_per_frame)
        all_points = depth_result.all_points
        if len(all_points) > self.config.max_points_per_frame:
            # Sample points evenly across layers
            points_per_layer = self.config.max_points_per_frame // len(depth_result.layers)
            selected_points = []
            for layer in depth_result.layers:
                layer_points = layer.points[:points_per_layer]
                selected_points.extend(layer_points)
            all_points = selected_points[:self.config.max_points_per_frame]
        
        # Step 3: SAM2 segmentation
        sam2_start = time.time()
        if all_points:
            self.sam2_processor.set_image(frame)
            sam2_result = self.sam2_processor.segment_from_points(all_points)
        else:
            # Create empty result if no points
            from slivs_sam2_core import SAM2ProcessingResult
            sam2_result = SAM2ProcessingResult(
                segments=[],
                all_masks=np.zeros(frame.shape[:2], dtype=bool),
                point_prompts=[],
                processing_time=0.0,
                frame_shape=frame.shape[:2]
            )
        sam2_time = time.time() - sam2_start
        
        total_time = time.time() - start_time
        
        # Update statistics
        self.stats['depth_processing_time'] = depth_time
        self.stats['sam2_processing_time'] = sam2_time
        self.stats['total_processing_time'] = total_time
        
        return depth_result, sam2_result
    
    def run_camera_demo(self):
        """Run live camera demo with SLIVS processing"""
        print("Starting SLIVS camera demo...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Toggle point display")
        print("  'd' - Toggle depth display")
        print("  's' - Toggle segment display")
        print("  'r' - Reset view settings")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.config.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.config.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        
        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process frame
                depth_result, sam2_result = self.process_frame(frame)
                
                # Update frame count
                self.stats['frame_count'] += 1
                fps_frame_count += 1
                
                # Calculate FPS every 30 frames
                if fps_frame_count >= 30:
                    elapsed_time = time.time() - fps_start_time
                    self.stats['fps'] = fps_frame_count / elapsed_time
                    fps_start_time = time.time()
                    fps_frame_count = 0
                    
                    # Performance logging
                    if self.config.enable_performance_logging:
                        total_points = len(depth_result.all_points)
                        total_segments = len(sam2_result.segments)
                        print(f"Frame {self.stats['frame_count']}: "
                              f"FPS={self.stats['fps']:.1f}, "
                              f"Points={total_points}, "
                              f"Segments={total_segments}, "
                              f"Depth={self.stats['depth_processing_time']*1000:.1f}ms, "
                              f"SAM2={self.stats['sam2_processing_time']*1000:.1f}ms")
                
                # Create visualization
                composite = self.visualizer.create_composite_display(
                    frame, depth_result, sam2_result, self.stats
                )
                
                # Display
                cv2.imshow('SLIVS Real-Time Test', composite)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.config.show_points = not self.config.show_points
                    print(f"Point display: {'ON' if self.config.show_points else 'OFF'}")
                elif key == ord('d'):
                    self.config.show_depth_layers = not self.config.show_depth_layers
                    print(f"Depth display: {'ON' if self.config.show_depth_layers else 'OFF'}")
                elif key == ord('s'):
                    self.config.show_segments = not self.config.show_segments
                    print(f"Segment display: {'ON' if self.config.show_segments else 'OFF'}")
                elif key == ord('r'):
                    self.config.show_points = True
                    self.config.show_depth_layers = True
                    self.config.show_segments = True
                    print("Display settings reset")
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        finally:
            self.cleanup()
    
    def run_image_test(self, image_path: str):
        """Test on a single image"""
        print(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Process frame
        depth_result, sam2_result = self.process_frame(frame)
        
        # Create visualization
        composite = self.visualizer.create_composite_display(
            frame, depth_result, sam2_result, self.stats
        )
        
        # Display
        cv2.imshow('SLIVS Image Test', composite)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Print results
        print(f"\nResults:")
        print(f"  Depth points found: {len(depth_result.all_points)}")
        print(f"  Segments created: {len(sam2_result.segments)}")
        print(f"  Depth processing time: {depth_result.processing_time*1000:.1f}ms")
        print(f"  SAM2 processing time: {sam2_result.processing_time*1000:.1f}ms")
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description="SLIVS Real-Time Testing Script")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--image", type=str, help="Test on single image instead of camera")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--sam2-model", type=str, default="sam2.1_hiera_tiny",
                       choices=["sam2.1_hiera_tiny", "sam2.1_hiera_small", 
                               "sam2.1_hiera_base_plus", "sam2.1_hiera_large"],
                       help="SAM2 model size")
    parser.add_argument("--depth-model", type=str, default="Intel/dpt-swinv2-tiny-256",
                       help="MiDAS depth model")
    parser.add_argument("--max-points", type=int, default=50,
                       help="Maximum points per frame for SAM2")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"],
                       help="Processing device (auto-detect if not specified)")
    parser.add_argument("--no-logging", action="store_true",
                       help="Disable performance logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = SLIVSTestConfig(
        camera_id=args.camera,
        frame_width=args.width,
        frame_height=args.height,
        sam2_model=args.sam2_model,
        depth_model=args.depth_model,
        max_points_per_frame=args.max_points,
        device=args.device,
        enable_performance_logging=not args.no_logging
    )
    
    try:
        # Initialize test system
        test_system = SLIVSRealTimeTest(config)
        
        if args.image:
            # Single image test
            test_system.run_image_test(args.image)
        else:
            # Live camera test
            test_system.run_camera_demo()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()