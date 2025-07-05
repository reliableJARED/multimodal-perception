#!/usr/bin/env python3
"""
SLIVS Real-Time Segmentation Demo
Demonstrates the SLIVSSam2Processor with live camera feed and point-based segmentation.

Process:
1. Capture and warm up camera
2. Use default point at center of image as initial prompt (10 frames)
3. Use cluster of 3 points all 10 pixels apart at center (10 frames)
4. Keep looping between single point and 3-point cluster
5. Display real-time video feed with segmented overlay, points, confidence, and bounding box
"""

import cv2
import numpy as np
import time
import sys
import os
from typing import List, Tuple, Optional

# Import your SLIVS processor
from slivs_segment_core import SLIVSSam2Processor, SAM2Config, SegmentResult


class SLIVSRealtimeDemo:
    """Real-time segmentation demo using SLIVS SAM2 processor"""
    
    def __init__(self, camera_id: int = 0):
        """Initialize the demo with camera and processor"""
        self.camera_id = camera_id
        self.cap = None
        self.processor = None
        self.frame_count = 0
        self.current_mode = "single_point"  # "single_point" or "three_points"
        self.mode_frame_count = 0
        self.frames_per_mode = 10
        
        # Display settings
        self.overlay_alpha = 0.3
        self.point_radius = 5
        self.point_color = (0, 255, 0)  # Green for points
        self.bbox_color = (255, 0, 0)   # Red for bounding box
        self.overlay_color = (255, 0, 0)  # Blue for segment overlay
        
        # Performance tracking
        self.fps_history = []
        self.processing_times = []
        
    def setup_camera(self):
        """Initialize and warm up the camera"""
        print("Starting camera...")
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Warm up camera - discard first 10 frames
        print("Warming up camera (discarding 10 frames)...")
        for i in range(10):
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError(f"Failed to capture warmup frame {i}")
            print(f"Warmup frame {i+1}/10")
        
        print("Camera warmup complete!")
        
    def setup_processor(self):
        """Initialize the SLIVS SAM2 processor"""
        print("Initializing SLIVS SAM2 processor...")
        
        # Use tiny model for real-time performance
        config = SAM2Config(
            model_name="sam2.1_hiera_tiny",
            device=None,  # Auto-detect best device
            multimask_output=True,
            min_mask_confidence=0.5,
            use_highest_confidence=True
        )
        
        self.processor = SLIVSSam2Processor(config)
        print("SLIVS processor initialized!")
        
    def get_current_points(self, frame_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get the current point prompts based on mode and frame shape"""
        height, width = frame_shape[:2]
        center_x, center_y = width // 2, height // 2
        
        if self.current_mode == "single_point":
            return [(center_x, center_y)]
        else:  # three_points in L shape
            return [
                (center_x, center_y),           # Center point
                (center_x + 25, center_y),      # Right point (horizontal arm)
                (center_x, center_y + 25)       # Bottom point (vertical arm)
            ]
    
    def update_mode(self):
        """Update the segmentation mode (single point vs three points)"""
        self.mode_frame_count += 1
        
        if self.mode_frame_count >= self.frames_per_mode:
            # Switch modes
            if self.current_mode == "single_point":
                self.current_mode = "three_points"
                print(f"Frame {self.frame_count}: Switching to 3-point L-shape mode")
            else:
                self.current_mode = "single_point"
                print(f"Frame {self.frame_count}: Switching to single point mode")
            
            self.mode_frame_count = 0
    
    def draw_points(self, frame: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw the point prompts on the frame"""
        result = frame.copy()
        
        for i, (x, y) in enumerate(points):
            # Draw point circle
            cv2.circle(result, (x, y), self.point_radius, self.point_color, -1)
            # Draw point border
            cv2.circle(result, (x, y), self.point_radius + 1, (0, 0, 0), 1)
            # Draw point number
            cv2.putText(result, str(i+1), (x-5, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def draw_bounding_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Draw the bounding box on the frame"""
        result = frame.copy()
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(result, (x1, y1), (x2, y2), self.bbox_color, 2)
        
        # Draw corner markers
        corner_size = 10
        cv2.line(result, (x1, y1), (x1 + corner_size, y1), self.bbox_color, 3)
        cv2.line(result, (x1, y1), (x1, y1 + corner_size), self.bbox_color, 3)
        cv2.line(result, (x2, y1), (x2 - corner_size, y1), self.bbox_color, 3)
        cv2.line(result, (x2, y1), (x2, y1 + corner_size), self.bbox_color, 3)
        cv2.line(result, (x1, y2), (x1 + corner_size, y2), self.bbox_color, 3)
        cv2.line(result, (x1, y2), (x1, y2 - corner_size), self.bbox_color, 3)
        cv2.line(result, (x2, y2), (x2 - corner_size, y2), self.bbox_color, 3)
        cv2.line(result, (x2, y2), (x2, y2 - corner_size), self.bbox_color, 3)
        
        return result
    
    def draw_segment_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Draw the segmented area overlay on the frame"""
        result = frame.copy()
        
        # Create colored overlay
        overlay = np.zeros_like(frame)
        overlay[mask] = self.overlay_color
        
        # Blend overlay with original frame
        cv2.addWeighted(result, 1 - self.overlay_alpha, overlay, self.overlay_alpha, 0, result)
        
        # Draw segment outline
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, self.overlay_color, 2)
        
        return result
    
    def draw_info_panel(self, frame: np.ndarray, segment_result: SegmentResult, 
                       points: List[Tuple[int, int]], fps: float, processing_time: float) -> np.ndarray:
        """Draw information panel with statistics"""
        result = frame.copy()
        
        # Panel background
        panel_height = 120
        panel_color = (0, 0, 0)
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (result.shape[1], panel_height), panel_color, -1)
        cv2.addWeighted(result, 0.7, overlay, 0.3, 0, result)
        
        # Text settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (255, 255, 255)
        line_height = 18
        
        # Information text
        texts = [
            f"Frame: {self.frame_count}  Mode: {self.current_mode}  Mode Frame: {self.mode_frame_count + 1}/{self.frames_per_mode}",
            f"FPS: {fps:.1f}  Processing Time: {processing_time*1000:.1f}ms",
            f"Points: {len(points)}  Confidence: {segment_result.confidence:.3f}",
            f"Segment Area: {segment_result.area} pixels",
            f"Bounding Box: {segment_result.bounding_box}",
            f"Segment ID: {segment_result.segment_id[:8]}..."
        ]
        
        for i, text in enumerate(texts):
            y = 15 + i * line_height
            cv2.putText(result, text, (10, y), font, font_scale, text_color, 1)
        
        return result
    
    def draw_confidence_above_points(self, frame: np.ndarray, points: List[Tuple[int, int]], 
                                   confidence: float) -> np.ndarray:
        """Draw confidence value above the points"""
        result = frame.copy()
        
        if not points:
            return result
            
        # Find the topmost point
        top_point = min(points, key=lambda p: p[1])
        x, y = top_point
        
        # Draw confidence text above the topmost point
        confidence_text = f"Conf: {confidence:.3f}"
        text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        
        # Background rectangle for text
        padding = 5
        rect_x1 = x - text_size[0] // 2 - padding
        rect_y1 = y - 35 - padding
        rect_x2 = x + text_size[0] // 2 + padding
        rect_y2 = y - 35 + text_size[1] + padding
        
        cv2.rectangle(result, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.rectangle(result, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 255, 255), 1)
        
        # Draw text
        cv2.putText(result, confidence_text, (x - text_size[0] // 2, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame and return the visualization"""
        try:
            start_time = time.time()
            
            # Get current points based on mode
            points = self.get_current_points(frame.shape)
            
            # Set image in processor
            self.processor.set_image(frame)
            
            # Segment from points
            segment_result = self.processor.segment_from_point(points)
            
            # Check if segmentation was successful
            if segment_result.confidence < self.processor.config.min_mask_confidence:
                print(f"Frame {self.frame_count}: Low confidence segment ({segment_result.confidence:.3f})")
                return None
            
            processing_time = time.time() - start_time
            
            # Calculate FPS
            current_fps = 1.0 / processing_time if processing_time > 0 else 0
            self.fps_history.append(current_fps)
            if len(self.fps_history) > 30:  # Keep last 30 FPS measurements
                self.fps_history.pop(0)
            avg_fps = np.mean(self.fps_history)
            
            # Create visualization
            result = frame.copy()
            
            # Draw segment overlay
            result = self.draw_segment_overlay(result, segment_result.mask)
            
            # Draw points
            result = self.draw_points(result, points)
            
            # Draw confidence above points
            result = self.draw_confidence_above_points(result, points, segment_result.confidence)
            
            # Draw bounding box
            result = self.draw_bounding_box(result, segment_result.bounding_box)
            
            # Draw info panel
            result = self.draw_info_panel(result, segment_result, points, avg_fps, processing_time)
            
            return result
            
        except Exception as e:
            print(f"Error processing frame {self.frame_count}: {e}")
            return None
    
    def run(self):
        """Main demo loop"""
        try:
            # Setup
            self.setup_camera()
            self.setup_processor()
            
            print("Starting real-time demo...")
            print("Press 'q' to quit, 'r' to reset modes, 's' to save screenshot")
            
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                self.frame_count += 1
                
                # Process frame
                result = self.process_frame(frame)
                
                if result is not None:
                    # Display result
                    cv2.imshow('SLIVS Real-Time Segmentation Demo', result)
                else:
                    # Display original frame if processing failed
                    cv2.imshow('SLIVS Real-Time Segmentation Demo', frame)
                
                # Update mode
                self.update_mode()
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user")
                    break
                elif key == ord('r'):
                    print("Resetting modes...")
                    self.current_mode = "single_point"
                    self.mode_frame_count = 0
                elif key == ord('s'):
                    if result is not None:
                        filename = f"slivs_screenshot_{int(time.time())}.png"
                        cv2.imwrite(filename, result)
                        print(f"Screenshot saved as {filename}")
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Error in demo: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Demo cleanup complete")


def main():
    """Main function to run the demo"""
    try:
        demo = SLIVSRealtimeDemo(camera_id=0)
        demo.run()
    except Exception as e:
        print(f"Failed to run demo: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()