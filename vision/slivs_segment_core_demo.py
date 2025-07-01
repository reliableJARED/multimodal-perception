#!/usr/bin/env python3
"""
Simple SLIVS SAM2 Camera Demo
Real-time camera segmentation with fixed point prompt
"""

import cv2
import numpy as np
import time

# Import SLIVS modules
from slivs_segment_core import SLIVSSam2Processor, SAM2Config


def create_visualization(frame, sam2_result, current_point):
    """Create simple visualization of segments"""
    vis = frame.copy()
    
    # Colors for segments
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
    ]
    
    if sam2_result.segments:
        for i, segment in enumerate(sam2_result.segments):
            color = colors[i % len(colors)]
            
            # Fill segment with semi-transparent color
            mask_colored = np.zeros_like(vis)
            mask_colored[segment.mask] = color
            vis = cv2.addWeighted(vis, 1.0, mask_colored, 0.3, 0)
            
            # Draw segment boundary
            contours, _ = cv2.findContours(
                segment.mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, color, 2)
            
            # Show confidence
            x1, y1, x2, y2 = segment.bounding_box
            conf_text = f"Conf: {segment.confidence:.2f}"
            cv2.putText(vis, conf_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw current point prompt (larger and more visible)
    cv2.circle(vis, current_point, 8, (255, 255, 255), -1)
    cv2.circle(vis, current_point, 9, (0, 0, 0), 2)
    
    return vis


def main():
    print("=== SLIVS SAM2 Camera Demo ===")
    print("Initializing SAM2...")
    
    # Initialize SAM2 processor with tiny model
    config = SAM2Config(model_name="sam2.1_hiera_tiny")
    processor = SLIVSSam2Processor(config)
    
    print("Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # 10 fixed point prompts (x: 300-1200, y: 400-800)
    point_prompts = [
        (400, 400),
        (500, 500),
        (600, 600),
        (700, 700),
        (800, 500),
        (900, 600),
        (1000, 700),
        (1100, 500),
        (350, 450),
        (450, 750)
    ]
    
    print(f"Using {len(point_prompts)} rotating point prompts")
    print("Press 'q' to quit")
    
    frame_count = 0
    total_time = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            
            # Get current point prompt (rotate every 10 frames)
            point_index = ((frame_count - 1) // 10) % len(point_prompts)
            current_point = point_prompts[point_index]
            
            # Set image and run segmentation
            processor.set_image(frame)
            
            start_time = time.time()
            try:
                sam2_result = processor.segment_from_points([current_point])
                processing_time = time.time() - start_time
                total_time += processing_time
                
                # Create visualization
                vis_frame = create_visualization(frame, sam2_result, current_point)
                
                # Add FPS info
                fps = 1.0 / processing_time if processing_time > 0 else 0
                fps_text = f"FPS: {fps:.1f}"
                segments_text = f"Segments: {len(sam2_result.segments)}"
                point_text = f"Point: {current_point}"
                
                cv2.putText(vis_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, segments_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(vis_frame, point_text, (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Print progress
                if frame_count % 30 == 0:
                    avg_fps = frame_count / total_time
                    print(f"Frame {frame_count}: Avg FPS: {avg_fps:.1f}, Current point: {current_point}")
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                vis_frame = frame.copy()
                cv2.putText(vis_frame, f"Error: {str(e)}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.circle(vis_frame, current_point, 5, (255, 255, 255), -1)
            
            # Show result
            cv2.imshow('SLIVS SAM2 Demo', vis_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nDemo interrupted")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if frame_count > 0:
            avg_fps = frame_count / total_time
            print(f"\nDemo complete:")
            print(f"  Frames processed: {frame_count}")
            print(f"  Average FPS: {avg_fps:.1f}")


if __name__ == "__main__":
    main()