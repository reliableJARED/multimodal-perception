"""
Real-time camera capture + MiDaS depth map demo
Opens camera, processes depth in real-time, displays 6 panels live
Original frame + 5 depth sections with erosion

https://huggingface.co/collections/Intel/dpt-31-65b2a13eb0a5a381b6df9b6b
https://huggingface.co/Intel/dpt-swinv2-large-384

Controls:
- Press 'q' to quit
- Press 's' to save current frame composite
"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time

class RealTimeDepthProcessor:
    def __init__(self):
        # Load MiDaS model
        print("Loading MiDaS model...")
        #self.model_name = "Intel/dpt-swinv2-large-384" # Better quality but slower
        self.model_name = "Intel/dpt-swinv2-tiny-256" # Faster for real-time
        
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        
        # Create erosion kernel
        self.erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Define depth ranges
        self.ranges = [
            (0, 50, "0-50"),
            (51, 100, "51-100"), 
            (101, 150, "101-150"),
            (151, 200, "151-200"),
            (201, 255, "201-255")
        ]
        
        print(f"Model loaded on {self.device}")
    
    def process_depth(self, frame):
        """Process frame and return depth map"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Resize to original dimensions
        height, width = frame.shape[:2]
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to 0-255 range
        depth_map = prediction.squeeze().cpu().numpy()
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_normalized
    
    def create_depth_sections(self, depth_normalized):
        """Create 5 eroded depth sections"""
        sections = []
        
        for min_val, max_val, range_name in self.ranges:
            # Create a copy of the depth map
            section_depth = depth_normalized.copy()
            
            # Create mask for pixels outside the current range
            mask = (section_depth < min_val) | (section_depth > max_val)
            
            # Set pixels outside range to black (0)
            section_depth[mask] = 0
            
            # Create binary mask of non-zero pixels for erosion
            binary_mask = (section_depth > 0).astype(np.uint8) * 255
            
            # Apply erosion to remove thin outlines and small artifacts
            eroded_mask = cv2.erode(binary_mask, self.erosion_kernel, iterations=2)
            
            # Apply eroded mask back to the section
            section_depth[eroded_mask == 0] = 0
            
            sections.append(section_depth)
        
        return sections
    
    def create_display_composite(self, original_frame, depth_normalized, depth_sections):
        """Create composite display with original + depth + 5 sections"""
        height, width = original_frame.shape[:2]
        
        # Create composite image: 2 rows x 3 columns
        composite_height = height * 2
        composite_width = width * 3
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
        # Convert depth images to 3-channel for display
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        # All images to display
        display_images = [original_frame, depth_colored] + [cv2.applyColorMap(section, cv2.COLORMAP_JET) for section in depth_sections]
        
        # Labels for each section
        labels = ["Original", "Full Depth", "0-50", "51-100", "101-150", "151-200", "201-255"]
        
        # Positions in 2x3 grid (we'll use 6 out of 6 spots)
        positions = [
            (0, 0),           # Original - top left
            (0, width),       # Full depth - top middle  
            (0, width*2),     # 0-50 - top right
            (height, 0),      # 51-100 - bottom left
            (height, width),  # 101-150 - bottom middle
            (height, width*2) # 151-200 - bottom right
        ]
        
        # Place first 6 images (original + depth + first 4 sections)
        for i in range(6):
            if i < len(display_images):
                row, col = positions[i]
                composite[row:row+height, col:col+width] = display_images[i]
                
                # Add text label
                cv2.putText(composite, labels[i], (col + 10, row + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return composite
    
    def run_realtime(self):
        """Main real-time processing loop"""
        print("=== Real-time MiDaS Depth Processing ===")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        fps_timer = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                start_time = time.time()
                
                # Process depth
                depth_normalized = self.process_depth(frame)
                
                # Create depth sections with erosion
                depth_sections = self.create_depth_sections(depth_normalized)
                
                # Create display composite
                display_composite = self.create_display_composite(frame, depth_normalized, depth_sections)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_timer)
                    fps_timer = current_time
                    print(f"FPS: {fps:.1f}")
                
                # Add processing time to display
                process_time = time.time() - start_time
                cv2.putText(display_composite, f"Process: {process_time*1000:.1f}ms", 
                           (10, display_composite.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display composite
                cv2.imshow('Real-time Depth Sections', display_composite)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"realtime_depth_composite_{timestamp}.jpg"
                    cv2.imwrite(filename, display_composite)
                    print(f"Saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    processor = RealTimeDepthProcessor()
    processor.run_realtime()

if __name__ == "__main__":
    main()