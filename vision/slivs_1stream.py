"""
Real-time camera capture + MiDaS depth map with layered visualization
Displays 6-panel window (3x2) showing full depth + 5 depth layers in real-time

https://huggingface.co/collections/Intel/dpt-31-65b2a13eb0a5a381b6df9b6b
https://huggingface.co/Intel/dpt-swinv2-large-384

Controls:
- Press 'q' to quit
- Press 's' to save current composite frame
"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time

class RealTimeDepthLayering:
    def __init__(self):
        # Easy to modify constants at the top of __init__
        self.DEPTH_LAYER_1_MIN = 0     # Furthest objects  
        self.DEPTH_LAYER_1_MAX = 25    # (0-25)
        self.DEPTH_LAYER_2_MIN = 26    # (26-50)
        self.DEPTH_LAYER_2_MAX = 50
        self.DEPTH_LAYER_3_MIN = 51    # (51-75)
        self.DEPTH_LAYER_3_MAX = 75
        self.DEPTH_LAYER_4_MIN = 76    # (76-150)
        self.DEPTH_LAYER_4_MAX = 150
        self.DEPTH_LAYER_5_MIN = 151   # Closest objects
        self.DEPTH_LAYER_5_MAX = 255   # (151-255)
        
        # Display settings
        self.PANEL_WIDTH = 320   # Width of each panel
        self.PANEL_HEIGHT = 240  # Height of each panel
        self.FONT_SCALE = 0.6
        self.FONT_THICKNESS = 1
        
        # Initialize camera
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera")
        
        # Set camera resolution (optional - adjust as needed)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load MiDaS model
        print("Loading MiDaS model...")
        self.model_name = "Intel/dpt-swinv2-tiny-256"  # Fast model for real-time
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        
        print(f"Using device: {self.device}")
        print("System ready! Press 'q' to quit, 's' to save frame")
    
    def process_depth(self, frame):
        """Process frame through MiDaS and return normalized depth map"""
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
    
    def create_depth_layers(self, depth_normalized):
        """Create 5 depth layer masks based on defined ranges"""
        layers = []
        layer_configs = [
            (self.DEPTH_LAYER_1_MIN, self.DEPTH_LAYER_1_MAX, f"Layer 1: {self.DEPTH_LAYER_1_MIN}-{self.DEPTH_LAYER_1_MAX}"),
            (self.DEPTH_LAYER_2_MIN, self.DEPTH_LAYER_2_MAX, f"Layer 2: {self.DEPTH_LAYER_2_MIN}-{self.DEPTH_LAYER_2_MAX}"),
            (self.DEPTH_LAYER_3_MIN, self.DEPTH_LAYER_3_MAX, f"Layer 3: {self.DEPTH_LAYER_3_MIN}-{self.DEPTH_LAYER_3_MAX}"),
            (self.DEPTH_LAYER_4_MIN, self.DEPTH_LAYER_4_MAX, f"Layer 4: {self.DEPTH_LAYER_4_MIN}-{self.DEPTH_LAYER_4_MAX}"),
            (self.DEPTH_LAYER_5_MIN, self.DEPTH_LAYER_5_MAX, f"Layer 5: {self.DEPTH_LAYER_5_MIN}-{self.DEPTH_LAYER_5_MAX}")
        ]
        
        for min_val, max_val, label in layer_configs:
            # Create layer mask
            layer_depth = depth_normalized.copy()
            mask = (layer_depth < min_val) | (layer_depth > max_val)
            layer_depth[mask] = 0
            layers.append((layer_depth, label))
        
        return layers
    
    def create_composite_display(self, depth_normalized, layers):
        """Create 3x2 composite display with full depth + 5 layers"""
        # Resize depth map to panel size
        depth_resized = cv2.resize(depth_normalized, (self.PANEL_WIDTH, self.PANEL_HEIGHT))
        
        # Create composite image: 2 rows x 3 columns
        composite_height = self.PANEL_HEIGHT * 2
        composite_width = self.PANEL_WIDTH * 3
        composite = np.zeros((composite_height, composite_width), dtype=np.uint8)
        
        # Panel positions (row, col)
        positions = [
            (0, 0),                              # Full depth - top left
            (0, self.PANEL_WIDTH),               # Layer 1 - top middle  
            (0, self.PANEL_WIDTH * 2),           # Layer 2 - top right
            (self.PANEL_HEIGHT, 0),              # Layer 3 - bottom left
            (self.PANEL_HEIGHT, self.PANEL_WIDTH), # Layer 4 - bottom middle
            (self.PANEL_HEIGHT, self.PANEL_WIDTH * 2) # Layer 5 - bottom right
        ]
        
        # Place full depth map (top-left)
        row, col = positions[0]
        composite[row:row+self.PANEL_HEIGHT, col:col+self.PANEL_WIDTH] = depth_resized
        cv2.putText(composite, "Full Depth", (col + 5, row + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, (255, 255, 255), self.FONT_THICKNESS)
        
        # Place depth layers
        for i, ((layer_depth, label), (row, col)) in enumerate(zip(layers, positions[1:])):
            # Resize layer to panel size
            layer_resized = cv2.resize(layer_depth, (self.PANEL_WIDTH, self.PANEL_HEIGHT))
            
            # Place layer in composite
            composite[row:row+self.PANEL_HEIGHT, col:col+self.PANEL_WIDTH] = layer_resized
            
            # Add label
            cv2.putText(composite, label, (col + 5, row + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, (255, 255, 255), self.FONT_THICKNESS)
        
        return composite
    
    def save_current_frame(self, composite):
        """Save current composite frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"depth_layers_realtime_{timestamp}.jpg"
        cv2.imwrite(filename, composite)
        print(f"Frame saved as: {filename}")
    
    def run(self):
        """Main real-time processing loop"""
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0.0
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process depth
                depth_normalized = self.process_depth(frame)
                
                # Create depth layers
                layers = self.create_depth_layers(depth_normalized)
                
                # Create composite display
                composite = self.create_composite_display(depth_normalized, layers)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    current_fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    print(f"FPS: {current_fps:.1f}")
                
                # Add FPS to display (use actual calculated FPS)
                cv2.putText(composite, f"FPS: {current_fps:.1f}", 
                           (composite.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display composite
                cv2.imshow('Real-time Depth Layers', composite)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_current_frame(composite)
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")

def main():
    print("=== Real-time MiDaS Depth Layering System ===")
    
    try:
        depth_system = RealTimeDepthLayering()
        depth_system.run()
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()