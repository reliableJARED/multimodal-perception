"""
Real-time camera capture + MiDaS depth map with layered visualization + Grid-based point selection
Displays 6-panel window (3x2) showing full depth + 5 depth layers with segment points in real-time

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
import math

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
        
        # Grid and point selection settings
        self.TARGET_SQUARES = 100       # Target number of grid squares
        self.MIN_FILL_THRESHOLD = 0.7  # 70% non-black pixel threshold
        self.POINT_COLOR = (0, 255, 0) # Green points
        self.POINT_SIZE = 3            # Point radius
        
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
    
    def calculate_optimal_grid(self, height, width):
        """Calculate optimal grid size based on GCD to get close to target squares"""
        gcd = math.gcd(height, width)
        
        # Find divisors of the GCD
        divisors = []
        for i in range(1, int(math.sqrt(gcd)) + 1):
            if gcd % i == 0:
                divisors.append(i)
                if i != gcd // i:
                    divisors.append(gcd // i)
        
        divisors.sort()
        
        # Find the divisor that gives us closest to target squares
        best_divisor = divisors[0]
        best_squares = float('inf')
        
        for divisor in divisors:
            squares_h = height // divisor
            squares_w = width // divisor
            total_squares = squares_h * squares_w
            
            # We want at least 20 squares but not too many more
            if total_squares >= self.TARGET_SQUARES:
                if total_squares < best_squares:
                    best_squares = total_squares
                    best_divisor = divisor
        
        # If no divisor gives us enough squares, use the largest one
        if best_squares == float('inf'):
            best_divisor = divisors[-1]
            squares_h = height // best_divisor
            squares_w = width // best_divisor
            best_squares = squares_h * squares_w
        
        square_size = best_divisor
        squares_h = height // square_size
        squares_w = width // square_size
        
        return square_size, squares_h, squares_w, best_squares
    
    def find_segment_points(self, layer_mask):
        """Find points in grid squares with sufficient non-black pixels"""
        height, width = layer_mask.shape
        square_size, squares_h, squares_w, total_squares = self.calculate_optimal_grid(height, width)
        
        # First pass: find all qualifying squares
        qualifying_squares = []
        
        # Go through each square in the grid
        for row in range(squares_h):
            for col in range(squares_w):
                # Calculate square boundaries
                start_y = row * square_size
                end_y = min(start_y + square_size, height)
                start_x = col * square_size
                end_x = min(start_x + square_size, width)
                
                # Extract the square
                square = layer_mask[start_y:end_y, start_x:end_x]
                
                # Count non-black pixels (assuming black = 0)
                non_black_pixels = np.count_nonzero(square)
                total_pixels = square.size
                
                # Check if square meets the fill threshold
                if total_pixels > 0 and (non_black_pixels / total_pixels) >= self.MIN_FILL_THRESHOLD:
                    center_y = start_y + (end_y - start_y) // 2
                    center_x = start_x + (end_x - start_x) // 2
                    qualifying_squares.append((row, col, center_x, center_y))
        
        # Second pass: combine points using neighbor check
        points = self.combine_neighboring_points(qualifying_squares, squares_h, squares_w)
        
        return points, (square_size, squares_h, squares_w, total_squares)
    
    def combine_neighboring_points(self, qualifying_squares, squares_h, squares_w):
        """Simple combination: group into 3x3 windows and combine rows/columns within each group"""
        # Create a grid to track which squares have points
        grid = np.zeros((squares_h, squares_w), dtype=bool)
        square_centers = {}
        
        # Mark qualifying squares in grid and store their centers
        for row, col, center_x, center_y in qualifying_squares:
            grid[row, col] = True
            square_centers[(row, col)] = (center_x, center_y)
        
        # Track which squares have been processed
        processed = np.zeros((squares_h, squares_w), dtype=bool)
        combined_points = []
        
        # Process the grid in 3x3 windows
        for window_row in range(0, squares_h, 3):
            for window_col in range(0, squares_w, 3):
                # Extract points in this 3x3 window
                window_points = []
                for r in range(window_row, min(window_row + 3, squares_h)):
                    for c in range(window_col, min(window_col + 3, squares_w)):
                        if grid[r, c] and not processed[r, c]:
                            window_points.append((r, c))
                
                if not window_points:
                    continue
                
                # Combine points within this 3x3 window
                combined_in_window = self.combine_points_in_3x3_window(window_points, square_centers)
                combined_points.extend(combined_in_window)
                
                # Mark all points in this window as processed
                for r, c in window_points:
                    processed[r, c] = True
        
        return combined_points
    
    def combine_points_in_3x3_window(self, window_points, square_centers):
        """Combine points within a single 3x3 window by rows and columns"""
        if len(window_points) <= 1:
            # Single point or no points - just return as is
            return [square_centers[point] for point in window_points]
        
        # Convert to relative coordinates within the 3x3 window
        min_row = min(r for r, c in window_points)
        min_col = min(c for r, c in window_points)
        
        # Create a 3x3 local grid
        local_grid = {}
        for r, c in window_points:
            local_r = r - min_row
            local_c = c - min_col
            local_grid[(local_r, local_c)] = (r, c)
        
        # Special case: if we have a full 3x3 grid (9 points), combine to single center
        if len(window_points) == 9 and (1, 1) in local_grid:
            center_point = local_grid[(1, 1)]  # Center of 3x3
            return [square_centers[center_point]]
        
        combined_points = []
        used_points = set()
        
        # Check for 3-in-a-row (horizontal combinations)
        for row in range(3):
            row_points = [(row, col) for col in range(3) if (row, col) in local_grid]
            if len(row_points) >= 3:
                # Combine to center column (col=1)
                original_coords = [local_grid[p] for p in row_points]
                center_point = local_grid[(row, 1)]  # Middle of the row
                combined_points.append(square_centers[center_point])
                used_points.update(original_coords)
        
        # Check for 3-in-a-column (vertical combinations)
        for col in range(3):
            col_points = [(row, col) for row in range(3) if (row, col) in local_grid]
            if len(col_points) >= 3:
                # Combine to center row (row=1)
                original_coords = [local_grid[p] for p in col_points]
                # Only combine if not already used by row combination
                if not any(coord in used_points for coord in original_coords):
                    center_point = local_grid[(1, col)]  # Middle of the column
                    combined_points.append(square_centers[center_point])
                    used_points.update(original_coords)
        
        # Add any remaining individual points that weren't combined
        for local_pos, original_pos in local_grid.items():
            if original_pos not in used_points:
                combined_points.append(square_centers[original_pos])
        
        return combined_points
    
    def draw_points_on_layer(self, layer_mask, points):
        """Draw points on the layer mask and return colored version"""
        # Convert grayscale to BGR for colored points
        layer_colored = cv2.cvtColor(layer_mask, cv2.COLOR_GRAY2BGR)
        
        # Draw points
        for point in points:
            cv2.circle(layer_colored, point, self.POINT_SIZE, self.POINT_COLOR, -1)
        
        return layer_colored
    
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
        """Create 5 depth layer masks based on defined ranges with segment points"""
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
            
            # Find segment points for this layer
            points, grid_info = self.find_segment_points(layer_depth)
            
            # Create colored version with points
            layer_with_points = self.draw_points_on_layer(layer_depth, points)
            
            layers.append((layer_with_points, label, points, grid_info))
        
        return layers
    
    def create_composite_display(self, depth_normalized, layers):
        """Create 3x2 composite display with full depth + 5 layers with points"""
        # Resize depth map to panel size and convert to BGR
        depth_resized = cv2.resize(depth_normalized, (self.PANEL_WIDTH, self.PANEL_HEIGHT))
        depth_colored = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
        
        # Create composite image: 2 rows x 3 columns
        composite_height = self.PANEL_HEIGHT * 2
        composite_width = self.PANEL_WIDTH * 3
        composite = np.zeros((composite_height, composite_width, 3), dtype=np.uint8)
        
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
        composite[row:row+self.PANEL_HEIGHT, col:col+self.PANEL_WIDTH] = depth_colored
        cv2.putText(composite, "Full Depth", (col + 5, row + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, (255, 255, 255), self.FONT_THICKNESS)
        
        # Place depth layers with points
        for i, ((layer_with_points, label, points, grid_info), (row, col)) in enumerate(zip(layers, positions[1:])):
            # Resize layer to panel size
            layer_resized = cv2.resize(layer_with_points, (self.PANEL_WIDTH, self.PANEL_HEIGHT))
            
            # Place layer in composite
            composite[row:row+self.PANEL_HEIGHT, col:col+self.PANEL_WIDTH] = layer_resized
            
            # Add label with point count
            point_count = len(points)
            square_size, squares_h, squares_w, total_squares = grid_info
            label_with_info = f"{label} ({point_count}pts)"
            cv2.putText(composite, label_with_info, (col + 5, row + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.FONT_SCALE, (255, 255, 255), self.FONT_THICKNESS)
            
            # Add grid info
            grid_info_text = f"Grid:{squares_h}x{squares_w} ({total_squares})"
            cv2.putText(composite, grid_info_text, (col + 5, row + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return composite
    
    def save_current_frame(self, composite):
        """Save current composite frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"depth_layers_points_{timestamp}.jpg"
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
                
                # Create depth layers with segment points
                layers = self.create_depth_layers(depth_normalized)
                
                # Create composite display
                composite = self.create_composite_display(depth_normalized, layers)
                
                # Calculate and display FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update FPS every 30 frames
                    current_fps = 30 / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    
                    # Print detailed info every 30 frames
                    total_points = sum(len(layer[2]) for layer in layers)
                    print(f"FPS: {current_fps:.1f}, Total Points: {total_points}")
                
                # Add FPS to display
                cv2.putText(composite, f"FPS: {current_fps:.1f}", 
                           (composite.shape[1] - 100, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display composite
                cv2.imshow('Real-time Depth Layers with Segment Points', composite)
                
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
    print("=== Real-time MiDaS Depth Layering with Segment Points ===")
    print("Grid-based point selection: ~20 squares target, 70% fill threshold")
    
    try:
        depth_system = RealTimeDepthLayering()
        depth_system.run()
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()