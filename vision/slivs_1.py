"""
Simple camera capture + MiDaS depth map demo
Opens camera, warms up, takes picture, processes depth, saves jpg
Now saves 5 sectioned depth maps with specified ranges

https://huggingface.co/collections/Intel/dpt-31-65b2a13eb0a5a381b6df9b6b
https://huggingface.co/Intel/dpt-swinv2-large-384

"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time

def create_depth_sections_composite(depth_normalized, timestamp):
    """
    Create a single composite image with 5 depth map sections:
    - Section 1: 0-50
    - Section 2: 51-100  
    - Section 3: 101-150
    - Section 4: 151-200
    - Section 5: 201-255
    Arranged in a 2x3 grid (original + 5 sections)
    Applies erosion to remove outline artifacts
    """
    
    # Define the ranges
    ranges = [
        (0, 50, "0-50"),
        (51, 100, "51-100"), 
        (101, 150, "101-150"),
        (151, 200, "151-200"),
        (201, 255, "201-255")
    ]
    
    # Create erosion kernel (adjust size as needed)
    # 3x3 kernel will remove thin outlines, 5x5 for thicker outlines
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Create individual sections
    sections = []
    
    # Add original as first image
    sections.append(depth_normalized.copy())
    
    for min_val, max_val, range_name in ranges:
        # Create a copy of the depth map
        section_depth = depth_normalized.copy()
        
        # Create mask for pixels outside the current range
        mask = (section_depth < min_val) | (section_depth > max_val)
        
        # Set pixels outside range to black (0)
        section_depth[mask] = 0
        
        # Create binary mask of non-zero pixels for erosion
        binary_mask = (section_depth > 0).astype(np.uint8) * 255
        
        # Apply erosion to remove thin outlines and small artifacts
        eroded_mask = cv2.erode(binary_mask, erosion_kernel, iterations=2)
        
        # Apply eroded mask back to the section
        section_depth[eroded_mask == 0] = 0
        
        sections.append(section_depth)
        
        print(f"Created and eroded depth section {range_name}")
    
    # Get dimensions
    height, width = depth_normalized.shape
    
    # Create composite image: 2 rows x 3 columns
    composite_height = height * 2
    composite_width = width * 3
    composite = np.zeros((composite_height, composite_width), dtype=np.uint8)
    
    # Labels for each section
    labels = ["Original", "0-50", "51-100", "101-150", "151-200", "201-255"]
    
    # Arrange images in 2x3 grid
    positions = [
        (0, 0),      # Original - top left
        (0, width),  # 0-50 - top middle  
        (0, width*2), # 51-100 - top right
        (height, 0),  # 101-150 - bottom left
        (height, width), # 151-200 - bottom middle
        (height, width*2) # 201-255 - bottom right
    ]
    
    for i, (section, label, (row, col)) in enumerate(zip(sections, labels, positions)):
        # Place section in composite
        composite[row:row+height, col:col+width] = section
        
        # Add text label
        cv2.putText(composite, label, (col + 10, row + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Save composite image
    composite_filename = f"depth_sections_composite_{timestamp}.jpg"
    cv2.imwrite(composite_filename, composite)
    
    print(f"Applied erosion with {erosion_kernel.shape} kernel to remove outlines")
    
    return composite_filename

def main():
    print("=== Camera Capture + MiDaS Depth Demo ===")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()
    
    # Warm up camera - discard first 5 frames
    print("Warming up camera...")
    for i in range(5):
        ret, frame = cap.read()
    
    # Capture image
    print("Capturing image...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture image")
        exit()
    
    # Load MiDaS model
    print("Loading MiDaS model...")
    #model_name = "Intel/dpt-swinv2-large-384" #This model has moderately less quality compared to large models, but has a better speed-performance trade-off
    model_name = "Intel/dpt-swinv2-tiny-256" # This model is recommended for deployment on embedded devices

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    
    # Process depth map
    print("Processing depth map...")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
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
    
    # Save original depth map
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_filename = f"depth_map_original_{timestamp}.jpg"
    cv2.imwrite(original_filename, depth_normalized)
    print(f"Original depth map saved as: {original_filename}")
    
    # Create and save composite depth sections
    print("\nCreating composite depth sections...")
    composite_filename = create_depth_sections_composite(depth_normalized, timestamp)
    
    print(f"\nFiles saved:")
    print(f"- Original: {original_filename}")
    print(f"- Composite sections: {composite_filename}")

if __name__ == "__main__":
    main()