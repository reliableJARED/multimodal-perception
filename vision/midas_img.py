"""
Simple camera capture + MiDaS depth map demo
Opens camera, warms up, takes picture, processes depth, saves jpg

https://huggingface.co/collections/Intel/dpt-31-65b2a13eb0a5a381b6df9b6b
https://huggingface.co/Intel/dpt-swinv2-large-384

"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time

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
    
    # Save depth map as JPG
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    depth_filename = f"depth_map_{timestamp}.jpg"
    cv2.imwrite(depth_filename, depth_normalized)
    
    print(f"Depth map saved as: {depth_filename}")

if __name__ == "__main__":
    main()