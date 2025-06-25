"""
Simple MiDaS depth estimation demo - Records raw video first, then processes to depth map
Due to MiDaS processing time, we can't do real-time depth estimation at full framerate.
Solution: Record raw footage first, then process each frame to create depth video.


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
    print("=== MiDaS Depth Video Demo ===")
    print("Stage 1: Collecting raw footage for 5 seconds...")
    
    # Initialize camera and collect raw frames
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    raw_frames = []
    start_time = time.time()
    
    # Stage 1: Record raw video frames for 5 seconds
    while time.time() - start_time < 5:
        ret, frame = cap.read()
        if ret:
            raw_frames.append(frame.copy())
        elapsed = time.time() - start_time
        print(f"\rCollecting frames... {elapsed:.1f}/5.0s ({len(raw_frames)} frames)", end="", flush=True)
    
    cap.release()
    print(f"\nCollected {len(raw_frames)} frames")
    
    # Stage 2: Load MiDaS model (after collecting raw footage)
    print("\nStage 2: Loading MiDaS model...")
    #model_name = "Intel/dpt-swinv2-large-384" #This model has moderately less quality compared to large models, but has a better speed-performance trade-off
    model_name = "Intel/dpt-swinv2-tiny-256" # This model is recommended for deployment on embedded devices

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Model loaded on {device}")
    
    # Stage 3: Process each frame to create depth video
    print("Stage 3: Converting footage to depth map...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"depth_video_{timestamp}.mp4"
    
    # Setup video writer for depth output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Process each collected frame
    for i, frame in enumerate(raw_frames):
        # Convert BGR to RGB for MiDaS processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Run MiDaS depth prediction on this frame
        inputs = processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Resize depth prediction to match original frame size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert depth map to displayable format (0-255 grayscale)
        depth_map = prediction.squeeze().cpu().numpy()
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Convert grayscale depth to BGR for video output
        depth_bgr = cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2BGR)
        
        # Write processed depth frame to output video
        video_writer.write(depth_bgr)
        
        # Show processing progress
        progress = (i + 1) / len(raw_frames) * 100
        print(f"\rProcessing frames... {i+1}/{len(raw_frames)} ({progress:.1f}%)", end="", flush=True)
    
    # Cleanup and finish
    video_writer.release()
    
    print(f"\n\nDepth video saved as: {output_file}")
    print(f"Original footage: {len(raw_frames)} frames over 5 seconds")
    print(f"Final video: {len(raw_frames)} depth-processed frames")


if __name__ == "__main__":
    main()