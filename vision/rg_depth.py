"""
Simple MiDaS depth estimation demo - streaming video with RG-Depth channel modification

https://huggingface.co/collections/Intel/dpt-31-65b2a13eb0a5a381b6df9b6b
https://huggingface.co/Intel/dpt-swinv2-large-384

"""
import cv2
import torch
import numpy as np
import os
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

class OfflineMiDaS:
    def __init__(self, model_name="Intel/dpt-swinv2-large-384"):
        self.model_name = model_name
        self.cache_dir = Path("models")
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = None
        self.model = None
        
    def download_and_load_model(self):
        """Download and load the model and processor from Hugging Face"""
        print(f"Loading {self.model_name} model from Hugging Face...")
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        try:
            # Load processor and model from Hugging Face
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            self.model = AutoModelForDepthEstimation.from_pretrained(
                self.model_name, 
                cache_dir=self.cache_dir
            )
            
            # Move model to appropriate device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_model(self):
        """Load the model (downloads if not cached)"""
        if self.model is None or self.processor is None:
            self.download_and_load_model()
        return self.model, self.processor

    def predict_depth(self, image):
        """Predict depth from an image"""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR format from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            image = Image.fromarray(image)
        
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],  # PIL size is (width, height), we need (height, width)
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy and normalize
        output = prediction.squeeze().cpu().numpy()
        return output

def create_rgdepth_frame(rgb_frame, depth_map):
    """
    Create RG-Depth frame by replacing blue channel with depth data
    
    Args:
        rgb_frame: Original RGB frame (BGR format from OpenCV)
        depth_map: Normalized depth map (0-255)
    
    Returns:
        RG-Depth frame where blue channel is replaced with depth
    """
    # Make a copy to avoid modifying the original
    rgdepth_frame = rgb_frame.copy()
    
    # Replace blue channel (index 0 in BGR format) with depth
    rgdepth_frame[:, :, 0] = depth_map
    
    return rgdepth_frame

def create_combined_display(rgb_frame, depth_frame, rgdepth_frame, window_width=1200):
    """
    Create a combined display with RGB, Depth, and RG-Depth frames side by side
    
    Args:
        rgb_frame: Original RGB frame
        depth_frame: Depth visualization frame
        rgdepth_frame: RG-Depth combined frame
        window_width: Total width for the combined display
    
    Returns:
        Combined frame for display
    """
    # Calculate individual frame width
    frame_width = window_width // 3
    
    # Get original frame dimensions
    original_height, original_width = rgb_frame.shape[:2]
    
    # Calculate new height maintaining aspect ratio
    frame_height = int(frame_width * original_height / original_width)
    
    # Resize all frames to the same dimensions
    rgb_resized = cv2.resize(rgb_frame, (frame_width, frame_height))
    depth_resized = cv2.resize(depth_frame, (frame_width, frame_height))
    rgdepth_resized = cv2.resize(rgdepth_frame, (frame_width, frame_height))
    
    # Convert depth to 3-channel for consistent concatenation
    if len(depth_resized.shape) == 2:
        depth_resized = cv2.cvtColor(depth_resized, cv2.COLOR_GRAY2BGR)
    
    # Add labels to each frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    
    # Add text labels
    cv2.putText(rgb_resized, "RGB Original", (10, 30), font, font_scale, text_color, font_thickness)
    cv2.putText(depth_resized, "Depth Map", (10, 30), font, font_scale, text_color, font_thickness)
    cv2.putText(rgdepth_resized, "RG-Depth", (10, 30), font, font_scale, text_color, font_thickness)
    
    # Concatenate frames horizontally
    combined_frame = np.hstack([rgb_resized, depth_resized, rgdepth_resized])
    
    return combined_frame

def main():
    # Initialize Intel DPT 3.1 models
    #model_name = "Intel/dpt-swinv2-large-384" #This model has moderately less quality compared to large models, but has a better speed-performance trade-off
    model_name = "Intel/dpt-swinv2-tiny-256" # This model is recommended for deployment on embedded devices
    
    # Initialize MiDaS handler
    midas_handler = OfflineMiDaS(model_name=model_name)
    
    # Load model
    try:
        model, processor = midas_handler.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    print("Display shows: RGB Original | Depth Map | RG-Depth Combined")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        try:
            # Predict depth
            depth_map = midas_handler.predict_depth(frame)
            
            #Normalize depth map to 0-255 for display
            # In RG-Depth frame: closer objects will have more blue (higher values)
            # Farther objects will have less blue, appearing more red/green
            depth_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # Create RG-Depth frame (replace blue channel with depth)
            rgdepth_frame = create_rgdepth_frame(frame, depth_normalized)
            
            # Create combined display
            combined_display = create_combined_display(frame, depth_normalized, rgdepth_frame)
            
            # Show the combined display
            cv2.imshow('RGB | Depth | RG-Depth Combined View', combined_display)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Show only the original frame if there's an error
            cv2.imshow('RGB | Depth | RG-Depth Combined View', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")

if __name__ == "__main__":
    main()