"""
Simple MiDaS depth estimation demo - streaming video

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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        try:
            # Predict depth
            depth_map = midas_handler.predict_depth(frame)
            
            # Normalize depth map to 0-255 for display
            # Smaller depth values (closer objects) will be whiter
            # Larger depth values (farther objects) will be darker
            depth_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # Show both the original frame and the depth map
            cv2.imshow('Original', frame)
            cv2.imshow('Depth Map', depth_normalized)
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            cv2.imshow('Original', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")

if __name__ == "__main__":
    main()