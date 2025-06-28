"""
Simple DINOv2 depth estimation demo - streaming video

Using Facebook's DPT-DINOv2 model for depth estimation
https://huggingface.co/facebook/dpt-dinov2-base-kitti

@misc{oquab2023dinov2,
      title={DINOv2: Learning Robust Visual Features without Supervision}, 
      author={Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski},
      year={2023},
      eprint={2304.07193},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


"""
import cv2
import torch
import numpy as np
import os
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

class OfflineDINOv2Depth:
    def __init__(self, model_name="facebook/dpt-dinov2-base-kitti"):
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
            
            print(f"DINOv2 depth model loaded successfully on {self.device}")
            
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
    # Initialize DINOv2 depth estimation model
    # Available models:
    # - facebook/dpt-dinov2-base-kitti (recommended for general use)
    # - facebook/dpt-dinov2-small-kitti (smaller, faster)
    # - facebook/dpt-dinov2-large-kitti (larger, potentially better quality)
    model_name = "facebook/dpt-dinov2-small-kitti"
    
    # Initialize DINOv2 depth handler
    depth_handler = OfflineDINOv2Depth(model_name=model_name)
    
    # Load model
    try:
        model, processor = depth_handler.load_model()
        print("DINOv2 depth model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Press 'q' to quit")
    print("DINOv2 depth estimation running...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        try:
            # Predict depth using DINOv2
            depth_map = depth_handler.predict_depth(frame)
            
            # Normalize depth map to 0-255 for display
            # Smaller depth values (closer objects) will be whiter
            # Larger depth values (farther objects) will be darker
            depth_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # Apply colormap for better visualization (optional)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Show both the original frame and the depth map
            cv2.imshow('Original', frame)
            cv2.imshow('Depth Map (Grayscale)', depth_normalized)
            cv2.imshow('Depth Map (Colored)', depth_colored)
            
        except Exception as e:
            print(f"Error during depth prediction: {e}")
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