"""
Integrated SLIVS Pipeline: RGB → Depth → HS-Depth → SAM2 Segmentation
Real-time processing with 2x3 display panel showing all processing stages
Uses HSV color space with Hue-Saturation-Depth format instead of RGB-Depth
"""
import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import urllib.request
import sys
import time

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class IntegratedSLIVSPipelineHSD:
    def __init__(self, 
                 depth_model_name="Intel/dpt-swinv2-tiny-256",
                 sam_model_name="sam2.1_hiera_tiny"):
        """
        Initialize the integrated SLIVS pipeline with HS-Depth format
        
        Args:
            depth_model_name: MiDAS model for depth estimation
            sam_model_name: SAM2 model for segmentation
        """
        self.depth_model_name = depth_model_name
        self.sam_model_name = sam_model_name
        self.cache_dir = Path("models")
        self.device = self._setup_device()
        
        # Model components
        self.depth_processor = None
        self.depth_model = None
        self.sam_predictor = None
        
        print(f"Using device: {self.device}")
        print("Pipeline Format: RGB → HSV → HS-Depth → SAM2")
        
    def _setup_device(self):
        """Setup appropriate device for processing"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def _download_sam_checkpoint(self, checkpoint_path, model_name):
        """Download SAM2 checkpoint if it doesn't exist"""
        model_urls = {
            "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "sam2.1_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", 
            "sam2.1_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        }
        
        if model_name not in model_urls:
            raise ValueError(f"Unknown model: {model_name}")
        
        url = model_urls[model_name]
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        print(f"Downloading {model_name} model...")
        
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                downloaded_mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rProgress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()
        
        try:
            urllib.request.urlretrieve(url, checkpoint_path, show_progress)
            print(f"\n✓ Successfully downloaded {model_name}")
            return True
        except Exception as e:
            print(f"\n✗ Failed to download checkpoint: {e}")
            return False
    
    def _get_sam_config(self, model_name):
        """Get SAM2 config filename"""
        config_map = {
            "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
        return config_map[model_name]
    
    def load_models(self):
        """Load both MiDAS and SAM2 models"""
        print("Loading models...")
        
        # Load MiDAS depth model
        print(f"Loading {self.depth_model_name} depth model...")
        self.cache_dir.mkdir(exist_ok=True)
        
        try:
            self.depth_processor = AutoImageProcessor.from_pretrained(
                self.depth_model_name, 
                cache_dir=self.cache_dir
            )
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(
                self.depth_model_name, 
                cache_dir=self.cache_dir
            )
            self.depth_model.to(self.device)
            self.depth_model.eval()
            print("✓ Depth model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading depth model: {e}")
            raise
        
        # Load SAM2 model
        print(f"Loading {self.sam_model_name} segmentation model...")
        checkpoint_path = f"./models/checkpoints/{self.sam_model_name}.pt"
        model_cfg = self._get_sam_config(self.sam_model_name)
        
        # Download checkpoint if needed
        if not os.path.exists(checkpoint_path):
            if not self._download_sam_checkpoint(checkpoint_path, self.sam_model_name):
                raise RuntimeError("Failed to download SAM2 checkpoint")
        
        try:
            sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            print("✓ SAM2 model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading SAM2 model: {e}")
            raise
            
        print("✓ All models loaded successfully!")
    
    def predict_depth(self, image):
        """Predict depth from RGB image using MiDAS"""
        if self.depth_model is None or self.depth_processor is None:
            raise RuntimeError("Depth model not loaded")
        
        # Convert BGR to RGB if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Preprocess and predict
        inputs = self.depth_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        return prediction.squeeze().cpu().numpy()
    
    def rgb_to_hsv_frame(self, rgb_frame):
        """Convert RGB frame to HSV"""
        # Ensure input is in RGB format
        if isinstance(rgb_frame, np.ndarray) and rgb_frame.dtype == np.uint8:
            # Convert from BGR to RGB if needed (OpenCV default is BGR)
            if len(rgb_frame.shape) == 3:
                # Simple heuristic: if this came from cv2, it's likely BGR
                rgb_frame_corrected = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame_corrected = rgb_frame
        else:
            rgb_frame_corrected = rgb_frame
            
        # Convert RGB to HSV
        hsv_frame = cv2.cvtColor(rgb_frame_corrected, cv2.COLOR_RGB2HSV)
        return hsv_frame
    
    def create_hs_depth_frame(self, rgb_frame, depth_map):
        """Create HS-Depth frame by replacing V channel with depth"""
        # First convert RGB to HSV
        hsv_frame = self.rgb_to_hsv_frame(rgb_frame)
        
        # Ensure depth_map is numpy array and proper dtype
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()
        
        depth_map = depth_map.astype(np.float32)
        
        # Normalize depth to 0-255 (same range as V channel)
        depth_normalized = cv2.normalize(
            depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        # Create HS-Depth frame
        hsd_frame = hsv_frame.copy().astype(np.uint8)
        
        # Replace V (Value/Brightness) channel with depth
        hsd_frame[:, :, 2] = depth_normalized  # V channel is index 2 in HSV
        
        return hsd_frame, hsv_frame
    
    def hsd_to_display_rgb(self, hsd_frame):
        """Convert HS-Depth frame to RGB for display purposes"""
        # Convert HS-D back to RGB (treating depth as value channel)
        # Note: This is for visualization only
        display_rgb = cv2.cvtColor(hsd_frame, cv2.COLOR_HSV2RGB)
        return display_rgb
    
    def segment_with_sam2(self, image, point_coords, format_type="RGB"):
        """Perform SAM2 segmentation with point prompt
        
        Args:
            image: Input image (RGB, HSV, or HS-Depth format)
            point_coords: [x, y] coordinates for point prompt
            format_type: "RGB", "HSV", or "HSD" to indicate input format
        """
        if self.sam_predictor is None:
            raise RuntimeError("SAM2 model not loaded")
        
        # Prepare image for SAM2 (needs RGB format)
        if format_type == "HSD":
            # For HS-Depth, we have H, S, and Depth channels
            # We'll use it directly but note this is experimental
            # SAM2 expects RGB but we're testing if it can work with HS-D
            image_for_sam = image.copy()
            print(f"⚠️  Using HS-Depth format directly with SAM2 (experimental)")
        elif format_type == "HSV":
            # Convert HSV to RGB for SAM2
            image_for_sam = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:  # RGB
            # Ensure RGB format
            if len(image.shape) == 3 and image.shape[2] == 3:
                if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                    # Check if this looks like BGR (most likely from cv2)
                    image_for_sam = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_for_sam = image
            else:
                image_for_sam = image
        
        # Set image for predictor
        self.sam_predictor.set_image(image_for_sam)
        
        # Prepare point inputs
        input_points = np.array([point_coords])
        input_labels = np.array([1])  # Positive point
        
        # Run segmentation
        if self.device.type == "mps":
            with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.float16):
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )
        else:
            with torch.inference_mode():
                masks, scores, logits = self.sam_predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )
        
        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def create_segmentation_overlay(self, base_frame, mask, point_coords, score, frame_label=""):
        """Create segmentation overlay with mask and point"""
        overlay = base_frame.copy().astype(np.uint8)
        
        # Ensure mask is boolean numpy array
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        mask = mask.astype(bool)
        
        # Ensure point_coords are integers
        point_coords = [int(point_coords[0]), int(point_coords[1])]
        
        # Create colored mask overlay
        mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
        overlay[mask] = (overlay[mask].astype(np.float32) * 0.7 + mask_color.astype(np.float32) * 0.3).astype(np.uint8)
        
        # Draw point
        cv2.circle(overlay, tuple(point_coords), 8, (0, 0, 255), -1)  # Red point
        cv2.circle(overlay, tuple(point_coords), 10, (255, 255, 255), 2)  # White border
        
        # Add score and label text
        score_text = f"Score: {score:.3f}"
        cv2.putText(overlay, score_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        if frame_label:
            cv2.putText(overlay, frame_label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)  # Yellow text
        
        return overlay
    
    def create_combined_display(self, rgb_frame, depth_frame, hsv_frame, hsd_frame, 
                              hsd_segmentation_frame, rgb_segmentation_frame, window_width=1800):
        """Create display panel with RGB, Depth, HSV, HS-Depth, HS-D Segmentation, and RGB Segmentation"""
        # Calculate frame dimensions for 2x3 layout
        frame_width = window_width // 3
        
        # Get original frame dimensions
        original_height, original_width = rgb_frame.shape[:2]
        
        # Calculate new height maintaining aspect ratio
        frame_height = int(frame_width * original_height / original_width)
        
        # Resize all frames to the same dimensions
        rgb_resized = cv2.resize(rgb_frame, (frame_width, frame_height))
        
        # Convert depth to 3-channel for display
        if len(depth_frame.shape) == 2:
            depth_display = cv2.applyColorMap(depth_frame, cv2.COLORMAP_PLASMA)
        else:
            depth_display = depth_frame
        depth_resized = cv2.resize(depth_display, (frame_width, frame_height))
        
        # Convert HSV to RGB for display
        hsv_display = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2RGB)
        hsv_resized = cv2.resize(hsv_display, (frame_width, frame_height))
        
        # Convert HS-Depth to RGB for display
        hsd_display = self.hsd_to_display_rgb(hsd_frame)
        hsd_resized = cv2.resize(hsd_display, (frame_width, frame_height))
        
        hsd_seg_resized = cv2.resize(hsd_segmentation_frame, (frame_width, frame_height))
        rgb_seg_resized = cv2.resize(rgb_segmentation_frame, (frame_width, frame_height))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)
        
        cv2.putText(rgb_resized, "Original RGB", (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(depth_resized, "Depth Map", (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(hsv_resized, "HSV Format", (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(hsd_resized, "HS-Depth", (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(hsd_seg_resized, "SAM2 on HS-D", (10, 30), font, font_scale, text_color, font_thickness)
        cv2.putText(rgb_seg_resized, "SAM2 on RGB", (10, 30), font, font_scale, text_color, font_thickness)
        
        # Create 2x3 layout
        top_row = np.hstack([rgb_resized, depth_resized, hsv_resized])
        bottom_row = np.hstack([hsd_resized, hsd_seg_resized, rgb_seg_resized])
        
        # Combine rows
        combined = np.vstack([top_row, bottom_row])
        
        return combined
    
    def process_frame(self, frame):
        """Process single frame through complete HS-Depth pipeline"""
        height, width = frame.shape[:2]
        center_point = [width // 2, height // 2]
        
        try:
            # Step 1: Depth estimation
            depth_map = self.predict_depth(frame)
            
            # Ensure depth_map is numpy array with proper dtype
            if isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()
            depth_map = depth_map.astype(np.float32)
            
            # Create normalized depth for display
            depth_normalized = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            
            # Step 2: Create HSV and HS-Depth frames
            hsd_frame, hsv_frame = self.create_hs_depth_frame(frame, depth_map)
            
            # Step 3a: SAM2 segmentation on HS-Depth (experimental)
            mask_hsd, score_hsd = self.segment_with_sam2(hsd_frame, center_point, format_type="HSD")
            
            # Step 3b: SAM2 segmentation on original RGB (for comparison)
            mask_rgb, score_rgb = self.segment_with_sam2(frame, center_point, format_type="RGB")
            
            # Step 4a: Create HS-D segmentation overlay (convert HS-D to RGB for display)
            hsd_display = self.hsd_to_display_rgb(hsd_frame)
            segmentation_overlay_hsd = self.create_segmentation_overlay(
                hsd_display, mask_hsd, center_point, score_hsd, "HS-D Input"
            )
            
            # Step 4b: Create RGB segmentation overlay
            segmentation_overlay_rgb = self.create_segmentation_overlay(
                frame, mask_rgb, center_point, score_rgb, "RGB Input"
            )
            
            # Step 5: Create combined display
            combined_display = self.create_combined_display(
                frame, depth_normalized, hsv_frame, hsd_frame,
                segmentation_overlay_hsd, segmentation_overlay_rgb
            )
            
            return combined_display, {
                'depth_map': depth_map,
                'hsv_frame': hsv_frame,
                'hsd_frame': hsd_frame,
                'mask_hsd': mask_hsd,
                'score_hsd': score_hsd,
                'mask_rgb': mask_rgb,
                'score_rgb': score_rgb,
                'center_point': center_point
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()
            
            # Return error display
            error_text = f"Processing Error: {str(e)}"
            error_frame = frame.copy()
            cv2.putText(error_frame, error_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 0, 255), 2)
            return error_frame, None

def main():
    print("Integrated SLIVS Pipeline with HS-Depth Format")
    print("=" * 70)
    print("Pipeline: RGB → HSV → HS-Depth → SAM2 Comparison")
    print("Display Layout (2x3):")
    print("Top Row:    RGB Original | Depth Map | HSV Format")
    print("Bottom Row: HS-Depth | SAM2 on HS-D | SAM2 on RGB")
    print("Point prompt: Center of image (red circle)")
    print("⚠️  Note: SAM2 + HS-Depth is experimental - SAM2 was trained on RGB")
    print("Compare segmentation quality between RGB vs HS-Depth input")
    print("Controls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current 6-panel comparison image")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = IntegratedSLIVSPipelineHSD(
        depth_model_name="Intel/dpt-swinv2-tiny-256",  # Fast model for real-time
        sam_model_name="sam2.1_hiera_tiny"  # Fast SAM2 model
    )
    
    # Load models
    try:
        pipeline.load_models()
    except Exception as e:
        print(f"Failed to load models: {e}")
        return
    
    # Initialize camera
    print("\nInitializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("✓ Camera opened successfully!")
    print("\nStarting real-time HS-Depth processing...")
    
    frame_count = 0
    fps_start_time = time.time()
    save_counter = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Process frame through HS-Depth pipeline
        display_frame, results = pipeline.process_frame(frame)
        
        # Calculate and display FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_start_time)
            fps_start_time = current_time
            print(f"FPS: {fps:.1f}")
            
            # Print comparison scores if available
            if results:
                print(f"Scores - RGB: {results['score_rgb']:.3f}, HS-D: {results['score_hsd']:.3f}")
        
        # Add FPS counter and instructions to display
        if results:
            fps_text = f"Frame: {frame_count} | 's'=save | RGB:{results['score_rgb']:.3f} vs HS-D:{results['score_hsd']:.3f}"
            cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show combined display
        cv2.imshow('SLIVS HS-Depth Pipeline: RGB vs HS-Depth Segmentation', display_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the current 6-panel display
            save_counter += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"slivs_hsd_comparison_{timestamp}_{save_counter:03d}.png"
            
            try:
                cv2.imwrite(filename, display_frame)
                print(f"\n✓ Saved HS-Depth comparison image: {filename}")
                
                # Also save individual results if available
                if results:
                    # Save individual data
                    data_filename = f"slivs_hsd_data_{timestamp}_{save_counter:03d}.npz"
                    np.savez(data_filename,
                            depth_map=results['depth_map'],
                            hsv_frame=results['hsv_frame'],
                            hsd_frame=results['hsd_frame'],
                            mask_hsd=results['mask_hsd'],
                            mask_rgb=results['mask_rgb'],
                            score_hsd=results['score_hsd'],
                            score_rgb=results['score_rgb'],
                            center_point=results['center_point'])
                    print(f"✓ Saved HS-Depth data: {data_filename}")
                    
            except Exception as e:
                print(f"\n✗ Failed to save image: {e}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "=" * 70)
    print("HS-Depth Pipeline stopped successfully!")
    if save_counter > 0:
        print(f"Total images saved: {save_counter}")
    print("=" * 70)

if __name__ == "__main__":
    main()