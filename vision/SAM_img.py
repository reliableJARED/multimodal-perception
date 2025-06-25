"""
Enhanced SAM2 Object Detection with Automatic Mask Generation
Uses Facebook's SAM2 for better object detection and mask generation
"""
import cv2
import numpy as np
import time
import torch
from PIL import Image
import matplotlib.pyplot as plt
import random

# Try different SAM2 import approaches
USE_HF = False
try:
    # Try Hugging Face approach first
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    USE_HF = True
    print("Using Hugging Face SAM2 integration")
except ImportError:
    try:
        # Try direct SAM2 approach
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        USE_HF = False
        print("Using direct SAM2 integration")
    except ImportError as e:
        print(f"SAM2 import failed: {e}")
        print("Please install SAM2: pip install git+https://github.com/facebookresearch/sam2.git")
        exit(1)

def generate_distinct_colors(n):
    """Generate n visually distinct colors"""
    colors = []
    
    # Use HSV color space for better distribution
    for i in range(n):
        hue = int(360 * i / n)
        saturation = 255
        value = 255
        
        # Convert HSV to BGR
        hsv_color = np.uint8([[[hue // 2, saturation, value]]])  # OpenCV uses 0-179 for hue
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr_color)))
    
    # Shuffle for better visual separation
    random.shuffle(colors)
    return colors

def show_masks_on_image(image, masks):
    """Create a colored mask overlay on the image"""
    if len(masks) == 0:
        return image
    
    # Sort masks by area (largest first) for better layering
    sorted_masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    
    # Generate distinct colors for each mask
    colors = generate_distinct_colors(len(sorted_masks))
    
    # Create overlay
    overlay = image.copy()
    
    for i, mask_data in enumerate(sorted_masks):
        mask = mask_data['segmentation']
        color = colors[i % len(colors)]
        
        # Apply color to mask region
        overlay[mask] = color
    
    return overlay

def main():
    print("Setting up SAM2...")
    
    # Download model and config if needed
    sam2_checkpoint, model_cfg = download_sam2_checkpoint()
    
    # Configure device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps" 
        print("Using Apple Metal GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    print("Loading SAM2 model...")
    
    try:
        # Build SAM2 model
        print(f"Loading model from: {sam2_checkpoint}")
        print(f"Using config: {model_cfg}")
        
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        
        # Create automatic mask generator with optimized parameters
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=32,           # More points = more objects detected
            pred_iou_thresh=0.8,          # Higher = better quality masks
            stability_score_thresh=0.85,   # Higher = more stable masks
            crop_n_layers=1,              # Process image crops for small objects
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,     # Filter out tiny masks
        )
        
    except Exception as e:
        print(f"Error loading SAM2 model: {e}")
        if USE_HF:
            print("Make sure you have installed SAM2 and huggingface_hub:")
            print("pip install git+https://github.com/facebookresearch/sam2.git")
            print("pip install huggingface_hub")
        else:
            print("Make sure you have:")
            print("1. Installed SAM2: pip install git+https://github.com/facebookresearch/sam2.git")
            print("2. Downloaded the checkpoint files")
            print("3. Correct paths to checkpoint and config files")
        return
    
    print("Capturing frame...")
    
    # Capture single frame
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution for better results
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Discard first few frames to let camera adjust
    for _ in range(10):
        cap.read()
    
    # Capture test frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture frame")
        return
    
    print("Running SAM2 automatic mask generation...")
    start_time = time.time()
    
    # Convert BGR to RGB for SAM2
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Generate masks automatically
    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                masks = mask_generator.generate(rgb_frame)
        else:
            masks = mask_generator.generate(rgb_frame)
    
    inference_time = time.time() - start_time
    print(f"Segmentation completed in {inference_time:.2f}s")
    print(f"Found {len(masks)} objects")
    
    if len(masks) == 0:
        print("No objects detected!")
        return
    
    # Filter masks by quality and size
    filtered_masks = []
    for mask in masks:
        # Filter by stability score and area
        if (mask.get('stability_score', 0) > 0.85 and 
            mask.get('area', 0) > 500 and 
            mask.get('predicted_iou', 0) > 0.8):
            filtered_masks.append(mask)
    
    print(f"High-quality masks after filtering: {len(filtered_masks)}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Create colored segmentation overlay
    colored_overlay = show_masks_on_image(rgb_frame, filtered_masks)
    
    # Convert back to BGR for OpenCV saving
    colored_overlay_bgr = cv2.cvtColor(colored_overlay, cv2.COLOR_RGB2BGR)
    
    # Create pure mask image (white objects on black background)
    pure_mask = np.zeros(frame.shape, dtype=np.uint8)
    colors = generate_distinct_colors(len(filtered_masks))
    
    for i, mask_data in enumerate(filtered_masks):
        mask = mask_data['segmentation']
        color = colors[i % len(colors)]
        pure_mask[mask] = color
    
    # Create blended overlay
    blended = cv2.addWeighted(frame, 0.6, colored_overlay_bgr, 0.4, 0)
    
    # Save results
    timestamp = int(time.time())
    original_file = f"original_{timestamp}.jpg"
    pure_mask_file = f"masks_{timestamp}.jpg"
    overlay_file = f"overlay_{timestamp}.jpg"
    blended_file = f"blended_{timestamp}.jpg"
    
    cv2.imwrite(original_file, frame)
    cv2.imwrite(pure_mask_file, pure_mask)
    cv2.imwrite(overlay_file, colored_overlay_bgr)
    cv2.imwrite(blended_file, blended)
    
    print(f"Saved files:")
    print(f"  Original: {original_file}")
    print(f"  Pure masks: {pure_mask_file}")
    print(f"  Colored overlay: {overlay_file}")
    print(f"  Blended: {blended_file}")
    
    # Print mask statistics
    print("\nMask Statistics:")
    for i, mask in enumerate(filtered_masks):
        area = mask.get('area', 0)
        stability = mask.get('stability_score', 0)
        iou = mask.get('predicted_iou', 0)
        print(f"  Object {i+1}: Area={area:,} pixels, Stability={stability:.3f}, IoU={iou:.3f}")

def download_sam2_checkpoint():
    """Download SAM2 checkpoint and config if needed"""
    import os
    import urllib.request
    import subprocess
    import sys
    
    # Use /models directory
    models_dir = "./models"
    checkpoint_dir = os.path.join(models_dir, "checkpoints")
    
    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, "sam2.1_hiera_large.pt")
    
    # Download checkpoint if it doesn't exist
    if not os.path.exists(checkpoint_path):
        print("Downloading SAM2 checkpoint (~800MB)...")
        checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        urllib.request.urlretrieve(checkpoint_url, checkpoint_path)
        print(f"Checkpoint downloaded to: {checkpoint_path}")
    else:
        print(f"Using existing checkpoint: {checkpoint_path}")
    
    # Clone SAM2 repo for configs if not exists
    sam2_repo_dir = os.path.join(models_dir, "sam2_repo")
    if not os.path.exists(sam2_repo_dir):
        print("Downloading SAM2 repository for config files...")
        try:
            subprocess.run([
                "git", "clone", "https://github.com/facebookresearch/sam2.git", sam2_repo_dir
            ], check=True, capture_output=True)
            print(f"SAM2 repository cloned to: {sam2_repo_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e}")
            print("Trying alternative method...")
            # Fallback: download config file directly
            config_dir = os.path.join(models_dir, "configs", "sam2.1")
            os.makedirs(config_dir, exist_ok=True)
            config_path = os.path.join(config_dir, "sam2.1_hiera_l.yaml")
            
            config_url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
            urllib.request.urlretrieve(config_url, config_path)
            print(f"Config downloaded to: {config_path}")
            return checkpoint_path, config_path
    else:
        print(f"Using existing SAM2 repository: {sam2_repo_dir}")
    
    # Use config from cloned repo
    config_path = os.path.join(sam2_repo_dir, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml")
    
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        print("Downloading config file directly...")
        config_dir = os.path.join(models_dir, "configs", "sam2.1")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, "sam2.1_hiera_l.yaml")
        
        config_url = "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
        urllib.request.urlretrieve(config_url, config_path)
        print(f"Config downloaded to: {config_path}")
    
    return checkpoint_path, config_path

if __name__ == "__main__":
    main()