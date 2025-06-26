"""
Camera capture with SAM2 segmentation analysis
Captures image from camera, performs segmentation For the specific point input, saves results, and exits
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import urllib.request
import sys
import time

def download_checkpoint(checkpoint_path, model_name="sam2.1_hiera_large"):
    """
    Download SAM2 checkpoint if it doesn't exist
    
    Args:
        checkpoint_path: Path where checkpoint should be saved
        model_name: Name of the model to download
    """
    # Available model URLs
    model_urls = {
        "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", 
        "sam2.1_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    }
    
    if model_name not in model_urls:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_urls.keys())}")
    
    url = model_urls[model_name]
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print(f"Downloading {model_name} model...")
    print(f"URL: {url}")
    print(f"Saving to: {checkpoint_path}")
    
    def show_progress(block_num, block_size, total_size):
        """Show download progress"""
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

def check_and_download_checkpoint(checkpoint_path, model_name="sam2.1_hiera_large"):
    """
    Check if checkpoint exists, download if missing
    
    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Name of model to download if missing
        
    Returns:
        bool: True if checkpoint is available, False otherwise
    """
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint found: {checkpoint_path}")
        return True
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Attempting to download...")
        return download_checkpoint(checkpoint_path, model_name)

def get_model_config(model_name):
    """
    Get the config filename for a given model
    
    Args:
        model_name: Name of the model
        
    Returns:
        str: Config filename
    """
    config_map = {
        "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
    
    if model_name not in config_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(config_map.keys())}")
    
    return config_map[model_name]

def capture_and_analyze_image(model_name="sam2.1_hiera_large"):
    """
    Capture image from camera and perform SAM2 segmentation analysis
    
    Args:
        model_name: Which SAM2 model to use (tiny, small, base_plus, large)
    """
    
    # ===== CAMERA SETUP =====
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None, None
    
    print("Camera opened successfully!")
    print("Discarding first 5 frames...")
    
    # Discard the first 5 frames as requested
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i+1}")
        time.sleep(0.1)  # Small delay between frames
    
    print("Capturing image...")
    
    # Capture the actual frame we want to keep
    ret, captured_frame = cap.read()
    
    # Release the camera immediately after capture
    cap.release()
    
    if not ret:
        print("Error: Could not capture image")
        return None, None
    
    print("✓ Image captured successfully!")
    
    # Save the captured image
    cv2.imwrite('captured_image.jpg', captured_frame)
    print("✓ Captured image saved as 'captured_image.jpg'")
    
    # ===== SAM2 MODEL SETUP =====
    print(f"\nSetting up SAM2 model: {model_name}")
    checkpoint = f"./models/checkpoints/{model_name}.pt"
    #SAM2 uses Hydra configuration management, which expects config files to be in the SAM2 package directory. not local directory
    model_cfg = get_model_config(model_name)
    
    print(f"Config: {model_cfg}")
    
    # Check and download checkpoint if needed
    if not check_and_download_checkpoint(checkpoint, model_name):
        print("Failed to obtain checkpoint file. Exiting.")
        return None, None
    
    # Determine device (MPS for Apple Silicon, CPU for Intel Macs)
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Build predictor with correct device
    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # ===== IMAGE PROCESSING =====
    # Convert captured image to RGB for SAM2
    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    
    # Set image for predictor
    print("Processing image with SAM2...")
    predictor.set_image(image_rgb)
    
    # Define point prompt (center of image as default)
    height, width = image_rgb.shape[:2]
    center_x, center_y = width // 2, height // 2
    input_points = np.array([[center_x, center_y]])  # [x, y] coordinate
    input_labels = np.array([1])  # 1 = positive point (include object)
    
    print(f"Using center point for segmentation: ({center_x}, {center_y})")
    
    # Run segmentation with appropriate precision for device
    if device == "mps":
        # Use float16 for MPS (Apple Silicon)
        with torch.inference_mode(), torch.autocast(device, dtype=torch.float16):
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
    else:
        # CPU inference without autocast
        with torch.inference_mode():
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
    
    print(f"✓ Segmentation complete! Generated {len(masks)} masks.")
    print(f"Mask scores: {scores}")
    
    # ===== DISPLAY AND SAVE RESULTS =====
    plt.figure(figsize=(15, 5))
    
    # Original image with point
    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.scatter(input_points[0, 0], input_points[0, 1], color='red', s=100, marker='*')
    plt.title('Captured Image + Point')
    plt.axis('off')
    
    # Best mask (highest score)
    best_idx = np.argmax(scores)
    plt.subplot(1, 4, 2)
    plt.imshow(image_rgb)
    plt.imshow(masks[best_idx], alpha=0.5, cmap='Reds')
    plt.title(f'Best Mask (Score: {scores[best_idx]:.3f})')
    plt.axis('off')
    
    # Mask only
    plt.subplot(1, 4, 3)
    plt.imshow(masks[best_idx], cmap='gray')
    plt.title('Mask Only')
    plt.axis('off')
    
    # All masks combined
    plt.subplot(1, 4, 4)
    combined_mask = np.zeros_like(masks[0], dtype=float)
    for i, mask in enumerate(masks):
        combined_mask += mask * (i + 1) / len(masks)
    plt.imshow(image_rgb)
    plt.imshow(combined_mask, alpha=0.6, cmap='viridis')
    plt.title('All Masks Combined')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save the segmentation results
    segmentation_filename = 'segmentation_results.png'
    plt.savefig(segmentation_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Segmentation results saved as '{segmentation_filename}'")
    
    # Show the results briefly
    plt.show(block=False)
    plt.pause(3)  # Show for 3 seconds
    plt.close()
    
    # Save individual masks as numpy arrays
    masks_filename = 'segmentation_masks.npz'
    np.savez(masks_filename, masks=masks, scores=scores, input_points=input_points)
    print(f"✓ Mask data saved as '{masks_filename}'")
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE!")
    print("="*50)
    print(f"Files saved:")
    print(f"  - captured_image.jpg (original capture)")
    print(f"  - {segmentation_filename} (visualization)")
    print(f"  - {masks_filename} (mask data)")
    print("="*50)
    
    return masks, scores

# Main execution
if __name__ == "__main__":
    print("Camera SAM2 Segmentation Tool")
    print("="*40)
    
    # You can change the model size here:
    # "sam2.1_hiera_tiny"      - Fastest, smallest file (~38MB)
    # "sam2.1_hiera_small"     - Good balance (~183MB)  
    # "sam2.1_hiera_base_plus" - Better quality (~312MB)
    # "sam2.1_hiera_large"     - Best quality (~899MB)
    
    model_choice = "sam2.1_hiera_tiny"  # Change this to try different models
    
    try:
        masks, scores = capture_and_analyze_image(model_choice)
        if masks is not None:
            print("Program completed successfully!")
        else:
            print("Program failed - check error messages above")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()
        print("Program ended.")