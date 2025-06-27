"""
Camera capture with SAM2 segmentation analysis (Hugging Face version)
Captures image from camera, performs segmentation For the specific point input, saves results, and exits

https://huggingface.co/facebook/sam2.1-hiera-tiny

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import sys
import time

def capture_and_analyze_image(model_name="facebook/sam2-hiera-large"):
    """
    Capture image from camera and perform SAM2 segmentation analysis using Hugging Face models
    
    Args:
        model_name: Which SAM2 model to use from Hugging Face Hub
                   Options:
                   - "facebook/sam2-hiera-tiny"
                   - "facebook/sam2-hiera-small" 
                   - "facebook/sam2-hiera-base-plus"
                   - "facebook/sam2-hiera-large"
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
    
    # ===== SAM2 MODEL SETUP (HUGGING FACE VERSION) =====
    print(f"\nSetting up SAM2 model from Hugging Face: {model_name}")
    
    # Determine device (MPS for Apple Silicon, CUDA for NVIDIA GPUs, CPU for others)
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (NVIDIA GPU)")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
        dtype = torch.float16
    else:
        device = "cpu"
        print("Using CPU")
        dtype = torch.float32
    
    # Load predictor directly from Hugging Face Hub (will cache locally for offline use)
    print("Loading SAM2 model from Hugging Face Hub...")
    try:
        predictor = SAM2ImagePredictor.from_pretrained(model_name, device=device)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None, None
    
    # ===== IMAGE PROCESSING =====
    # Convert captured image to RGB for SAM2
    image_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
    
    # Set image for predictor
    predictor.set_image(image_rgb)
    
    # Define point prompt (center of image as default)
    height, width = image_rgb.shape[:2]
    center_x, center_y = width // 2, height // 2
    input_points = np.array([[center_x, center_y]])  # [x, y] coordinate
    input_labels = np.array([1])  # 1 = positive point (include object)
    
    print(f"Using center point for segmentation: ({center_x}, {center_y})")
    
    # Run segmentation with appropriate precision for device
    print("Running segmentation...")
    try:
        if device == "cpu":
            # CPU inference without autocast
            with torch.inference_mode():
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )
        else:
            # GPU inference with autocast
            with torch.inference_mode(), torch.autocast(device, dtype=dtype):
                masks, scores, logits = predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True,
                )
    except Exception as e:
        print(f"✗ Segmentation failed: {e}")
        return None, None
    
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
    try:
        #specify the model (change this line to use different models):
        model_choice = "facebook/sam2-hiera-large"  # Change this to try different models
        
        print(f"\nUsing model: {model_choice}")
        
        masks, scores = capture_and_analyze_image(model_choice)
        if masks is not None:
            print("Program completed successfully!")
        else:
            print("Program failed - check error messages above")
            
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure all OpenCV windows are closed
        cv2.destroyAllWindows()
        print("Program ended.")