"""
Simple SAM Test using Transformers - Base Model with Center Point
Clean test using Hugging Face transformers implementation
Uses center point as default prompt for meaningful segmentation
"""
import cv2
import numpy as np
import time
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

def main():
    print("Loading SAM Base model...")
    
    # Load Facebook's SAM Base model via transformers
    # Force CPU for now due to MPS float64 issues
    device = "cpu"  # torch.backends.mps.is_available() causes float64 issues
    print(f"Using device: {device}")
    
    model = SamModel.from_pretrained("facebook/sam-vit-base")
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model.to(device)
    
    print("Capturing frame...")
    
    # Capture single frame
    cap = cv2.VideoCapture(0)
    
    # Discard first 5 frames
    for _ in range(5):
        cap.read()
    
    # Capture test frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to capture frame")
        return
    
    print("Running SAM segmentation on a point at the center of the image...")
    start_time = time.time()
    
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Get image dimensions for center point
    h, w = rgb_frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    # Use center point as input prompt
    input_points = [[[center_x, center_y]]]
    input_labels = [[1]]
    
    inputs = processor(
        pil_image, 
        input_points=input_points, 
        input_labels=input_labels, 
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get masks
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    
    inference_time = time.time() - start_time
    print(f"Segmentation completed in {inference_time:.2f}s")
    
    # Select best mask using IoU scores
    masks_np = masks[0].squeeze().numpy()
    iou_scores = outputs.iou_scores.cpu().numpy().squeeze()
    
    if len(masks_np.shape) == 3:
        best_mask_idx = np.argmax(iou_scores)
        best_mask = masks_np[best_mask_idx]
    else:
        best_mask = masks_np
    
    # Create colored segmentation
    segmentation = np.zeros_like(frame)
    
    if best_mask.sum() > 100:
        # Use green for the segmented object
        color = (0, 255, 0)
        segmentation[best_mask > 0] = color
    
    # Save results
    timestamp = int(time.time())
    original_file = f"original_{timestamp}.jpg"
    segmentation_file = f"segmentation_{timestamp}.jpg"
    overlay_file = f"overlay_{timestamp}.jpg"
    
    cv2.imwrite(original_file, frame)
    cv2.imwrite(segmentation_file, segmentation)
    
    # Create overlay
    overlay = cv2.addWeighted(frame, 0.7, segmentation, 0.3, 0)
    cv2.imwrite(overlay_file, overlay)
    
    print(f"Saved: {original_file}, {segmentation_file}, {overlay_file}")

if __name__ == "__main__":
    main()