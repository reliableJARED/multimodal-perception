"""
Simple SAM Test using Transformers - Base Model
Clean test using Hugging Face transformers implementation
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
    
    print("Running SAM segmentation...")
    start_time = time.time()
    
    # Convert BGR to RGB and to PIL
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # Use automatic mask generation approach - no input points needed
    inputs = processor(pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get masks - for automatic segmentation we need to use the mask decoder differently
    # Let's try the simpler approach with no points first
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    
    inference_time = time.time() - start_time
    print(f"Segmentation completed in {inference_time:.2f}s")
    
    # Convert masks to numpy and filter
    if len(masks) > 0:
        masks_np = masks[0].squeeze().numpy()
        if len(masks_np.shape) == 2:  # Single mask
            masks_np = [masks_np]
        elif len(masks_np.shape) == 3:  # Multiple masks
            masks_np = [masks_np[i] for i in range(masks_np.shape[0])]
    else:
        masks_np = []
    
    print(f"Found {len(masks_np)} masks")
    
    # Create colored segmentation
    segmentation = np.zeros_like(frame)
    
    for i, mask in enumerate(masks_np):
        if mask.sum() > 500:  # Only use masks with sufficient area
            # Generate color
            hue = int(180 * i / len(masks_np))
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            color = tuple(map(int, color))
            
            # Apply color to mask
            segmentation[mask > 0] = color
    
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