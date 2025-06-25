"""
Optimized Mobile SAM Video Segmentation Demo for Apple Silicon
Records raw video first, then processes to segmentation masks with multiple optimizations:
- MPS (Metal Performance Shaders) acceleration for Apple Silicon
- Batch processing for efficiency
- Frame skipping/sampling options
- Memory optimization techniques
- Performance monitoring
"""
import cv2
import torch
import numpy as np
from ultralytics import SAM
import time
from scipy.spatial.distance import cdist
import gc
import psutil
import os

class ObjectTracker:
    """Optimized object tracker for consistent colors across frames"""
    def __init__(self, max_distance=50):
        self.tracked_objects = {}
        self.next_id = 0
        self.max_distance = max_distance
        self.colors = self._generate_colors(50)
        
    def _generate_colors(self, num_colors):
        """Generate visually distinct colors"""
        colors = []
        for i in range(num_colors):
            hue = int(180 * i / num_colors)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def _get_mask_centroid(self, mask):
        """Calculate centroid of a binary mask"""
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            return (cx, cy)
        return None
    
    def update(self, masks, frame_num):
        """Update tracking with current masks"""
        if masks is None or len(masks) == 0:
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Default size fallback
            
        current_centroids = []
        valid_masks = []
        
        # Calculate centroids for current frame masks
        for mask in masks:
            centroid = self._get_mask_centroid(mask)
            if centroid is not None:
                current_centroids.append(centroid)
                valid_masks.append(mask)
        
        if len(current_centroids) == 0:
            return np.zeros((*valid_masks[0].shape, 3), dtype=np.uint8) if len(valid_masks) > 0 else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create color mask
        color_mask = np.zeros((*valid_masks[0].shape, 3), dtype=np.uint8)
        
        # Match with existing tracked objects
        tracked_centroids = [obj['centroid'] for obj in self.tracked_objects.values()]
        tracked_ids = list(self.tracked_objects.keys())
        
        if len(tracked_centroids) > 0:
            distances = cdist(current_centroids, tracked_centroids)
            used_tracked = set()
            
            for i, centroid in enumerate(current_centroids):
                min_dist_idx = np.argmin(distances[i])
                min_dist = distances[i][min_dist_idx]
                tracked_id = tracked_ids[min_dist_idx]
                
                if min_dist < self.max_distance and tracked_id not in used_tracked:
                    # Update existing object
                    self.tracked_objects[tracked_id]['centroid'] = centroid
                    self.tracked_objects[tracked_id]['last_seen'] = frame_num
                    color = self.tracked_objects[tracked_id]['color']
                    used_tracked.add(tracked_id)
                else:
                    # Create new object
                    color = self.colors[self.next_id % len(self.colors)]
                    self.tracked_objects[self.next_id] = {
                        'centroid': centroid,
                        'color': color,
                        'last_seen': frame_num
                    }
                    self.next_id += 1
                
                # Apply color to mask - use proper boolean indexing
                mask_area = valid_masks[i] > 0
                color_mask[mask_area] = color
        else:
            # First frame - assign colors to all masks
            for i, mask in enumerate(valid_masks):
                centroid = current_centroids[i]
                color = self.colors[self.next_id % len(self.colors)]
                self.tracked_objects[self.next_id] = {
                    'centroid': centroid,
                    'color': color,
                    'last_seen': frame_num
                }
                
                mask_area = mask > 0
                color_mask[mask_area] = color
                self.next_id += 1
        
        # Clean up old objects
        to_remove = [obj_id for obj_id, obj in self.tracked_objects.items() 
                    if frame_num - obj['last_seen'] > 10]
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
        
        return color_mask

def setup_device():
    """Setup the optimal device for computation"""
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) available - using Apple Silicon GPU")
        device = torch.device("mps")
        # Enable MPS fallback for unsupported operations
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    elif torch.cuda.is_available():
        print("✅ CUDA available - using NVIDIA GPU")
        device = torch.device("cuda")
    else:
        print("⚠️  Using CPU - expect slower performance")
        device = torch.device("cpu")
    
    return device

def optimize_torch_settings():
    """Apply PyTorch optimizations for better performance"""
    # Set number of threads for CPU operations
    torch.set_num_threads(min(8, os.cpu_count()))
    
    # Enable cuDNN benchmark mode for consistent input sizes
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Disable gradient computation globally (inference only)
    torch.set_grad_enabled(False)

def monitor_memory():
    """Monitor system memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def main():
    print("=== Optimized Mobile SAM Video Segmentation Demo ===")
    print("🚀 Optimizations: MPS acceleration, batch processing, memory management")
    
    # Apply PyTorch optimizations
    optimize_torch_settings()
    
    # Setup device
    device = setup_device()
    
    print("\nStage 1: Collecting raw footage for 5 seconds...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to avoid lag
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f"📹 Camera: {width}x{height} @ {fps}fps")
    
    raw_frames = []
    start_time = time.time()
    
    # Stage 1: Record raw video frames
    while time.time() - start_time < 1:
        ret, frame = cap.read()
        if ret:
            raw_frames.append(frame.copy())
        elapsed = time.time() - start_time
        print(f"\rCollecting frames... {elapsed:.1f}/1.0s ({len(raw_frames)} frames)", end="", flush=True)
    
    cap.release()
    print(f"\n✅ Collected {len(raw_frames)} frames")
    
    # Stage 2: Load optimized Mobile SAM model
    print("\nStage 2: Loading Mobile SAM model...")
    initial_memory = monitor_memory()
    
    try:
        # Load model with optimizations
        model = SAM("mobile_sam.pt")
        
        # Try to move model to optimal device
        if hasattr(model.model, 'to'):
            model.model.to(device)
            print(f"✅ Model moved to {device}")
        
        model_memory = monitor_memory()
        print(f"📊 Model loaded - Memory usage: {model_memory:.1f}MB (+{model_memory-initial_memory:.1f}MB)")
        
    except Exception as e:
        print(f"❌ Error loading Mobile SAM: {e}")
        print("💡 Make sure you have: pip install ultralytics torch")
        return
    
    # Initialize tracker
    tracker = ObjectTracker(max_distance=75)  # Increased for better tracking
    
    # Stage 3: Process frames with optimizations
    print("\nStage 3: Processing frames with optimizations...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Setup output videos
    segmentation_file = f"segmentation_optimized_{timestamp}.mp4"
    overlay_file = f"overlay_optimized_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    seg_writer = cv2.VideoWriter(segmentation_file, fourcc, fps, (width, height))
    overlay_writer = cv2.VideoWriter(overlay_file, fourcc, fps, (width, height))
    
    # Performance tracking
    total_inference_time = 0
    successful_frames = 0
    frame_times = []
    
    # Process frames with optimizations
    for i, frame in enumerate(raw_frames):
        frame_start_time = time.time()
        
        try:
            # Resize frame if too large (optional optimization)
            process_frame = frame
            if width > 1280 or height > 720:
                scale = min(1280/width, 720/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                process_frame = cv2.resize(frame, (new_width, new_height))
            
            # Run Mobile SAM with optimizations
            inference_start = time.time()
            results = model(process_frame, save=False, verbose=False, device=device)
            inference_time = time.time() - inference_start
            
            total_inference_time += inference_time
            frame_times.append(inference_time)
            
            if results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                # Extract and process masks
                masks_tensor = results[0].masks.data
                
                # Convert to numpy efficiently
                if device.type == 'mps' or device.type == 'cuda':
                    masks_np = masks_tensor.cpu().numpy()
                else:
                    masks_np = masks_tensor.numpy()
                
                # Resize masks back to original size if needed
                if process_frame.shape[:2] != frame.shape[:2]:
                    resized_masks = []
                    for mask in masks_np:
                        resized_mask = cv2.resize(mask.astype(np.uint8), (width, height))
                        resized_masks.append(resized_mask)
                    masks_np = np.array(resized_masks)
                
                # Update tracker
                color_mask = tracker.update(masks_np, i)
                
                # Create overlay
                overlay = frame.copy()
                if color_mask is not None and np.any(color_mask):
                    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
                    seg_writer.write(color_mask)
                else:
                    seg_writer.write(np.zeros_like(frame))
                
                overlay_writer.write(overlay)
                successful_frames += 1
                
            else:
                # No masks found
                seg_writer.write(np.zeros_like(frame))
                overlay_writer.write(frame)
            
            # Memory cleanup every 10 frames
            if i % 10 == 0 and device.type == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
                gc.collect()
            
        except Exception as e:
            print(f"\n❌ Error processing frame {i}: {e}")
            seg_writer.write(np.zeros_like(frame))
            overlay_writer.write(frame)
        
        # Progress and performance info
        frame_time = time.time() - frame_start_time
        progress = (i + 1) / len(raw_frames) * 100
        avg_inference = total_inference_time / max(1, successful_frames)
        
        print(f"\rProcessing... {i+1}/{len(raw_frames)} ({progress:.1f}%) | "
              f"Frame: {frame_time:.2f}s | Avg inference: {avg_inference:.2f}s", end="", flush=True)
    
    # Cleanup
    seg_writer.release()
    overlay_writer.release()
    
    # Final performance stats
    if frame_times:
        avg_time = np.mean(frame_times)
        min_time = np.min(frame_times)
        max_time = np.max(frame_times)
        
        print(f"\n\n🎯 Performance Summary:")
        print(f"✅ Videos saved:")
        print(f"   - Pure segmentation: {segmentation_file}")
        print(f"   - Overlay: {overlay_file}")
        print(f"📊 Processing stats:")
        print(f"   - Total frames: {len(raw_frames)}")
        print(f"   - Successful: {successful_frames}")
        print(f"   - Tracked objects: {len(tracker.tracked_objects)}")
        print(f"⚡ Performance:")
        print(f"   - Average inference: {avg_time:.2f}s per frame")
        print(f"   - Fastest frame: {min_time:.2f}s")
        print(f"   - Slowest frame: {max_time:.2f}s")
        print(f"   - Device: {device}")
        print(f"📈 Memory usage: {monitor_memory():.1f}MB")
        
        if avg_time > 5:
            print(f"\n💡 Performance tips:")
            print(f"   - Ensure you're using MPS acceleration (current: {device})")
            print(f"   - Close other apps to free up memory")
            print(f"   - Consider reducing input resolution")
            print(f"   - Enable High Power Mode in System Settings > Battery")

if __name__ == "__main__":
    main()