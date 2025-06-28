"""
Streamlined SAM2 + Optical Flow Object Tracking

Simple tracking approach:
1. Initialize with SAM2 segment
2. Use optical flow on segment pixels for frame-to-frame tracking
3. Use average optical flow to predict new centroid for SAM2
4. Stop tracking when object is lost (no recovery)
"""

import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import time

class StreamlinedSAMTracker:
    def __init__(self, sam_model_name="sam2.1_hiera_tiny"):
        # SAM2 setup
        self.sam_checkpoint = f"./models/checkpoints/{sam_model_name}.pt"
        self.sam_config = self.get_sam_config(sam_model_name)
        
        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA GPU")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        # Initialize models
        self.sam_predictor = None
        
        # Tracking state
        self.current_mask = None
        self.current_centroid = None
        self.prev_frame_gray = None
        self.segment_points = None
        
        # Tracking parameters
        self.tracking_active = False
        
        # Optical flow parameters
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        self.load_models()
    
    def get_sam_config(self, model_name):
        """Get SAM2 config path"""
        config_map = {
            "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
        return config_map[model_name]
    
    def load_models(self):
        """Load SAM2 model"""
        print("Loading models...")
        
        # Load SAM2
        if not os.path.exists(self.sam_checkpoint):
            print(f"SAM2 checkpoint not found: {self.sam_checkpoint}")
            return False
        
        sam2_model = build_sam2(self.sam_config, self.sam_checkpoint, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam2_model)
        print("✓ SAM2 loaded")
        
        return True
    
    def get_segment_points(self, mask, num_points=100):
        """Extract points from segment mask for optical flow tracking"""
        y_coords, x_coords = np.where(mask)
        
        if len(y_coords) == 0:
            return None
        
        total_points = len(y_coords)
        if total_points <= num_points:
            points = np.column_stack((x_coords, y_coords)).astype(np.float32)
        else:
            indices = np.linspace(0, total_points - 1, num_points, dtype=int)
            points = np.column_stack((x_coords[indices], y_coords[indices])).astype(np.float32)
        
        return points
    
    def segment_with_point(self, image, point):
        """Get SAM2 segmentation for given point"""
        self.sam_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        input_points = np.array([point])
        input_labels = np.array([1])
        
        if self.device == "mps":
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16):
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
        
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]
    
    def get_mask_centroid(self, mask):
        """Calculate centroid of mask"""
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
        return (int(np.mean(x_coords)), int(np.mean(y_coords)))
    
    def initialize_tracking(self, frame, click_point):
        """Initialize tracking with first frame and click point"""
        print("Initializing tracking...")
        
        mask, score = self.segment_with_point(frame, click_point)
        self.current_mask = mask
        self.current_centroid = self.get_mask_centroid(mask)
        
        # Initialize optical flow tracking points
        self.segment_points = self.get_segment_points(mask)
        self.prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.tracking_active = True
        
        print(f"✓ Tracking initialized with score: {score:.3f}")
        print(f"✓ Extracted {len(self.segment_points)} tracking points")
        
        return mask, score
    
    def track_frame(self, frame):
        """Main tracking function"""
        if not self.tracking_active or self.segment_points is None:
            return None, None, None
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_frame_gray, current_gray, self.segment_points, None, **self.lk_params)
        
        # Filter good points
        good_mask = status.flatten() == 1
        good_points = new_points[good_mask]
        
        if len(good_points) < 10:  # Too few points tracked
            print("Too few optical flow points, stopping tracking")
            self.tracking_active = False
            return None, None, None
        
        # Calculate new centroid from optical flow
        predicted_centroid = tuple(np.mean(good_points, axis=0).astype(int))
        
        # Get new segmentation at predicted location
        try:
            mask, score = self.segment_with_point(frame, predicted_centroid)
            
            # Update tracking state
            self.segment_points = self.get_segment_points(mask)
            self.prev_frame_gray = current_gray
            self.current_mask = mask
            self.current_centroid = self.get_mask_centroid(mask)
            
            return mask, score, predicted_centroid
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            self.tracking_active = False
            return None, None, predicted_centroid
    
    def reset_tracking(self):
        """Reset tracking state"""
        self.tracking_active = False
        self.current_mask = None
        self.current_centroid = None
        self.segment_points = None
        self.prev_frame_gray = None
    
    def cleanup(self):
        """Cleanup resources"""
        self.tracking_active = False

def mouse_callback(event, x, y, flags, param):
    """Mouse callback for clicking to initialize tracking"""
    tracker, frame_ref, tracking_flag = param
    
    if event == cv2.EVENT_LBUTTONDOWN and not tracking_flag[0]:
        print(f"Clicked at ({x}, {y})")
        mask, score = tracker.initialize_tracking(frame_ref[0], (x, y))
        
        if mask is not None:
            # Show initialization result
            overlay = frame_ref[0].copy()
            if mask.dtype != bool:
                mask = mask.astype(bool)
            overlay[mask] = [0, 255, 0]
            result = cv2.addWeighted(frame_ref[0], 0.7, overlay, 0.3, 0)
            cv2.imshow('Streamlined Tracker', result)
            cv2.waitKey(500)  # Show result briefly
            tracking_flag[0] = True
            print("Tracking initialized successfully!")
        else:
            print("Failed to initialize tracking")

def main():
    """Main function"""
    tracker = StreamlinedSAMTracker()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Streamlined SAM2 + Optical Flow Tracker")
    print("Instructions:")
    print("- Click on object to start tracking")
    print("- Press 'r' to reset tracking")
    print("- Press 'q' to quit")
    print("- Press 'space' to track center point")
    
    tracking_initialized = [False]
    current_frame = [None]
    
    cv2.namedWindow('Streamlined Tracker')
    cv2.setMouseCallback('Streamlined Tracker', mouse_callback, (tracker, current_frame, tracking_initialized))
    
    frame_count = 0
    fps_timer = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame[0] = frame
            display_frame = frame.copy()
            frame_count += 1
            
            if tracking_initialized[0] and tracker.tracking_active:
                mask, score, centroid = tracker.track_frame(frame)
                
                if mask is not None:
                    # Draw successful tracking
                    overlay = display_frame.copy()
                    if mask.dtype != bool:
                        mask = mask.astype(bool)
                    overlay[mask] = [0, 255, 0]
                    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                    
                    if centroid:
                        cv2.circle(display_frame, centroid, 5, (0, 255, 0), -1)
                    
                    cv2.putText(display_frame, f"Tracking: {score:.2f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Tracking Lost", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    tracking_initialized[0] = False
            else:
                cv2.putText(display_frame, "Click on object to track", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show FPS
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_timer)
                fps_timer = time.time()
                print(f"FPS: {fps:.1f}")
            
            cv2.putText(display_frame, f"Frame: {frame_count}", 
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Streamlined Tracker', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracking_initialized[0] = False
                tracker.reset_tracking()
                print("Tracking reset - click to reinitialize")
            elif key == ord(' ') and not tracking_initialized[0]:
                h, w = frame.shape[:2]
                center_point = (w//2, h//2)
                mask, score = tracker.initialize_tracking(frame, center_point)
                if mask is not None:
                    tracking_initialized[0] = True
                    print("Tracking initialized at center")
    
    finally:
        tracker.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()