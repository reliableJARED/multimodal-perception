"""
Enhanced Combined Optical Flow + Depth + Segmentation Pipeline

Improvements:
- Blob-based tracking with movement averaging
- Depth-based validation for tracking points
- Outlier detection and correction
- Better contour tracking of depth blobs
- Mean-based depth filtering with binary mask visualization
"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time
import matplotlib.pyplot as plt
import os
import urllib.request
import sys
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
import threading
from sklearn.cluster import DBSCAN

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class EnhancedCombinedPipeline:
    def __init__(self, sam_model_name="sam2.1_hiera_tiny"):
        self.sam_model_name = sam_model_name
        self.midas_processor = None
        self.midas_model = None
        self.sam_predictor = None
        self.device = None
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Tracking state
        self.tracking_points = None
        self.depth_range = None
        self.segment_center = None
        self.colors = None
        self.reference_depth_map = None
        self.tracking_history = []  # Store recent tracking positions
        self.max_history = 5
        self.lost_points_count = 0
        self.total_frames = 0
        
        # Blob tracking parameters
        self.movement_threshold = 5.0  # Minimum movement to consider valid
        self.depth_tolerance = 0.15  # Tolerance for depth validation
        self.outlier_threshold = 2.0  # Standard deviations for outlier detection
        self.min_tracking_confidence = 0.7  # Minimum ratio of good points to maintain tracking
        self.auto_reset_threshold = 0.3  # Auto-reset if confidence drops below this
        
        # Performance monitoring
        self.performance_stats = {
            'corrections': 0,
            'outliers_detected': 0,
            'depth_failures': 0,
            'auto_resets': 0
        }
        
        # Initialize models
        self.setup_models()
    
    def download_checkpoint(self, checkpoint_path, model_name):
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
    
    def get_model_config(self, model_name):
        """Get the config filename for a given model"""
        config_map = {
            "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
        return config_map[model_name]
    
    def setup_models(self):
        """Initialize all models"""
        print("Setting up models...")
        
        # Setup device
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("Using CUDA GPU")
        else:
            self.device = "cpu"
            print("Using CPU")
        
        # Setup MiDaS
        print("Loading MiDaS model...")
        model_name = "Intel/dpt-swinv2-tiny-256"
        self.midas_processor = AutoImageProcessor.from_pretrained(model_name)
        self.midas_model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.midas_model.to(self.device).eval()
        
        # Setup SAM2
        print("Loading SAM2 model...")
        checkpoint = f"./models/checkpoints/{self.sam_model_name}.pt"
        model_cfg = self.get_model_config(self.sam_model_name)
        
        if not os.path.exists(checkpoint):
            if not self.download_checkpoint(checkpoint, self.sam_model_name):
                raise RuntimeError("Failed to obtain SAM2 checkpoint")
        
        sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam2_model)
        
        print("✓ All models loaded successfully!")
    
    def countdown_and_capture(self, cap):
        """Step 1: Countdown from 10 and capture frame"""
        print("=== Step 1: Countdown and Capture ===")
        
        # Warm up camera
        for i in range(5):
            ret, frame = cap.read()
            if not ret:
                return None
        
        # Countdown
        for i in range(10, 0, -1):
            ret, frame = cap.read()
            if not ret:
                return None
            
            # Create countdown display
            display_frame = frame.copy()
            height, width = display_frame.shape[:2]
            
            # Draw countdown text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3
            color = (0, 255, 0)
            thickness = 5
            
            text = str(i)
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            cv2.putText(display_frame, text, (text_x, text_y), font, font_scale, color, thickness)
            
            # Draw center crosshair
            center_x, center_y = width // 2, height // 2
            cv2.line(display_frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 2)
            cv2.line(display_frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 2)
            
            cv2.imshow('Countdown', display_frame)
            cv2.waitKey(1000)  # Wait 1 second
        
        # Capture final frame
        ret, frame = cap.read()
        if ret:
            print("✓ Frame captured!")
            return frame
        return None
    
    def generate_depth_map(self, frame):
        """Step 2: Generate depth map using MiDaS"""
        print("=== Step 2: Generating Depth Map ===")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        inputs = self.midas_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.midas_model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Resize to original dimensions
        height, width = frame.shape[:2]
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        
        depth_map = prediction.squeeze().cpu().numpy()
        print("✓ Depth map generated!")
        return depth_map
    
    def segment_center_point(self, frame):
        """Step 3: Segment center point using SAM2"""
        print("=== Step 3: Segmenting Center Point ===")
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image_rgb)
        
        # Use center point as prompt
        height, width = image_rgb.shape[:2]
        center_x, center_y = width // 2, height // 2
        self.segment_center = (center_x, center_y)
        
        input_points = np.array([[center_x, center_y]])
        input_labels = np.array([1])
        
        # Run segmentation
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
        
        best_mask = masks[np.argmax(scores)]
        print(f"✓ Segmentation complete! Score: {scores[np.argmax(scores)]:.3f}")
        return best_mask
    
    def extract_segment_depth(self, depth_map, mask):
        """Step 4: Extract segmented area from depth map"""
        print("=== Step 4: Extracting Segment Depth ===")
        
        # Ensure mask is boolean type
        if mask.dtype != bool:
            mask = mask.astype(bool)
            print(f"Converted mask from {mask.dtype} to boolean")
        
        print(f"Mask shape: {mask.shape}, Depth map shape: {depth_map.shape}")
        print(f"Mask true pixels: {np.sum(mask)}")
        
        segment_depths = depth_map[mask]
        if len(segment_depths) == 0:
            print("Warning: No depth data in segment")
            return None, None
        
        min_depth = np.min(segment_depths)
        max_depth = np.max(segment_depths)
        self.depth_range = (min_depth, max_depth)
        
        print(f"✓ Depth range: {min_depth:.3f} - {max_depth:.3f}")
        return segment_depths, self.depth_range
    
    def disperse_border_points(self, mask, num_points=20):
        """Step 5: Disperse 20 points evenly on segment border"""
        print("=== Step 5: Dispersing Border Points ===")
        
        # Ensure mask is boolean, then convert to uint8 for contour detection
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("Warning: No contours found")
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get contour points
        contour_points = largest_contour.reshape(-1, 2)
        
        if len(contour_points) < num_points:
            print(f"Warning: Only {len(contour_points)} contour points available")
            selected_points = contour_points
        else:
            # Distribute points evenly along contour
            contour_length = len(contour_points)
            indices = np.linspace(0, contour_length - 1, num_points, dtype=int)
            selected_points = contour_points[indices]
        
        # Convert to format expected by optical flow
        self.tracking_points = selected_points.astype(np.float32).reshape(-1, 1, 2)
        
        # Generate colors for visualization
        self.colors = np.random.randint(0, 255, (len(selected_points), 3))
        
        # Initialize tracking history
        self.tracking_history = [self.tracking_points.copy()]
        
        print(f"✓ {len(selected_points)} border points dispersed")
        return self.tracking_points
    
    def calculate_movement_vectors(self, old_points, new_points, status):
        """Calculate movement vectors for tracked points"""
        if old_points is None or new_points is None:
            return None, None
        
        # Filter good points
        good_old = old_points[status == 1]
        good_new = new_points[status == 1]
        
        if len(good_old) == 0 or len(good_new) == 0:
            return None, None
        
        # Calculate movement vectors
        movement_vectors = good_new.reshape(-1, 2) - good_old.reshape(-1, 2)
        movement_magnitudes = np.linalg.norm(movement_vectors, axis=1)
        
        return movement_vectors, movement_magnitudes
    
    def detect_outliers_and_correct(self, old_points, new_points, status, depth_map):
        """Detect outlier points and correct them using blob movement averaging"""
        if old_points is None or new_points is None:
            return new_points, status
        
        # Calculate movement vectors
        movement_vectors, movement_magnitudes = self.calculate_movement_vectors(old_points, new_points, status)
        
        if movement_vectors is None:
            return new_points, status
        
        corrected_points = new_points.copy()
        corrections_made = 0
        
        # Detect outliers using movement magnitude
        if len(movement_magnitudes) > 3:  # Need enough points for meaningful statistics
            mean_movement = np.mean(movement_magnitudes)
            std_movement = np.std(movement_magnitudes)
            
            # Points that moved too little or too much are potential outliers
            outlier_mask = np.abs(movement_magnitudes - mean_movement) > (self.outlier_threshold * std_movement)
            
            if np.sum(outlier_mask) > 0:
                outliers_detected = np.sum(outlier_mask)
                self.performance_stats['outliers_detected'] += outliers_detected
                print(f"Detected {outliers_detected} outlier points")
                
                # Calculate average movement vector from good points
                good_movement_vectors = movement_vectors[~outlier_mask]
                if len(good_movement_vectors) > 0:
                    avg_movement = np.mean(good_movement_vectors, axis=0)
                    
                    # Correct outlier points
                    good_indices = np.where(status == 1)[0]
                    outlier_indices = good_indices[outlier_mask]
                    
                    for idx in outlier_indices:
                        # Apply average movement to outlier point
                        corrected_points[idx] = old_points[idx] + avg_movement.reshape(1, 1, 2)
                        corrections_made += 1
                        print(f"Corrected point {idx} using average movement vector")
                
                self.performance_stats['corrections'] += corrections_made
        
        return corrected_points, status
    
    def validate_points_with_depth(self, points, depth_map):
        """Validate tracking points using depth information"""
        if points is None or self.depth_range is None:
            return points, np.ones(len(points), dtype=bool)
        
        valid_mask = np.ones(len(points), dtype=bool)
        min_depth, max_depth = self.depth_range
        depth_span = max_depth - min_depth
        depth_failures = 0
        
        # Expand depth range for validation
        expanded_min = min_depth - (depth_span * self.depth_tolerance)
        expanded_max = max_depth + (depth_span * self.depth_tolerance)
        
        for i, point in enumerate(points):
            x, y = point.ravel().astype(int)
            
            # Check if point is within image bounds
            if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
                point_depth = depth_map[y, x]
                
                # Check if depth is within expected range
                if not (expanded_min <= point_depth <= expanded_max):
                    valid_mask[i] = False
                    depth_failures += 1
                    print(f"Point {i} failed depth validation: {point_depth:.3f} not in [{expanded_min:.3f}, {expanded_max:.3f}]")
            else:
                valid_mask[i] = False
                print(f"Point {i} out of bounds: ({x}, {y})")
        
        self.performance_stats['depth_failures'] += depth_failures
        return points, valid_mask
    
    def enhanced_optical_flow_tracking(self, old_gray, new_gray, points, depth_map):
        """Enhanced optical flow tracking with blob-based correction"""
        if points is None or len(points) == 0:
            return None, None
        
        self.total_frames += 1
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray, new_gray, points, None, **self.lk_params
        )
        
        if new_points is None:
            return None, None
        
        # Debug status array
        print(f"Debug: status shape: {status.shape}, dtype: {status.dtype}")
        print(f"Debug: status sample: {status[:3] if len(status) > 3 else status}")
        
        # Ensure status is 1D boolean array
        if len(status.shape) > 1:
            status = status.flatten()
        status = status.astype(bool)
        
        # Detect outliers and correct using blob movement
        corrected_points, status = self.detect_outliers_and_correct(points, new_points, status, depth_map)
        
        # Validate points using depth information
        validated_points, depth_valid = self.validate_points_with_depth(corrected_points, depth_map)
        
        # Combine optical flow status with depth validation
        final_status = status & depth_valid
        
        # Check if we have enough good points to continue tracking
        good_point_ratio = np.sum(final_status) / len(final_status)
        if good_point_ratio < self.min_tracking_confidence:
            print(f"Warning: Low tracking confidence ({good_point_ratio:.2f})")
            
            # Auto-reset if confidence is critically low
            if good_point_ratio < self.auto_reset_threshold:
                print("Tracking confidence critically low - marking for auto-reset")
                self.performance_stats['auto_resets'] += 1
                return None, None
        
        # Update tracking history
        self.tracking_history.append(validated_points.copy())
        if len(self.tracking_history) > self.max_history:
            self.tracking_history.pop(0)
        
        return validated_points, final_status
    
    def get_depth_blob_inside_points(self, depth_map, points):
        """Get depth blob inside tracked points with enhanced contour detection"""
        if points is None or len(points) < 3:
            return None, None
        
        # Create mask from points using convex hull
        hull_points = points.reshape(-1, 2).astype(np.int32)
        mask = np.zeros(depth_map.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [hull_points], 255)
        
        # Extract depth values inside the polygon
        inside_depths = depth_map[mask == 255]
        
        if len(inside_depths) == 0:
            return None, None
        
        return inside_depths, mask
    
    def create_mean_based_depth_mask(self, depth_map, points, depth_range):
        """Create binary mask based on mean depth ± half the relative depth window"""
        if points is None or len(points) < 3 or depth_range is None:
            return None, None, None
        
        # Get depth values inside tracking perimeter
        inside_depths, spatial_mask = self.get_depth_blob_inside_points(depth_map, points)
        
        if inside_depths is None or len(inside_depths) == 0:
            return None, None, None
        
        # Calculate mean depth inside the perimeter
        mean_depth = np.mean(inside_depths)
        
        # Calculate half the relative depth window
        min_depth, max_depth = depth_range
        relative_depth_window = max_depth - min_depth
        half_window = relative_depth_window / 2.0
        
        # Create depth range around mean
        depth_min = mean_depth - half_window
        depth_max = mean_depth + half_window
        
        print(f"Mean depth: {mean_depth:.3f}, Window: {depth_min:.3f} - {depth_max:.3f}")
        
        # Create binary mask for entire image
        # Points in depth range = white (255), others = black (0)
        binary_mask = np.zeros(depth_map.shape, dtype=np.uint8)
        depth_in_range = (depth_map >= depth_min) & (depth_map <= depth_max)
        binary_mask[depth_in_range] = 255
        
        # Apply spatial constraint (only inside tracking perimeter)
        final_mask = np.zeros_like(binary_mask)
        final_mask[spatial_mask == 255] = binary_mask[spatial_mask == 255]
        
        # Find contours in the final mask
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        if contours:
            min_area = 50  # Minimum contour area
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            return filtered_contours, final_mask, (depth_min, depth_max)
        
        return None, final_mask, (depth_min, depth_max)
    
    def create_enhanced_depth_contour(self, depth_map, mask, depth_range, tolerance=0.1):
        """Create enhanced depth contour with better filtering"""
        if mask is None or depth_range is None:
            return None, None
        
        min_depth, max_depth = depth_range
        depth_span = max_depth - min_depth
        
        # Use adaptive tolerance based on depth span
        adaptive_tolerance = max(tolerance, depth_span * 0.05)
        expanded_min = min_depth - (depth_span * adaptive_tolerance)
        expanded_max = max_depth + (depth_span * adaptive_tolerance)
        
        # Create depth-filtered mask
        depth_mask = np.logical_and(
            depth_map >= expanded_min,
            depth_map <= expanded_max
        )
        
        # Combine with spatial mask
        combined_mask = np.logical_and(mask == 255, depth_mask)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        if contours:
            min_area = 100  # Minimum contour area
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            return filtered_contours, combined_mask
        
        return None, combined_mask
    
    def run_pipeline(self):
        """Main pipeline execution"""
        print("=" * 60)
        print("Starting Enhanced Combined Pipeline")
        print("=" * 60)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        try:
            # Step 1: Countdown and capture
            initial_frame = self.countdown_and_capture(cap)
            if initial_frame is None:
                print("Failed to capture initial frame")
                return
            
            # Step 2: Generate depth map
            self.reference_depth_map = self.generate_depth_map(initial_frame)
            
            # Step 3: Segment center point
            mask = self.segment_center_point(initial_frame)
            
            # Step 4: Extract segment depth
            segment_depths, depth_range = self.extract_segment_depth(self.reference_depth_map, mask)
            if segment_depths is None:
                print("Failed to extract segment depth")
                return
            
            # Step 5: Disperse border points
            border_points = self.disperse_border_points(mask)
            if border_points is None:
                print("Failed to disperse border points")
                return
            
            # Initialize tracking
            old_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
            
            print("\n" + "=" * 60)
            print("Starting enhanced real-time tracking...")
            print("Press 'q' to quit, 'r' to reset, 'd' to show depth mask")
            print("=" * 60)
            
            show_depth_mask = False
            
            # Main tracking loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Enhanced optical flow tracking
                new_points, status = self.enhanced_optical_flow_tracking(
                    old_gray, frame_gray, self.tracking_points, self.reference_depth_map
                )
                
                # Handle auto-reset condition
                if new_points is None and self.tracking_points is not None:
                    print("Auto-resetting due to tracking failure...")
                    # Quick re-initialization without full countdown
                    mask = self.segment_center_point(frame)
                    if mask is not None:
                        segment_depths, depth_range = self.extract_segment_depth(self.reference_depth_map, mask)
                        if segment_depths is not None:
                            border_points = self.disperse_border_points(mask)
                            if border_points is not None:
                                new_points = border_points
                                status = np.ones(len(border_points), dtype=bool)
                                print("Auto-reset successful!")
                
                if new_points is not None and len(new_points) > 3:
                    # Create mean-based depth mask and contours
                    mean_contours, mean_mask, mean_depth_range = self.create_mean_based_depth_mask(
                        self.reference_depth_map, new_points, self.depth_range
                    )
                    
                    if show_depth_mask and mean_mask is not None:
                        # Show the binary depth mask
                        cv2.imshow('Depth Mask', mean_mask)
                    
                    # Draw tracking points with status indication
                    for i, point in enumerate(new_points):
                        x, y = point.ravel().astype(int)
                        
                        # Debug and safely handle status array
                        if status is not None and i < len(status):
                            status_val = status[i]
                            # Handle different status formats
                            if isinstance(status_val, (np.ndarray, list)):
                                is_good = bool(status_val.item()) if hasattr(status_val, 'item') else bool(status_val[0])
                            else:
                                is_good = bool(status_val)
                        else:
                            is_good = False
                        
                        color = self.colors[i % len(self.colors)].tolist() if is_good else [128, 128, 128]
                        radius = 4 if is_good else 2
                        cv2.circle(frame, (x, y), radius, color, -1)
                        
                        # Draw point index
                        cv2.putText(frame, str(i), (x + 5, y - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    
                    # Draw tracking perimeter (convex hull)
                    hull_points = new_points.reshape(-1, 2).astype(np.int32)
                    cv2.polylines(frame, [hull_points], True, (0, 255, 0), 2)
                    
                    # Draw mean-based depth contours
                    if mean_contours:
                        cv2.drawContours(frame, mean_contours, -1, (255, 0, 255), 3)  # Magenta contours
                        
                        # Fill the largest contour with semi-transparent color
                        if len(mean_contours) > 0:
                            largest_contour = max(mean_contours, key=cv2.contourArea)
                            overlay = frame.copy()
                            cv2.fillPoly(overlay, [largest_contour], (255, 100, 255))  # Light magenta fill
                            cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
                    
                    # Update tracking points
                    self.tracking_points = new_points
                
                # Add enhanced info overlay with performance stats
                good_points = np.sum(status) if status is not None else 0
                total_points = len(self.tracking_points) if self.tracking_points is not None else 0
                confidence = good_points / total_points if total_points > 0 else 0
                
                # Main tracking info
                cv2.putText(frame, f"Tracking: {good_points}/{total_points} points", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Depth info
                if mean_depth_range and new_points is not None and len(new_points) > 3:
                    depth_min, depth_max = mean_depth_range
                    cv2.putText(frame, f"Mean depth range: {depth_min:.2f} - {depth_max:.2f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                
                # Performance statistics
                cv2.putText(frame, f"Corrections: {self.performance_stats['corrections']}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Outliers: {self.performance_stats['outliers_detected']}", 
                           (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Depth fails: {self.performance_stats['depth_failures']}", 
                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.putText(frame, f"Auto resets: {self.performance_stats['auto_resets']}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Original depth and control info
                cv2.putText(frame, f"Original depth range: {self.depth_range[0]:.2f} - {self.depth_range[1]:.2f}" if self.depth_range else "No depth range", 
                           (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Frame: {self.total_frames}", 
                           (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Press 'r' to reset, 'q' to quit, 'd' for depth mask, 's' for stats", 
                           (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Enhanced Pipeline - Mean-Based Blob Tracking', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Resetting pipeline...")
                    # Re-run initial setup
                    initial_frame = self.countdown_and_capture(cap)
                    if initial_frame is not None:
                        self.reference_depth_map = self.generate_depth_map(initial_frame)
                        mask = self.segment_center_point(initial_frame)
                        segment_depths, depth_range = self.extract_segment_depth(self.reference_depth_map, mask)
                        border_points = self.disperse_border_points(mask)
                        old_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
                        self.tracking_history = []
                        # Reset performance stats
                        self.performance_stats = {
                            'corrections': 0,
                            'outliers_detected': 0,
                            'depth_failures': 0,
                            'auto_resets': 0
                        }
                        self.total_frames = 0
                elif key == ord('d'):
                    # Toggle depth mask display
                    show_depth_mask = not show_depth_mask
                    if show_depth_mask:
                        print("Showing depth mask window")
                    else:
                        print("Hiding depth mask window")
                        cv2.destroyWindow('Depth Mask')
                elif key == ord('s'):
                    # Print detailed statistics
                    print("\n" + "="*50)
                    print("PERFORMANCE STATISTICS")
                    print("="*50)
                    print(f"Total frames processed: {self.total_frames}")
                    print(f"Total corrections made: {self.performance_stats['corrections']}")
                    print(f"Total outliers detected: {self.performance_stats['outliers_detected']}")
                    print(f"Total depth validation failures: {self.performance_stats['depth_failures']}")
                    print(f"Total auto-resets: {self.performance_stats['auto_resets']}")
                    if self.total_frames > 0:
                        print(f"Correction rate: {self.performance_stats['corrections']/self.total_frames:.3f} per frame")
                        print(f"Outlier rate: {self.performance_stats['outliers_detected']/self.total_frames:.3f} per frame")
                    print("="*50)
                
                # Update for next iteration
                old_gray = frame_gray.copy()
        
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"An error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Enhanced pipeline ended.")

def main():
    """Main function"""
    try:
        pipeline = EnhancedCombinedPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()