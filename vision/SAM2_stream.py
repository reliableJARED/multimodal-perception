"""
Depth-Aware SAM2 Object Tracker
Combines SAM2 segmentation, MiDAS depth estimation, and multi-modal template matching
for robust object tracking across video frames.
"""
import cv2
import torch
import numpy as np
import time
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

@dataclass
class ObjectState:
    """Stores the current state of the tracked object"""
    centroid: np.ndarray  # [x, y] position
    velocity: np.ndarray  # [dx, dy] velocity
    depth_centroid: float  # Average depth of object
    depth_range: Tuple[float, float]  # (min_depth, max_depth)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    confidence: float  # Tracking confidence
    last_seen_frame: int  # Frame number when last successfully tracked

@dataclass
class Templates:
    """Stores template data for matching"""
    color_template: np.ndarray  # RGB color template
    depth_template: np.ndarray  # Depth template
    template_bbox: Tuple[int, int, int, int]  # Template bounding box
    template_centroid: np.ndarray  # Template center

class DepthAwareSAM2Tracker:
    def __init__(self, 
                 sam_model="facebook/sam2-hiera-small",
                 depth_model="Intel/dpt-swinv2-tiny-256",
                 search_radius=50,
                 template_size=(64, 64),
                 resegment_interval=15):
        """
        Initialize the depth-aware tracker
        
        Args:
            sam_model: SAM2 model name
            depth_model: MiDAS depth model name
            search_radius: Pixel radius for template search
            template_size: Size of templates (width, height)
            resegment_interval: Frames between SAM2 re-segmentation
        """
        self.search_radius = search_radius
        self.template_size = template_size
        self.resegment_interval = resegment_interval
        
        # Initialize models
        self._setup_models(sam_model, depth_model)
        
        # Tracking state
        self.object_state: Optional[ObjectState] = None
        self.templates: Optional[Templates] = None
        self.frame_count = 0
        self.tracking_initialized = False
        
        # Tracking parameters
        self.velocity_smoothing = 0.7
        self.depth_tolerance = 0.15  # 15% tolerance for depth matching
        self.confidence_threshold = 0.6
        self.max_frames_lost = 10
        
    def _setup_models(self, sam_model: str, depth_model: str):
        """Initialize SAM2 and MiDAS models"""
        print("Setting up models...")
        
        # Device selection
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device = "mps" 
            self.dtype = torch.float16
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            
        print(f"Using device: {self.device}")
        
        # Load SAM2
        try:
            self.sam_predictor = SAM2ImagePredictor.from_pretrained(sam_model, device=self.device)
            print(f"✓ SAM2 model loaded: {sam_model}")
        except Exception as e:
            print(f"✗ Failed to load SAM2: {e}")
            raise
            
        # Load MiDAS
        try:
            self.depth_processor = AutoImageProcessor.from_pretrained(depth_model)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(depth_model)
            self.depth_model.to(self.device)
            self.depth_model.eval()
            print(f"✓ MiDAS model loaded: {depth_model}")
        except Exception as e:
            print(f"✗ Failed to load MiDAS: {e}")
            raise
    
    def predict_depth(self, image: np.ndarray) -> np.ndarray:
        """Predict depth map from RGB image"""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL
        pil_image = Image.fromarray(image_rgb)
        
        # Process
        inputs = self.depth_processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            predicted_depth = outputs.predicted_depth
            
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        return prediction.squeeze().cpu().numpy()
    
    def initialize_tracking(self, frame: np.ndarray, point: Tuple[int, int]) -> bool:
        """Initialize tracking with a point click"""
        print(f"Initializing tracking at point {point}")
        
        try:
            # Get initial segmentation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(frame_rgb)
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            if len(masks) == 0:
                print("No masks generated by SAM2")
                return False
            
            # Use best mask
            best_mask = masks[np.argmax(scores)]
            
            # Ensure mask is boolean
            if best_mask.dtype != bool:
                best_mask = best_mask.astype(bool)
                
            print(f"Mask shape: {best_mask.shape}, dtype: {best_mask.dtype}")
            print(f"Mask has {np.sum(best_mask)} pixels")
            
            if np.sum(best_mask) == 0:
                print("Empty mask generated")
                return False
            
            # Get depth map
            print("Getting depth map...")
            depth_map = self.predict_depth(frame)
            print(f"Depth map shape: {depth_map.shape}, dtype: {depth_map.dtype}")
            
            # Initialize object state and templates
            print("Creating object state...")
            self._create_object_state(frame, best_mask, depth_map)
            print("Creating templates...")
            self._create_templates(frame, best_mask, depth_map)
            
            self.frame_count = 0
            self.tracking_initialized = True
            
            print(f"✓ Tracking initialized with confidence: {self.object_state.confidence:.3f}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize tracking: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_object_state(self, frame: np.ndarray, mask: np.ndarray, depth_map: np.ndarray):
        """Create initial object state from mask and depth"""
        print(f"Creating object state - mask dtype: {mask.dtype}, shape: {mask.shape}")
        
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask.astype(bool)
        
        # Calculate centroid
        y_coords, x_coords = np.where(mask)
        if len(x_coords) == 0:
            raise ValueError("No pixels found in mask")
        
        print(f"Found {len(x_coords)} mask pixels")
        centroid = np.array([float(np.mean(x_coords)), float(np.mean(y_coords))])
        
        # Calculate bounding box
        bbox = (int(x_coords.min()), int(y_coords.min()), 
                int(x_coords.max() - x_coords.min()), 
                int(y_coords.max() - y_coords.min()))
        
        print(f"Bbox: {bbox}, Centroid: {centroid}")
        
        # Calculate depth statistics
        try:
            object_depths = depth_map[mask]
            print(f"Extracted {len(object_depths)} depth values")
            
            if len(object_depths) == 0:
                depth_centroid = float(np.median(depth_map))
                depth_range = (float(np.min(depth_map)), float(np.max(depth_map)))
            else:
                depth_centroid = float(np.median(object_depths))
                depth_range = (float(np.percentile(object_depths, 10)), float(np.percentile(object_depths, 90)))
                
            print(f"Depth centroid: {depth_centroid}, range: {depth_range}")
            
        except Exception as e:
            print(f"Error extracting depth: {e}")
            depth_centroid = 1.0
            depth_range = (0.0, 2.0)
        
        self.object_state = ObjectState(
            centroid=centroid,
            velocity=np.array([0.0, 0.0]),
            depth_centroid=depth_centroid,
            depth_range=depth_range,
            bbox=bbox,
            confidence=1.0,
            last_seen_frame=0
        )
    
    def _create_templates(self, frame: np.ndarray, mask: np.ndarray, depth_map: np.ndarray):
        """Create color and depth templates from segmented object"""
        print(f"Creating templates - mask dtype: {mask.dtype}, shape: {mask.shape}")
        
        # Ensure mask is boolean
        if mask.dtype != bool:
            mask = mask.astype(bool)
            
        # Get bounding box with some padding
        center_x, center_y = int(self.object_state.centroid[0]), int(self.object_state.centroid[1])
        
        # Extract template region
        tw, th = self.template_size
        x1 = max(0, center_x - tw // 2)
        y1 = max(0, center_y - th // 2)
        x2 = min(frame.shape[1], center_x + tw // 2)
        y2 = min(frame.shape[0], center_y + th // 2)
        
        # Ensure we have valid coordinates
        if x2 <= x1 or y2 <= y1:
            # Fallback to center region
            x1, y1 = max(0, center_x - 32), max(0, center_y - 32)
            x2, y2 = min(frame.shape[1], center_x + 32), min(frame.shape[0], center_y + 32)
        
        print(f"Template region: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Extract templates
        color_template = frame[y1:y2, x1:x2].copy()
        depth_template = depth_map[y1:y2, x1:x2].copy()
        
        # Extract template mask and ensure it's boolean
        template_mask = mask[y1:y2, x1:x2]
        if template_mask.dtype != bool:
            template_mask = template_mask.astype(bool)
        
        print(f"Template mask shape: {template_mask.shape}, dtype: {template_mask.dtype}")
        print(f"Template mask has {np.sum(template_mask)} True pixels")
        
        # Mask templates to object region only
        if template_mask.any():
            # Create inverted mask safely
            inv_mask = ~template_mask
            color_template[inv_mask] = 0
            depth_template[inv_mask] = 0
        
        self.templates = Templates(
            color_template=color_template,
            depth_template=depth_template,
            template_bbox=(int(x1), int(y1), int(x2-x1), int(y2-y1)),
            template_centroid=np.array([float(center_x - x1), float(center_y - y1)])
        )
        
        print(f"Templates created successfully - color: {color_template.shape}, depth: {depth_template.shape}")
    
    def track_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Track object in new frame"""
        if not self.tracking_initialized:
            return None, {"status": "not_initialized"}
        
        self.frame_count += 1
        
        # Get depth map
        depth_map = self.predict_depth(frame)
        
        # Predict next position based on velocity
        predicted_pos = self.object_state.centroid + self.object_state.velocity
        
        # Search for object using templates
        match_result = self._search_with_templates(frame, depth_map, predicted_pos)
        
        # Update object state
        if match_result["confidence"] > self.confidence_threshold:
            self._update_object_state(match_result)
            
            # Re-segment periodically or when confidence is low
            if (self.frame_count % self.resegment_interval == 0 or 
                match_result["confidence"] < 0.8):
                return self._resegment_object(frame, predicted_pos), match_result
            else:
                # Return predicted mask based on template matching
                return self._generate_predicted_mask(frame.shape[:2], match_result), match_result
        else:
            # Tracking lost
            frames_lost = self.frame_count - self.object_state.last_seen_frame
            if frames_lost > self.max_frames_lost:
                self.tracking_initialized = False
                return None, {"status": "tracking_lost", "frames_lost": frames_lost}
            
            # Try to re-segment at predicted position
            return self._resegment_object(frame, predicted_pos), match_result
    
    def _search_with_templates(self, frame: np.ndarray, depth_map: np.ndarray, 
                             predicted_pos: np.ndarray) -> Dict[str, Any]:
        """Search for object using color and depth templates"""
        if self.templates is None:
            return {"confidence": 0.0}
        
        # Define search region
        center_x, center_y = int(predicted_pos[0]), int(predicted_pos[1])
        search_x1 = max(0, center_x - self.search_radius)
        search_y1 = max(0, center_y - self.search_radius)
        search_x2 = min(frame.shape[1], center_x + self.search_radius)
        search_y2 = min(frame.shape[0], center_y + self.search_radius)
        
        if search_x2 <= search_x1 or search_y2 <= search_y1:
            return {"confidence": 0.0}
        
        # Extract search regions
        color_search = frame[search_y1:search_y2, search_x1:search_x2]
        depth_search = depth_map[search_y1:search_y2, search_x1:search_x2]
        
        # Template matching for color
        color_result = cv2.matchTemplate(color_search, self.templates.color_template, 
                                       cv2.TM_CCOEFF_NORMED)
        
        # Template matching for depth
        depth_result = cv2.matchTemplate(depth_search, self.templates.depth_template, 
                                       cv2.TM_CCOEFF_NORMED)
        
        # Find best matches
        _, color_max, _, color_loc = cv2.minMaxLoc(color_result)
        _, depth_max, _, depth_loc = cv2.minMaxLoc(depth_result)
        
        # Combine scores (weighted)
        color_weight = 0.7
        depth_weight = 0.3
        
        # Check if color and depth matches are close
        loc_distance = np.linalg.norm(np.array(color_loc) - np.array(depth_loc))
        max_distance = 20  # pixels
        
        if loc_distance <= max_distance:
            # Use average location and combined confidence
            combined_confidence = color_weight * color_max + depth_weight * depth_max
            avg_loc = ((np.array(color_loc) + np.array(depth_loc)) / 2).astype(int)
        else:
            # Use color matching result if depth is inconsistent
            combined_confidence = color_max * 0.8  # Penalty for depth inconsistency
            avg_loc = color_loc
        
        # Convert back to global coordinates
        global_x = search_x1 + avg_loc[0] + self.templates.template_bbox[2] // 2
        global_y = search_y1 + avg_loc[1] + self.templates.template_bbox[3] // 2
        
        # Verify depth consistency
        global_y_int, global_x_int = int(global_y), int(global_x)
        if 0 <= global_y_int < depth_map.shape[0] and 0 <= global_x_int < depth_map.shape[1]:
            depth_at_match = depth_map[global_y_int, global_x_int]
            depth_diff = abs(depth_at_match - self.object_state.depth_centroid) / max(self.object_state.depth_centroid, 1e-6)
            
            if depth_diff > self.depth_tolerance:
                combined_confidence *= 0.7  # Penalty for depth inconsistency
        else:
            combined_confidence *= 0.5  # Penalty for out-of-bounds
        
        return {
            "confidence": combined_confidence,
            "position": np.array([float(global_x), float(global_y)]),
            "color_score": color_max,
            "depth_score": depth_max,
            "depth_consistency": 1.0 - min(depth_diff / self.depth_tolerance, 1.0) if 'depth_diff' in locals() else 0.5
        }
    
    def _update_object_state(self, match_result: Dict[str, Any]):
        """Update object state based on tracking result"""
        new_pos = match_result["position"]
        
        # Update velocity with smoothing
        velocity = new_pos - self.object_state.centroid
        self.object_state.velocity = (self.velocity_smoothing * velocity + 
                                    (1 - self.velocity_smoothing) * self.object_state.velocity)
        
        # Update position and confidence
        self.object_state.centroid = new_pos
        self.object_state.confidence = match_result["confidence"]
        self.object_state.last_seen_frame = self.frame_count
    
    def _resegment_object(self, frame: np.ndarray, position: np.ndarray) -> Optional[np.ndarray]:
        """Re-segment object using SAM2 at predicted position"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(frame_rgb)
            
            point = (int(position[0]), int(position[1]))
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=np.array([point]),
                point_labels=np.array([1]),
                multimask_output=True
            )
            
            if len(masks) > 0:
                best_mask = masks[np.argmax(scores)]
                
                # Ensure mask is boolean
                if best_mask.dtype != bool:
                    best_mask = best_mask.astype(bool)
                
                # Update templates with new segmentation
                depth_map = self.predict_depth(frame)
                self._create_object_state(frame, best_mask, depth_map)
                self._create_templates(frame, best_mask, depth_map)
                
                return best_mask
            
        except Exception as e:
            print(f"Re-segmentation failed: {e}")
        
        return None
    
    def _generate_predicted_mask(self, frame_shape: Tuple[int, int], 
                               match_result: Dict[str, Any]) -> np.ndarray:
        """Generate predicted mask based on template matching"""
        mask = np.zeros(frame_shape, dtype=bool)
        
        if self.templates is None:
            return mask
        
        # Create mask at predicted position
        pos = match_result["position"]
        tw, th = self.templates.template_bbox[2], self.templates.template_bbox[3]
        
        x1 = max(0, int(pos[0] - tw // 2))
        y1 = max(0, int(pos[1] - th // 2))
        x2 = min(frame_shape[1], int(pos[0] + tw // 2))
        y2 = min(frame_shape[0], int(pos[1] + th // 2))
        
        if x2 > x1 and y2 > y1:
            # Simple rectangular mask for now - could be improved with template shape
            mask[y1:y2, x1:x2] = True
        
        return mask
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get current tracking state for debugging"""
        if not self.tracking_initialized or self.object_state is None:
            return {"initialized": False}
        
        return {
            "initialized": True,
            "frame_count": self.frame_count,
            "centroid": self.object_state.centroid.tolist(),
            "velocity": self.object_state.velocity.tolist(),
            "confidence": self.object_state.confidence,
            "depth_centroid": self.object_state.depth_centroid,
            "depth_range": self.object_state.depth_range,
            "frames_since_last_seen": self.frame_count - self.object_state.last_seen_frame
        }

def main():
    """Demo the depth-aware tracker with automatic center initialization"""
    tracker = DepthAwareSAM2Tracker()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting automatic tracking at center. Press 'q' to quit, 'r' to reset.")
    
    tracking_active = False
    initialization_attempted = False
    
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        
        # Auto-initialize at center on first frame
        if not initialization_attempted:
            height, width = frame.shape[:2]
            center_point = (width // 2, height // 2)
            
            print(f"Auto-initializing tracking at center point {center_point}")
            if tracker.initialize_tracking(frame, center_point):
                tracking_active = True
                print("Automatic tracking started!")
            else:
                print("Failed to initialize tracking at center")
            
            initialization_attempted = True
        
        if tracking_active:
            # Track object
            mask, track_info = tracker.track_frame(frame)
            
            if mask is not None:
                # Ensure mask is boolean for indexing
                if mask.dtype != bool:
                    mask = mask.astype(bool)
                
                # Overlay mask
                overlay = display_frame.copy()
                overlay[mask] = [0, 255, 0]  # Green overlay
                display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                
                # Draw tracking info
                debug_info = tracker.get_debug_info()
                if debug_info["initialized"]:
                    centroid = debug_info["centroid"]
                    cv2.circle(display_frame, tuple(map(int, centroid)), 5, (0, 0, 255), -1)
                    
                    # Display info
                    info_text = [
                        f"Conf: {debug_info['confidence']:.3f}",
                        f"Vel: ({debug_info['velocity'][0]:.1f}, {debug_info['velocity'][1]:.1f})",
                        f"Depth: {debug_info['depth_centroid']:.2f}",
                        f"Frame: {debug_info['frame_count']}"
                    ]
                    
                    for i, text in enumerate(info_text):
                        cv2.putText(display_frame, text, (10, 30 + i * 25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                    # Draw center point indicator
                    center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                    cv2.circle(display_frame, (center_x, center_y), 3, (255, 0, 0), -1)
                    cv2.putText(display_frame, "Auto-init center", (center_x + 10, center_y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                tracking_active = False
                cv2.putText(display_frame, "Tracking Lost - Press 'r' to restart", 
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            if initialization_attempted:
                cv2.putText(display_frame, "Press 'r' to restart tracking at center", 
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Initializing tracking...", 
                          (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            # Always show center point
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            cv2.circle(display_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.putText(display_frame, "Tracking center", (center_x + 10, center_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Calculate and display FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30 / (time.time() - fps_start_time)
            fps_start_time = time.time()
            print(f"FPS: {fps:.1f}")
        
        cv2.imshow('Depth-Aware Tracker', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            tracking_active = False
            tracker.tracking_initialized = False
            initialization_attempted = False
            print("Tracking reset - will auto-initialize at center")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()