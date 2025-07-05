"""
SLIVS SAM2 Processor - Modular Pipeline Component
Provides SAM2 segmentation integration for point-based object detection

Integrates with SLIVSDepthProcessor to create comprehensive object understanding

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and Rädle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Dollár, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
"""

import torch
import numpy as np
import cv2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import os
import urllib.request
import sys
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import uuid


@dataclass
class SAM2Config:
    """Configuration for SAM2 model"""
    model_name: str = "sam2.1_hiera_tiny"  # tiny, small, base_plus, large
    checkpoint_dir: str = "./models/checkpoints"
    device: Optional[str] = None
    multimask_output: bool = True
    min_mask_confidence: float = 0.5 #if No mask at this level nothing is returned.
    use_highest_confidence: bool = True


@dataclass
class SegmentResult:
    """Result from SAM2 segmentation for a single point"""
    segment_id: str
    mask: np.ndarray
    confidence: float
    point_prompt: Tuple[int, int]
    bounding_box: Tuple[int, int, int, int]  # [x1, y1, x2, y2]
    area: int
    logits: Optional[np.ndarray] = None


@dataclass
class SAM2ProcessingResult:
    """Complete result from SAM2 processing"""
    segments: List[SegmentResult]
    all_masks: np.ndarray  # Combined masks for all segments
    point_prompts: List[Tuple[int, int]]
    processing_time: float
    frame_shape: Tuple[int, int]  # [height, width]


class SLIVSSam2Processor:
    """
    Modular SAM2 processing class for SLIVS pipeline.
    Handles SAM2 model loading and point-based segmentation API wrapping.
    """
    
    def __init__(self, config: SAM2Config = None):
        """
        Initialize the SAM2 processor.
        
        Args:
            config: SAM2 configuration object
        """
        if config is None:
            print(f"SLIVSSam2Processor instance using default configuration SAM2Config")
            self.config = SAM2Config()
        else:
            print(f"SLIVSSam2Processor instance using custom configuration SAM2Config")
            self.config = config
        
        # Model URLs for downloading
        self.model_urls = {
            "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
            "sam2.1_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", 
            "sam2.1_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
        }
        
        # Config file mapping
        #Hydra manages the config loading - SAM2 uses Facebook's Hydra framework (THESE ARE NOT LOCAL DIRECTORIES - DON"T CHANGE)
        self.config_map = {
            "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
        
        # Initialize device
        if self.config.device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        # Initialize model
        self.predictor = None
        self.current_frame = None
        self._load_model()
        
        print(f"SLIVS SAM2 Processor initialized with {self.config.model_name} on {self.device}")
    
    def _download_checkpoint(self, checkpoint_path: str) -> bool:
        """Download SAM2 checkpoint if it doesn't exist."""
        if self.config.model_name not in self.model_urls:
            raise ValueError(f"Unknown model: {self.config.model_name}")
        
        url = self.model_urls[self.config.model_name]
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
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
            print(f"\n✓ Successfully downloaded {self.config.model_name}")
            return True
        except Exception as e:
            print(f"\n✗ Failed to download checkpoint: {e}")
            return False
    
    def _load_model(self):
        """Load the SAM2 model and create predictor."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, 
            f"{self.config.model_name}.pt"
        )
        
        if not os.path.exists(checkpoint_path):
            if not self._download_checkpoint(checkpoint_path):
                raise RuntimeError("Failed to obtain SAM2 checkpoint")
        
        model_cfg = self.config_map[self.config.model_name]
        sam2_model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
    
    def set_image(self, frame: np.ndarray):
        """
        Set the current image for segmentation.
        
        Args:
            frame: Input RGB frame (H, W, 3)
        """
        if frame.shape[2] == 3:
            # Assume BGR input (OpenCV format), convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        self.current_frame = rgb_frame
        self.predictor.set_image(rgb_frame)
    
    def segment_from_point(self, point: Tuple[int, int], 
                          label: int = 1) -> SegmentResult:
        """
        Generate segmentation for a single point prompt.
        Returns the highest confidence mask when multimask_output=True.
        
        Args:
            point: Point coordinates (x, y)
            label: Point label (1 = positive, 0 = negative)
            
        Returns:
            SegmentResult with the best mask for the point
        """
        if self.current_frame is None:
            raise RuntimeError("No image set. Call set_image() first.")
        
        input_points = np.array([point])
        input_labels = np.array([label])
        
        # Run segmentation
        with torch.inference_mode():
            if self.device == "mps":
                with torch.autocast(self.device, dtype=torch.float16):
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_points,
                        point_labels=input_labels,
                        multimask_output=self.config.multimask_output,
                    )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=self.config.multimask_output,
                )
        
        # Select best mask
        if self.config.use_highest_confidence and len(scores) > 1:
            best_idx = np.argmax(scores)
        else:
            best_idx = 0
            
        best_mask = masks[best_idx].astype(bool)
        best_score = scores[best_idx]
        best_logits = logits[best_idx] if logits is not None else None
        
        # Calculate bounding box and area
        bbox = self._calculate_bounding_box(best_mask)
        area = np.sum(best_mask)
        
        return SegmentResult(
            segment_id=str(uuid.uuid4()),
            mask=best_mask,
            confidence=float(best_score),
            point_prompt=point,
            bounding_box=bbox,
            area=int(area),
            logits=best_logits
        )
    
    def segment_from_points(self, points: List[Tuple[int, int]], 
                           labels: Optional[List[int]] = None) -> SAM2ProcessingResult:
        """
        Generate segmentation for multiple point prompts.
        
        Args:
            points: List of point coordinates [(x1, y1), (x2, y2), ...]
            labels: List of point labels (1 = positive, 0 = negative). 
                   If None, all points are treated as positive.
            
        Returns:
            SAM2ProcessingResult with all segments
        """
        if self.current_frame is None:
            raise RuntimeError("No image set. Call set_image() first.")
        
        if not points:
            return SAM2ProcessingResult(
                segments=[],
                all_masks=np.zeros(self.current_frame.shape[:2], dtype=bool),
                point_prompts=[],
                processing_time=0.0,
                frame_shape=self.current_frame.shape[:2]
            )
        
        start_time = time.time()
        
        # Set default labels if not provided
        if labels is None:
            labels = [1] * len(points)
        
        segments = []
        all_masks = np.zeros(self.current_frame.shape[:2], dtype=bool)
        
        # Process each point
        for point, label in zip(points, labels):
            try:
                segment = self.segment_from_point(point, label)
                
                # Only include segments above confidence threshold
                if segment.confidence >= self.config.min_mask_confidence:
                    segments.append(segment)
                    all_masks = all_masks | segment.mask.astype(bool)
                    
            except Exception as e:
                print(f"Warning: Failed to segment point {point}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        return SAM2ProcessingResult(
            segments=segments,
            all_masks=all_masks,
            point_prompts=points,
            processing_time=processing_time,
            frame_shape=self.current_frame.shape[:2]
        )
    
    # TODO: Implement multipoint prompting (multiple points per segment)

    # TODO: Implement automatic segmentation (segment everything)

    # TODO: Implement bounding box prompting

    def _calculate_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Calculate bounding box for a mask."""
        if not np.any(mask):
            return (0, 0, 0, 0)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def get_segment_by_id(self, segment_id: str, 
                         segments: List[SegmentResult]) -> Optional[SegmentResult]:
        """Get a segment by its ID."""
        for segment in segments:
            if segment.segment_id == segment_id:
                return segment
        return None
    
    def filter_segments_by_area(self, segments: List[SegmentResult], 
                               min_area: int = 100, 
                               max_area: Optional[int] = None) -> List[SegmentResult]:
        """Filter segments by area constraints."""
        filtered = []
        for segment in segments:
            if segment.area >= min_area:
                if max_area is None or segment.area <= max_area:
                    filtered.append(segment)
        return filtered
    
    def filter_segments_by_confidence(self, segments: List[SegmentResult], 
                                    min_confidence: float = 0.5) -> List[SegmentResult]:
        """Filter segments by confidence threshold."""
        return [seg for seg in segments if seg.confidence >= min_confidence]
    
    def update_config(self, new_config: SAM2Config):
        """
        Update configuration and reload model if necessary.
        
        Args:
            new_config: New configuration
        """
        model_changed = new_config.model_name != self.config.model_name
        device_changed = new_config.device != self.config.device
        
        self.config = new_config
        
        if model_changed or device_changed:
            print("Configuration changed, reloading model...")
            self._load_model()