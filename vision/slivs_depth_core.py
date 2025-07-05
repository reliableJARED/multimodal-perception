"""
SLIVS Depth Layer Processor - Modular Pipeline Component
Provides depth estimation, layer creation, and point detection for SAM integration

@article{DBLP:journals/corr/abs-2103-13413,
  author    = {Ren{\'{e}} Reiner Birkl, Diana Wofk, Matthias Muller},
  title     = {MiDaS v3.1 - A Model Zoo for Robust Monocular Relative Depth Estimation},
  journal   = {CoRR},
  volume    = {abs/2307.14460},
  year      = {2021},
  url       = {https://arxiv.org/abs/2307.14460},
  eprinttype = {arXiv},
  eprint    = {2307.14460},
  timestamp = {Wed, 26 Jul 2023},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2307-14460.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time
import math
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

@dataclass
class DepthLayerConfig:
    """Configuration for a single depth layer"""
    min_depth: int
    max_depth: int
    label: str

@dataclass
class GridInfo:
    """Information about the grid used for point detection"""
    square_size: int
    squares_height: int
    squares_width: int
    total_squares: int

@dataclass
class LayerResult:
    """Result for a single depth layer"""
    layer_mask: np.ndarray
    points: List[Tuple[int, int]]
    grid_info: GridInfo
    config: DepthLayerConfig

@dataclass
class DepthProcessingResult:
    """Complete result from depth processing"""
    full_depth_map: np.ndarray
    layers: List[LayerResult]
    all_points: List[Tuple[int, int]]
    processing_time: float

class SLIVSDepthProcessor:
    """
    Modular depth processing class for SLIVS pipeline.
    Handles depth estimation, layer creation, and point detection.
    """
    
    def __init__(self, 
                 model_name: str = "Intel/dpt-swinv2-tiny-256",
                 target_squares: int = 100,
                 min_fill_threshold: float = 0.7,
                 device: Optional[str] = None,
                 depth_layer_config: Optional[Dict[str, DepthLayerConfig]] = None):
        """
        Initialize the depth processor.
        
        Args:
            model_name: MiDaS model to use
            target_squares: Target number of grid squares for point detection
            min_fill_threshold: Minimum fill ratio for valid squares (0.0-1.0)
            device: Device to use ('cuda', 'cpu', or None for auto)
            depth_layer_config: Custom depth layer configuration. If None, uses default layers.

        """
        self.target_squares = target_squares
        self.min_fill_threshold = min_fill_threshold
        
        # Set depth layer configuration
        if depth_layer_config is None:
            # Default depth layer range and name configuration,
            self.depth_layers = {
                "furthest": DepthLayerConfig(0, 25, "Furthest"),
                "far": DepthLayerConfig(26, 50, "Far"),
                "mid": DepthLayerConfig(51, 75, "Mid"),
                "near": DepthLayerConfig(76, 150, "Near"),
                "closest": DepthLayerConfig(151, 255, "Closest"),
            }
        else:
            self.depth_layers = depth_layer_config
        
        # Initialize device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load MiDaS model
        print(f"Loading MiDaS model: {model_name}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device).eval()
        
        print(f"SLIVS Depth Processor initialized on {self.device}")
        print(f"Configured with {len(self.depth_layers)} depth layers")
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate depth estimation using MiDaS.
        
        Args:
            frame: Input RGB frame (H, W, 3)
            
        Returns:
            Normalized depth map (H, W) with values 0-255
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Resize to original dimensions
        height, width = frame.shape[:2]
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to 0-255 range
        depth_map = prediction.squeeze().cpu().numpy()
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_normalized
    
    def calculate_optimal_grid(self, height: int, width: int) -> GridInfo:
        """
        Calculate optimal grid size based on GCD to get close to target squares.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            GridInfo with optimal grid parameters
        """
        gcd = math.gcd(height, width)
        
        # Find divisors of the GCD
        divisors = []
        for i in range(1, int(math.sqrt(gcd)) + 1):
            if gcd % i == 0:
                divisors.append(i)
                if i != gcd // i:
                    divisors.append(gcd // i)
        
        divisors.sort()
        
        # Find the divisor that gives us closest to target squares
        best_divisor = divisors[0]
        best_squares = float('inf')
        
        for divisor in divisors:
            squares_h = height // divisor
            squares_w = width // divisor
            total_squares = squares_h * squares_w
            
            # We want at least target squares but not too many more
            if total_squares >= self.target_squares:
                if total_squares < best_squares:
                    best_squares = total_squares
                    best_divisor = divisor
        
        # If no divisor gives us enough squares, use the largest one
        if best_squares == float('inf'):
            best_divisor = divisors[-1]
        
        square_size = best_divisor
        squares_h = height // square_size
        squares_w = width // square_size
        total_squares = squares_h * squares_w
        
        return GridInfo(square_size, squares_h, squares_w, total_squares)
    
    def find_layer_points(self, layer_mask: np.ndarray) -> Tuple[List[Tuple[int, int]], GridInfo]:
        """
        Find points in grid squares with sufficient non-black pixels.
        
        Args:
            layer_mask: Binary layer mask (H, W)
            
        Returns:
            Tuple of (points_list, grid_info)
        """
        height, width = layer_mask.shape
        grid_info = self.calculate_optimal_grid(height, width)
        
        # First pass: find all qualifying squares
        qualifying_squares = []
        
        # Go through each square in the grid
        for row in range(grid_info.squares_height):
            for col in range(grid_info.squares_width):
                # Calculate square boundaries
                start_y = row * grid_info.square_size
                end_y = min(start_y + grid_info.square_size, height)
                start_x = col * grid_info.square_size
                end_x = min(start_x + grid_info.square_size, width)
                
                # Extract the square
                square = layer_mask[start_y:end_y, start_x:end_x]
                
                # Count non-black pixels (assuming black = 0)
                non_black_pixels = np.count_nonzero(square)
                total_pixels = square.size
                
                # Check if square meets the fill threshold
                if total_pixels > 0 and (non_black_pixels / total_pixels) >= self.min_fill_threshold:
                    center_y = start_y + (end_y - start_y) // 2
                    center_x = start_x + (end_x - start_x) // 2
                    qualifying_squares.append((row, col, center_x, center_y))
        
        # Second pass: combine points using neighbor check
        points = self._combine_neighboring_points(qualifying_squares, grid_info)
        
        return points, grid_info
    
    def _combine_neighboring_points(self, qualifying_squares: List[Tuple], grid_info: GridInfo) -> List[Tuple[int, int]]:
        """
        Combine neighboring points using 3x3 window analysis.
        
        Args:
            qualifying_squares: List of (row, col, center_x, center_y) tuples
            grid_info: Grid information
            
        Returns:
            List of combined points as (x, y) tuples
        """
        # Create a grid to track which squares have points
        grid = np.zeros((grid_info.squares_height, grid_info.squares_width), dtype=bool)
        square_centers = {}
        
        # Mark qualifying squares in grid and store their centers
        for row, col, center_x, center_y in qualifying_squares:
            grid[row, col] = True
            square_centers[(row, col)] = (center_x, center_y)
        
        # Track which squares have been processed
        processed = np.zeros((grid_info.squares_height, grid_info.squares_width), dtype=bool)
        combined_points = []
        
        # Process the grid in 3x3 windows
        for window_row in range(0, grid_info.squares_height, 3):
            for window_col in range(0, grid_info.squares_width, 3):
                # Extract points in this 3x3 window
                window_points = []
                for r in range(window_row, min(window_row + 3, grid_info.squares_height)):
                    for c in range(window_col, min(window_col + 3, grid_info.squares_width)):
                        if grid[r, c] and not processed[r, c]:
                            window_points.append((r, c))
                
                if not window_points:
                    continue
                
                # Combine points within this 3x3 window
                combined_in_window = self._combine_points_in_3x3_window(window_points, square_centers)
                combined_points.extend(combined_in_window)
                
                # Mark all points in this window as processed
                for r, c in window_points:
                    processed[r, c] = True
        
        return combined_points
    
    def _combine_points_in_3x3_window(self, window_points: List[Tuple], square_centers: Dict) -> List[Tuple[int, int]]:
        """
        Combine points within a single 3x3 window by rows and columns.
        
        Args:
            window_points: Points in the 3x3 window
            square_centers: Mapping of grid positions to center coordinates
            
        Returns:
            List of combined points
        """
        if len(window_points) <= 1:
            # Single point or no points - just return as is
            return [square_centers[point] for point in window_points]
        
        # Convert to relative coordinates within the 3x3 window
        min_row = min(r for r, c in window_points)
        min_col = min(c for r, c in window_points)
        
        # Create a 3x3 local grid
        local_grid = {}
        for r, c in window_points:
            local_r = r - min_row
            local_c = c - min_col
            local_grid[(local_r, local_c)] = (r, c)
        
        # Special case: if we have a full 3x3 grid (9 points), combine to single center
        if len(window_points) == 9 and (1, 1) in local_grid:
            center_point = local_grid[(1, 1)]  # Center of 3x3
            return [square_centers[center_point]]
        
        combined_points = []
        used_points = set()
        
        # Check for 3-in-a-row (horizontal combinations)
        for row in range(3):
            row_points = [(row, col) for col in range(3) if (row, col) in local_grid]
            if len(row_points) >= 3:
                # Combine to center column (col=1)
                original_coords = [local_grid[p] for p in row_points]
                center_point = local_grid[(row, 1)]  # Middle of the row
                combined_points.append(square_centers[center_point])
                used_points.update(original_coords)
        
        # Check for 3-in-a-column (vertical combinations)
        for col in range(3):
            col_points = [(row, col) for row in range(3) if (row, col) in local_grid]
            if len(col_points) >= 3:
                # Combine to center row (row=1)
                original_coords = [local_grid[p] for p in col_points]
                # Only combine if not already used by row combination
                if not any(coord in used_points for coord in original_coords):
                    center_point = local_grid[(1, col)]  # Middle of the column
                    combined_points.append(square_centers[center_point])
                    used_points.update(original_coords)
        
        # Add any remaining individual points that weren't combined
        for local_pos, original_pos in local_grid.items():
            if original_pos not in used_points:
                combined_points.append(square_centers[original_pos])
        
        return combined_points
    
    def create_layer_mask(self, depth_map: np.ndarray, config: DepthLayerConfig) -> np.ndarray:
        """
        Create a layer mask for a specific depth range.
        
        Args:
            depth_map: Full depth map (H, W)
            config: Layer configuration
            
        Returns:
            Layer mask (H, W) with non-zero values only in the depth range
        """
        layer_depth = depth_map.copy()
        mask = (layer_depth < config.min_depth) | (layer_depth > config.max_depth)
        layer_depth[mask] = 0
        return layer_depth
    
    def process_frame(self, frame: np.ndarray) -> DepthProcessingResult:
        """
        Process a single frame to extract depth layers and points.
        
        Args:
            frame: Input RGB frame (H, W, 3)
            
        Returns:
            DepthProcessingResult with all processing outputs
        """
        start_time = time.time()
        
        # Generate depth map
        depth_map = self.estimate_depth(frame)
        
        # Process each depth layer
        layers = []
        all_points = []
        
        for layer_name, config in self.depth_layers.items():
            # Create layer mask
            layer_mask = self.create_layer_mask(depth_map, config)
            
            # Find points for this layer
            points, grid_info = self.find_layer_points(layer_mask)
            
            # Create layer result
            layer_result = LayerResult(
                layer_mask=layer_mask,
                points=points,
                grid_info=grid_info,
                config=config
            )
            
            layers.append(layer_result)
            all_points.extend(points)
        
        processing_time = time.time() - start_time
        
        return DepthProcessingResult(
            full_depth_map=depth_map,
            layers=layers,
            all_points=all_points,
            processing_time=processing_time
        )
    
    def get_depth_map(self, frame: np.ndarray) -> np.ndarray:
        """
        Get just the depth map without layer processing.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Depth map (H, W)
        """
        return self.estimate_depth(frame)
    
    def get_layer_masks(self, depth_map: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get layer masks for a given depth map.
        
        Args:
            depth_map: Input depth map
            
        Returns:
            Dictionary mapping layer names to masks
        """
        masks = {}
        for layer_name, config in self.depth_layers.items():
            masks[layer_name] = self.create_layer_mask(depth_map, config)
        return masks
    
    def get_all_points(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """
        Get all points across all layers for a frame.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            List of all points as (x, y) tuples
        """
        result = self.process_frame(frame)
        return result.all_points
    
    def get_layer_points(self, frame: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Get points organized by layer.
        
        Args:
            frame: Input RGB frame
            
        Returns:
            Dictionary mapping layer names to point lists
        """
        result = self.process_frame(frame)
        layer_points = {}
        for i, (layer_name, _) in enumerate(self.depth_layers.items()):
            layer_points[layer_name] = result.layers[i].points
        return layer_points

