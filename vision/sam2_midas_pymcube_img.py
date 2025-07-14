"""
Combined Camera Capture + MiDaS Depth + SAM2 Segmentation + Mesh Generation
Workflow:
1. Capture frame when camera is warmed up
2. Generate depth map
3. Get segment for center point prompt
4. Isolate depth pixels corresponding to segment
5. Generate 3D mesh using PyMCubes
6. Display: RGB original, depth frame, segment overlay, 3D mesh view
"""

import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import urllib.request
import sys

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Try to import PyMCubes
try:
    import mcubes
    MCUBES_AVAILABLE = True
    print("✓ PyMCubes available for mesh generation")
except ImportError:
    MCUBES_AVAILABLE = False
    print("✗ PyMCubes not available. Install with: pip install PyMCubes")
    print("  Mesh generation will be skipped")

def download_checkpoint(checkpoint_path, model_name="sam2.1_hiera_tiny"):
    """Download SAM2 checkpoint if it doesn't exist"""
    model_urls = {
        "sam2.1_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "sam2.1_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt", 
        "sam2.1_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "sam2.1_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    }
    
    if model_name not in model_urls:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_urls.keys())}")
    
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

def check_and_download_checkpoint(checkpoint_path, model_name="sam2.1_hiera_tiny"):
    """Check if checkpoint exists, download if missing"""
    if os.path.exists(checkpoint_path):
        print(f"✓ Checkpoint found: {checkpoint_path}")
        return True
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Attempting to download...")
        return download_checkpoint(checkpoint_path, model_name)

def get_model_config(model_name):
    """Get the config filename for a given model"""
    config_map = {
        "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
    
    if model_name not in config_map:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(config_map.keys())}")
    
    return config_map[model_name]

def capture_camera_frame():
    """Step 1: Capture frame when camera is warmed up"""
    print("=== Step 1: Camera Capture ===")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return None
    
    print("Warming up camera (discarding first 5 frames)...")
    for i in range(5):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: Could not read frame {i+1}")
        time.sleep(0.1)
    
    # Capture the actual frame
    print("Capturing frame...")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not capture frame")
        return None
    
    print("✓ Frame captured successfully!")
    cv2.imwrite('captured_frame.jpg', frame)
    return frame

def generate_depth_map(frame):
    """Step 2: Generate depth map using MiDaS"""
    print("=== Step 2: Depth Map Generation ===")
    
    # Load MiDaS model
    print("Loading MiDaS model...")
    model_name = "Intel/dpt-swinv2-tiny-256"  # Fast model for real-time use
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Using device: {device}")
    
    # Process depth map
    print("Processing depth map...")
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Resize to original dimensions
    height, width = frame.shape[:2]
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
    )
    
    # Convert to numpy array (keep original depth values)
    depth_map = prediction.squeeze().cpu().numpy()
    
    # Create normalized version for display
    depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    print("✓ Depth map generated successfully!")
    return depth_map, depth_normalized

def segment_center_point(frame, sam_model_name="sam2.1_hiera_tiny"):
    """Step 3: Get segment for center point prompt using SAM2"""
    print("=== Step 3: SAM2 Segmentation ===")
    
    # Setup SAM2 model
    checkpoint = f"./models/checkpoints/{sam_model_name}.pt"
    model_cfg = get_model_config(sam_model_name)
    
    if not check_and_download_checkpoint(checkpoint, sam_model_name):
        print("Failed to obtain checkpoint file.")
        return None, None
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Using CPU")
    
    # Build predictor
    print("Loading SAM2 model...")
    sam2_model = build_sam2(model_cfg, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    # Convert to RGB for SAM2
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    # Use center point as prompt
    height, width = image_rgb.shape[:2]
    center_x, center_y = width // 2, height // 2
    input_points = np.array([[center_x, center_y]])
    input_labels = np.array([1])  # positive point
    
    print(f"Using center point: ({center_x}, {center_y})")
    
    # Run segmentation
    if device == "mps":
        with torch.inference_mode(), torch.autocast(device, dtype=torch.float16):
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
    else:
        with torch.inference_mode():
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
    
    # Get best mask
    best_mask = masks[np.argmax(scores)]
    best_score = scores[np.argmax(scores)]
    
    print(f"✓ Segmentation complete! Best mask score: {best_score:.3f}")
    return best_mask, input_points

def isolate_segment_depth(depth_map, mask):
    """Step 4: Isolate depth pixels corresponding to segment"""
    print("=== Step 4: Depth Isolation ===")
    
    # Ensure mask is boolean type
    if mask.dtype != bool:
        mask = mask.astype(bool)
        print(f"Converted mask from {mask.dtype} to boolean")
    
    print(f"Mask shape: {mask.shape}, Depth map shape: {depth_map.shape}")
    print(f"Mask true pixels: {np.sum(mask)}")
    print(f"Depth map dtype: {depth_map.dtype}, range: {np.min(depth_map):.6f} - {np.max(depth_map):.6f}")
    
    # Create isolated depth map (segment only) - preserve f32 precision
    isolated_depth = np.zeros_like(depth_map, dtype=np.float32)
    isolated_depth[mask] = depth_map[mask]
    
    # Get statistics of the segmented region
    segment_depths = depth_map[mask]
    if len(segment_depths) > 0:
        min_depth = np.min(segment_depths)
        max_depth = np.max(segment_depths)
        mean_depth = np.mean(segment_depths)
        std_depth = np.std(segment_depths)
        print(f"✓ Segment depth stats (f32):")
        print(f"  Range: {min_depth:.6f} - {max_depth:.6f}")
        print(f"  Mean: {mean_depth:.6f}, Std: {std_depth:.6f}")
        print(f"  Pixels: {len(segment_depths)}")
    else:
        print("Warning: No pixels found in segment")
    
    return isolated_depth

def create_3d_point_cloud(depth_map, mask, rgb_frame):
    """Create 3D point cloud visualization of the isolated segment"""
    print("=== Creating 3D Point Cloud ===")
    
    # Ensure mask is boolean type
    if mask.dtype != bool:
        mask = mask.astype(bool)
    
    # Get coordinates where mask is True
    y_coords, x_coords = np.where(mask)
    
    if len(y_coords) == 0:
        print("Warning: No points to visualize")
        return None
    
    # Get corresponding depth and color values
    z_coords = depth_map[y_coords, x_coords]
    
    # Get RGB colors for the points
    rgb_frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    colors = rgb_frame_rgb[y_coords, x_coords] / 255.0
    
    # Subsample points for better performance (take every Nth point)
    subsample_factor = max(1, len(y_coords) // 5000)  # Limit to ~5000 points
    if subsample_factor > 1:
        indices = np.arange(0, len(y_coords), subsample_factor)
        x_coords = x_coords[indices]
        y_coords = y_coords[indices]
        z_coords = z_coords[indices]
        colors = colors[indices]
    
    print(f"✓ 3D point cloud prepared with {len(x_coords)} points")
    return x_coords, y_coords, z_coords, colors

def create_3d_mesh(depth_map, mask, rgb_frame, mesh_resolution=1.0):
    """Create 3D mesh from segmented depth data using PyMCubes"""
    print("=== Creating 3D Mesh ===")
    
    if not MCUBES_AVAILABLE:
        print("PyMCubes not available, skipping mesh generation")
        return None, None, None
    
    # Ensure mask is boolean type
    if mask.dtype != bool:
        mask = mask.astype(bool)
    
    if np.sum(mask) == 0:
        print("Warning: Empty mask, cannot generate mesh")
        return None, None, None
    
    # Get bounding box of the mask to reduce computation
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return None, None, None
    
    min_y, max_y = np.min(y_indices), np.max(y_indices)
    min_x, max_x = np.min(x_indices), np.max(x_indices)
    
    # Add minimal padding around the bounding box
    padding = 2
    min_y = max(0, min_y - padding)
    max_y = min(mask.shape[0], max_y + padding)
    min_x = max(0, min_x - padding)
    max_x = min(mask.shape[1], max_x + padding)
    
    print(f"Bounding box: ({min_x}, {min_y}) to ({max_x}, {max_y})")
    
    # Extract the region of interest
    roi_mask = mask[min_y:max_y, min_x:max_x]
    roi_depth = depth_map[min_y:max_y, min_x:max_x]
    
    # Downsample the ROI for mesh generation to reduce complexity
    downsample_factor = 4  # Reduce resolution by factor of 4
    if roi_mask.shape[0] > 50 and roi_mask.shape[1] > 50:
        roi_mask_small = roi_mask[::downsample_factor, ::downsample_factor]
        roi_depth_small = roi_depth[::downsample_factor, ::downsample_factor]
        print(f"Downsampled from {roi_mask.shape} to {roi_mask_small.shape}")
    else:
        roi_mask_small = roi_mask
        roi_depth_small = roi_depth
        downsample_factor = 1
    
    height, width = roi_mask_small.shape
    
    # Determine volume depth based on depth range in the segment
    segment_depths = roi_depth_small[roi_mask_small]
    if len(segment_depths) == 0:
        return None, None, None
    
    min_depth, max_depth = np.min(segment_depths), np.max(segment_depths)
    depth_range = max(max_depth - min_depth, 0.001)  # Avoid division by zero
    
    # Create a much smaller volume with reasonable resolution
    volume_depth = max(10, min(30, int(depth_range * 20)))  # Much smaller volume
    
    print(f"Creating volume: {width} x {height} x {volume_depth}")
    print(f"Depth range: {min_depth:.3f} - {max_depth:.3f}")
    
    # Initialize volume (1 = inside object, 0 = outside)
    volume = np.zeros((height, width, volume_depth), dtype=np.float32)
    
    # Create a surface-based volume instead of solid volume
    # This creates a thin shell representing the surface
    for y in range(height):
        for x in range(width):
            if roi_mask_small[y, x]:
                # Normalize depth to volume coordinates
                if depth_range > 0:
                    normalized_depth = (roi_depth_small[y, x] - min_depth) / depth_range
                else:
                    normalized_depth = 0.5
                
                # Convert to volume z-coordinate
                z_coord = int(normalized_depth * (volume_depth - 1))
                z_coord = max(0, min(volume_depth - 1, z_coord))
                
                # Create a thin surface instead of filling from bottom
                # Add thickness around the surface point
                surface_thickness = 2
                for dz in range(-surface_thickness, surface_thickness + 1):
                    z_surface = z_coord + dz
                    if 0 <= z_surface < volume_depth:
                        # Use gaussian-like falloff for smooth surface
                        distance = abs(dz) / surface_thickness
                        volume[y, x, z_surface] = max(volume[y, x, z_surface], 1.0 - distance)
    
    print("Volume filled, running marching cubes...")
    
    try:
        # Run marching cubes to extract the surface with a higher isovalue
        vertices, triangles = mcubes.marching_cubes(volume, 0.3)
        
        # Transform vertices back to original coordinate system
        # Account for downsampling
        vertices[:, 0] = (vertices[:, 0] * downsample_factor) + min_y  # Y in image coordinates
        vertices[:, 1] = (vertices[:, 1] * downsample_factor) + min_x  # X in image coordinates  
        vertices[:, 2] = (vertices[:, 2] / volume_depth) * depth_range + min_depth  # Depth
        
        # Simplify mesh if it's still too complex
        max_vertices = 10000
        if len(vertices) > max_vertices:
            print(f"Mesh too complex ({len(vertices)} vertices), decimating...")
            # Simple decimation by subsampling
            decimation_factor = len(vertices) // max_vertices
            vertex_indices = np.arange(0, len(vertices), decimation_factor)
            
            # Create mapping from old to new vertex indices
            vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(vertex_indices)}
            
            # Keep only selected vertices
            vertices = vertices[vertex_indices]
            
            # Update triangles to reference new vertex indices
            new_triangles = []
            for triangle in triangles:
                # Check if all vertices of the triangle are in our kept set
                if all(v in vertex_map for v in triangle):
                    new_triangle = [vertex_map[v] for v in triangle]
                    new_triangles.append(new_triangle)
            
            triangles = np.array(new_triangles)
            print(f"Decimated to {len(vertices)} vertices, {len(triangles)} faces")
        
        # Get colors for vertices by sampling from RGB image
        vertex_colors = []
        rgb_frame_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        
        for vertex in vertices:
            y_img = int(np.clip(vertex[0], 0, rgb_frame.shape[0] - 1))
            x_img = int(np.clip(vertex[1], 0, rgb_frame.shape[1] - 1))
            color = rgb_frame_rgb[y_img, x_img] / 255.0
            vertex_colors.append(color)
        
        vertex_colors = np.array(vertex_colors)
        
        print(f"✓ Mesh generated: {len(vertices)} vertices, {len(triangles)} faces")
        
        return vertices, triangles, vertex_colors
        
    except Exception as e:
        print(f"Mesh generation failed: {e}")
        return None, None, None

def display_results(rgb_frame, depth_normalized, mask, input_points, isolated_depth, point_cloud_data, mesh_data, depth_original=None):
    """Step 6: Display all results in panels including mesh"""
    print("=== Step 6: Results Display ===")
    
    # Convert BGR to RGB for display
    rgb_display = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(24, 12))
    
    # Panel 1: Original RGB image
    ax1 = plt.subplot(2, 4, 1)
    plt.imshow(rgb_display)
    plt.scatter(input_points[0, 0], input_points[0, 1], color='red', s=100, marker='*')
    plt.title('Original RGB + Center Point', fontsize=14)
    plt.axis('off')
    
    # Panel 2: Depth frame (normalized for display)
    ax2 = plt.subplot(2, 4, 2)
    plt.imshow(depth_normalized, cmap='viridis')
    plt.title('Depth Map (Normalized)', fontsize=14)
    plt.axis('off')
    plt.colorbar(shrink=0.8)
    
    # Panel 3: Segment overlay on RGB
    ax3 = plt.subplot(2, 4, 3)
    plt.imshow(rgb_display)
    plt.imshow(mask, alpha=0.5, cmap='Reds')
    plt.title('Segment Overlay', fontsize=14)
    plt.axis('off')
    
    # Panel 4: Isolated depth (segment only) - use original f32 values for display
    ax4 = plt.subplot(2, 4, 4)
    if depth_original is not None:
        # Create mask for display - only show non-zero values in isolated depth
        display_mask = isolated_depth > 0
        isolated_for_display = np.ma.masked_where(~display_mask, isolated_depth)
        im4 = plt.imshow(isolated_for_display, cmap='viridis')
        plt.title('Isolated Segment Depth (f32)', fontsize=14)
        plt.colorbar(im4, shrink=0.8, label='Depth (original units)')
    else:
        # Fallback to normalized version
        isolated_normalized = cv2.normalize(isolated_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imshow(isolated_normalized, cmap='viridis')
        plt.title('Isolated Segment Depth', fontsize=14)
        plt.colorbar(shrink=0.8)
    plt.axis('off')
    
    # Panel 5: 3D point cloud visualization
    ax5 = plt.subplot(2, 4, 5, projection='3d')
    if point_cloud_data is not None:
        x_coords, y_coords, z_coords, colors = point_cloud_data
        # Flip y-axis to match image coordinates
        scatter = ax5.scatter(x_coords, -y_coords, z_coords, c=colors, s=1, alpha=0.8)
        ax5.set_xlabel('X (pixels)')
        ax5.set_ylabel('Y (pixels)')
        ax5.set_zlabel('Depth (f32)')
        ax5.set_title('3D Point Cloud', fontsize=12)
        ax5.set_facecolor('black')
        
        # Add depth statistics to title
        if len(z_coords) > 0:
            depth_range = f"Depth: {np.min(z_coords):.3f} - {np.max(z_coords):.3f}"
            ax5.set_title(f'3D Point Cloud\n{depth_range}', fontsize=10)
    else:
        ax5.text(0.5, 0.5, 0.5, 'No 3D data', ha='center', va='center')
        ax5.set_title('3D Point Cloud (No Data)', fontsize=12)
    
    # Panel 6: 3D Mesh visualization
    ax6 = plt.subplot(2, 4, 6, projection='3d')
    if mesh_data is not None and mesh_data[0] is not None:
        vertices, triangles, vertex_colors = mesh_data
        
        # Create mesh collection
        mesh_triangles = []
        triangle_colors = []
        
        # Subsample triangles for display if too many
        max_triangles = 2000
        if len(triangles) > max_triangles:
            indices = np.random.choice(len(triangles), max_triangles, replace=False)
            display_triangles = triangles[indices]
        else:
            display_triangles = triangles
        
        for triangle in display_triangles:
            # Get triangle vertices (flip y-axis)
            tri_verts = vertices[triangle]
            tri_verts_flipped = tri_verts.copy()
            tri_verts_flipped[:, 1] = -tri_verts_flipped[:, 1]  # Flip Y
            mesh_triangles.append(tri_verts_flipped)
            
            # Average color of triangle vertices
            avg_color = np.mean(vertex_colors[triangle], axis=0)
            triangle_colors.append(avg_color)
        
        # Create and add mesh
        mesh_collection = Poly3DCollection(mesh_triangles, alpha=0.8)
        mesh_collection.set_facecolors(triangle_colors)
        mesh_collection.set_edgecolors('none')
        ax6.add_collection3d(mesh_collection)
        
        # Set axis limits
        ax6.set_xlim(np.min(vertices[:, 1]), np.max(vertices[:, 1]))
        ax6.set_ylim(-np.max(vertices[:, 0]), -np.min(vertices[:, 0]))
        ax6.set_zlim(np.min(vertices[:, 2]), np.max(vertices[:, 2]))
        
        ax6.set_xlabel('X (pixels)')
        ax6.set_ylabel('Y (pixels)')
        ax6.set_zlabel('Depth (f32)')
        ax6.set_title(f'3D Mesh\n{len(vertices)} vertices, {len(triangles)} faces', fontsize=10)
        ax6.set_facecolor('black')
        
    else:
        ax6.text(0.5, 0.5, 0.5, 'No mesh data\n(PyMCubes required)', ha='center', va='center')
        ax6.set_title('3D Mesh (No Data)', fontsize=12)
    
    # Panel 7: Mask only
    ax7 = plt.subplot(2, 4, 7)
    plt.imshow(mask, cmap='gray')
    plt.title('Segment Mask', fontsize=14)
    plt.axis('off')
    
    # Panel 8: Statistics
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Compile statistics
    stats_text = "Analysis Statistics:\n\n"
    
    # Mask statistics
    mask_bool = mask.astype(bool) if mask.dtype != bool else mask
    mask_pixels = np.sum(mask_bool)
    total_pixels = mask.shape[0] * mask.shape[1]
    mask_percentage = (mask_pixels / total_pixels) * 100
    stats_text += f"Segment: {mask_pixels:,} pixels\n"
    stats_text += f"Coverage: {mask_percentage:.1f}%\n\n"
    
    # Depth statistics
    if depth_original is not None:
        # Ensure mask is boolean for indexing
        mask_bool = mask.astype(bool) if mask.dtype != bool else mask
        segment_depths = depth_original[mask_bool]
        if len(segment_depths) > 0:
            stats_text += f"Depth Stats (f32):\n"
            stats_text += f"  Min: {np.min(segment_depths):.4f}\n"
            stats_text += f"  Max: {np.max(segment_depths):.4f}\n"
            stats_text += f"  Mean: {np.mean(segment_depths):.4f}\n"
            stats_text += f"  Std: {np.std(segment_depths):.4f}\n\n"
    
    # Point cloud statistics
    if point_cloud_data is not None:
        x_coords, y_coords, z_coords, colors = point_cloud_data
        stats_text += f"Point Cloud: {len(x_coords):,} points\n\n"
    
    # Mesh statistics
    if mesh_data is not None and mesh_data[0] is not None:
        vertices, triangles, vertex_colors = mesh_data
        stats_text += f"Mesh Generated:\n"
        stats_text += f"  Vertices: {len(vertices):,}\n"
        stats_text += f"  Faces: {len(triangles):,}\n"
        
        # Mesh volume estimation (rough)
        if len(triangles) > 0:
            volume_estimate = len(vertices) * 0.001  # Rough estimate
            stats_text += f"  Volume (est.): {volume_estimate:.3f}\n"
    else:
        if not MCUBES_AVAILABLE:
            stats_text += "Mesh: PyMCubes not available\n"
        else:
            stats_text += "Mesh: Generation failed\n"
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    ax8.set_title('Statistics', fontsize=14)
    
    plt.tight_layout()
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"depth_segment_mesh_results_{timestamp}.png"
    plt.savefig(results_filename, dpi=150, bbox_inches='tight')
    print(f"✓ Results saved as: {results_filename}")
    
    # Save mesh data if available
    if mesh_data is not None and mesh_data[0] is not None:
        mesh_filename = f"segment_mesh_{timestamp}.npz"
        vertices, triangles, vertex_colors = mesh_data
        np.savez(mesh_filename, 
                 vertices=vertices, 
                 triangles=triangles, 
                 vertex_colors=vertex_colors)
        print(f"✓ Mesh data saved as: {mesh_filename}")
    
    # Show results
    plt.show()

def main():
    """Main function combining all steps"""
    print("=" * 60)
    print("Combined Camera Depth + Segmentation + Mesh Analysis")
    print("=" * 60)
    
    try:
        # Step 1: Capture frame
        frame = capture_camera_frame()
        if frame is None:
            return
        
        # Step 2: Generate depth map
        depth_map, depth_normalized = generate_depth_map(frame)
        
        # Step 3: Segment center point
        mask, input_points = segment_center_point(frame)
        if mask is None:
            return
        
        # Step 4: Isolate segment depth
        isolated_depth = isolate_segment_depth(depth_map, mask)
        
        # Step 5: Create 3D point cloud
        point_cloud_data = create_3d_point_cloud(depth_map, mask, frame)
        
        # Step 6: Generate 3D mesh using PyMCubes
        mesh_data = create_3d_mesh(depth_map, mask, frame)
        
        # Step 7: Display results
        display_results(frame, depth_normalized, mask, input_points, isolated_depth, point_cloud_data, mesh_data, depth_map)
        
        print("=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Files saved:")
        print("  - captured_frame.jpg")
        print("  - depth_segment_mesh_results_[timestamp].png")
        if mesh_data is not None and mesh_data[0] is not None:
            print("  - segment_mesh_[timestamp].npz")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        print("Program ended.")

if __name__ == "__main__":
    main()