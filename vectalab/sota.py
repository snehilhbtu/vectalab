"""
Vectalab SOTA Vectorization Module.

State-of-the-art image vectorization with intelligent preprocessing,
adaptive settings, and iterative optimization.

Features:
- Automatic image type detection (logo, icon, photo)
- Color quantization for cleaner output
- Adaptive vtracer settings
- Path simplification and merging
- Iterative quality optimization loop

Usage:
    from vectalab.sota import vectorize_smart
    
    svg_path, metrics = vectorize_smart("logo.png", "output.svg")
"""

import numpy as np
import cv2
from PIL import Image
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import tempfile
import os
import re
import xml.etree.ElementTree as ET

# Try imports
try:
    import vtracer
    VTRACER_AVAILABLE = True
except ImportError:
    VTRACER_AVAILABLE = False

try:
    import cairosvg
    from io import BytesIO
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# IMAGE ANALYSIS
# ============================================================================

class ImageAnalyzer:
    """Analyze image characteristics to determine optimal vectorization settings."""
    
    @staticmethod
    def analyze(image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image and return characteristics.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Dictionary with image characteristics
        """
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3)
        
        # Count unique colors
        unique_colors = len(np.unique(pixels, axis=0))
        
        # Color distribution
        color_counts = Counter(map(tuple, pixels))
        total_pixels = len(pixels)
        
        # Coverage by top N colors
        top_10_coverage = sum(c for _, c in color_counts.most_common(10)) / total_pixels
        top_50_coverage = sum(c for _, c in color_counts.most_common(50)) / total_pixels
        
        # Color variance (indicates complexity)
        color_variance = np.std(pixels, axis=0).mean()
        
        # Edge density (indicates detail level)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Dominant colors
        dominant_colors = [color for color, _ in color_counts.most_common(20)]
        
        # Determine image type
        if color_variance < 40 and top_10_coverage > 0.85:
            image_type = "logo"
            complexity = "simple"
        elif color_variance < 60 and top_50_coverage > 0.90:
            image_type = "icon"
            complexity = "medium"
        elif color_variance < 80 and edge_density < 0.15:
            image_type = "illustration"
            complexity = "medium"
        else:
            image_type = "photo"
            complexity = "complex"
        
        return {
            "width": w,
            "height": h,
            "unique_colors": unique_colors,
            "top_10_coverage": top_10_coverage,
            "top_50_coverage": top_50_coverage,
            "color_variance": color_variance,
            "edge_density": edge_density,
            "dominant_colors": dominant_colors,
            "image_type": image_type,
            "complexity": complexity,
        }


# ============================================================================
# COLOR QUANTIZATION
# ============================================================================

def quantize_colors_kmeans(image: np.ndarray, n_colors: int = 16) -> np.ndarray:
    """
    Reduce image colors using K-means clustering.
    
    Args:
        image: RGB image
        n_colors: Target number of colors
        
    Returns:
        Quantized image
    """
    if not SKLEARN_AVAILABLE:
        return quantize_colors_simple(image, n_colors)
    
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.astype(np.uint8)
    
    # Reconstruct image
    quantized = centers[labels].reshape(h, w, 3)
    
    return quantized


def quantize_colors_simple(image: np.ndarray, n_colors: int = 16) -> np.ndarray:
    """
    Simple color quantization using color reduction.
    
    Args:
        image: RGB image
        n_colors: Approximate target number of colors
        
    Returns:
        Quantized image
    """
    # Reduce color depth
    factor = max(1, 256 // int(np.cbrt(n_colors) + 1))
    quantized = (image // factor) * factor
    
    return quantized


def quantize_colors_median_cut(image: np.ndarray, n_colors: int = 16) -> np.ndarray:
    """
    Quantize using PIL's median cut algorithm.
    
    Args:
        image: RGB image
        n_colors: Target number of colors
        
    Returns:
        Quantized image
    """
    pil_image = Image.fromarray(image)
    quantized_pil = pil_image.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    quantized_rgb = quantized_pil.convert('RGB')
    return np.array(quantized_rgb)


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess image based on its type.
    
    Args:
        image: RGB image
        analysis: Image analysis results
        
    Returns:
        Preprocessed image
    """
    image_type = analysis["image_type"]
    
    if image_type == "logo":
        # For logos: aggressive color quantization, denoising
        # Determine optimal color count
        if analysis["top_10_coverage"] > 0.95:
            n_colors = 8
        elif analysis["top_10_coverage"] > 0.90:
            n_colors = 12
        else:
            n_colors = 16
        
        # Quantize colors
        processed = quantize_colors_median_cut(image, n_colors)
        
        # Light blur to remove JPEG artifacts
        processed = cv2.bilateralFilter(processed, 5, 50, 50)
        
    elif image_type == "icon":
        # For icons: moderate quantization
        n_colors = 32
        processed = quantize_colors_median_cut(image, n_colors)
        processed = cv2.bilateralFilter(processed, 3, 30, 30)
        
    elif image_type == "illustration":
        # For illustrations: light quantization
        n_colors = 64
        processed = quantize_colors_kmeans(image, n_colors)
        
    else:
        # For photos: minimal preprocessing
        processed = cv2.bilateralFilter(image, 5, 75, 75)
    
    return processed


# ============================================================================
# ADAPTIVE VTRACER SETTINGS
# ============================================================================

def get_adaptive_vtracer_settings(analysis: Dict[str, Any], 
                                   quality: str = "balanced") -> Dict[str, Any]:
    """
    Get vtracer settings optimized for the image type.
    
    Args:
        analysis: Image analysis results
        quality: Quality level ("compact", "balanced", "quality")
        
    Returns:
        vtracer settings dictionary
    """
    image_type = analysis["image_type"]
    
    # Base settings by image type
    if image_type == "logo":
        base = {
            'colormode': 'color',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 8,
            'color_precision': 4,
            'layer_difference': 32,
            'corner_threshold': 60,
            'length_threshold': 4.0,
            'max_iterations': 10,
            'splice_threshold': 45,
            'path_precision': 3,
        }
    elif image_type == "icon":
        base = {
            'colormode': 'color',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 6,
            'color_precision': 5,
            'layer_difference': 24,
            'corner_threshold': 50,
            'length_threshold': 3.5,
            'max_iterations': 12,
            'splice_threshold': 45,
            'path_precision': 4,
        }
    elif image_type == "illustration":
        base = {
            'colormode': 'color',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 4,
            'color_precision': 6,
            'layer_difference': 16,
            'corner_threshold': 45,
            'length_threshold': 3.0,
            'max_iterations': 15,
            'splice_threshold': 45,
            'path_precision': 5,
        }
    else:  # photo
        base = {
            'colormode': 'color',
            'hierarchical': 'stacked',
            'mode': 'spline',
            'filter_speckle': 2,
            'color_precision': 6,
            'layer_difference': 8,
            'corner_threshold': 30,
            'length_threshold': 3.0,
            'max_iterations': 20,
            'splice_threshold': 45,
            'path_precision': 6,
        }
    
    # Adjust for quality level
    if quality == "compact":
        base['filter_speckle'] = min(base['filter_speckle'] * 2, 16)
        base['layer_difference'] = min(base['layer_difference'] * 2, 64)
        base['color_precision'] = max(base['color_precision'] - 2, 3)
        base['path_precision'] = max(base['path_precision'] - 1, 2)
    elif quality == "quality":
        base['filter_speckle'] = max(base['filter_speckle'] // 2, 1)
        base['layer_difference'] = max(base['layer_difference'] // 2, 4)
        base['color_precision'] = min(base['color_precision'] + 1, 8)
    
    return base


# ============================================================================
# SVG POST-PROCESSING
# ============================================================================

def optimize_svg_paths(svg_content: str, simplify_epsilon: float = 1.0) -> str:
    """
    Optimize SVG paths by simplifying and cleaning.
    
    Args:
        svg_content: SVG content string
        simplify_epsilon: Simplification tolerance
        
    Returns:
        Optimized SVG content
    """
    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError:
        return svg_content
    
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    
    # Process all path elements
    for path in root.iter(f'{ns}path'):
        d = path.get('d', '')
        if d:
            # Round coordinates
            d = _round_path_coords(d, precision=1)
            # Simplify if possible
            d = _simplify_path_commands(d)
            path.set('d', d)
    
    # Merge paths with same fill color
    svg_content = ET.tostring(root, encoding='unicode')
    
    return svg_content


def _round_path_coords(d: str, precision: int = 1) -> str:
    """Round all coordinates in path data."""
    def round_num(match):
        num = float(match.group(0))
        if abs(num - round(num)) < 0.01:
            return str(int(round(num)))
        return f'{num:.{precision}f}'.rstrip('0').rstrip('.')
    
    return re.sub(r'-?\d+\.?\d*', round_num, d)


def _simplify_path_commands(d: str) -> str:
    """Simplify path commands (e.g., use relative commands, remove redundant points)."""
    # Remove extra spaces
    d = re.sub(r'\s+', ' ', d.strip())
    # Remove space after command letters
    d = re.sub(r'([MmLlHhVvCcSsQqTtAaZz])\s+', r'\1', d)
    # Use comma as separator
    d = re.sub(r'\s+', ',', d)
    # Remove commas after commands
    d = re.sub(r'([MmLlHhVvCcSsQqTtAaZz]),', r'\1', d)
    
    return d


def merge_same_color_paths(svg_content: str) -> str:
    """
    Merge adjacent paths with the same fill color.
    
    Args:
        svg_content: SVG content string
        
    Returns:
        SVG with merged paths
    """
    try:
        root = ET.fromstring(svg_content)
    except ET.ParseError:
        return svg_content
    
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    
    # Group paths by fill color
    color_groups = {}
    paths_to_remove = []
    
    for path in root.iter(f'{ns}path'):
        fill = path.get('fill', 'black')
        d = path.get('d', '')
        
        if fill not in color_groups:
            color_groups[fill] = {'element': path, 'paths': [d]}
        else:
            color_groups[fill]['paths'].append(d)
            paths_to_remove.append(path)
    
    # Merge paths
    for fill, group in color_groups.items():
        if len(group['paths']) > 1:
            merged_d = ' '.join(group['paths'])
            group['element'].set('d', merged_d)
    
    # Remove merged paths
    for path in paths_to_remove:
        parent = None
        for elem in root.iter():
            if path in list(elem):
                parent = elem
                break
        if parent is not None:
            parent.remove(path)
    
    return ET.tostring(root, encoding='unicode')


def apply_scour_optimization(svg_content: str) -> str:
    """Apply scour optimization."""
    try:
        from scour import scour
        from scour.scour import scourString
        
        options = scour.sanitizeOptions(options={
            'enable_viewboxing': True,
            'strip_ids': True,
            'strip_comments': True,
            'shorten_ids': True,
            'indent_type': 'none',
            'newlines': False,
            'digits': 1,
            'remove_metadata': True,
            'remove_titles': True,
            'remove_descriptions': True,
            'remove_descriptive_elements': True,
            'enable_comment_stripping': True,
        })
        
        result = scourString(svg_content, options)
        result = re.sub(r'<\?xml[^?]*\?>\s*', '', result)
        return result.strip()
    except Exception as e:
        print(f"Scour failed: {e}")
        return svg_content


# ============================================================================
# QUALITY MEASUREMENT
# ============================================================================

def render_svg_to_array(svg_content: str, width: int, height: int) -> np.ndarray:
    """Render SVG to numpy array."""
    if not CAIROSVG_AVAILABLE:
        raise ImportError("cairosvg required for rendering")
    
    png_data = cairosvg.svg2png(
        bytestring=svg_content.encode('utf-8'),
        output_width=width,
        output_height=height
    )
    
    img = Image.open(BytesIO(png_data)).convert('RGB')
    return np.array(img)


def compute_quality_metrics(original: np.ndarray, svg_content: str) -> Dict[str, float]:
    """
    Compute quality metrics between original and rendered SVG.
    
    Args:
        original: Original RGB image
        svg_content: SVG content string
        
    Returns:
        Dictionary with quality metrics
    """
    h, w = original.shape[:2]
    
    try:
        rendered = render_svg_to_array(svg_content, w, h)
    except Exception as e:
        return {"error": str(e)}
    
    # SSIM
    if SKIMAGE_AVAILABLE:
        ssim_value = ssim(original, rendered, channel_axis=2, data_range=255)
    else:
        ssim_value = 0.0
    
    # PSNR
    mse = np.mean((original.astype(float) - rendered.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255 ** 2 / mse)
    else:
        psnr = float('inf')
    
    # Mean Absolute Error
    mae = np.mean(np.abs(original.astype(float) - rendered.astype(float)))
    
    # File size
    file_size = len(svg_content.encode('utf-8'))
    
    # Path count
    path_count = svg_content.count('<path')
    
    return {
        "ssim": ssim_value,
        "psnr": psnr,
        "mae": mae,
        "file_size": file_size,
        "path_count": path_count,
    }


# ============================================================================
# MAIN VECTORIZATION FUNCTION
# ============================================================================

def vectorize_smart(
    input_path: str,
    output_path: str,
    target_ssim: float = 0.92,
    max_file_size: int = 100_000,  # 100KB default
    max_iterations: int = 5,
    verbose: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Smart vectorization with automatic optimization.
    
    This function:
    1. Analyzes the image to determine optimal settings
    2. Preprocesses the image (color quantization, denoising)
    3. Vectorizes with adaptive settings
    4. Iteratively optimizes until quality targets are met
    
    Args:
        input_path: Path to input image
        output_path: Path for output SVG
        target_ssim: Minimum SSIM quality (0.0-1.0)
        max_file_size: Maximum file size in bytes
        max_iterations: Maximum optimization iterations
        verbose: Print progress
        
    Returns:
        Tuple of (output_path, metrics_dict)
    """
    if not VTRACER_AVAILABLE:
        raise ImportError("vtracer required for vectorization")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not load image: {input_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    if verbose:
        print(f"Input: {input_path} ({w}x{h})")
    
    # Analyze image
    analysis = ImageAnalyzer.analyze(image_rgb)
    
    if verbose:
        print(f"Image type: {analysis['image_type']} ({analysis['complexity']})")
        print(f"Unique colors: {analysis['unique_colors']:,}")
        print(f"Top 10 colors cover: {analysis['top_10_coverage']*100:.1f}%")
    
    # Preprocess image
    processed = preprocess_image(image_rgb, analysis)
    
    if verbose:
        processed_colors = len(np.unique(processed.reshape(-1, 3), axis=0))
        print(f"After preprocessing: {processed_colors} colors")
    
    # Save preprocessed image temporarily
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    # Iterative optimization
    best_result = None
    best_score = -float('inf')
    
    quality_levels = ["compact", "balanced", "quality"]
    
    for iteration in range(max_iterations):
        quality = quality_levels[min(iteration, len(quality_levels) - 1)]
        
        if verbose:
            print(f"\nIteration {iteration + 1}/{max_iterations} (quality: {quality})")
        
        # Get adaptive settings
        settings = get_adaptive_vtracer_settings(analysis, quality)
        
        # Vectorize
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
            tmp_svg_path = tmp_svg.name
        
        vtracer.convert_image_to_svg_py(tmp_path, tmp_svg_path, **settings)
        
        # Read and optimize SVG
        with open(tmp_svg_path, 'r') as f:
            svg_content = f.read()
        
        # Optimize SVG
        svg_content = optimize_svg_paths(svg_content)
        svg_content = merge_same_color_paths(svg_content)
        svg_content = apply_scour_optimization(svg_content)
        
        # Measure quality
        metrics = compute_quality_metrics(image_rgb, svg_content)
        
        if verbose:
            print(f"  SSIM: {metrics.get('ssim', 0):.4f}")
            print(f"  File size: {metrics.get('file_size', 0):,} bytes")
            print(f"  Paths: {metrics.get('path_count', 0)}")
        
        # Score based on quality and size
        ssim_val = metrics.get('ssim', 0)
        file_size = metrics.get('file_size', float('inf'))
        
        # Calculate score (higher is better)
        size_score = max(0, 1 - file_size / max_file_size) if file_size < max_file_size * 3 else -1
        quality_score = ssim_val
        score = quality_score * 0.7 + size_score * 0.3
        
        if score > best_score:
            best_score = score
            best_result = {
                'svg_content': svg_content,
                'metrics': metrics,
                'settings': settings,
                'quality': quality,
            }
        
        # Check if targets met
        if ssim_val >= target_ssim and file_size <= max_file_size:
            if verbose:
                print(f"  ✅ Targets met!")
            break
        
        # Cleanup temp SVG
        try:
            os.remove(tmp_svg_path)
        except:
            pass
        
        # Adjust settings for next iteration if needed
        if ssim_val < target_ssim and iteration < max_iterations - 1:
            # Need more quality - will use higher quality preset next
            pass
        elif file_size > max_file_size and iteration < max_iterations - 1:
            # Need smaller size - try more aggressive quantization
            n_colors = max(4, int(len(np.unique(processed.reshape(-1, 3), axis=0)) * 0.7))
            processed = quantize_colors_median_cut(processed, n_colors)
            cv2.imwrite(tmp_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    # Cleanup temp files
    try:
        os.remove(tmp_path)
    except:
        pass
    
    if best_result is None:
        raise RuntimeError("Vectorization failed")
    
    # Write final output
    with open(output_path, 'w') as f:
        f.write(best_result['svg_content'])
    
    final_metrics = best_result['metrics']
    final_metrics['quality_preset'] = best_result['quality']
    final_metrics['image_type'] = analysis['image_type']
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"FINAL RESULT")
        print(f"{'='*50}")
        print(f"Output: {output_path}")
        print(f"File size: {final_metrics['file_size']:,} bytes ({final_metrics['file_size']/1024:.1f} KB)")
        print(f"Paths: {final_metrics['path_count']}")
        print(f"SSIM: {final_metrics['ssim']:.4f} ({final_metrics['ssim']*100:.2f}%)")
    
    return output_path, final_metrics


def vectorize_logo(
    input_path: str,
    output_path: str,
    target_size_kb: int = 50,
    verbose: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimized vectorization for logos.
    
    Creates small, clean SVG files optimized for logos.
    
    Args:
        input_path: Path to input image
        output_path: Path for output SVG
        target_size_kb: Target file size in KB
        verbose: Print progress
        
    Returns:
        Tuple of (output_path, metrics_dict)
    """
    return vectorize_smart(
        input_path,
        output_path,
        target_ssim=0.90,  # Logos don't need pixel-perfect
        max_file_size=target_size_kb * 1024,
        max_iterations=5,
        verbose=verbose,
    )


def vectorize_icon(
    input_path: str,
    output_path: str,
    verbose: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimized vectorization for icons.
    
    Args:
        input_path: Path to input image
        output_path: Path for output SVG
        verbose: Print progress
        
    Returns:
        Tuple of (output_path, metrics_dict)
    """
    return vectorize_smart(
        input_path,
        output_path,
        target_ssim=0.92,
        max_file_size=100_000,
        max_iterations=4,
        verbose=verbose,
    )


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python -m vectalab.sota <input_image> <output.svg>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        svg_path, metrics = vectorize_smart(input_path, output_path)
        print(f"\nSuccess! Output saved to {svg_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
