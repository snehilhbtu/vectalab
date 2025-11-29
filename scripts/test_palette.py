#!/usr/bin/env python3
"""
Test logo vectorization with color palette reduction.
Reduces to 16 colors before vectorization for cleaner output.
"""

import cv2
import numpy as np
from PIL import Image
import vtracer
import tempfile
import os
from io import BytesIO
import cairosvg
from skimage.metrics import structural_similarity as ssim


def reduce_to_palette(image_rgb: np.ndarray, n_colors: int = 16) -> np.ndarray:
    """Reduce image to fixed color palette using PIL's quantize."""
    pil_img = Image.fromarray(image_rgb)
    # Quantize to palette
    quantized = pil_img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    # Convert back to RGB
    rgb_img = quantized.convert('RGB')
    return np.array(rgb_img)


def vectorize_with_palette(
    input_path: str,
    output_path: str,
    n_colors: int = 16,
    verbose: bool = True
):
    """Vectorize logo with palette reduction."""
    
    # Load image
    original = cv2.imread(input_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    h, w = original_rgb.shape[:2]
    
    if verbose:
        orig_colors = len(np.unique(original_rgb.reshape(-1, 3), axis=0))
        print(f"Original: {w}x{h}, {orig_colors:,} colors")
    
    # Reduce to palette
    reduced = reduce_to_palette(original_rgb, n_colors)
    
    if verbose:
        new_colors = len(np.unique(reduced.reshape(-1, 3), axis=0))
        print(f"Reduced to {new_colors} colors")
    
    # Save reduced image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(reduced, cv2.COLOR_RGB2BGR))
    
    # Vectorize with quality settings
    settings = {
        'colormode': 'color',
        'hierarchical': 'stacked',
        'mode': 'spline',
        'filter_speckle': 4,
        'color_precision': 6,
        'layer_difference': 16,
        'corner_threshold': 60,
        'length_threshold': 4.0,
        'max_iterations': 10,
        'splice_threshold': 45,
        'path_precision': 5,
    }
    
    try:
        vtracer.convert_image_to_svg_py(tmp_path, output_path, **settings)
        
        with open(output_path, 'r') as f:
            svg_content = f.read()
        
        # Render and compare
        png_data = cairosvg.svg2png(bytestring=svg_content.encode(), output_width=w, output_height=h)
        rendered = np.array(Image.open(BytesIO(png_data)).convert('RGB'))
        
        # Metrics vs original
        ssim_orig = ssim(original_rgb, rendered, channel_axis=2, data_range=255)
        # Metrics vs reduced (should be higher)
        ssim_reduced = ssim(reduced, rendered, channel_axis=2, data_range=255)
        
        path_count = svg_content.count('<path')
        file_size = len(svg_content.encode())
        
        if verbose:
            print(f"\nResults:")
            print(f"  File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"  Paths: {path_count}")
            print(f"  SSIM vs original: {ssim_orig*100:.2f}%")
            print(f"  SSIM vs reduced:  {ssim_reduced*100:.2f}%")
        
        # Save reduced image for comparison
        reduced_path = output_path.replace('.svg', '_reduced.png')
        cv2.imwrite(reduced_path, cv2.cvtColor(reduced, cv2.COLOR_RGB2BGR))
        
        return {
            'ssim_original': ssim_orig,
            'ssim_reduced': ssim_reduced,
            'paths': path_count,
            'file_size': file_size,
        }
        
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    input_file = 'examples/ELITIZON_LOGO.jpg'
    
    # Test different color counts
    for n_colors in [8, 16, 32]:
        output_file = f'examples/ELITIZON_LOGO_{n_colors}colors.svg'
        print(f"\n{'='*50}")
        print(f"Testing with {n_colors} colors")
        print('='*50)
        vectorize_with_palette(input_file, output_file, n_colors=n_colors)
