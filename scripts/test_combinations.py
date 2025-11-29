#!/usr/bin/env python3
"""
Test different preprocessing + vectorization combinations.
Find the optimal balance for logo vectorization.
"""

import vtracer
import cv2
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
import tempfile
import os


def test_combination(original_rgb, processed, settings, name):
    """Test a preprocessing + settings combination."""
    h, w = original_rgb.shape[:2]
    
    # Save processed image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    
    # Vectorize
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
        svg_path = tmp_svg.name
    
    try:
        vtracer.convert_image_to_svg_py(tmp_path, svg_path, **settings)
        
        with open(svg_path, 'r') as f:
            svg_content = f.read()
        
        # Render
        png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=w, output_height=h)
        rendered = np.array(Image.open(BytesIO(png_data)).convert('RGB'))
        
        # Metrics
        ssim_val = ssim(original_rgb, rendered, channel_axis=2, data_range=255)
        path_count = svg_content.count('<path')
        file_size = len(svg_content.encode('utf-8'))
        
        diff = np.abs(original_rgb.astype(float) - rendered.astype(float))
        problem_pixels = np.sum(np.mean(diff, axis=2) > 50)
        
        return {
            'name': name,
            'ssim': ssim_val,
            'paths': path_count,
            'size_kb': file_size / 1024,
            'problems': problem_pixels,
            'svg_content': svg_content,
            'rendered': rendered,
        }
    finally:
        os.remove(tmp_path)
        os.remove(svg_path)


# Load original
input_file = 'examples/ELITIZON_LOGO.jpg'
original = cv2.imread(input_file)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
h, w = original_rgb.shape[:2]

print(f"Testing combinations on {input_file} ({w}x{h})")
print("=" * 70)

# Preprocessing options
preprocessors = {
    'none': lambda img: img,
    'bilateral_light': lambda img: cv2.bilateralFilter(img, 5, 30, 30),
    'bilateral_medium': lambda img: cv2.bilateralFilter(img, 7, 50, 50),
    'median_3': lambda img: cv2.medianBlur(img, 3),
    'gaussian_light': lambda img: cv2.GaussianBlur(img, (3, 3), 0.5),
}

# vtracer settings options  
settings_options = {
    'quality': {
        'colormode': 'color', 'hierarchical': 'stacked', 'mode': 'spline',
        'filter_speckle': 2, 'color_precision': 7, 'layer_difference': 8,
        'corner_threshold': 40, 'length_threshold': 2.5, 'max_iterations': 15,
        'splice_threshold': 40, 'path_precision': 6,
    },
    'ultra': {
        'colormode': 'color', 'hierarchical': 'stacked', 'mode': 'spline',
        'filter_speckle': 1, 'color_precision': 8, 'layer_difference': 4,
        'corner_threshold': 30, 'length_threshold': 2.0, 'max_iterations': 20,
        'splice_threshold': 30, 'path_precision': 8,
    },
}

results = []

for preproc_name, preproc_fn in preprocessors.items():
    processed = preproc_fn(original_rgb)
    
    for settings_name, settings in settings_options.items():
        name = f"{preproc_name}+{settings_name}"
        print(f"Testing: {name}...", end=" ", flush=True)
        
        result = test_combination(original_rgb, processed, settings, name)
        results.append(result)
        
        print(f"SSIM={result['ssim']*100:.2f}%, paths={result['paths']}, "
              f"size={result['size_kb']:.1f}KB, problems={result['problems']}")

# Find best result
print("\n" + "=" * 70)
print("BEST RESULTS BY SSIM:")
results_sorted = sorted(results, key=lambda x: x['ssim'], reverse=True)

for i, r in enumerate(results_sorted[:5]):
    print(f"{i+1}. {r['name']}: SSIM={r['ssim']*100:.2f}%, "
          f"paths={r['paths']}, size={r['size_kb']:.1f}KB")

# Save best result
best = results_sorted[0]
print(f"\nSaving best result: {best['name']}")

output_svg = 'examples/ELITIZON_LOGO_BEST.svg'
with open(output_svg, 'w') as f:
    f.write(best['svg_content'])

cv2.imwrite('examples/ELITIZON_LOGO_BEST_rendered.png', 
            cv2.cvtColor(best['rendered'], cv2.COLOR_RGB2BGR))

# Save comparison
comparison = np.hstack([original_rgb, best['rendered']])
cv2.imwrite('examples/BEST_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

print(f"Saved to {output_svg}")
