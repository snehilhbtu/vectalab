#!/usr/bin/env python3
"""Test ultra quality vtracer settings."""

import vtracer
import cv2
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO
from skimage.metrics import structural_similarity as ssim

# Maximum quality settings
settings = {
    'colormode': 'color',
    'hierarchical': 'stacked',
    'mode': 'spline',
    'filter_speckle': 1,
    'color_precision': 8,
    'layer_difference': 1,
    'corner_threshold': 20,
    'length_threshold': 1.5,
    'max_iterations': 30,
    'splice_threshold': 20,
    'path_precision': 8,
}

input_file = 'examples/ELITIZON_LOGO.jpg'
output_file = 'examples/ELITIZON_LOGO_ULTRA.svg'

print("Vectorizing with ultra settings...")
vtracer.convert_image_to_svg_py(input_file, output_file, **settings)

# Check quality
with open(output_file, 'r') as f:
    svg_content = f.read()

original = cv2.imread(input_file)
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
h, w = original_rgb.shape[:2]

print("Rendering SVG...")
png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=w, output_height=h)
rendered = np.array(Image.open(BytesIO(png_data)).convert('RGB'))

# Save rendered
cv2.imwrite('examples/ELITIZON_LOGO_ULTRA_rendered.png', cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

ssim_val = ssim(original_rgb, rendered, channel_axis=2, data_range=255)
path_count = svg_content.count('<path')
file_size = len(svg_content.encode('utf-8'))

print(f"\nResults:")
print(f"SSIM: {ssim_val*100:.2f}%")
print(f"Paths: {path_count}")
print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")

diff = np.abs(original_rgb.astype(float) - rendered.astype(float))
problem_pixels = np.sum(np.mean(diff, axis=2) > 50)
print(f"Problem pixels (>50): {problem_pixels:,}")

# Save comparison
comparison = np.hstack([original_rgb, rendered])
cv2.imwrite('examples/ULTRA_comparison.png', cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
print(f"\nSaved comparison to examples/ULTRA_comparison.png")
