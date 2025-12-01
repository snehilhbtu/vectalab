"""
ARCHIVED: historic comparison helper

This script compared older/fixed SVG outputs to an original JPEG — kept here for historical debugging.
"""

#!/usr/bin/env python3
"""Compare old vs new SVG output quality."""

import cv2
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO
from skimage.metrics import structural_similarity as ssim

# Load original
original = cv2.imread('examples/ELITIZON_LOGO.jpg')
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
h, w = original_rgb.shape[:2]

print(f"Original: {w}x{h}")
print("=" * 60)

svgs_to_test = [
    ('FINAL (old broken)', 'examples/ELITIZON_LOGO_FINAL.svg'),
    ('BEST (new fixed)', 'examples/ELITIZON_LOGO_BEST.svg'),
]

for name, svg_path in svgs_to_test:
    with open(svg_path, 'r') as f:
        svg_content = f.read()
    
    file_size = len(svg_content.encode('utf-8'))
    path_count = svg_content.count('<path')
    
    # Render
    png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=w, output_height=h)
    rendered = np.array(Image.open(BytesIO(png_data)).convert('RGB'))
    
    # Metrics
    ssim_val = ssim(original_rgb, rendered, channel_axis=2, data_range=255)
    diff = np.abs(original_rgb.astype(float) - rendered.astype(float))
    mae = np.mean(diff)
    max_err = np.max(diff)
    
    diff_gray = np.mean(diff, axis=2)
    problem_50 = np.sum(diff_gray > 50)
    problem_100 = np.sum(diff_gray > 100)
    
    print(f"\n{name}:")
    print(f"  File: {file_size/1024:.1f} KB, Paths: {path_count}")
    print(f"  SSIM: {ssim_val*100:.2f}%")
    print(f"  MAE: {mae:.2f}, Max Error: {max_err:.0f}")
    print(f"  Problem pixels >50: {problem_50:,}")
    print(f"  Problem pixels >100: {problem_100:,}")

print("\n" + "=" * 60)
print("The BEST version should have text intact, unlike FINAL")
