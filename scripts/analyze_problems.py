#!/usr/bin/env python3
"""Analyze where the quality problems are in the vectorized output."""

import cv2
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO

# Load files
original = cv2.imread('examples/ELITIZON_LOGO.jpg')
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
h, w = original_rgb.shape[:2]

with open('examples/ELITIZON_LOGO_BEST.svg', 'r') as f:
    svg_content = f.read()

png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), output_width=w, output_height=h)
rendered = np.array(Image.open(BytesIO(png_data)).convert('RGB'))

# Compute difference
diff = np.abs(original_rgb.astype(float) - rendered.astype(float))
diff_gray = np.mean(diff, axis=2)

# Find problem regions
problem_mask = diff_gray > 50

# Analyze by region
regions = {
    'logo_icon': (180, 370, 80, 180),  # (x1, x2, y1, y2)
    'text_eli': (370, 520, 230, 330),
    'text_ti': (520, 570, 230, 330),
    'text_zon': (570, 730, 230, 330),
    'text_ltd': (740, 920, 230, 330),
}

print("Problem pixel analysis by region:")
print("=" * 50)

for name, (x1, x2, y1, y2) in regions.items():
    region_mask = problem_mask[y1:y2, x1:x2]
    problem_count = np.sum(region_mask)
    total_pixels = (x2-x1) * (y2-y1)
    pct = problem_count / total_pixels * 100
    
    region_diff = diff_gray[y1:y2, x1:x2]
    max_err = np.max(region_diff)
    mean_err = np.mean(region_diff)
    
    print(f"{name:15} Problems: {problem_count:5} ({pct:5.2f}%) "
          f"MaxErr: {max_err:5.1f} MeanErr: {mean_err:4.2f}")

# Create visualization
vis = original_rgb.copy()
vis[problem_mask] = [255, 0, 0]  # Mark problems in red

cv2.imwrite('examples/problem_regions.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"\nSaved problem visualization to examples/problem_regions.png")

# Check if any text is completely missing
print("\n" + "=" * 50)
print("Checking for missing content...")

# Sample specific pixels in text regions
test_points = [
    ('E in Eli', 385, 270),
    ('l in Eli', 430, 280),
    ('i in Eli', 455, 280),
    ('t in ti', 495, 280),
    ('i in ti', 540, 270),
    ('z in zon', 590, 280),
    ('o in zon', 630, 280),
    ('n in zon', 700, 280),
]

for name, x, y in test_points:
    orig_px = original_rgb[y, x]
    rend_px = rendered[y, x]
    diff_px = np.abs(orig_px.astype(float) - rend_px.astype(float))
    err = np.mean(diff_px)
    print(f"{name:12} Orig:{orig_px} Rend:{rend_px} Err:{err:.1f}")
