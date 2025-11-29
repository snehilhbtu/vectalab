#!/usr/bin/env python3
"""Profile the vectorization pipeline to find bottlenecks."""

import time

# First warm up imports
print('Warming up imports...')
import_start = time.time()

import cv2
import numpy as np
import tempfile
import os
import vtracer
from vectalab.premium import (
    edge_aware_denoise, 
    sharpen_corners, 
    reduce_to_clean_palette, 
    render_svg_to_array, 
    compute_ssim
)

print(f'Import time: {time.time() - import_start:.2f}s')

print()
print('='*50)
print('ACTUAL PROCESSING TIME (after imports)')
print('='*50)

total_start = time.time()

image = cv2.imread('examples/test_logo_benchmark.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(f'Image size: {image_rgb.shape}')

start = time.time()
processed = edge_aware_denoise(image_rgb)
processed = sharpen_corners(processed, strength=0.3)
print(f'Edge denoise: {time.time() - start:.3f}s')

start = time.time()
reduced, palette = reduce_to_clean_palette(processed, 8, snap_to_standard=True)
print(f'Palette:      {time.time() - start:.3f}s')

start = time.time()
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    tmp_path = f.name
cv2.imwrite(tmp_path, cv2.cvtColor(reduced, cv2.COLOR_RGB2BGR))
with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
    svg_path = f.name
vtracer.convert_image_to_svg_py(
    tmp_path, svg_path, 
    colormode='color', 
    hierarchical='stacked', 
    mode='spline', 
    filter_speckle=2
)
with open(svg_path, 'r') as f:
    svg = f.read()
print(f'Vtracer:      {time.time() - start:.3f}s')

start = time.time()
rendered = render_svg_to_array(svg, 400, 200)
ssim_val = compute_ssim(image_rgb, rendered)
print(f'SSIM:         {time.time() - start:.3f}s')

print()
print(f'TOTAL:        {time.time() - total_start:.3f}s')
print(f'SSIM:         {ssim_val*100:.2f}%')
print(f'SVG size:     {len(svg):,} bytes')

os.remove(tmp_path)
os.remove(svg_path)

print()
print('='*50)
print('PROFILING COMPLETE')
print('='*50)
