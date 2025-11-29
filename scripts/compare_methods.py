#!/usr/bin/env python3
"""Compare all vectorization methods on the ELITIZON logo."""

import os
import sys
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/vmagic')

from vectalab.quality import vectorize_logo_clean, vectorize_optimal
from vectalab.premium import vectorize_premium, vectorize_logo_premium
import numpy as np
from PIL import Image
import cairosvg
from io import BytesIO
from skimage.metrics import structural_similarity as ssim

INPUT = "/Users/raphaelmansuy/Github/03-working/vmagic/examples/ELITIZON_LOGO.jpg"
OUTPUT_DIR = "/Users/raphaelmansuy/Github/03-working/vmagic/examples"

# Load original
orig = np.array(Image.open(INPUT).convert('RGB'))
h, w = orig.shape[:2]

def measure_svg(svg_path):
    """Measure SVG quality metrics."""
    with open(svg_path, 'r') as f:
        svg = f.read()
    
    # File size
    size = os.path.getsize(svg_path)
    
    # Path count
    paths = svg.count('<path')
    
    # Render and compare
    png_data = cairosvg.svg2png(url=svg_path, output_width=w, output_height=h)
    rendered = np.array(Image.open(BytesIO(png_data)).convert('RGB'))
    score = ssim(orig, rendered, channel_axis=2)
    
    return score, size, paths

print("=" * 70)
print("COMPARISON: ELITIZON LOGO VECTORIZATION METHODS")
print("=" * 70)
print(f"Input: ELITIZON_LOGO.jpg ({w}x{h})")
print()

results = []

# 1. Logo command (palette reduction)
print("1. Logo command (palette reduction)...")
out1 = f"{OUTPUT_DIR}/test_logo.svg"
vectorize_logo_clean(INPUT, out1, verbose=False)
s1, sz1, p1 = measure_svg(out1)
results.append(("Logo", s1, sz1, p1))
print(f"   SSIM: {s1*100:.2f}%, Size: {sz1/1024:.1f} KB, Paths: {p1}")

# 2. Optimal command (bilateral + quality)
print("2. Optimal command (bilateral + quality)...")
out2 = f"{OUTPUT_DIR}/test_optimal.svg"
vectorize_optimal(INPUT, out2, verbose=False)
s2, sz2, p2 = measure_svg(out2)
results.append(("Optimal", s2, sz2, p2))
print(f"   SSIM: {s2*100:.2f}%, Size: {sz2/1024:.1f} KB, Paths: {p2}")

# 3. Premium command (SOTA)
print("3. Premium command (SOTA auto)...")
out3 = f"{OUTPUT_DIR}/test_premium.svg"
vectorize_premium(INPUT, out3, verbose=False)
s3, sz3, p3 = measure_svg(out3)
results.append(("Premium", s3, sz3, p3))
print(f"   SSIM: {s3*100:.2f}%, Size: {sz3/1024:.1f} KB, Paths: {p3}")

# 4. Premium Logo mode
print("4. Premium Logo mode...")
out4 = f"{OUTPUT_DIR}/test_premium_logo.svg"
vectorize_logo_premium(INPUT, out4, verbose=False)
s4, sz4, p4 = measure_svg(out4)
results.append(("Premium Logo", s4, sz4, p4))
print(f"   SSIM: {s4*100:.2f}%, Size: {sz4/1024:.1f} KB, Paths: {p4}")

print()
print("=" * 70)
print(f"{'Method':<15} {'SSIM':>10} {'Size':>10} {'Paths':>8}")
print("-" * 70)
for name, s, sz, p in results:
    print(f"{name:<15} {s*100:>9.2f}% {sz/1024:>9.1f}KB {p:>8}")
print("=" * 70)

# Find best
best = max(results, key=lambda x: x[1])
print(f"\nBest quality: {best[0]} ({best[1]*100:.2f}% SSIM)")

smallest = min(results, key=lambda x: x[2])
print(f"Smallest file: {smallest[0]} ({smallest[2]/1024:.1f} KB)")

fewest = min(results, key=lambda x: x[3])
print(f"Fewest paths: {fewest[0]} ({fewest[3]} paths)")

# Cleanup test files
for out in [out1, out2, out3, out4]:
    try:
        os.remove(out)
    except:
        pass
