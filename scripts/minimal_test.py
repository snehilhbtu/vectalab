#!/usr/bin/env python3
"""
Minimal vectorization test - just 2 icons with ultra-fast settings.
"""

import os
import subprocess

def run_vectalab_simple(input_png, output_svg):
    """Run with minimal settings for fast testing."""
    try:
        cmd = [
            "vectalab", "convert",
            input_png, output_svg,
            "--method", "hifi",
            "--quality", "figma",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True
        return False
    except:
        return False

# Just test 2 icons
icons = [
    ("test_data/png_mono/circle.png", "test_data/vectalab_mono/circle.svg"),
    ("test_data/png_multi/github.png", "test_data/vectalab_multi/github.svg"),
]

print("Minimal vectorization test (2 icons)...")
for png_path, svg_path in icons:
    name = os.path.basename(png_path)
    if run_vectalab_simple(png_path, svg_path):
        print(f"✓ {name}")
    else:
        print(f"✗ {name}")

print("\n✓ Test complete!")
