#!/usr/bin/env python3
"""
Test suite for vtracer-based vectorization.
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2
import pytest
from skimage.metrics import structural_similarity as ssim

try:
    import vtracer
    VTRACER_AVAILABLE = True
except ImportError:
    VTRACER_AVAILABLE = False

try:
    import cairosvg
    from PIL import Image
    import io
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False


def get_test_image_path():
    """Find test image path."""
    candidates = [
        Path(__file__).parent.parent / "examples" / "ELITIZON_LOGO.jpg",
        Path(__file__).parent.parent / "test_case" / "ELITIZON_LOGO.jpg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


@pytest.fixture
def test_image_path():
    path = get_test_image_path()
    if path is None:
        pytest.skip("Test image not found")
    return path


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.skipif(not VTRACER_AVAILABLE, reason="vtracer not installed")
class TestVtracer:
    """Test vtracer vectorization."""
    
    def test_basic_vectorization(self, test_image_path, temp_output_dir):
        """Test basic vtracer conversion."""
        svg_path = os.path.join(temp_output_dir, "output.svg")
        
        vtracer.convert_image_to_svg_py(
            test_image_path, svg_path,
            colormode='color',
            hierarchical='stacked',
            mode='spline'
        )
        
        assert os.path.exists(svg_path)
        assert os.path.getsize(svg_path) > 0
    
    @pytest.mark.skipif(not CAIROSVG_AVAILABLE, reason="cairosvg not installed")
    def test_vectorization_quality(self, test_image_path, temp_output_dir):
        """Test vtracer achieves good quality."""
        svg_path = os.path.join(temp_output_dir, "output.svg")
        
        # Load original
        original = cv2.imread(test_image_path)
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        h, w = original_rgb.shape[:2]
        
        # Vectorize with ultra settings
        vtracer.convert_image_to_svg_py(
            test_image_path, svg_path,
            colormode='color',
            hierarchical='stacked',
            mode='polygon',
            filter_speckle=0,
            color_precision=8,
            layer_difference=1,
        )
        
        # Render back
        png_data = cairosvg.svg2png(url=svg_path, output_width=w, output_height=h)
        rendered = np.array(Image.open(io.BytesIO(png_data)).convert('RGB'))
        
        # Calculate SSIM
        ssim_val = ssim(original_rgb, rendered, channel_axis=2, data_range=255)
        
        # vtracer should achieve at least 99% SSIM
        assert ssim_val >= 0.99, f"SSIM {ssim_val:.4f} below 0.99"


def main():
    """Run tests manually."""
    if not VTRACER_AVAILABLE:
        print("vtracer not installed")
        return 1
    
    test_image = get_test_image_path()
    if not test_image:
        print("Test image not found")
        return 1
    
    print("Testing vtracer vectorization...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        svg_path = os.path.join(tmpdir, "output.svg")
        
        vtracer.convert_image_to_svg_py(
            test_image, svg_path,
            colormode='color',
            hierarchical='stacked',
            mode='polygon',
            filter_speckle=0,
            color_precision=8,
            layer_difference=1,
        )
        
        print(f"SVG created: {os.path.getsize(svg_path)/1024:.1f} KB")
        
        if CAIROSVG_AVAILABLE:
            original = cv2.imread(test_image)
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            h, w = original_rgb.shape[:2]
            
            png_data = cairosvg.svg2png(url=svg_path, output_width=w, output_height=h)
            rendered = np.array(Image.open(io.BytesIO(png_data)).convert('RGB'))
            
            ssim_val = ssim(original_rgb, rendered, channel_axis=2, data_range=255)
            print(f"SSIM: {ssim_val:.4f} ({ssim_val*100:.2f}%)")
        
        print("✅ vtracer test passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
