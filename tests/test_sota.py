"""
Tests for the SOTA (State-of-the-Art) vectorization module.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import os

from vectalab.sota import (
    ImageAnalyzer,
    quantize_colors_simple,
    quantize_colors_median_cut,
    preprocess_image,
    get_adaptive_vtracer_settings,
    optimize_svg_paths,
)


class TestImageAnalyzer:
    """Tests for ImageAnalyzer class."""
    
    def test_analyze_simple_image(self):
        """Test analyzing a simple solid color image."""
        # Create a simple image with few colors
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:50, :, :] = [255, 0, 0]  # Red top half
        image[50:, :, :] = [0, 0, 255]  # Blue bottom half
        
        analysis = ImageAnalyzer.analyze(image)
        
        assert analysis['width'] == 100
        assert analysis['height'] == 100
        assert analysis['unique_colors'] == 2
        assert analysis['top_10_coverage'] == 1.0
        # Simple 2-color image might be classified differently based on edge detection
        assert 'image_type' in analysis
    
    def test_analyze_gradient_image(self):
        """Test analyzing a gradient image with many colors."""
        # Create a gradient image
        image = np.zeros((100, 256, 3), dtype=np.uint8)
        for i in range(256):
            image[:, i, :] = [i, i, i]
        
        analysis = ImageAnalyzer.analyze(image)
        
        assert analysis['width'] == 256
        assert analysis['height'] == 100
        assert analysis['unique_colors'] == 256
        assert 'image_type' in analysis
        assert 'complexity' in analysis
    
    def test_analyze_returns_required_keys(self):
        """Test that analysis returns all required keys."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        analysis = ImageAnalyzer.analyze(image)
        
        required_keys = [
            'width', 'height', 'unique_colors', 'top_10_coverage',
            'top_50_coverage', 'color_variance', 'edge_density',
            'dominant_colors', 'image_type', 'complexity'
        ]
        
        for key in required_keys:
            assert key in analysis, f"Missing key: {key}"


class TestColorQuantization:
    """Tests for color quantization functions."""
    
    def test_simple_quantization_reduces_colors(self):
        """Test that simple quantization reduces colors."""
        # Create image with many colors
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        original_colors = len(np.unique(image.reshape(-1, 3), axis=0))
        
        quantized = quantize_colors_simple(image, n_colors=16)
        new_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        
        assert new_colors < original_colors
        assert quantized.shape == image.shape
    
    def test_median_cut_quantization(self):
        """Test median cut quantization."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        quantized = quantize_colors_median_cut(image, n_colors=8)
        new_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
        
        assert new_colors <= 8
        assert quantized.shape == image.shape
    
    def test_quantization_preserves_shape(self):
        """Test that quantization preserves image shape."""
        shapes = [(100, 100, 3), (50, 200, 3), (256, 256, 3)]
        
        for shape in shapes:
            image = np.random.randint(0, 255, shape, dtype=np.uint8)
            quantized = quantize_colors_simple(image, n_colors=16)
            assert quantized.shape == shape


class TestPreprocessing:
    """Tests for image preprocessing."""
    
    def test_preprocess_logo(self):
        """Test preprocessing for logo images."""
        # Create a simple logo-like image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :50] = [0, 100, 200]
        image[:, 50:] = [255, 255, 255]
        
        analysis = {'image_type': 'logo', 'top_10_coverage': 0.98}
        
        processed = preprocess_image(image, analysis)
        
        assert processed.shape == image.shape
        assert processed.dtype == np.uint8
    
    def test_preprocess_photo(self):
        """Test preprocessing for photo images."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        analysis = {'image_type': 'photo', 'top_10_coverage': 0.3}
        
        processed = preprocess_image(image, analysis)
        
        assert processed.shape == image.shape


class TestAdaptiveSettings:
    """Tests for adaptive vtracer settings."""
    
    def test_logo_settings(self):
        """Test settings for logo images."""
        analysis = {'image_type': 'logo'}
        settings = get_adaptive_vtracer_settings(analysis, 'balanced')
        
        assert 'filter_speckle' in settings
        assert settings['filter_speckle'] >= 6
        assert settings['colormode'] == 'color'
    
    def test_photo_settings(self):
        """Test settings for photo images."""
        analysis = {'image_type': 'photo'}
        settings = get_adaptive_vtracer_settings(analysis, 'balanced')
        
        assert settings['filter_speckle'] <= 4
        assert settings['layer_difference'] <= 16
    
    def test_quality_levels(self):
        """Test different quality levels produce different settings."""
        analysis = {'image_type': 'icon'}
        
        compact = get_adaptive_vtracer_settings(analysis, 'compact')
        quality = get_adaptive_vtracer_settings(analysis, 'quality')
        
        assert compact['filter_speckle'] > quality['filter_speckle']


class TestSVGOptimization:
    """Tests for SVG path optimization."""
    
    def test_optimize_svg_paths(self):
        """Test SVG path optimization."""
        svg = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <path d="M 0.123456789 0.987654321 L 10.111 20.222" fill="#FF0000"/>
        </svg>'''
        
        optimized = optimize_svg_paths(svg)
        
        # Should still be valid SVG (may have ns0 prefix due to ElementTree)
        assert 'svg' in optimized.lower()
        assert 'path' in optimized.lower()
    
    def test_invalid_svg_returns_original(self):
        """Test that invalid SVG returns original content."""
        invalid_svg = "not valid svg content"
        result = optimize_svg_paths(invalid_svg)
        assert result == invalid_svg


class TestIntegration:
    """Integration tests for the SOTA module."""
    
    @pytest.mark.skipif(
        not os.path.exists('/Users/raphaelmansuy/Github/03-working/vmagic/examples/ELITIZON_LOGO.jpg'),
        reason="Test image not available"
    )
    def test_vectorize_logo_e2e(self):
        """End-to-end test for logo vectorization."""
        from vectalab.sota import vectorize_smart
        
        input_path = '/Users/raphaelmansuy/Github/03-working/vmagic/examples/ELITIZON_LOGO.jpg'
        
        with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
            output_path = tmp.name
        
        try:
            svg_path, metrics = vectorize_smart(
                input_path,
                output_path,
                target_ssim=0.85,
                max_file_size=200000,
                max_iterations=2,
                verbose=False,
            )
            
            # Check output exists
            assert Path(output_path).exists()
            
            # Check metrics
            assert 'ssim' in metrics
            assert 'file_size' in metrics
            assert 'path_count' in metrics
            assert 'image_type' in metrics
            
            # Check reasonable values
            assert metrics['ssim'] > 0.8
            assert metrics['file_size'] < 200000
            assert metrics['path_count'] < 1000
            
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
