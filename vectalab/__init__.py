"""
Vectalab - Professional High-Fidelity Image Vectorization

Vectalab converts raster images (PNG, JPG) to scalable vector graphics (SVG)
with optimized output for design tools like Figma and Illustrator.

Website: https://vectalab.com
"""

from .core import Vectalab
from .bayesian import optimize_vectorization, BayesianVectorRenderer
from .hifi import (
    vectorize_high_fidelity, 
    vectorize_for_figma,
    vectorize_with_quality,
    render_svg_to_png,
    compute_quality_metrics,
    list_presets,
)
from .optimize import (
    SVGOptimizer,
    create_figma_optimizer,
    create_quality_optimizer,
    optimize_svg_file,
    optimize_svg_string,
    get_vtracer_preset,
    VTRACER_PRESETS,
)
from .sota import (
    vectorize_smart,
    vectorize_logo,
    vectorize_icon,
    ImageAnalyzer,
)
from .quality import (
    vectorize_optimal,
    vectorize_quality,
    vectorize_logo_clean,
    compare_and_visualize,
    compute_pixel_metrics,
    analyze_image,
    reduce_to_palette,
    QUALITY_PRESETS,
)
from .premium import (
    vectorize_premium,
    vectorize_logo_premium,
    vectorize_photo_premium,
    edge_aware_denoise,
    reduce_to_clean_palette,
)

__version__ = "0.4.0"
__author__ = "Vectalab Contributors"

__all__ = [
    # Core
    'Vectalab', 
    'optimize_vectorization', 
    'BayesianVectorRenderer',
    # High-fidelity vectorization
    'vectorize_high_fidelity',
    'vectorize_for_figma',
    'vectorize_with_quality',
    'render_svg_to_png',
    'compute_quality_metrics',
    'list_presets',
    # Optimization
    'SVGOptimizer',
    'create_figma_optimizer',
    'create_quality_optimizer',
    'optimize_svg_file',
    'optimize_svg_string',
    'get_vtracer_preset',
    'VTRACER_PRESETS',
    # SOTA (Smart vectorization)
    'vectorize_smart',
    'vectorize_logo',
    'vectorize_icon',
    'ImageAnalyzer',
    # Quality-first vectorization
    'vectorize_optimal',
    'vectorize_quality',
    'vectorize_logo_clean',
    'compare_and_visualize',
    'compute_pixel_metrics',
    'analyze_image',
    'reduce_to_palette',
    'QUALITY_PRESETS',
    # Premium (SOTA) vectorization
    'vectorize_premium',
    'vectorize_logo_premium',
    'vectorize_photo_premium',
    'edge_aware_denoise',
    'reduce_to_clean_palette',
]
