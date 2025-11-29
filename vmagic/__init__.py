from .core import VMagic
from .bayesian import optimize_vectorization, BayesianVectorRenderer
from .hifi import vectorize_high_fidelity, render_svg_to_png

__all__ = [
    'VMagic', 
    'optimize_vectorization', 
    'BayesianVectorRenderer',
    'vectorize_high_fidelity',
    'render_svg_to_png'
]
