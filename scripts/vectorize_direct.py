#!/usr/bin/env python3
"""
Direct vectorization using the Vectalab core library (no CLI).
"""

import sys
sys.path.insert(0, '/Users/raphaelmansuy/Github/03-working/vmagic')

from pathlib import Path
from vectalab.core import Vectalab

def vectorize_icon(input_png, output_svg, method="sam"):
    """Vectorize using the core library directly."""
    try:
        input_path = str(Path(input_png).absolute())
        output_path = str(Path(output_svg).absolute())
        
        # Create Vectalab instance
        vl = Vectalab(method=method, device="auto")
        
        # Vectorize
        vl.vectorize(input_path, output_path)
        
        print(f"✓ {Path(input_png).name}")
        return True
    except Exception as e:
        print(f"✗ {Path(input_png).name}: {e}")
        return False

def main():
    test_set_mono = ['circle.png', 'square.png', 'star.png']
    test_set_multi = ['github.png', 'google.png', 'apple.png']
    
    print("Vectorization using core library (SAM method)...")
    print("\nMonochrome:")
    for filename in test_set_mono:
        png_path = f"test_data/png_mono/{filename}"
        svg_path = f"test_data/vectalab_mono/{filename.replace('.png', '.svg')}"
        vectorize_icon(png_path, svg_path, method="sam")
    
    print("\nMulti-color:")
    for filename in test_set_multi:
        png_path = f"test_data/png_multi/{filename}"
        svg_path = f"test_data/vectalab_multi/{filename.replace('.png', '.svg')}"
        vectorize_icon(png_path, svg_path, method="sam")
    
    print("\n✓ Complete!")

if __name__ == "__main__":
    main()