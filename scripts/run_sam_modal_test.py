#!/usr/bin/env python3
"""
Full test suite execution using SAM method with Modal cloud offloading.
"""

import sys
import os
import modal
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vectalab.core import Vectalab

def vectorize_icon(input_png, output_svg, vl):
    """Vectorize using the provided Vectalab instance."""
    try:
        input_path = str(Path(input_png).absolute())
        output_path = str(Path(output_svg).absolute())
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Vectorize
        vl.vectorize(input_path, output_path)
        
        print(f"✓ {Path(input_png).name}")
        return True
    except Exception as e:
        print(f"✗ {Path(input_png).name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Enable Modal logs
    # modal.enable_output() # Optional: enable if debugging is needed
    
    # Full test set from protocol
    test_set_mono = [
        'circle.png', 'square.png', 'triangle.png', 'star.png', 'heart.png', 
        'user.png', 'home.png', 'search.png', 'settings.png', 'camera.png', 
        'cloud.png', 'sun.png', 'moon.png', 'wind.png', 'cloud-rain.png', 
        'coffee.png', 'code.png', 'terminal.png', 'cpu.png', 'database.png'
    ]
    
    test_set_multi = [
        'github.png', 'twitter.png', 'facebook.png', 'instagram.png', 'youtube.png', 
        'linkedin.png', 'google.png', 'apple.png', 'microsoft.png', 'amazon.png', 
        'slack.png', 'spotify.png', 'netflix.png', 'airbnb.png', 'dropbox.png', 
        'trello.png', 'atlassian.png', 'jira.png', 'bitbucket.png', 'gitlab.png'
    ]
    
    print("Initializing Vectalab with SAM (Modal) method...")
    try:
        # Initialize Vectalab once to reuse the Modal app session if possible
        # Note: The current implementation creates a new app run for each segmentation call
        # but the object initialization happens here.
        vl = Vectalab(
            method="bayesian", # Using Bayesian method which uses SAM for initialization
            model_type="vit_h", 
            use_modal=True,
            device="cpu" # Local device for non-SAM parts
        )
    except Exception as e:
        print(f"Failed to initialize Vectalab: {e}")
        return

    print("\n=== Monochrome Icons ===")
    for filename in test_set_mono:
        png_path = f"test_data/png_mono/{filename}"
        svg_path = f"test_data/vectalab_mono/{filename.replace('.png', '.svg')}"
        if os.path.exists(png_path):
            vectorize_icon(png_path, svg_path, vl)
        else:
            print(f"⚠ Missing: {png_path}")
    
    print("\n=== Multi-Color Icons ===")
    for filename in test_set_multi:
        png_path = f"test_data/png_multi/{filename}"
        svg_path = f"test_data/vectalab_multi/{filename.replace('.png', '.svg')}"
        if os.path.exists(png_path):
            vectorize_icon(png_path, svg_path, vl)
        else:
            print(f"⚠ Missing: {png_path}")
    
    print("\n✓ Test suite complete!")

if __name__ == "__main__":
    main()
