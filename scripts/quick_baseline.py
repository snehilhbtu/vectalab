"""
This script was archived & moved to `scripts/archived/quick_baseline.py`.

If you need to restore the original, retrieve it from `scripts/archived/` and move it back.
"""

import os
import subprocess

def run_vectalab(input_png, output_svg):
    """Run Vectalab with fast settings."""
    try:
        cmd = [
            "vectalab", "convert",
            input_png, output_svg,
            "--method", "hifi",
            "--quality", "figma",  # Fastest preset
            "--force",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"✓ {os.path.basename(input_png)}")
            return True
        else:
            print(f"✗ {os.path.basename(input_png)}")
            return False
    except Exception as e:
        print(f"✗ {os.path.basename(input_png)}: {e}")
        return False

def main():
    # Quick test set - just 3 monochrome and 3 multi-color
    test_set_mono = ['circle.png', 'square.png', 'star.png']
    test_set_multi = ['github.png', 'google.png', 'apple.png']
    
    print("Quick vectorization baseline (figma preset)...")
    print("\nMonochrome:")
    for filename in test_set_mono:
        png_path = f"test_data/png_mono/{filename}"
        svg_path = f"test_data/vectalab_mono/{filename.replace('.png', '.svg')}"
        run_vectalab(png_path, svg_path)
    
    print("\nMulti-color:")
    for filename in test_set_multi:
        png_path = f"test_data/png_multi/{filename}"
        svg_path = f"test_data/vectalab_multi/{filename.replace('.png', '.svg')}"
        run_vectalab(png_path, svg_path)
    
    print("\n✓ Quick baseline complete!")

if __name__ == "__main__":
    main()