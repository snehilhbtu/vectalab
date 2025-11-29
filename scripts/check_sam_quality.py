#!/usr/bin/env python3
import sys
import os
import numpy as np
from PIL import Image
import cairosvg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def check_quality(original_png, generated_svg):
    print(f"Checking quality for {generated_svg}...")
    
    # Render SVG to PNG
    temp_png = "temp_check.png"
    try:
        cairosvg.svg2png(url=generated_svg, write_to=temp_png)
    except Exception as e:
        print(f"Error rendering SVG: {e}")
        return

    # Load images
    img1 = Image.open(original_png).convert('RGB')
    img2 = Image.open(temp_png).convert('RGB')
    
    if img1.size != img2.size:
        print(f"Resizing rendered image from {img2.size} to {img1.size}")
        img2 = img2.resize(img1.size)
        
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    
    s = ssim(arr1, arr2, channel_axis=2, data_range=255)
    p = psnr(arr1, arr2, data_range=255)
    
    print(f"SSIM: {s*100:.2f}%")
    print(f"PSNR: {p:.2f} dB")
    
    os.remove(temp_png)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python check_sam_quality.py <original_png> <generated_svg>")
        sys.exit(1)
    check_quality(sys.argv[1], sys.argv[2])
