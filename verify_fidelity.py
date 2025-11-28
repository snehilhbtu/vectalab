import sys
import os
import cv2
import numpy as np
import cairosvg
from skimage.metrics import structural_similarity as ssim

def compare_images(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None:
        raise ValueError(f"Could not load {img1_path}")
    if img2 is None:
        raise ValueError(f"Could not load {img2_path}")
        
    # Resize img2 to match img1 if needed
    if img1.shape != img2.shape:
        print(f"Resizing {img2.shape} to {img1.shape}")
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to grayscale for SSIM
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Calculate SSIM
    score, diff = ssim(gray1, gray2, full=True)
    print(f"SSIM Score: {score:.4f}")
    
    # Calculate Pixel Difference
    diff_pixels = np.sum(np.abs(img1.astype(int) - img2.astype(int)))
    total_pixels = img1.size
    avg_diff = diff_pixels / total_pixels
    print(f"Average Pixel Difference: {avg_diff:.4f}")
    
    return score

def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_fidelity.py <original_png> <generated_svg>")
        sys.exit(1)
        
    original_path = sys.argv[1]
    svg_path = sys.argv[2]
    rasterized_path = "rasterized_check.png"
    
    print(f"Rasterizing {svg_path} to {rasterized_path}...")
    try:
        cairosvg.svg2png(url=svg_path, write_to=rasterized_path)
    except Exception as e:
        print(f"Error rasterizing SVG: {e}")
        sys.exit(1)
        
    print("Comparing images...")
    score = compare_images(original_path, rasterized_path)
    
    if score >= 0.99:
        print("SUCCESS: Fidelity check passed (> 99%)")
        sys.exit(0)
    else:
        print("FAILURE: Fidelity check failed (< 99%)")
        sys.exit(1)

if __name__ == "__main__":
    main()
