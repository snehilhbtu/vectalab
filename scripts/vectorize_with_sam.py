#!/usr/bin/env python3
"""
Vectorize images using Segment Anything Model (SAM).
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import svgwrite
from skimage import measure
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vectalab.segmentation import SAMSegmenter

def mask_to_polygon(mask, epsilon=2.0):
    """Convert binary mask to polygon."""
    # Find contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 10:
            continue
            
        # Simplify
        poly = cv2.approxPolyDP(contour, epsilon, True)
        if len(poly) < 3:
            continue
            
        polygons.append(poly.reshape(-1, 2))
        
    return polygons

def get_average_color(image, mask):
    """Get average color of masked region."""
    masked_img = image[mask]
    if len(masked_img) == 0:
        return (0, 0, 0)
    return tuple(map(int, np.mean(masked_img, axis=0)))

def vectorize_image_sam(image_path, output_path, model_type="vit_b", device="cpu"):
    print(f"Loading image {image_path}...")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Initializing SAM ({model_type})...")
    # Point to the local checkpoint we just downloaded
    checkpoint_path = f"sam_{model_type}.pth"
    if not os.path.exists(checkpoint_path):
        # Try looking in current dir if not found
        checkpoint_path = os.path.join(os.getcwd(), f"sam_{model_type}.pth")
        
    # Configure for more detail
    custom_args = {
        "points_per_side": 128,
        "pred_iou_thresh": 0.70,
        "stability_score_thresh": 0.80,
        "min_mask_region_area": 10,
        "crop_n_layers": 2,
    }
    
    segmenter = SAMSegmenter(model_type=model_type, checkpoint_path=checkpoint_path, device=device, **custom_args)
    
    print("Generating masks...")
    masks = segmenter.segment(image)
    print(f"Found {len(masks)} masks.")
    
    # Sort masks by area (largest first to draw background first)
    masks.sort(key=lambda x: x['area'], reverse=True)
    
    # Create SVG
    dwg = svgwrite.Drawing(output_path, size=(image.shape[1], image.shape[0]))
    
    # Add background rect (average color of whole image)
    avg_bg = tuple(map(int, np.mean(image, axis=(0, 1))))
    dwg.add(dwg.rect(insert=(0, 0), size=('100%', '100%'), fill=f"rgb({avg_bg[0]},{avg_bg[1]},{avg_bg[2]})"))
    
    print("Converting to SVG...")
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = get_average_color(image, mask)
        polygons = mask_to_polygon(mask)
        
        rgb_color = f"rgb({color[0]},{color[1]},{color[2]})"
        
        for poly in polygons:
            points = [(float(p[0]), float(p[1])) for p in poly]
            dwg.add(dwg.polygon(points=points, fill=rgb_color, stroke="none"))
            
    dwg.save()
    print(f"Saved to {output_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python vectorize_with_sam.py <input_image> <output_svg>")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    # MPS has float64 issues with SAM, so default to CPU for stability
    device = "cpu"
    print(f"Using device: {device}")
    
    vectorize_image_sam(input_path, output_path, device=device)
