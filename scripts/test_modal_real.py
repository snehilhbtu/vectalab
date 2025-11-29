import os
import sys
import cv2
import numpy as np
import modal
from vectalab.segmentation import SAMSegmenter

def test_modal_sam():
    modal.enable_output()
    print("Testing Modal SAM integration...")
    
    # Create a dummy image
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 100, (255, 255, 255), -1)
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
    
    print("Initializing SAMSegmenter with use_modal=True...")
    try:
        segmenter = SAMSegmenter(model_type="vit_h", use_modal=True)
        
        print("Running segmentation...")
        masks = segmenter.segment(img)
        
        print(f"Success! Received {len(masks)} masks.")
        for i, mask in enumerate(masks):
            print(f"Mask {i}: area={mask['area']}, bbox={mask['bbox']}")
            
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_modal_sam()
