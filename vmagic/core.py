import os
import cv2
import numpy as np
from pathlib import Path
from .segmentation import SAMSegmenter
from .tracing import Tracer
from .output import SVGWriter

class VMagic:
    def __init__(self, model_type="vit_h", checkpoint_path=None, device="cpu"):
        self.segmenter = SAMSegmenter(model_type, checkpoint_path, device)
        self.tracer = Tracer()
        self.writer = SVGWriter()

    def vectorize(self, image_path, output_path):
        """
        Main pipeline:
        1. Load Image
        2. Segment Image (SAM)
        3. Trace Segments (Potrace)
        4. Save to SVG
        """
        print(f"Processing {image_path}...")
        
        # 1. Load Image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Segment Image
        print("Running segmentation...")
        masks = self.segmenter.segment(image)
        print(f"Found {len(masks)} segments.")

        # 3. Trace Segments
        print("Tracing paths...")
        paths = self.tracer.trace(image, masks)

        # 4. Save Output
        print(f"Saving to {output_path}...")
        self.writer.save(paths, output_path, image.shape[:2])
        print("Done.")
