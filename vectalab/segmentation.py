import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class SAMSegmenter:
    def __init__(self, model_type="vit_h", checkpoint_path=None, device="cpu", use_modal=False, **kwargs):
        self.device = device
        self.model_type = model_type
        self.use_modal = use_modal
        
        if self.use_modal:
            try:
                from .modal_sam import app, ModalSAM
                self.app = app
                self.ModalSAM = ModalSAM
                self.kwargs = kwargs
                print("Initialized SAM with Modal backend.")
                return
            except ImportError:
                print("Warning: Modal not found or import failed. Falling back to local execution.")
                self.use_modal = False

        self.checkpoint_path = checkpoint_path or self._get_default_checkpoint_path(model_type)
        
        if not os.path.exists(self.checkpoint_path):
            print(f"Checkpoint not found at {self.checkpoint_path}. Downloading...")
            self._download_checkpoint(model_type, self.checkpoint_path)

        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        elif device == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = 'cpu'

        print(f"Loading SAM model ({model_type}) from {self.checkpoint_path} to {device}...")
        self.sam = sam_model_registry[model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=device)
        
        # Default parameters
        generator_args = {
            "points_per_side": 32,
            "pred_iou_thresh": 0.86,
            "stability_score_thresh": 0.92,
            "crop_n_layers": 1,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": 100,
        }
        # Update with provided kwargs
        generator_args.update(kwargs)
        
        print(f"Initializing Mask Generator with args: {generator_args}")
        
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            **generator_args
        )

    def _get_default_checkpoint_path(self, model_type):
        # Default to current directory or a cache directory
        return f"sam_{model_type}.pth"

    def _download_checkpoint(self, model_type, path):
        urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        }
        url = urls.get(model_type)
        if not url:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Downloading {url} to {path}...")
        import requests
        response = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def segment(self, image):
        """
        Returns a list of masks.
        Each mask is a dict with keys: 'segmentation', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'
        """
        if self.use_modal:
            print("Running segmentation on Modal...")
            masks = None
            try:
                import pickle
                kwargs_bytes = pickle.dumps(self.kwargs)
                with self.app.run():
                    # Pass kwargs as bytes
                    model = self.ModalSAM(model_type=self.model_type, kwargs_bytes=kwargs_bytes)
                    masks = model.generate_masks.remote(image)
            except Exception as e:
                print(f"Modal execution failed: {e}")
                raise e
            
            if masks is None:
                raise RuntimeError("Modal execution failed to return masks.")
                
            return masks

        masks = self.mask_generator.generate(image)
        # Sort by area (largest first) to handle layering if needed, 
        # but for vectorization, we might want smallest first to draw on top.
        # Let's return as is, the core logic can decide.
        return masks
