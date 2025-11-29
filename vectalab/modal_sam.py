import os
import numpy as np

try:
    import modal
except ImportError:
    modal = None

# Define the Modal App
if modal:
    app = modal.App("vectalab-sam")

    # Define the image with dependencies
    sam_image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install(
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libpotrace0",
            "libpotrace-dev",
            "pkg-config",
            "libagg-dev",
            "libcairo2",
            "wget"
        )
        .pip_install(
            "segment-anything",
            "torch",
            "torchvision",
            "opencv-python",
            "numpy",
            "requests",
            "pypotrace",
            "svgwrite",
            "scour",
            "scikit-learn",
            "scikit-image",
            "scipy",
            "vtracer",
            "cairosvg",
            "Pillow"
        )
        .run_commands(
            "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O /root/sam_vit_h.pth",
            "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O /root/sam_vit_l.pth",
            "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O /root/sam_vit_b.pth"
        )
    )

    @app.cls(image=sam_image, gpu="A10G", timeout=600)
    class ModalSAM:
        model_type: str = modal.parameter(default="vit_h")
        kwargs_bytes: bytes = modal.parameter(default=b"")

        @modal.enter()
        def load_model(self):
            import torch
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
            import requests
            import os
            import pickle

            # Deserialize kwargs
            kwargs = {}
            if self.kwargs_bytes:
                try:
                    kwargs = pickle.loads(self.kwargs_bytes)
                except Exception as e:
                    print(f"Failed to deserialize kwargs: {e}")

            # Check for pre-downloaded checkpoint in /root/
            checkpoint_path = f"/root/sam_{self.model_type}.pth"
            
            # Fallback to local directory if not found (though it should be there)
            if not os.path.exists(checkpoint_path):
                checkpoint_path = f"sam_{self.model_type}.pth"
            
            if not os.path.exists(checkpoint_path):
                urls = {
                    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                }
                url = urls.get(self.model_type)
                if url:
                    print(f"Downloading {url}...")
                    response = requests.get(url, stream=True)
                    with open(checkpoint_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
            
            print(f"Loading SAM model ({self.model_type}) from {checkpoint_path}...")
            self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            self.sam.to(device="cuda")
            
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
            
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                **generator_args
            )

        @modal.method()
        def generate_masks(self, image_np):
            print(f"Generating masks for image shape: {image_np.shape}")
            masks = self.mask_generator.generate(image_np)
            # Masks contain boolean arrays which are serializable
            return masks
