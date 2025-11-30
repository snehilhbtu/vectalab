"""
Perceptual Quality Metrics for Vectalab.

This module implements advanced perceptual metrics using deep learning models
to evaluate the visual similarity between images in a way that correlates
better with human perception than traditional metrics like MSE or SSIM.

Metrics included:
1. LPIPS (Learned Perceptual Image Patch Similarity)
2. DISTS (Deep Image Structure and Texture Similarity)
3. GMSD (Gradient Magnitude Similarity Deviation)
"""

import numpy as np
import torch
import logging
from typing import Optional, Union
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Global model instances
_LPIPS_MODEL = None
_DISTS_MODEL = None
_GMSD_MODEL = None

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

def get_lpips_model(net: str = 'alex'):
    """
    Get or initialize the LPIPS model.
    
    Args:
        net: Network backbone to use ('alex', 'vgg', 'squeeze'). 
             'alex' is faster and widely used.
             
    Returns:
        LPIPS model instance or None if initialization fails.
    """
    global _LPIPS_MODEL
    if _LPIPS_MODEL is not None:
        return _LPIPS_MODEL

    try:
        import lpips
        # Initialize model
        # spatial=False returns a single scalar value
        model = lpips.LPIPS(net=net, spatial=False)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model.cuda()
        elif torch.backends.mps.is_available():
             # MPS (Metal Performance Shaders) for macOS
             model.to('mps')
             
        model.eval()
        _LPIPS_MODEL = model
        return _LPIPS_MODEL
    except ImportError:
        logger.warning("LPIPS package not found. Install with 'pip install lpips'.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LPIPS model: {e}")
        return None

def preprocess_for_lpips(img: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    """
    Preprocess image for LPIPS model.
    
    Args:
        img: Input image (numpy array or PIL Image)
        
    Returns:
        Torch tensor normalized to [-1, 1] with shape (1, 3, H, W)
    """
    if isinstance(img, np.ndarray):
        # Convert numpy array to PIL Image if needed
        if img.dtype == np.uint8:
            img = Image.fromarray(img)
        else:
            # Assuming float 0-1
            img = Image.fromarray((img * 255).astype(np.uint8))
            
    if isinstance(img, Image.Image):
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to tensor
        # LPIPS expects values in [-1, 1]
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        tensor = transform(img).unsqueeze(0) # Add batch dimension
        
        return tensor
    
    raise ValueError(f"Unsupported image type: {type(img)}")

def preprocess_for_piq(img: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    """
    Preprocess image for PIQ metrics (0-1 range).
    """
    if isinstance(img, np.ndarray):
        if img.dtype == np.uint8:
            img = Image.fromarray(img)
        else:
            img = Image.fromarray((img * 255).astype(np.uint8))
            
    if isinstance(img, Image.Image):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        import torchvision.transforms as transforms
        transform = transforms.ToTensor() # Converts to [0, 1]
        tensor = transform(img).unsqueeze(0)
        return tensor
    
    raise ValueError(f"Unsupported image type: {type(img)}")

def calculate_dists(img1: Union[np.ndarray, Image.Image], 
                   img2: Union[np.ndarray, Image.Image]) -> float:
    """
    Calculate DISTS (Deep Image Structure and Texture Similarity).
    Lower is better (0.0 means identical).
    """
    global _DISTS_MODEL
    try:
        import piq
        
        if _DISTS_MODEL is None:
            _DISTS_MODEL = piq.DISTS()
            _DISTS_MODEL.to(get_device())
            _DISTS_MODEL.eval()
            
        t1 = preprocess_for_piq(img1).to(get_device())
        t2 = preprocess_for_piq(img2).to(get_device())
        
        with torch.no_grad():
            score = _DISTS_MODEL(t1, t2)
            
        return float(score.item())
    except ImportError:
        logger.warning("piq package not found. Install with 'pip install piq'.")
        return None
    except Exception as e:
        logger.error(f"Error calculating DISTS: {e}")
        return None

def calculate_gmsd(img1: Union[np.ndarray, Image.Image], 
                  img2: Union[np.ndarray, Image.Image]) -> float:
    """
    Calculate GMSD (Gradient Magnitude Similarity Deviation).
    Lower is better (0.0 means identical).
    """
    try:
        import piq
        
        t1 = preprocess_for_piq(img1).to(get_device())
        t2 = preprocess_for_piq(img2).to(get_device())
        
        with torch.no_grad():
            score = piq.gmsd(t1, t2)
            
        return float(score.item())
    except ImportError:
        logger.warning("piq package not found. Install with 'pip install piq'.")
        return None
    except Exception as e:
        logger.error(f"Error calculating GMSD: {e}")
        return None

def calculate_lpips(img1: Union[np.ndarray, Image.Image], 
                   img2: Union[np.ndarray, Image.Image],
                   net: str = 'alex') -> float:
    """
    Calculate LPIPS distance between two images.
    Lower is better (0.0 means identical).
    
    Args:
        img1: First image (Reference)
        img2: Second image (Distorted/Vectorized)
        net: Backbone network ('alex', 'vgg', 'squeeze')
        
    Returns:
        LPIPS distance (float) or None if model unavailable.
    """
    model = get_lpips_model(net)
    if model is None:
        return None
        
    try:
        # Preprocess images
        t1 = preprocess_for_lpips(img1)
        t2 = preprocess_for_lpips(img2)
        
        # Move tensors to same device as model
        device = next(model.parameters()).device
        t1 = t1.to(device)
        t2 = t2.to(device)
        
        # Compute distance
        with torch.no_grad():
            dist = model(t1, t2)
            
        return float(dist.item())
    except Exception as e:
        logger.error(f"Error calculating LPIPS: {e}")
        return None
