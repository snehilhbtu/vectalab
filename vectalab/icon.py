"""
Vectalab Icon Module - Specialized processing for icons.

This module provides specialized vectorization strategies for icons,
particularly monochrome geometric icons on transparent backgrounds.
"""

import os
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
import tempfile
import shutil

# Import premium vectorization
try:
    from vectalab.premium import vectorize_logo_premium
    PREMIUM_AVAILABLE = True
except ImportError:
    PREMIUM_AVAILABLE = False

def is_monochrome_icon(img_path):
    """
    Check if the image is a monochrome icon on transparent background.
    Returns (bool, color_tuple).
    """
    try:
        img = Image.open(img_path).convert('RGBA')
        arr = np.array(img)
        alpha = arr[:, :, 3]
        
        # If mostly opaque (e.g. > 95%), it's likely not a transparent icon
        if np.mean(alpha > 10) > 0.95: 
            return False, None
        
        # Check colors of visible pixels
        visible_mask = alpha > 10
        if not np.any(visible_mask): 
            return False, None
        
        visible = arr[visible_mask]
        
        # Get average color and variance
        avg_color = np.mean(visible[:, :3], axis=0)
        std_color = np.std(visible[:, :3], axis=0)
        
        # Low variance implies monochrome
        if np.max(std_color) > 30: 
            return False, None
            
        return True, tuple(map(int, avg_color))
    except Exception:
        return False, None

def process_geometric_icon(input_path, output_path, color, verbose=False):
    """
    Process a geometric icon using the inversion strategy.
    
    1. Create inverted image (White shape on Black bg)
    2. Run premium vectorization with shape detection
    3. Post-process SVG to restore original color and transparency
    """
    if not PREMIUM_AVAILABLE:
        raise ImportError("vectalab.premium module is required for geometric icon processing")

    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_input = Path(temp_dir) / f"temp_inverted_{input_path.name}.png"
        temp_output = Path(temp_dir) / f"temp_output.svg"
        
        try:
            # 1. Create inverted image
            orig_img = Image.open(input_path).convert('RGBA')
            # Create Black background
            bg = Image.new('RGB', orig_img.size, (0, 0, 0))
            # Paste White shape
            white_shape = Image.new('RGB', orig_img.size, (255, 255, 255))
            bg.paste(white_shape, mask=orig_img.split()[3])
            bg.save(temp_input)
            
            # 2. Run premium vectorization
            # We use vectorize_logo_premium with detect_shapes=True
            vectorize_logo_premium(
                str(temp_input),
                str(temp_output),
                use_svgo=True, # Enable SVGO for cleaner output
                detect_shapes=True,
                verbose=verbose
            )
            
            # 3. Post-process SVG
            if temp_output.exists():
                tree = ET.parse(temp_output)
                root = tree.getroot()
                
                # Handle namespaces
                # Register namespace to avoid ns0: prefixes if possible, 
                # but ET handling of namespaces is tricky.
                # We'll just iterate and check tags ending with 'path'
                
                hex_color = '#{:02x}{:02x}{:02x}'.format(*color)
                
                # We need to collect elements to remove
                # Since we can't easily remove from iterator, we'll build a list of (parent, child)
                # But ET doesn't give parent.
                # We'll iterate over children of root (usually paths are direct children in vectalab output)
                # If they are in groups, we might need recursive search, but let's assume flat for now
                # or just handle direct children as per benchmark implementation.
                
                # Actually, let's do a recursive search for paths to modify/remove
                # But removing requires access to parent.
                
                # Strategy: 
                # 1. Identify elements to keep (white paths) and modify their color.
                # 2. Identify elements to remove (black paths).
                # 3. Rebuild the tree or remove carefully.
                
                # Recursive function to handle groups
                def process_node(node):
                    # Iterate over a copy of children to allow modification
                    for child in list(node):
                        tag = child.tag.split('}')[-1]
                        if tag == 'path':
                            fill = child.get('fill')
                            if fill is None:
                                fill = '#000000'
                            fill = fill.lower()
                            
                            # Check for black (background)
                            if fill == '#000000' or fill == '#000':
                                node.remove(child)
                            # Check for white (shape)
                            elif fill == '#ffffff' or fill == '#fff':
                                child.set('fill', hex_color)
                        elif tag == 'g':
                            process_node(child)
                            # Remove empty groups
                            if len(list(child)) == 0:
                                node.remove(child)

                process_node(root)
                
                # Save result
                tree.write(output_path, encoding='utf-8', xml_declaration=True)
                return True, {"method": "geometric_icon", "original_color": hex_color}
            else:
                return False, {"error": "Vectorization failed"}
                
        except Exception as e:
            if verbose:
                print(f"Error in process_geometric_icon: {e}")
            return False, {"error": str(e)}
