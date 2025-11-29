#!/usr/bin/env python3
"""
Download SAM model checkpoint (ViT-B).
"""

import os
import requests
import sys

def download_sam_vit_b():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    filename = "sam_vit_b.pth"
    
    if os.path.exists(filename):
        print(f"{filename} already exists.")
        # Check size
        size = os.path.getsize(filename)
        print(f"Size: {size / 1024 / 1024:.2f} MB")
        if size > 300 * 1024 * 1024: # Should be ~375MB
            print("Size looks correct.")
            return
        else:
            print("Size looks too small. Re-downloading...")
    
    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB
        wrote = 0
        
        with open(filename, "wb") as f:
            for data in response.iter_content(block_size):
                wrote += len(data)
                f.write(data)
                sys.stdout.write(f"\rDownloaded {wrote / 1024 / 1024:.2f} MB")
                sys.stdout.flush()
        
        print("\nDownload complete.")
        
    except Exception as e:
        print(f"\nError downloading: {e}")

if __name__ == "__main__":
    download_sam_vit_b()
