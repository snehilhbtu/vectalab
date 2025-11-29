#!/usr/bin/env python3
"""
Download test SVG icons for the test protocol.
"""

import os
import requests

# Feather icons (monochrome)
FEATHER_ICONS = [
    'circle', 'square', 'triangle', 'star', 'heart',
    'user', 'home', 'search', 'settings', 'camera'
]

# Multi-color icons from gilbarbara/logos
MULTI_COLOR_ICONS = {
    'github': 'github-icon',
    'twitter': 'twitter',
    'facebook': 'facebook',
    'instagram': 'instagram-icon',
    'youtube': 'youtube-icon',
    'linkedin': 'linkedin-icon',
    'google': 'google-icon',
    'apple': 'apple',
    'microsoft': 'microsoft-icon',
    'amazon': 'amazon-icon'
}

def download_svg(url, output_path):
    """Download SVG from URL to output path."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_path, 'w') as f:
            f.write(response.text)
        print(f"Downloaded: {output_path}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    # Download Feather icons
    for icon in FEATHER_ICONS:
        url = f"https://raw.githubusercontent.com/feathericons/feather/master/icons/{icon}.svg"
        output = f"test_data/svg_mono/{icon}.svg"
        download_svg(url, output)

    # Download Multi-color icons
    for name, filename in MULTI_COLOR_ICONS.items():
        url = f"https://raw.githubusercontent.com/gilbarbara/logos/master/logos/{filename}.svg"
        output = f"test_data/svg_multi/{name}.svg"
        download_svg(url, output)

if __name__ == "__main__":
    main()