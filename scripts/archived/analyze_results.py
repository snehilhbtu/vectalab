"""
ARCHIVED: small helper to analyze JSON results

Preserved here for reference; this helped quickly inspect JSON results created by old benchmarking scripts.
"""

import json
import sys
from pathlib import Path

def analyze_results(json_path):
    with open(json_path, 'r') as f:
        results = json.load(f)
    
    print(f"Analyzing {len(results)} results from {json_path}")
    
    # Sort by Edge Accuracy (ascending)
    sorted_by_edge = sorted(results, key=lambda x: x['edge'])
    
    print("\n📉 Worst 5 by Edge Accuracy:")
    for r in sorted_by_edge[:5]:
        print(f"  • {r['icon']}: {r['edge']:.1f}% (SSIM: {r['ssim']:.1f}%, Topo: {r['topology']:.1f}%)")

    # Sort by Topology (ascending)
    sorted_by_topo = sorted(results, key=lambda x: x['topology'])
    
    print("\n📉 Worst 5 by Topology:")
    for r in sorted_by_topo[:5]:
        print(f"  • {r['icon']}: {r['topology']:.1f}% (Edge: {r['edge']:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_results.py <path_to_results.json>")
        sys.exit(1)
    analyze_results(sys.argv[1])
