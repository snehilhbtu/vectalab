#!/usr/bin/env python3
"""
Vectalab 80/20 Optimization Benchmark Script.

This script measures the impact of 80/20 optimizations:
1. SVGO post-processing (target: 30-50% size reduction)
2. Shape primitive detection (cleaner SVGs)
3. LAB color space metrics (perceptually accurate)
4. Coordinate precision control (10-15% size reduction)

Usage:
    python scripts/benchmark_80_20.py <input_image>
    python scripts/benchmark_80_20.py examples/test_logo.png
"""

import sys
import os
import tempfile
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


def run_benchmark(input_path: str):
    """Run comprehensive benchmark comparing before/after optimizations."""
    
    console.print(Panel.fit(
        "[bold cyan]Vectalab 80/20 Optimization Benchmark[/]",
        border_style="cyan"
    ))
    console.print()
    
    # Check if input exists
    if not os.path.exists(input_path):
        console.print(f"[red]Error: Input file not found: {input_path}[/]")
        return
    
    console.print(f"[dim]Input: {input_path}[/]")
    console.print()
    
    # Import modules
    try:
        from vectalab.premium import vectorize_premium
        from vectalab.optimizations import (
            check_svgo_available,
            compute_enhanced_quality_metrics,
            apply_all_optimizations,
        )
        import cairosvg
        from PIL import Image
        from io import BytesIO
        from skimage.metrics import structural_similarity as ssim
    except ImportError as e:
        console.print(f"[red]Error: Missing dependency: {e}[/]")
        return
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        console.print(f"[red]Error: Could not load image[/]")
        return
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    
    console.print(f"[dim]Image size: {w}x{h}[/]")
    
    # Check SVGO availability
    svgo_available = check_svgo_available()
    console.print(f"SVGO available: {'[green]✓[/]' if svgo_available else '[yellow]✗ (install Node.js for 30-50% more reduction)[/]'}")
    console.print()
    
    results = []
    
    # ==== Test 1: Baseline (no 80/20 optimizations) ====
    console.print("[bold]1. Running BASELINE (high precision, no SVGO)...[/]")
    
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
        baseline_path = f.name
    
    try:
        _, baseline_metrics = vectorize_premium(
            input_path,
            baseline_path,
            use_svgo=False,
            precision=8,  # High precision
            detect_shapes=False,
            use_lab_metrics=False,
            verbose=False,
        )
        
        results.append({
            'name': 'Baseline',
            'file_size': baseline_metrics['file_size'],
            'ssim_rgb': baseline_metrics['ssim'],
            'paths': baseline_metrics['path_count'],
        })
        console.print(f"   Size: {baseline_metrics['file_size']:,} bytes, SSIM: {baseline_metrics['ssim']*100:.2f}%")
        
    except Exception as e:
        console.print(f"   [red]Error: {e}[/]")
        baseline_metrics = {'file_size': 0, 'ssim': 0}
    
    # ==== Test 2: ALL 80/20 Optimizations ====
    console.print("\n[bold]2. Running ALL 80/20 OPTIMIZATIONS...[/]")
    
    with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as f:
        full_path = f.name
    
    try:
        _, full_metrics = vectorize_premium(
            input_path,
            full_path,
            use_svgo=True,
            precision=2,
            detect_shapes=True,
            use_lab_metrics=True,
            verbose=False,
        )
        
        size_reduction = (1 - full_metrics['file_size'] / baseline_metrics['file_size']) * 100 if baseline_metrics['file_size'] > 0 else 0
        
        results.append({
            'name': '80/20 Optimized',
            'file_size': full_metrics['file_size'],
            'ssim_rgb': full_metrics['ssim'],
            'ssim_lab': full_metrics.get('ssim_lab', 0),
            'delta_e': full_metrics.get('delta_e', 0),
            'paths': full_metrics['path_count'],
            'size_reduction': size_reduction,
            'shapes': full_metrics.get('optimizations', {}).get('shapes', {}),
        })
        console.print(f"   Size: {full_metrics['file_size']:,} bytes ({size_reduction:.1f}% reduction)")
        console.print(f"   SSIM RGB: {full_metrics['ssim']*100:.2f}%, SSIM LAB: {full_metrics.get('ssim_lab', 0)*100:.2f}%")
        console.print(f"   Delta E: {full_metrics.get('delta_e', 0):.2f}")
        
        # Shape detection results
        shapes = full_metrics.get('optimizations', {}).get('shapes', {})
        if shapes:
            circles = shapes.get('circles_detected', 0)
            rects = shapes.get('rectangles_detected', 0)
            ellipses = shapes.get('ellipses_detected', 0)
            if circles or rects or ellipses:
                console.print(f"   Shapes: {circles} circles, {rects} rectangles, {ellipses} ellipses")
        
    except Exception as e:
        console.print(f"   [red]Error: {e}[/]")
        import traceback
        traceback.print_exc()
    
    # ==== Results Summary ====
    console.print("\n")
    
    # Create comparison table
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan", title="📊 RESULTS")
    table.add_column("Configuration", style="bold")
    table.add_column("File Size", justify="right")
    table.add_column("Reduction", justify="right")
    table.add_column("SSIM", justify="right")
    table.add_column("Paths", justify="right")
    
    for r in results:
        size_kb = r['file_size'] / 1024
        reduction = r.get('size_reduction', 0)
        reduction_str = f"[green]-{reduction:.1f}%[/]" if reduction > 0 else "-"
        ssim_rgb = f"{r['ssim_rgb']*100:.2f}%"
        
        table.add_row(
            r['name'],
            f"{size_kb:.1f} KB",
            reduction_str,
            ssim_rgb,
            str(r['paths']),
        )
    
    console.print(table)
    
    # Print impact summary
    if len(results) >= 2:
        baseline_size = results[0]['file_size']
        final_size = results[-1]['file_size']
        total_reduction = (1 - final_size / baseline_size) * 100 if baseline_size > 0 else 0
        
        console.print()
        console.print(Panel(
            f"[bold green]✨ Total Size Reduction: {total_reduction:.1f}%[/]\n"
            f"[dim]{baseline_size:,} bytes → {final_size:,} bytes[/]",
            title="80/20 Impact",
            border_style="green"
        ))
    
    # Cleanup temp files
    for path in [baseline_path, full_path]:
        try:
            os.remove(path)
        except:
            pass
    
    return results


def main():
    if len(sys.argv) < 2:
        console.print("[bold]Vectalab 80/20 Optimization Benchmark[/]")
        console.print()
        console.print("Usage: python scripts/benchmark_80_20.py <input_image>")
        console.print()
        console.print("Example:")
        console.print("  python scripts/benchmark_80_20.py examples/test_logo.png")
        console.print("  python scripts/benchmark_80_20.py ~/Desktop/my_logo.jpg")
        sys.exit(1)
    
    input_path = sys.argv[1]
    run_benchmark(input_path)


if __name__ == "__main__":
    main()
