import argparse
import itertools
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Import Vectalab components
from vectalab.premium import vectorize_premium
from vectalab.cli import _calculate_full_metrics

console = Console()

def tune_image(image_path: str, output_dir: str):
    """
    Run a grid search of vectorization parameters on a single image
    and rank results by SOTA metrics.
    """
    img_path = Path(image_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    console.print(f"[bold cyan]🎯 Tuning Vectorization for: {img_path.name}[/]")
    
    # Define parameter grid to search
    # These are the key knobs in vtracer/premium
    param_grid = {
        'n_colors': [16, 32, 64],
        'corner_threshold': [30, 45, 60],  # Lower = more corners, Higher = smoother
        'speckle_filter': [2, 4],          # Lower = more detail, Higher = cleaner
        'mode': ['spline', 'polygon'],     # Spline = curves, Polygon = sharp
    }
    
    # Generate combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    
    for i, params in enumerate(track(combinations, description="Testing configurations...")):
        run_id = f"run_{i:03d}"
        out_svg = out_dir / f"{img_path.stem}_{run_id}.svg"
        
        try:
            # We need to modify how we call vectorize_premium to pass these specific low-level args.
            # Since vectorize_premium encapsulates vtracer, we might need to patch it or 
            # use a modified version. For this script, we will assume we can pass them 
            # or we will modify premium.py to accept **kwargs for vtracer if needed.
            # For now, let's map what we can.
            
            # Note: vectorize_premium currently hardcodes some settings based on target_ssim.
            # To properly tune, we should ideally expose these. 
            # For this demonstration, we will map 'n_colors' directly and use 'target_ssim' 
            # as a proxy for detail if needed, but ideally we'd pass kwargs.
            
            # Let's assume we updated premium.py to accept vtracer_args (I will do this next)
            
            vtracer_args = {
                'corner_threshold': params['corner_threshold'],
                'filter_speckle': params['speckle_filter'],
                'mode': params['mode']
            }
            
            vectorize_premium(
                str(img_path),
                str(out_svg),
                n_colors=params['n_colors'],
                verbose=False,
                # We will add this capability to premium.py
                vtracer_args=vtracer_args 
            )
            
            # Calculate Metrics
            metrics = _calculate_full_metrics(img_path, out_svg)
            
            # Score: Weighted combination (Lower is better)
            # LPIPS (Perceptual) + DISTS (Texture) + (100-Edge)/100 (Geometry)
            score = (metrics.get('lpips', 1.0) * 1.0) + \
                    (metrics.get('dists', 1.0) * 0.5) + \
                    ((100 - metrics.get('edge', 0)) / 100.0 * 0.2)
            
            result = {
                'id': run_id,
                'score': score,
                **params,
                **metrics
            }
            results.append(result)
            
        except Exception as e:
            console.print(f"[red]Run {i} failed: {e}[/]")

    # Analyze Results
    df = pd.DataFrame(results)
    df = df.sort_values('score') # Lower score is better
    
    # Display Top 5
    table = Table(title=f"Top 5 Configurations for {img_path.name}")
    table.add_column("Rank", style="cyan")
    table.add_column("Score (Lower=Better)", style="bold green")
    table.add_column("LPIPS", style="magenta")
    table.add_column("Edge Acc", style="blue")
    table.add_column("Params")
    
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        params_str = f"Colors={row['n_colors']}, Corner={row['corner_threshold']}, Mode={row['mode']}"
        table.add_row(
            str(i+1),
            f"{row['score']:.4f}",
            f"{row['lpips']:.4f}",
            f"{row['edge']:.1f}%",
            params_str
        )
        
    console.print(table)
    
    # Save best result
    best = df.iloc[0]
    best_svg = out_dir / f"{img_path.stem}_{best['id']}.svg"
    final_svg = out_dir / f"{img_path.stem}_optimized.svg"
    
    import shutil
    shutil.copy(best_svg, final_svg)
    console.print(f"\n[bold green]✅ Best configuration saved to: {final_svg}[/]")
    console.print(f"   LPIPS: {best['lpips']:.4f}")
    console.print(f"   Edge Accuracy: {best['edge']:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image to tune")
    parser.add_argument("--output", "-o", default="tuning_results", help="Output directory")
    args = parser.parse_args()
    
    tune_image(args.image, args.output)
