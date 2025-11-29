# Python API Reference

## Quick Reference

```python
# Most common usage
from vectalab import vectorize_premium
svg_path, metrics = vectorize_premium("input.png", "output.svg")

# Optimize existing SVG
from vectalab import optimize_with_svgo
optimized_svg, metrics = optimize_with_svgo(svg_content)
```

---

## Premium Vectorization (Recommended)

### `vectorize_premium`

State-of-the-art vectorization with all optimizations.

```python
from vectalab import vectorize_premium

svg_path, metrics = vectorize_premium(
    input_path: str,
    output_path: str,
    target_ssim: float = 0.98,
    max_iterations: int = 5,
    n_colors: int = None,        # Auto-detect if None
    use_svgo: bool = True,
    precision: int = 2,
    detect_shapes: bool = False,
    use_lab_metrics: bool = True,
    verbose: bool = False,
)
```

**Returns:** `Tuple[str, Dict]`
- `str`: Output file path
- `Dict`: Metrics with keys:
  - `ssim`: RGB SSIM (0-1)
  - `ssim_lab`: LAB SSIM (0-1)
  - `delta_e`: Color accuracy (lower is better)
  - `file_size`: Output size in bytes
  - `path_count`: Number of SVG paths
  - `optimizations`: Applied optimizations

**Example:**
```python
svg_path, metrics = vectorize_premium("logo.png", "logo.svg")
print(f"Quality: {metrics['ssim']*100:.1f}%")
print(f"Size: {metrics['file_size']/1024:.1f} KB")
print(f"Color accuracy: ΔE={metrics['delta_e']:.2f}")
```

### `vectorize_logo_premium`

Optimized for logos and icons.

```python
from vectalab import vectorize_logo_premium

svg_path, metrics = vectorize_logo_premium(
    input_path: str,
    output_path: str,
    use_svgo: bool = True,
    precision: int = 2,
    detect_shapes: bool = True,
    verbose: bool = False,
)
```

### `vectorize_photo_premium`

Optimized for photographs.

```python
from vectalab import vectorize_photo_premium

svg_path, metrics = vectorize_photo_premium(
    input_path: str,
    output_path: str,
    n_colors: int = 32,
    use_svgo: bool = True,
    precision: int = 2,
    verbose: bool = False,
)
```

---

## SVGO Optimization

### `optimize_with_svgo`

Optimize SVG content using SVGO.

```python
from vectalab import optimize_with_svgo

optimized_svg, metrics = optimize_with_svgo(
    svg_content: str,
    precision: int = 2,
    multipass: bool = True,
)
```

**Returns:** `Tuple[str, Dict]`
- `str`: Optimized SVG content
- `Dict`: Metrics with keys:
  - `svgo_applied`: bool
  - `original_size`: int (bytes)
  - `optimized_size`: int (bytes)
  - `reduction_percent`: float

**Example:**
```python
with open("input.svg") as f:
    svg_content = f.read()

optimized, metrics = optimize_with_svgo(svg_content, precision=2)
print(f"Reduced by {metrics['reduction_percent']:.1f}%")

with open("output.svg", "w") as f:
    f.write(optimized)
```

### `check_svgo_available`

Check if SVGO is installed.

```python
from vectalab import check_svgo_available

if check_svgo_available():
    print("SVGO ready")
else:
    print("Install: npm install -g svgo")
```

---

## Quality Metrics

### `compute_enhanced_quality_metrics`

Compute comprehensive quality metrics.

```python
from vectalab import compute_enhanced_quality_metrics

metrics = compute_enhanced_quality_metrics(
    original: np.ndarray,   # Original image (H, W, 3)
    rendered: np.ndarray,   # Rendered SVG (H, W, 3)
)
```

**Returns:** `Dict[str, float]`
- `ssim_rgb`: Standard SSIM
- `ssim_lab`: LAB color space SSIM
- `delta_e`: Average Delta E
- `quality_score`: Combined score (0-1)

### `compute_lab_ssim`

SSIM in perceptually uniform LAB color space.

```python
from vectalab import compute_lab_ssim

ssim = compute_lab_ssim(original, rendered)
```

### `compute_delta_e`

Average color difference (CIE76).

```python
from vectalab import compute_delta_e

delta_e = compute_delta_e(original, rendered)
# < 1: Imperceptible
# 1-2: Barely perceptible
# 2-10: Noticeable
# > 10: Different colors
```

---

## Shape Detection

### `detect_circles`

```python
from vectalab import detect_circles

circles = detect_circles(image, min_radius=5, max_radius=200)
# Returns: [{'cx': x, 'cy': y, 'r': radius, 'color': (r,g,b)}, ...]
```

### `detect_rectangles`

```python
from vectalab import detect_rectangles

rects = detect_rectangles(image, min_area=100)
# Returns: [{'x': x, 'y': y, 'width': w, 'height': h, 'color': (r,g,b)}, ...]
```

### `detect_ellipses`

```python
from vectalab import detect_ellipses

ellipses = detect_ellipses(image, min_area=100)
# Returns: [{'cx': x, 'cy': y, 'rx': rx, 'ry': ry, 'angle': deg, 'color': (r,g,b)}, ...]
```

---

## High-Fidelity Vectorization

### `vectorize_high_fidelity`

Iterative refinement for maximum quality.

```python
from vectalab import vectorize_high_fidelity

svg_path, ssim = vectorize_high_fidelity(
    input_path: str,
    output_path: str,
    target_ssim: float = 0.998,
    quality: str = "ultra",      # figma, balanced, quality, ultra
    max_iterations: int = 5,
    verbose: bool = True,
)
```

### `render_svg_to_png`

Render SVG for quality comparison.

```python
from vectalab import render_svg_to_png

png_path = render_svg_to_png(
    svg_path: str,
    png_path: str,
    scale: int = 1,
)
```

---

## Complete Pipeline Example

```python
from vectalab import (
    vectorize_premium,
    render_svg_to_png,
    compute_enhanced_quality_metrics,
)
import cv2

# 1. Vectorize
svg_path, metrics = vectorize_premium(
    "input.png",
    "output.svg",
    use_svgo=True,
    precision=2,
)

# 2. Render back for verification
render_svg_to_png(svg_path, "rendered.png")

# 3. Compute detailed metrics
original = cv2.cvtColor(cv2.imread("input.png"), cv2.COLOR_BGR2RGB)
rendered = cv2.cvtColor(cv2.imread("rendered.png"), cv2.COLOR_BGR2RGB)

quality = compute_enhanced_quality_metrics(original, rendered)

print(f"RGB SSIM:  {quality['ssim_rgb']*100:.2f}%")
print(f"LAB SSIM:  {quality['ssim_lab']*100:.2f}%")
print(f"Delta E:   {quality['delta_e']:.2f}")
```

---

## Batch Processing

```python
from vectalab import vectorize_premium
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process_image(input_path):
    output_path = input_path.with_suffix('.svg')
    try:
        _, metrics = vectorize_premium(str(input_path), str(output_path))
        return input_path.name, metrics['ssim'], metrics['file_size']
    except Exception as e:
        return input_path.name, None, str(e)

# Process all PNGs
images = list(Path("images").glob("*.png"))

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_image, images))

for name, ssim, size in results:
    if ssim:
        print(f"{name}: {ssim*100:.1f}% quality, {size/1024:.1f} KB")
    else:
        print(f"{name}: FAILED - {size}")
```
