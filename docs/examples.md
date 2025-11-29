# Examples & Recipes

## Common Workflows

### 1. Logo/Icon Vectorization (Most Common)

```bash
# CLI - Simple
vectalab premium logo.png

# CLI - Maximum compression
vectalab premium logo.png --precision 1 --mode logo
```

```python
# Python
from vectalab import vectorize_logo_premium

svg_path, metrics = vectorize_logo_premium("logo.png", "logo.svg")
```

**Expected results:**
- File size: 2-10 KB
- Quality: 97-99% SSIM
- Delta E: < 2

---

### 2. Optimize Downloaded SVG

```bash
# Check SVGO is installed
vectalab svgo-info

# Optimize (27-50% smaller)
vectalab optimize icon.svg icon_min.svg
```

```python
from vectalab import optimize_with_svgo

with open("icon.svg") as f:
    svg = f.read()

optimized, metrics = optimize_with_svgo(svg, precision=2)
print(f"Reduced: {metrics['reduction_percent']:.0f}%")
```

---

### 3. Photo Vectorization (Artistic)

```bash
vectalab premium photo.jpg --mode photo --colors 32
```

```python
from vectalab import vectorize_photo_premium

svg_path, metrics = vectorize_photo_premium(
    "photo.jpg", 
    "photo.svg",
    n_colors=32
)
```

---

### 4. Batch Processing

```bash
# Shell - process all PNGs
for f in *.png; do
    vectalab premium "$f" "${f%.png}.svg"
done
```

```python
# Python - parallel processing
from vectalab import vectorize_premium
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def convert(path):
    out = path.with_suffix('.svg')
    _, m = vectorize_premium(str(path), str(out))
    return path.name, m['ssim'], m['file_size']

with ProcessPoolExecutor(max_workers=4) as ex:
    results = list(ex.map(convert, Path(".").glob("*.png")))
```

---

### 5. Quality Verification

```python
from vectalab import (
    vectorize_premium,
    render_svg_to_png,
    compute_enhanced_quality_metrics,
)
import cv2

# Vectorize
_, metrics = vectorize_premium("input.png", "output.svg")

# Render and compare
render_svg_to_png("output.svg", "rendered.png")

orig = cv2.cvtColor(cv2.imread("input.png"), cv2.COLOR_BGR2RGB)
rend = cv2.cvtColor(cv2.imread("rendered.png"), cv2.COLOR_BGR2RGB)

quality = compute_enhanced_quality_metrics(orig, rend)

# Quality report
print(f"""
Quality Report
--------------
RGB SSIM:  {quality['ssim_rgb']*100:.2f}%
LAB SSIM:  {quality['ssim_lab']*100:.2f}%
Delta E:   {quality['delta_e']:.2f}
""")
```

---

## Optimization Tips

### File Size vs Quality

| Precision | Size | Quality | Use Case |
|-----------|------|---------|----------|
| 1 | Smallest | 95%+ | Icons, simple logos |
| 2 | Small | 97%+ | **Default - best balance** |
| 3 | Medium | 98%+ | Complex artwork |
| 4+ | Larger | 99%+ | Archival, print |

```bash
# Smallest file
vectalab premium image.png -p 1

# Best balance (default)
vectalab premium image.png

# Maximum quality
vectalab premium image.png -p 4 --no-svgo
```

---

### When SVGO Makes Things Worse

Rarely, SVGO can break SVGs with:
- Embedded images
- Complex filters
- CSS animations

```bash
# Skip SVGO
vectalab premium image.png --no-svgo
```

---

### Color Palette Control

```bash
# Force specific number of colors
vectalab premium logo.png --colors 4   # Minimal
vectalab premium logo.png --colors 8   # Logo default
vectalab premium logo.png --colors 32  # Photo
vectalab premium logo.png --colors 64  # Maximum detail
```

---

## Integration Examples

### Flask/FastAPI

```python
from fastapi import FastAPI, UploadFile
from vectalab import vectorize_premium
import tempfile

app = FastAPI()

@app.post("/vectorize")
async def vectorize(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_in.write(await file.read())
        tmp_in.flush()
        
        out_path = tmp_in.name.replace(".png", ".svg")
        _, metrics = vectorize_premium(tmp_in.name, out_path)
        
        with open(out_path) as f:
            svg = f.read()
    
    return {"svg": svg, "metrics": metrics}
```

### Django

```python
from django.http import HttpResponse
from vectalab import vectorize_premium
import tempfile

def vectorize_view(request):
    if request.method == 'POST':
        uploaded = request.FILES['image']
        
        with tempfile.NamedTemporaryFile(suffix='.png') as tmp:
            for chunk in uploaded.chunks():
                tmp.write(chunk)
            tmp.flush()
            
            svg_path = tmp.name.replace('.png', '.svg')
            vectorize_premium(tmp.name, svg_path)
            
            with open(svg_path) as f:
                svg = f.read()
        
        return HttpResponse(svg, content_type='image/svg+xml')
```

---

## Troubleshooting

### SVGO Not Found

```bash
# Check status
vectalab svgo-info

# Install
npm install -g svgo

# Verify
svgo --version  # Should show 4.0.0+
```

### Large File Output

```bash
# 1. Use lower precision
vectalab premium image.png -p 1

# 2. Reduce colors
vectalab premium image.png --colors 8

# 3. Ensure SVGO is running
vectalab svgo-info
```

### Low Quality Output

```bash
# 1. Use higher precision
vectalab premium image.png -p 4

# 2. More colors
vectalab premium image.png --colors 64

# 3. Check input quality
vectalab info image.png
```
