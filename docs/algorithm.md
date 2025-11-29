# Vectalab Algorithm

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        VECTALAB PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT                    CORE                      OPTIMIZATION        │
│  ─────                    ────                      ────────────        │
│                                                                         │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────────────────┐    │
│  │  PNG/   │───▶│  Image Analysis  │───▶│  80/20 Optimizations    │    │
│  │  JPG    │    │  (Type Detection)│    │  • SVGO (30-50%)        │    │
│  └─────────┘    └──────────────────┘    │  • Precision (10-15%)   │    │
│                         │               │  • Color snapping       │    │
│                         ▼               └─────────────────────────┘    │
│                 ┌──────────────────┐              │                    │
│                 │  Preprocessing   │              ▼                    │
│                 │  • Edge-aware    │    ┌─────────────────────────┐    │
│                 │  • Color quant   │    │      OUTPUT: SVG        │    │
│                 └──────────────────┘    │  • 70-80% smaller       │    │
│                         │               │  • 97%+ quality         │    │
│                         ▼               └─────────────────────────┘    │
│                 ┌──────────────────┐                                   │
│                 │    vtracer       │                                   │
│                 │  (Rust engine)   │                                   │
│                 └──────────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Processing Stages

### 1. Image Analysis

Automatic detection of image type to select optimal parameters.

```python
def analyze_image(image):
    """
    Analyze image characteristics.
    
    Features computed:
    - Color count (unique colors)
    - Edge density (gradient magnitude)
    - Color entropy (distribution)
    - Spatial frequency (detail level)
    """
    
    # Color analysis
    unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
    
    # Edge analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.mean(edges > 0)
    
    # Classification
    if unique_colors < 32 and edge_density < 0.1:
        return "logo"
    elif edge_density > 0.3:
        return "photo"
    else:
        return "artwork"
```

**Image Type → Parameter Mapping:**

| Type | Colors | Precision | SVGO |
|------|--------|-----------|------|
| Logo | 4-16 | 2 | Yes |
| Icon | 8-24 | 2 | Yes |
| Artwork | 16-48 | 3 | Yes |
| Photo | 32-64 | 3 | Yes |

---

### 2. Preprocessing

Edge-preserving color reduction.

```python
def preprocess_for_vectorization(image, n_colors):
    """
    1. Edge-aware denoising (preserve sharp boundaries)
    2. Color quantization (reduce palette)
    3. Color snapping (pure black/white)
    """
    
    # Bilateral filter: smooth colors, keep edges
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    
    # K-means color quantization
    pixels = denoised.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # Snap near-black/white to pure values
    palette = snap_colors(palette)
    
    return quantized_image, palette
```

**Color Snapping Rules:**
- RGB < (10, 10, 10) → Pure black (0, 0, 0)
- RGB > (245, 245, 245) → Pure white (255, 255, 255)
- Near-grayscale → True grayscale

---

### 3. Vectorization Core (vtracer)

Rust-based tracing engine for path generation.

```python
# vtracer parameters (optimized defaults)
settings = {
    'colormode': 'color',           # Full color
    'hierarchical': 'stacked',      # Layer ordering
    'mode': 'spline',               # Smooth curves
    'filter_speckle': 4,            # Remove noise
    'color_precision': 6,           # Color accuracy
    'layer_difference': 16,         # Color separation
    'corner_threshold': 60,         # Corner detection
    'length_threshold': 4.0,        # Minimum path length
    'max_iterations': 10,           # Optimization iterations
    'splice_threshold': 45,         # Path merging
    'path_precision': 3,            # Coordinate precision
}
```

**Key Parameters:**

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `filter_speckle` | Remove small artifacts | Higher = cleaner, may lose detail |
| `corner_threshold` | Detect corners | Higher = more curves, smoother |
| `path_precision` | Decimal places | Higher = larger file, more accurate |
| `layer_difference` | Color separation | Higher = fewer paths, less detail |

---

### 4. 80/20 Optimizations

High-impact, low-effort improvements.

#### SVGO (30-50% reduction)

```bash
svgo -p 2 --multipass -i input.svg -o output.svg
```

**What SVGO does:**
- Remove comments and metadata
- Collapse whitespace
- Shorten color values (#ffffff → #fff)
- Optimize path data
- Remove empty groups

#### Coordinate Precision (10-15% reduction)

```python
def reduce_precision(svg, decimals=2):
    """
    123.456789 → 123.46
    
    Most displays: 96 DPI = 0.26mm per pixel
    2 decimal places = 0.0026mm precision
    Far beyond human perception.
    """
    pattern = r'-?\d+\.\d+'
    
    def round_num(m):
        n = float(m.group(0))
        return f"{round(n, decimals):.{decimals}f}".rstrip('0').rstrip('.')
    
    return re.sub(pattern, round_num, svg)
```

---

## Quality Metrics

### SSIM (Structural Similarity)

```
SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ

Where:
- l = luminance comparison
- c = contrast comparison  
- s = structure comparison
```

**Interpretation:**
- 1.0 = Identical
- 0.99+ = Imperceptible difference
- 0.95+ = Excellent quality
- 0.90+ = Good quality
- < 0.90 = Noticeable degradation

### LAB SSIM (Perceptually Uniform)

Standard SSIM uses RGB which doesn't match human perception. LAB color space provides perceptually uniform distances.

```python
def compute_lab_ssim(original, rendered):
    # Convert to LAB (perceptually uniform)
    orig_lab = rgb2lab(original)
    rend_lab = rgb2lab(rendered)
    
    # Weighted SSIM (luminance most important)
    ssim_L = ssim(orig_lab[:,:,0], rend_lab[:,:,0])
    ssim_a = ssim(orig_lab[:,:,1], rend_lab[:,:,1])
    ssim_b = ssim(orig_lab[:,:,2], rend_lab[:,:,2])
    
    return 0.5 * ssim_L + 0.25 * ssim_a + 0.25 * ssim_b
```

### Delta E (Color Accuracy)

CIE76 color difference formula:

```
ΔE = √[(L₁-L₂)² + (a₁-a₂)² + (b₁-b₂)²]
```

**Interpretation:**
| ΔE | Perception |
|----|------------|
| < 1 | Imperceptible |
| 1-2 | Barely perceptible |
| 2-10 | Noticeable |
| 11-49 | Different colors |
| 100 | Opposite colors |

---

## Shape Detection

Computer vision techniques to identify geometric primitives.

### Circle Detection (Hough Transform)

```python
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,                    # Resolution ratio
    minDist=20,              # Minimum distance between centers
    param1=50,               # Edge detection threshold
    param2=30,               # Accumulator threshold
    minRadius=5,
    maxRadius=200
)
```

### Rectangle Detection (Contour Approximation)

```python
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:  # 4 vertices = rectangle
        x, y, w, h = cv2.boundingRect(approx)
```

---

## Performance Characteristics

### Benchmarks

| Image Size | Processing Time | Memory |
|------------|-----------------|--------|
| 100×100 | ~50ms | ~10MB |
| 500×500 | ~200ms | ~50MB |
| 1000×1000 | ~500ms | ~150MB |
| 2000×2000 | ~2s | ~500MB |

### Bottlenecks

| Stage | Time % | Optimization |
|-------|--------|--------------|
| Import (cairosvg) | 60% | One-time cost |
| vtracer | 20% | Rust, already optimized |
| SVGO | 15% | Node.js subprocess |
| Quality metrics | 5% | NumPy vectorized |

---

## Comparison with Other Tools

| Feature | Vectalab | Potrace | Vector Magic |
|---------|----------|---------|--------------|
| Color support | Full RGB | B/W only | Full RGB |
| Anti-aliasing | Preserved | Lost | Preserved |
| File size | Optimized | Small | Medium |
| Quality (SSIM) | 97-99% | 85-95% | 98-99% |
| Speed | Fast | Fastest | Slow |
| Price | Free/OSS | Free | $295 |
