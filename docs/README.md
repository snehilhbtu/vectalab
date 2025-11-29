# Vectalab Documentation

> **Professional High-Fidelity Image Vectorization**  
> Convert raster images (PNG, JPG) to optimized SVG with 97%+ quality

## Quick Start

```bash
# Install
pip install vectalab

# Install SVGO for best compression (optional but recommended)
npm install -g svgo

# Vectorize an image
vectalab premium logo.png

# Optimize existing SVG
vectalab optimize icon.svg
```

## Documentation Index

| Document | Description |
|----------|-------------|
| [CLI Reference](cli.md) | Complete command-line interface guide |
| [Python API](api.md) | Programmatic usage with Python |
| [Examples](examples.md) | Common workflows and recipes |
| [Algorithm](algorithm.md) | Technical deep-dive into vectorization |
| [Cloud Setup](modal_setup.md) | Modal.com integration for cloud acceleration |

## Which Command Should I Use?

```
┌─────────────────────────────────────────────────────────────────┐
│                    VECTALAB DECISION TREE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Do you have a raster image (PNG/JPG)?                         │
│  ├─ YES → Use 'premium' (best quality + smallest files)        │
│  │        $ vectalab premium image.png                         │
│  │                                                             │
│  └─ NO → Do you have an SVG to compress?                       │
│          ├─ YES → Use 'optimize' (SVGO compression)            │
│          │        $ vectalab optimize file.svg                 │
│          │                                                     │
│          └─ NO → Use 'info' to analyze your file               │
│                  $ vectalab info file.png                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Results

| Metric | Typical Value |
|--------|---------------|
| File size reduction | 70-80% vs baseline |
| SSIM quality | 97-99% |
| Delta E (color accuracy) | < 1 (imperceptible) |
| Processing time | 0.2-2s per image |
