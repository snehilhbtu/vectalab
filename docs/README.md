# Vectalab Documentation — concise index

Professional, high‑fidelity raster → SVG vectorization. This docs index is intentionally compact and focused: quick start, concise CLI/API reference, architecture overview, and reproducible benchmarks.

## Quick start (one-minute)

Install and run a best-effort vectorization with defaults:

```bash
pip install vectalab
# raster → compact, high-quality SVG (recommended for photos & complex images)
vectalab convert image.png

# recommended fast path for logos/icons (palette reduction + SVGO)
vectalab premium logo.png

# compress an existing SVG
vectalab optimize file.svg
```

## Documentation Index

| Document | Short summary |
|---|---:|
| [CLI Reference](cli.md) | Concise, accurate commands & examples — pick a command by use-case |
| [Python API](api.md) | Stable programmatic entrypoints and quick recipes |
| [Examples & Recipes](examples.md) | Copy‑pasteable, targeted workflows for users and integrators |
| [Architecture](algorithm.md) | Short technical overview + diagram (how pieces fit together) |
| [Benchmarks & Protocol](benchmarks.md) | Reproducible benchmarking (uses scripts/benchmark_runner.py) |
| [Cloud (Modal)](modal_setup.md) | How to enable remote SAM execution (if needed) |

## Which command should I run? (short)

- You have a raster image (PNG/JPG): use `convert` for general cases; use `premium` for highest quality + SVGO optimizations.
- You already have an SVG and want to shrink it: use `optimize`.
- Want multiple strategies and automatic selection? use `auto` or `smart`.

Suggested quick mapping:

- raster → vectalab convert (method=hifi by default) or vectalab premium (SOTA + SVGO)
- svg → vectalab optimize
- logos/icons → vectalab logo or vectalab premium --mode logo

For the full CLI reference and targeted examples see [CLI Reference](cli.md) and [Examples & Recipes](examples.md).

## Pragmatic expectations

These are typical, not guaranteed. Real results depend on input complexity and chosen presets.

- SSIM quality (RGB): high-quality presets aim for ≥ 0.99; premium defaults target 0.98.
- Delta‑E (color): usually under 2 for high-quality presets (perceptually near-exact colors).
- Typical file-size reductions: 30–80% depending on input and SVGO use.

If you need reproducible evaluation, use the benchmark scripts (see [Benchmarks & Protocol](benchmarks.md)).
