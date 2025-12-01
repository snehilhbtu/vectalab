# CLI Reference — quick, accurate

This page is a concise reference for the commands that matter for most users. The full, in-code help (CLI flags/descriptions) remains authoritative — run `vectalab <command> --help` to see exact options.

Core commands:

- convert — general-purpose raster→SVG converter (default method: hifi). Good for balanced quality and speed.
- premium — SOTA quality with 80/20 optimizations (SVGO, precision tuning, shape detection). Best for production-grade results.
- logo — focused palette reduction and simplification for logos/icons.
- optimize — compress an existing SVG using SVGO and precision reduction.
- smart / auto — multi-strategy runners: `smart` targets a size/quality, `auto` runs multiple strategies and picks the best.

Quick decision map (one-line):

• I have a PNG/JPG → `convert` (fast) or `premium` (highest quality + SVGO)

• I have an SVG → `optimize`

• I need to batch/auto-select the best result → `smart` or `auto`

---

## Example usage (most common)

vectalab convert image.png                   # general conversion (default hifi)
vectalab premium image.png                   # highest-quality production path (SVGO, precision)
vectalab logo icon.png -c 8                  # logo-optimized (forces palette size)
vectalab optimize icon.svg -p 1              # aggressively reduce precision and size
vectalab smart batch.png -s 50               # iterate to reach a target size

All commands are self-documenting; run `vectalab <command> --help` to see exact flags and defaults.

Notes/flags to remember

- --target-ssim / -t: controls the quality goal (0.90–1.0). For `convert` the CLI default target is 0.998; `premium` defaults to 0.98.
- --precision / -p: coordinate decimals (1–8). Lower reduces file size; 2 is a good default balance.
- --svgo/--no-svgo: premium + optimize rely on SVGO for major size savings — requires Node.js + svgo package.
- --colors / -c: force palette size for logo/photo flows.

Developer note: the CLI is implemented in `vectalab/cli.py` — keep this file as the authoritative reference for available options and behavior.

Troubleshooting quick tips

- SVGO not found: run `vectalab svgo-info` to check environment and install instructions.
- If conversion fails due to missing libs, check that `vtracer`, `cairosvg`, and `scikit-image` are installed.

### Output Metrics

```text
Quality (SSIM RGB):  97.65% ✅
Quality (SSIM LAB):  97.86%
Color Accuracy (ΔE): 0.93 (Imperceptible)
File Size:           2.5 KB
Size Reduction:      77.6%
```

- **Delta E interpretation:**

- < 1: Imperceptible (excellent)
- 1-2: Barely perceptible (good)
- 2-5: Noticeable on close inspection
- > 5: Clearly different

---

## `vectalab optimize`

Compress existing SVG files with SVGO.

```bash
vectalab optimize <input.svg> [output.svg] [options]
```

### When to Use

```
❌ WRONG: PNG → convert to SVG → vectorize again
   (Produces larger, lower quality files)

✅ RIGHT: PNG → premium → SVG
   OR
✅ RIGHT: existing SVG → optimize → smaller SVG
```

### optimize — Options

| Flag | Default | Description |
|------|---------|-------------|
| `--precision, -p` | 2 | Coordinate decimals |
| `--force, -f` | false | Skip confirmation |

### Example

```bash
# Optimize in-place
vectalab optimize icon.svg

# Save to new file
vectalab optimize icon.svg icon_min.svg -p 1
```

---

## `vectalab convert`

Basic high-fidelity vectorization.

```bash
vectalab convert <input> [output] [options]
```

### convert — Options

| Flag | Default | Description |
|------|---------|-------------|
| `--method, -m` | hifi | Method: `hifi`, `bayesian`, `sam` |
| `--quality, -q` | ultra | Preset: `figma`, `balanced`, `quality`, `ultra` |
| `--target, -t` | 0.998 | Target SSIM |
| `--device, -d` | auto | Device: `cpu`, `cuda`, `mps` |

### Quality Presets

| Preset | File Size | Quality | Use Case |
|--------|-----------|---------|----------|
| `figma` | Smallest | Good | Design tools |
| `balanced` | Medium | Better | General use |
| `quality` | Larger | High | Print-ready |
| `ultra` | Largest | Highest | Archival |

---

## `vectalab logo`

Optimized for flat logos and icons.

```bash
vectalab logo <input> [output] [options]
```

### logo — Options

| Flag | Default | Description |
|------|---------|-------------|
| `--colors, -c` | 8 | Palette size (2-64) |
| `--quality, -q` | 0.95 | Target SSIM |
| `--snap/--no-snap` | enabled | Snap to pure colors |

---

## `vectalab smart`

Auto-balance between file size and quality.

```bash
vectalab smart <input> [output] [options]
```

### smart — Options

| Flag | Default | Description |
|------|---------|-------------|
| `--size, -s` | 100 | Target size in KB |
| `--quality, -q` | 0.92 | Minimum SSIM |
| `--iterations, -i` | 5 | Max attempts |

---

## `vectalab info`

Analyze an image file.

```bash
vectalab info <file>
```

### Output

```text
📊 Image Analysis
├── Dimensions: 400×200
├── Colors: 3 unique
├── Type: Logo/Icon (flat colors)
├── Recommended: vectalab premium --mode logo
└── Expected size: ~2-5 KB
```

---

## `vectalab compare`

Compare two images.

```bash
vectalab compare <image1> <image2>
```

---

## `vectalab render`

Render SVG to PNG.

```bash
vectalab render <input.svg> [output.png] [--scale N]
```

---

## `vectalab svgo-info`

Check SVGO installation status.

```bash
vectalab svgo-info
```

### Output (when installed)

```text
╭─────────┬─────────────┬──────────╮
│ Node.js │ ✓ Installed │ v20.18.1 │
│ SVGO    │ ✓ Installed │ v4.0.0   │
╰─────────┴─────────────┴──────────╯
```

### Installing SVGO

```bash
# macOS
brew install node
npm install -g svgo

# Ubuntu
sudo apt install nodejs npm
npm install -g svgo

# Verify
svgo --version
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `VECTALAB_DEVICE` | Default device (cpu/cuda/mps) |
| `VECTALAB_VERBOSE` | Enable verbose output |

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (file not found, conversion failed) |
| 2 | Invalid arguments |
| 130 | Interrupted (Ctrl+C) |
