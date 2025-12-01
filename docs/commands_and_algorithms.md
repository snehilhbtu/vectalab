## Commands & Algorithms — concise reference

This page maps each public CLI command to the canonical implementation and the algorithmic primitives used under the hood. It is intentionally concise and citation-backed so integrators and researchers can read fast and act.

---

### Quick navigation
- convert / hifi alias — high‑fidelity tracing (HiFi / vtracer)
- premium / logo / smart / auto — high‑quality SOTA pipelines + 80/20 optimizations
- optimize — SVGO-based SVG compression
- render / info / compare / svgo-info / optimal — utilities and evaluation

---

### Core algorithm building blocks (one‑line summaries)
 vtracer (tracing engine): robust path extraction from raster edges; replaces pure bitmap→vector heuristics with geometric fitting and spline/polygon optimization. Canonical implementation and bindings: https://github.com/visioncortex/vtracer (and Python bindings https://github.com/etjones/vtracer_py). See the project docs for CLI options and modes (colormode, hierarchical scanning and path precision).
 HiFi (high-fidelity raster→vector): iterative raster-to-curve fit focusing on structure preservation rather than minimizing path count — tuned for pixel‑accurate structural similarity.
 Color quantization (k‑means / median cut): reduces palette before tracing for logos and limited‑color images (fewer colors → cleaner, simpler paths). Typical implementation uses k‑means clustering (scikit‑learn, sklearn.cluster) or optimized median‑cut implementations.
 Path simplification (Douglas–Peucker / Ramer–Douglas–Peucker): reduces point density while keeping geometric shape; useful for smaller SVGs with minimal visual change.
 Edge‑preserving denoising (bilateral / guided filters): removes texture noise while keeping strong edges for cleaner tracing; common implementations available in OpenCV and scikit‑image.
 Shape detection (Hough-based circle/rect detection): identifies primitives (circle, rect, ellipse) for smaller, semantically correct SVG primitives instead of traced paths; commonly implemented via OpenCV Hough transforms for circles/lines.
 Iterative refinement + perceptual metrics (SSIM, delta‑E, LPIPS, DISTS, GMSD): quality‑guided loops that trade off between SSIM (structure) and file size/complexity. The toolkit exposes SSIM as the primary stopping condition with optional LPIPS/DISTS for photos.
 SVGO (Node.js): post‑processing for multiline optimizations and minification of generated SVGs. See the official SVGO repo for rules and plugins.
 SSIM (structural similarity): fast, robust for structural preservation; used as the primary target in many vectorization loops. (Wang et al., 2004) — https://en.wikipedia.org/wiki/Structural_similarity (DOI: https://doi.org/10.1109/TIP.2003.819861)
 LPIPS / DISTS / GMSD: perceptual metrics that better match human judgments for texture/photorealism differences (useful for photos). Use LPIPS / DISTS for modern perceptual ranking.
   - LPIPS: perceptual similarity metric and reference implementation — https://github.com/richzhang/PerceptualSimilarity
  - DISTS: diversified image quality metric combining structure and texture (paper / repo) — arXiv: https://arxiv.org/abs/2004.07728 — implementation: https://github.com/dingkeyan93/DISTS
  - GMSD: gradient magnitude similarity deviation — arXiv/IEEE: https://arxiv.org/abs/1308.3052 (original: IEEE TIP 2013) — DOI: https://doi.org/10.1109/TIP.2013.2293423.
 Delta‑E (CIEDE2000): perceptual color difference — used for color‑critical workflows (logos, brand colours). See color‐difference overview: https://en.wikipedia.org/wiki/Color_difference
### Commands — concise behavior and internals
 - Segment Anything (SAM) segmentation (optional in some pipelines): https://github.com/facebookresearch/segment-anything
 - vtracer/VTracing-type tools (core tracing concepts): search for "vtracer" on GitHub for Rust binding notes and README
  - intent: clean logo/icon vectorization for minimal-colour graphics.
  - implementation: automatic color palette detection + k‑means quantization → contour extraction → path merging & color snapping.
  - trade-offs: very small SVGs and clean shapes; loses photographic detail and fine gradients.

- vectalab premium
  - intent: SOTA 80/20 vectorization (best balance of quality, size, runtime).
  - implementation: iterative hybrid pipeline: preprocess (edge‑aware filters), color quantization (optional), vtracer/HiFi tracing, shape detection, path merging, then optional SVGO.
  - options: target SSIM, precision (coordinate digits), shape detection, iterations, and SVGO integration.
  - trade-offs: configurable between file size and fidelity; SVGO reduces size further at the cost of an external Node.js dependency.

- vectalab smart
  - intent: recommended default for logos, icons and illustrations — automatic presets tuned for typical assets.
  - implementation: adaptive selection of quantization, tracing parameters, path simplification and a fixed optimization budget.
  - trade-offs: balances quality and size automatically — good default for most users.

- vectalab auto
  - intent: run multiple strategies in parallel (Logo Clean, Premium Logo, Premium Photo, Smart) and pick the best by SSIM/file size trade-off.
  - implementation: runs competitive strategies (multi‑worker) then scores outputs using quality metrics and size.
  - trade-offs: more CPU/memory but increases chance of producing the best result without hand‑tuning.

- vectalab optimize
  - intent: compress / minify existing SVG files.
  - implementation: wraps SVGO (Node.js) and exposes coordinate precision and multipass options.
  - trade-offs: very effective size reduction (30–50%) but requires externally-installed SVGO (and Node.js). Use when you already have an SVG and want compression only.

- vectalab render / info / compare / optimal / svgo-info
  - render: rasterize SVGs to PNG (for thumbnails / visual checks).
  ### Per‑command mini diagrams & code mappings (quick)

  Below are compact, single‑line diagrams and the canonical code entry‑points for each public CLI command. These are short, opinionated maps so maintainers and integrators can jump from what to run to where the work happens in the source.

  - vectalab convert (CLI handler: `convert` in `vectalab/cli.py`)

    Pipeline (mini):

      Input → validate_input_file → [auto|hifi|standard] selector → vectorize_high_fidelity / Vectalab.vectorize → calculate_full_metrics → (optional) SVGO → Output

    Canonical code paths:
    - CLI handler: `vectalab/cli.py::convert`
    - HiFi flow: `vectalab/hifi.py::vectorize_high_fidelity`
    - Auto fallback: `vectalab/auto.py::determine_auto_mode` → `vectalab/premium.py::vectorize_premium` or `vectorize_logo_premium`
    - Standard SAM/Bayesian: `vectalab/core.py::Vectalab.vectorize`

  - vectalab logo (CLI handler: `logo` in `vectalab/cli.py`)

    Pipeline (mini):

      Input → palette detection / k‑means → contour extraction (vtracer) → path merging & color snapping → metrics → Output

    Canonical code paths:
    - CLI handler: `vectalab/cli.py::logo`
    - Core work: `vectalab/quality.py::vectorize_logo_clean`
    - Tracer: `visioncortex/vtracer` (Rust) or Python bindings `etjones/vtracer_py` where used

  - vectalab premium (CLI handler: `premium` in `vectalab/cli.py`)

    Pipeline (mini):

      Input → edge-aware preprocess (bilateral/guided) → quantize (optional) → HiFi/vtracer trace → shape detection → iterative refinement (SSIM/Learned metrics) → (optional) SVGO → Output

    Canonical code paths:
    - CLI handler: `vectalab/cli.py::premium`
    - Pipeline implementation: `vectalab/premium.py::vectorize_premium`, `vectorize_logo_premium`, `vectorize_photo_premium`
    - SVGO integration: `vectalab/optimizations.py::optimize_with_svgo` and `check_svgo_available`

  - vectalab smart (CLI handler: `smart` in `vectalab/cli.py`)

    Pipeline (mini):

      Input → detect type (logo/illustration/photo) → adaptive settings → run `vectalab/sota::vectorize_smart` → metrics → Output

    Canonical code paths:
    - CLI handler: `vectalab/cli.py::smart`
    - Router/impl: `vectalab/sota.py::vectorize_smart`

  - vectalab auto (CLI handler: `auto` in `vectalab/cli.py`)

    Pipeline (mini):

      Input → run multiple strategies (Logo Clean / Premium Logo / Premium Photo / Smart) in parallel → evaluate SSIM + size → pick best → Output

    Canonical code paths:
    - CLI handler: `vectalab/cli.py::auto`
    - Mode detection: `vectalab/auto.py::determine_auto_mode`
    - Strategy implementations: `vectalab/premium.py`, `vectalab/sota.py`, `vectalab/hifi.py`

  - vectalab optimize (CLI handler: `optimize` in `vectalab/cli.py`)

    Pipeline (mini):

      Input SVG → read → optimize (SVGO preset / plugins) via `optimize_with_svgo` → write → Output

    Canonical code paths:
    - CLI handler: `vectalab/cli.py::optimize_svg`
    - Implementation: `vectalab/optimizations.py::optimize_with_svgo` and `check_svgo_available`

  - vectalab render / info / compare / optimal / svgo-info

    Short mappings:
    - `render` → `vectalab/cli.py::render` → `vectalab/hifi.py::render_svg_to_png`
    - `info` → `vectalab/cli.py::info` → lightweight CV checks + suggested CLI snippet
    - `compare` → `vectalab/cli.py::compare` → uses skimage.metrics (SSIM / PSNR) and OpenCV
    - `optimal` → `vectalab/cli.py::optimal` → `vectalab/quality.py::vectorize_optimal` + compare_and_visualize
    - `svgo-info` → helper in `vectalab/optimizations.py` (check Node/SVGO availability, install tips)

  Each of the mini‑diagrams above is intentionally terse — they point to the single canonical implementation file where the main algorithm lives. For visual aid, see the full pipeline SVG diagram: `docs/assets/pipeline.svg` (this doc references it above).
  - info: inspect images / metadata.
  - compare: compute and show image quality metrics between images/SVGs (SSIM, PSNR, LPIPS, DISTS, Delta‑E when configured).
  - optimal: search style/hyperparameter space to meet an SSIM target with smallest file size (quality-guided search + metric reporting).
  - svgo-info: test Node/SVGO availability and give install tips.

---

### Quality metrics — what they mean, and when to trust them
- SSIM (structural similarity): fast, robust for structural preservation; used as the primary target in many vectorization loops. (Wang et al., 2004)
- PSNR: classic pixel-wise signal-noise metric — useful for quick checks but poor perceptual correlation on complex images.
- LPIPS / DISTS / GMSD: perceptual metrics that better match human judgments for texture/photorealism differences (useful for photos). Use LPIPS / DISTS for modern perceptual ranking.
- Delta‑E (CIEDE2000): perceptual color difference — used for color-critical workflows (logos, brand colours).

---

### Practical guidance / rules of thumb
- Need pixel-perfect structure (diagrams, UI screenshots)? Use convert / premium with high SSIM targets and higher precision.
- Need small file size for icons/logos? Use logo mode and aggressively reduce palette (8–16 colors) + optimize.
- Need the best single result without tuning? Use auto — it runs several strategies and picks the best.
- Need maximum compression on an already‑generated SVG? Use optimize with SVGO and precision tuning.

---

### Where to look in the code (canonical sources)
- CLI command mapping and flags: `vectalab/cli.py` — the definitive surface
- High‑fidelity routines: `vectalab/hifi.py` and `vectalab/sota.py` (smart/auto/premium wrappers)
- Logo and premium internals: `vectalab/premium.py`, `vectalab/quality.py`, `vectalab/optimizations.py`

---

### Useful references (short)
 - SSIM: https://en.wikipedia.org/wiki/Structural_similarity (Wang et al., 2004; DOI: https://doi.org/10.1109/TIP.2003.819861)
- LPIPS: https://github.com/richzhang/PerceptualSimilarity (Zhang et al., CVPR 2018)
 - DISTS: https://arxiv.org/abs/2004.07728 (paper) / https://github.com/dingkeyan93/DISTS (code)
 - Douglas–Peucker (path simplification): https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
- Hough transform: https://en.wikipedia.org/wiki/Hough_transform
- Bilateral filter: https://en.wikipedia.org/wiki/Bilateral_filter
 - SVGO documentation: https://svgo.dev/ (canonical docs) — source: https://github.com/svg/svgo

---

If you want, I’ll now add the in‑line citations and a few small per‑command pipeline diagrams linking to `docs/assets/pipeline.svg` and add compact examples that line-up directly with `vectalab/cli.py` flags.

---

## Examples & quick command snippets

View a compact pipeline diagram: docs/assets/pipeline.svg

Basic conversions:

- Convert an image using high fidelity:
  ```bash
  vectalab convert photo.png --target 0.998 --quality ultra
  ```

- Logo mode (small, clean icons):
  ```bash
  vectalab logo icon.png -c 8
  ```

- Premium SOTA with SVGO enabled and shape detection (logo):
  ```bash
  vectalab premium artwork.png --mode logo --shapes --svgo
  ```

- Smart default (recommended):
  ```bash
  vectalab smart image.png -s 50 -q 0.92
  ```

- Auto (run several strategies in parallel):
  ```bash
  vectalab auto image.png -w 8
  ```

- Optimize an existing SVG (requires svgo):
  ```bash
  vectalab optimize logo.svg -p 1
  ```

- Run an optimization search for smallest file meeting SSIM target:
  ```bash
  vectalab optimal image.png --quality 0.95
  ```

---

If you'd like, I can add per-command pipeline mini diagrams, or cross-link the examples to the benchmark scripts in `scripts/` so users can reproduce outputs exactly.
