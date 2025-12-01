# Scripts directory

This folder contains a mixture of runnable tooling used across the project: benchmarking harnesses, test drivers, small utilities, and some older ad-hoc scripts.

Organization
- `scripts/benchmark_runner.py`, `scripts/benchmark_80_20.py` — main benchmarking tools used by the team.
- `scripts/run_sota_session.py` — higher-level session runner used for SOTA runs.
- `scripts/download_*.py` / `download_models.sh` — helpers for fetching golden data and model weights.
- `scripts/check_*` / `scripts/test_*.py` — various QA helpers and tests.
- `scripts/templates/` — bundled templates used by the report generator.

Archived / housekeeping
- Old ad-hoc or one-off scripts have been moved to `scripts/archived/` to keep the main `scripts/` area focused.

If a script you depend on was moved, you can restore it from `scripts/archived/`. If you'd like to remove or restore additional scripts, open a PR describing which files to archive or restore.

Files (grouped by category)
--------------------------

Benchmarking & reporting
- `benchmark_runner.py` — Full, repeatable benchmark & SOTA session runner that computes metrics and generates HTML reports.
- `benchmark_80_20.py` — Focused benchmark demonstrating the impact of 80/20 optimizations (SVGO, shape detection, precision).

Downloads & models
- `download_golden_dataset.py` — Populate `golden_data/` with curated icons/logos/illustrations used for benchmarking.
- `download_test_svgs.py` — Fetches a defined set of test SVGs (Feather icons, gilbarbara logos, W3C samples) into `test_data/`.
- `download_models.sh` — Shell helper for downloading model checkpoints into `models/` (curl/wget).
- `download_sam_model.py` — Python downloader specifically for the SAM ViT‑B checkpoint with progress reporting.

Tests / QA
- `check_imports.py` — Smoke-test ensuring core Python imports succeed in this environment.
- `check_alpha.py` — Quick inspector to detect whether PNG test files include an alpha channel.
- `check_sam_quality.py` — Render a SAM-generated SVG and compare it to the original PNG using SSIM/PSNR.
- `test_bayesian.py` — Targeted tests for the Bayesian vectorization method on complex scenes.
- `test_combinations.py` — Systematic preprocessing + vtracer settings search to find best combos.
- `test_modal_real.py` — Integration test for Modal SAM segmentation flow (cloud-backed).
- `test_palette.py` — Tests palette-reduction before logo vectorization.
- `test_ultra.py` — Runs vtracer ultra-quality settings for diagnostics.

Optimization & tuning
- `optimize_logo.py` — Grid-search logo conversion parameters (quality/colors) and persist the best candidate.
- `optimize_hifi_params.py` — Sweep and benchmark HIFI presets across complex scenes to find optimal settings.
- `profile_pipeline.py` — Profile pipeline stages (denoise, vtracer, render) to locate bottlenecks.

Vectorize / runner examples
- `run_sota_session.py` — Orchestrates timestamped parallel SOTA sessions, computes metrics, and generates reports.
- `vectorize_direct.py` — Example showing direct use of the `Vectalab` core API (no CLI).
- `vectorize_with_sam.py` — SAM-based segmentation → polygon → SVG pipeline implementation.
- `convert_svg_to_png.py` — Helper to render SVG files to PNG (useful for tests & comparisons).

Utilities & diagnostics
- `compare_results.py` — Compute SOTA metrics (SSIM, topology, edge, delta‑E) and compare outputs.
- `compare_methods.py` — Compare various vectorization methods on a target sample logo.
- `visualize_components.py` — Visualize connected components in PNGs and save a color-coded mask image.

Archived / ad-hoc (preserved for history)
- `minimal_test.py` — (Archived) Minimal smoke-test harness. See `scripts/archived/` for the original.
- `quick_baseline.py` — (Archived) Rapid baseline runner using fastest presets.
- `run_vectalab_test.py` — (Archived) Performance/time measurement harness.
- `run_full_sota_test.py` — (Archived) Full SOTA test harness (SAM + Modal cloud) used historically.
- `run_sam_modal_test.py` — (Archived) SAM + Modal end-to-end experiment harness.
- `run_optimization_modal.py` — (Archived) Modal-focused optimization helper.

Templates
- `templates/` — Jinja2 templates used by `benchmark_runner.py` and report generation.
