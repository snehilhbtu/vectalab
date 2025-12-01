# Benchmarks & Protocol (concise)

This page documents how to run reproducible benchmarks used during development. It references the benchmark scripts in `scripts/` and the curated Golden Dataset in `golden_data/`.

## Overview

- The test harness lives in `scripts/benchmark_runner.py` and `scripts/benchmark_80_20.py` for targeted 80/20 optimizations.
- The dataset used in experiments is stored in `/golden_data` (downloaded samples or curated snapshots).

How to run (minimal reproducible):

1. Prepare images (e.g., SVG → PNG rasterization for reproducible input)

1. Run the benchmark runner with a chosen mode (example):

```bash
python scripts/benchmark_runner.py --input-dir golden_data/icons --mode premium --quality balanced
```

1. For 80/20 optimization verification (SVGO, precision, etc):

```bash
python scripts/benchmark_80_20.py examples/test_logo.png
```

## What to expect

- Benchmarks will output CSV/JSON reports in `test_runs/` with metrics like SSIM (RGB/LAB), ΔE, file size, path count, and topology checks.
- When referencing benchmark results in docs, prefer the exact CSV/JSON artifact in `test_runs/` to avoid vague claims.

## Notes

- The old `vectalab-benchmark` binary is not provided by default — run the Python scripts above for consistent, auditable runs.
- If you need cluster or remote runs, adapt `scripts/benchmark_runner.py` to your environment or use the automated SOTA runners in `scripts/run_sota_session.py`.

If you want a deeper protocol for publication, we can add a dedicated `docs/benchmarks-protocol.md` that lists exact dataset versions, random seeds, and CI recipes.
