Actions:
- Created `docs/commands_and_algorithms.md` (initial draft) — concise mapping from CLI commands to underlying algorithmic primitives and tradeoffs.
- Embedded citation-style references for SSIM, LPIPS, DISTS, GMSD, Delta-E, SVGO, SAM and core tracing concepts.
- Added compact CLI examples and linked pipeline diagram (`docs/assets/pipeline.svg`).
- Ran tests — 44 passed, 0 failed (no regressions).

Decisions:
- Keep the doc concise and actionable, pointing to code (`vectalab/cli.py`) for exact CLI flags.
- Prioritize pragmatic examples and short pointers rather than long academic review.

Next steps:
- Optionally add inline pipeline mini‑diagrams per command and link to reproducible benchmark scripts in `scripts/`.

Lessons/insights:
- The CLI is the canonical surface; docs should remain short and reference the code for exact flags and behaviour.
