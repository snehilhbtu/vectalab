## 2025-12-01 12:00 — Documentation audit & refactor

Actions:
- Audited the docs/ folder and CLI implementation for mismatches and outdated content.
- Removed outdated `docs/protocol_v2.md` and replaced with concise `docs/benchmarks.md` referencing `scripts/benchmark_runner.py`.
- Condensed and modernized docs: `docs/README.md`, `docs/cli.md`, `docs/api.md`, `docs/examples.md`, `docs/modal_setup.md`, `docs/algorithm.md` (concise architecture) and added `docs/assets/pipeline.svg`.
- Updated top-level README to reference the new benchmarks docs and runner script.
- Ran full test suite: 44 passed, 0 failed.

Decisions:
- Keep the CLI file (`vectalab/cli.py`) as the single source of truth for options; docs are concise pointers to that reality.
- Replace verbose deep-dives with short, actionable content targeted at end-users, integrators, and developers.

Next steps / follow-ups:
- Optional: polish remaining markdown lint issues across the repository (non-blocking) and add more diagrams/examples for specific audiences.
- Consider adding a CI check to ensure docs link targets and code examples remain valid.

Lessons/insights:
- The CLI is the canonical interface; docs must be short, actionable, and point readers to the code for exact flags.
- Benchmarking should use the Python harness for reproducibility rather than a non-shipped CLI.
