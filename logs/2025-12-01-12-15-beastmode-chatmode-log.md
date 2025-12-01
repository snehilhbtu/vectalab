Per-command mini diagrams & code mappings added to docs/commands_and_algorithms.md

Actions performed:
- Added compact mini-diagrams (single-line flows) and exact handler mappings for the following CLI commands: convert, logo, premium, smart, auto, optimize, render, info, compare, optimal.
- Mapped CLI command -> canonical implementation files and key functions (e.g., convert -> vectalab/cli.py::convert -> vectalab/hifi.py::vectorize_high_fidelity).
- Confirmed full test-suite passes after edits (44 passed, 0 failed).

Why: Make it trivial for maintainers to trace CLI behavior to source-of-truth code and speed up onboarding.

Timestamp: 2025-12-01 12:15
