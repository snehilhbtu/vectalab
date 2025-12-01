Protect Mermaid labels with double quotes (parenthesis) — docs/architecture_and_pipelines.md

Actions performed:
- Updated `docs/architecture_and_pipelines.md` Mermaid nodes where labels contained parentheses so the label string is wrapped in double quotes. This prevents Mermaid parsing issues.
- Confirmed that all test-suite runs remain green after docs edits (44 passed, 0 failed).

Notes:
- Changes affected nodes such as: Premium, Opt, VTracer, SVGO, SAM, optional SVGO node, preprocess/quantization/shape/iterative-refine labels, and several labels in Smart/Auto diagrams.

Timestamp: 2025-12-01 12:30
