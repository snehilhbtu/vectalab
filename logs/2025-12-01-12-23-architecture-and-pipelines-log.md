Added a high-value architecture + per-pipeline Mermaid diagram document

Actions performed:
- Created `docs/architecture_and_pipelines.md` with an architecture diagram and compact per-pipeline Mermaid diagrams:
  - Overall architecture (CLI -> core modules -> external tools)
  - Convert / HiFi pipeline
  - Logo pipeline
  - Premium (SOTA) pipeline
  - Smart pipeline
  - Auto pipeline
- Applied a pastel theme via Mermaid `themeVariables` for pleasant, professional visuals and good contrast.
- Verified test-suite: 44 passed, 0 failed.

Why: Provide maintainers and integrators clear, copyable diagrams mapping CLI commands to canonical code paths and algorithm building blocks.

Timestamp: 2025-12-01 12:23
