# Architecture & Conversion Pipelines — diagrams (Mermaid)

This document shows a compact, high-value visual overview of the Vectalab architecture and the main conversion pipelines. Diagrams are intentionally simple and use a pastel theme for clarity and readability.

---

```mermaid
flowchart LR
  %% compact architecture: max 6 nodes
  CLI[vectalab CLI] --> Core[Core modules]
  Core --> VTracer[vtracer]
  Core --> SVGO[SVGO]
  Core --> Quality[Metrics]
  Auto[Auto orchestrator] --> Core

  %% colors removed for neutral rendering
```

---

## Per-pipeline diagrams — concise

Notes: each mini-diagram is annotated with the canonical handler/function from the codebase so maintainers can quickly find the implementation.

### Convert / HiFi pipeline

```mermaid
flowchart LR
  %% Convert / HiFi pipeline (<=6 nodes)
  IN[Input image] --> Selector{method}
  Selector --> HiFi[HiFi vectorize]
  Selector --> Auto[Auto strategy]
  HiFi --> Metrics[Quality metrics]
  Metrics --> Out[Output SVG]

  %% colors removed for neutral rendering
```

### Logo pipeline

```mermaid
flowchart LR
  %% Logo pipeline (<=6 nodes)
  IN[Input image] --> Detect[Detect logo]
  Detect --> Palette[Palette reduce]
  Palette --> Trace[vtracer trace]
  Trace --> Merge[merge paths]
  Merge --> Out[Output SVG]

  %% colors removed for neutral rendering
```

### Premium (SOTA + 80/20) pipeline

```mermaid
flowchart LR
  %% Premium pipeline (<=6 nodes)
  IN[Input] --> Pre[Preprocess]
  Pre --> Trace["Trace (vtracer)"]
  Trace --> Ref[Refine iterations]
  Ref --> Opt[Optional optimize]
  Opt --> Out[Output SVG]

  %% colors removed for neutral rendering
```

### Smart pipeline (targets size/SSIM budget)

```mermaid
flowchart LR
  %% Smart pipeline (<=6 nodes)
  IN[Input image] --> Detect[Detect type]
  Detect --> Tune[Adapt params]
  Tune --> Run[Run vectorization]
  Run --> Score[Measure quality]
  Score --> Out[Output or retry]

  %% colors removed for neutral rendering
```

### Auto pipeline (parallel strategies)

```mermaid
flowchart LR
  %% Auto pipeline (<=6 nodes)
  IN[Input] --> Fork[Parallel strategies]
  Fork --> Strategies[Run strategies]
  Strategies --> Eval[Evaluate results]
  Eval --> Select[Select best]
  Select --> Out[Output SVG]

  %% colors removed for neutral rendering
```

---

### How to embed or render these diagrams
- GitHub renders Mermaid blocks in README.md and many Markdown viewers support Mermaid. The diagrams above are intentionally compact so they read well in-line.

---

If you'd like, I can also export individual diagrams to PNG/SVG assets, or split the per-pipeline diagrams into separate files for in-page embedding in `docs/assets/`.
