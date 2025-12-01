Link verification and hardening pass (docs/commands_and_algorithms.md)

Actions performed:
- Verified all 17 external URLs referenced in docs/commands_and_algorithms.md for reachability and canonical sources.
- Hardened fragile links by:
  - Adding canonical vtracer repository: https://github.com/visioncortex/vtracer and Python bindings (etjones/vtracer_py).
  - Replaced GMSD advice with explicit DOI: https://doi.org/10.1109/TIP.2013.2293423 alongside the arXiv link.
  - Normalized Ramer–Douglas–Peucker link to the algorithm page.
  - Pointed SVGO to its canonical docs site (https://svgo.dev/) while keeping the GitHub source link.
- Ran full test-suite after edits: 44 tests passed, 0 failed.

Next steps:
- Final pass if the team wants additional references added (e.g., LPIPS/arXiv references inline), or per-command mini-diagrams linked to docs/assets/pipeline.svg.

Timestamp: 2025-12-01 12:06
