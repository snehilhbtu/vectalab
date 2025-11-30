# Auto Mode Implementation Log

## Date: 2025-11-30
## Task: Implement `vectalab auto` command

### Summary
Implemented a new `auto` command in the `vectalab` CLI that runs multiple vectorization strategies in parallel and selects the best one based on SSIM and file size.

### Changes
1.  **`vectalab/cli.py`**:
    - Added `auto` command using `typer`.
    - Added arguments: `input_path`, `output_path`, `target_ssim`, `workers`, `verbose`.
    - Implemented call to `vectorize_auto`.

2.  **`vectalab/sota.py`**:
    - Implemented `vectorize_auto` function.
    - Implemented `_run_strategy_wrapper` helper for `ProcessPoolExecutor`.
    - Defined 4 strategies:
        - Logo Clean (Ultra)
        - Premium Logo
        - Premium Photo
        - Smart Adaptive
    - Implemented scoring logic: `SSIM * 100 - (SizeKB / 100)`.

### Challenges
-   **Pickling Error**: Encountered `AttributeError: Can't get local object` when using `ProcessPoolExecutor` with a locally defined `run_strategy` function.
-   **Fix**: Refactored `run_strategy` to a module-level function `_run_strategy_wrapper` to ensure it is picklable.

### Verification
-   Ran `vectalab auto test_data/png_mono/camera.png output_camera.svg --verbose`.
-   Verified that all 4 strategies ran in parallel.
-   Verified that the best strategy was selected and the output file was created.

### Next Steps
-   Consider adding more strategies or tuning the existing ones.
-   Add unit tests for `vectorize_auto`.
