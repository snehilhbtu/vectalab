# Task Log: Centralized Auto Mode Logic

## Actions
- Created `vectalab/auto.py` to centralize the "Auto Mode" decision logic.
- Implemented `determine_auto_mode` function which encapsulates the logic for selecting between `geometric_icon`, `logo`, and `premium` modes.
- Refactored `vectalab/benchmark.py` to use `determine_auto_mode` and shared metric functions from `vectalab/quality.py`.
- Refactored `vectalab/cli.py` to use `determine_auto_mode` instead of duplicating the logic.

## Decisions
- **Single Source of Truth**: By moving the decision logic to `vectalab/auto.py`, we ensure that the CLI and the Benchmark tool will always behave identically. Any improvements to the auto-detection logic will now automatically benefit both tools.
- **Code Cleanup**: Removed significant code duplication from `cli.py` and `benchmark.py`.

## Results
- Verified `vectalab convert ... --method auto` works correctly with the new centralized logic (tested on `alert-octagon.png`).
- Verified `vectalab-benchmark ... --mode auto` works correctly with the new centralized logic (tested on `icons` folder).
- Both tools now produce identical results and metrics for the same inputs.

## Next Steps
- The codebase is now much cleaner and more maintainable.
- Future improvements to "Auto Mode" only need to be made in `vectalab/auto.py`.
