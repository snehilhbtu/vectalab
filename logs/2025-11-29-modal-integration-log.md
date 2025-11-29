# Modal Integration for SAM

## Context
The user requested to use Modal.com to run SAM (Segment Anything Model) because the local environment (macOS) does not support MPS for SAM, or at least it's problematic.

## Changes
1.  **Created `vectalab/modal_sam.py`**:
    -   Defines a Modal App `vectalab-sam`.
    -   Defines a `ModalSAM` class that wraps `SamAutomaticMaskGenerator`.
    -   Handles model downloading and caching in Modal.
    -   Exposes `generate_masks` method.

2.  **Updated `vectalab/segmentation.py`**:
    -   Added `use_modal` parameter to `SAMSegmenter`.
    -   If `use_modal=True`, it imports `ModalSAM` and runs segmentation remotely using `with app.run():`.

3.  **Updated `vectalab/core.py`**:
    -   Updated `Vectalab` class to accept `use_modal` and pass it to `SAMSegmenter`.

4.  **Updated `vectalab/cli.py`**:
    -   Added `--use-modal` flag to the `convert` command.
    -   Passed the flag to `Vectalab`.

5.  **Updated `requirements.txt`**:
    -   Added `modal>=0.50.0`.

## Usage
To use the Modal backend:
```bash
vectalab convert input.png --method sam --use-modal
```
This requires `modal` to be installed and configured (`modal setup`).
