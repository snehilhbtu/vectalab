# Task Log - Modal Model Pre-download

## Actions
- Modified `vectalab/modal_sam.py` to include `wget` in system dependencies.
- Added `.run_commands()` to the Modal image definition to pre-download SAM checkpoints (`vit_h`, `vit_l`, `vit_b`) to `/root/` during the image build phase.
- Updated `ModalSAM.load_model` to check for checkpoints in `/root/` first, eliminating runtime downloads.
- Verified the changes by running `scripts/run_optimization_modal.py`.

## Decisions
- Downloaded all three model variants (`vit_h`, `vit_l`, `vit_b`) to ensure flexibility without rebuilding the image if the user switches models.
- Placed models in `/root/` as it is the standard working directory for Modal containers.

## Next Steps
- The Modal image will now take longer to build initially (due to downloading ~4GB of models), but subsequent runs will be much faster as the model load time is reduced to just reading from disk.
- No further action needed for this task.

## Lessons
- Pre-baking large model weights into the container image is a best practice for serverless inference to minimize cold-start latency.
