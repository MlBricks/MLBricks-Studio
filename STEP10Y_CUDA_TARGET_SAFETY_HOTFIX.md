# Step 10Y — CUDA Target Safety Hotfix

Prevents invalid class targets from reaching CUDA cross-entropy kernels.

- COCO-derived classification rows with the explicit unlabeled sentinel `-1` are removed before accelerator transfer.
- Positive classification labels outside the configured classifier range fail on CPU with a clear message.
- Detection class IDs are validated against the Detection Head before tensor targets are built.
- Classification and detection losses perform a final target-range guard before `cross_entropy`.
- No dataset is bundled or cached; COCO128 remains cloud-only and temporary.

After a CUDA device-side assert, the current Python process must be restarted because the CUDA context is no longer trustworthy.
