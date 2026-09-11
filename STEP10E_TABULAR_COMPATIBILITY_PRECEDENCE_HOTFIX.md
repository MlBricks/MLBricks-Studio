# Step 10E — Tabular Compatibility Precedence Hotfix

Fixes the remaining Model Builder compatibility false-negative where a prepared tabular demo could still be reported as `Data: signal`.

## Root cause

`datasetModality()` checked `pipeline.signal_processing` before the dataset source/data contract. Older autosaved prepared-dataset metadata could contain a stale `signal_processing` snapshot, so a valid `Tabular Regression Demo` was classified as signal even though its source `demo_type` and `feature_N` columns were tabular.

## Fix

- Source `demo_type` is now authoritative.
- Explicit dataset modality is considered next.
- `feature_N` columns infer tabular modality.
- Processing-node flags are only fallback hints and cannot override the data contract.
- The behavior is mirrored in both frontend source and packaged Studio builder payload.
