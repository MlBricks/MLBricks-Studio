# Step 10C — Classical ML Tabular Runtime Hotfix

Fixes runtime input for KNN, Decision Tree, K-Means, PCA, and other feature-input models.

## Problem

Classical ML models were inferred as `signal` models because `feature_input` and `signal_input` shared the same runtime modality. The Runtime panel therefore showed signal controls and retained the default text prompt (`Once upon a time`). Numeric feature rows were reshaped as `[1, T, 1]`, while fitted classical models expect `[B, F]`.

## Changes

- `feature_input` now resolves to the `tabular` runtime modality.
- Classical-fit models force `tabular` even when an older local draft still stores `signal` runtime settings.
- Added Tabular / Numeric Features input type.
- Added Single Feature Row and Feature Batch modes.
- Added Inline Features and CSV / JSON / Text File sources.
- Added task actions:
  - Predict / Run Prediction
  - Predict Class
  - Assign Cluster
  - Transform Features
- Added visible `Feature Values` numeric editor.
- Removed prompt/instruction handling from tabular models.
- Non-text status panels show `Model Output` instead of `Generated Output`.
- Non-text trained models expose `Open Runtime` rather than `Open Generation`.
- Non-text toolbar uses `RUNTIME SETUP / RUNTIME STATUS`.
- Python universal input runtime now converts tabular data to `[B, F]` float tensors.
- Added JSON, CSV, NPY, and plain numeric file support for tabular inference.
- Added automatic zero-valued demo row sized from `feature_dim` when an old draft has no numeric runtime input.

## Verified models

- KNN — 8-feature classification
- Decision Tree — 8-feature classification
- K-Means — 2-feature cluster assignment
- PCA — 16-feature → 2-component transform

## Validation

- Frontend syntax checks pass.
- Python compile check passes.
- Full suite: **525 passed, 2 skipped, 0 failed**.
