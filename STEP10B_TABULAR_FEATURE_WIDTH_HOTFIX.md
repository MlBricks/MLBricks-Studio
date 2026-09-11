# Step 10B — Tabular Feature-Width Hotfix

Fixes a mismatch between the Machine Learning Gallery presets and their demo datasets:

- Linear Regression expected 4 input features while Tabular Regression Demo generated 8.
- Logistic Regression had the same hidden mismatch.
- The two demo presets now explicitly generate 4 features.
- Model requirements now expose `feature_dim` from Feature Input nodes.
- Training compatibility now shows a Feature width check for scalar `feature_N` datasets.
- Generic supervised training fails early with a clear feature-width message instead of a low-level PyTorch matmul error.

Regression suite: 520 passed, 2 skipped.
