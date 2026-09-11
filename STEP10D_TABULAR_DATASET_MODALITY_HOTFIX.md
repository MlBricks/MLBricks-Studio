# Step 10D — Tabular Dataset Modality Hotfix

## Problem
The Gallery preset correctly labeled `Tabular Regression Demo` as `Tabular`, but the training compatibility helper `datasetModality()` still mapped tabular demo types to `signal`. As a result, Linear Regression showed `Model: tabular · Data: signal` and disabled training even though feature width and rows were valid.

## Fix
- Tabular regression/classification, clustering, PCA/high-dimensional, and neuron-regression demos now resolve to the `tabular` runtime modality.
- Sequence and scientific signal demos continue to resolve to `signal`.
- Applied to both frontend source and packaged Studio static builder.
- Added regression tests so a tabular preset cannot silently regress back to signal compatibility.

## Result
`Linear Regression + Tabular Regression Demo` now passes the modality compatibility gate and exposes Train when the remaining checks pass.
