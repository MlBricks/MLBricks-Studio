# Step 2B — Classical ML Fit Runtime

This patch completes the fit-based Machine Learning Gallery models introduced in Step 2.

## Ready Gallery models

- KNN — editable neighbours, voting mode and distance p
- Decision Tree — editable depth, split/leaf limits and Gini/entropy criterion
- K-Means — K-Means++ initialization, editable cluster count, tolerance and iterations
- PCA — SVD-based dimensionality reduction with centering and optional whitening

## Runtime

The four algorithms are public `ML Core` Studio components and are implemented in PyTorch without a scikit-learn runtime dependency. Their fitted state is stored as module buffers so it participates in normal `state_dict` / MLBricks lifecycle persistence.

`train_builder_model` detects a classical fit component and routes the graph through a one-shot `classical_fit` lifecycle rather than the causal-LM gradient loop. Supervised fit reports training/validation accuracy; K-Means reports inertia and iterations; PCA reports explained variance.

## Data Gallery mappings

- KNN → Multiclass Classification Demo
- Decision Tree → Multiclass Classification Demo
- K-Means → Unlabeled Clustering Demo
- PCA → High-Dimensional Feature Demo

Tabular demo datasets are treated as numeric/signal-style inputs for Studio compatibility instead of falling through to text modality.

## Studio UI

Classical models use **Start Fit / Stop Fit**, **Fit Status**, **Fit Metrics**, and **Fit Log** semantics. Gradient-specific learning-rate/optimizer budget validation is skipped for classical-fit graphs. The same universal input runtime can execute the fitted graph after persistence/load.

## Tests

Coverage includes public component registration, KNN/tree/K-Means/PCA fitting, prediction/transformation, fitted-state round trips, and `train_builder_model` classical dispatch/metrics.
