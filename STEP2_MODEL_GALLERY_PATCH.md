# Step 2 — Educational Model Gallery

This patch builds on Step 1 (Data Gallery Foundation) and introduces a categorized Model Gallery.

## Added

- Model-category selector: All Models, Machine Learning, Deep Learning, Language, JEPA, Vision, Audio, Signal, Multimodal.
- Data-linked model cards with **Open Model** and **Open Data** actions.
- Editable, glass-box model templates built from public Studio components.
- Machine Learning templates:
  - Linear Regression — first-principles `Feature Input -> MatMul(W) -> Add(b) -> Output`.
  - Logistic Regression — first-principles linear score plus Sigmoid.
- Deep Learning templates:
  - Single Neuron
  - ANN
  - CNN
  - RNN
  - LSTM
  - GRU
  - Autoencoder
- Existing 50M/200M ESA/SOUP language models reorganized under the Language category.
- Data cards are linked to the recommended dataset/pipeline for every model preset.
- KNN, Decision Tree, K-Means, and PCA are staged visibly but intentionally disabled until the non-gradient classical fit runtime is implemented. Studio does not expose fake neural approximations for these algorithms.
- `Linear / Dense` now falls back to `torch.nn.Linear` when `mlbricks-kit` is unavailable, preserving offline educational usability while preferring the MLBricks implementation when installed.

## Validation

- Educational first-principles regression graphs execute.
- Logistic regression outputs valid probabilities.
- ANN, CNN, RNN, LSTM, GRU, and Autoencoder graph shapes are tested.
- Full test suite: **426 passed, 2 skipped**.
