# Step 3 — Generic Supervised Deep Learning Training

This patch extends MLBricks Studio with a generic non-text supervised training runtime so the educational Deep Learning Gallery models can be trained through the same Studio compiler/runtime used by custom graphs.

## Supported educational models

- Single Neuron
- ANN
- CNN
- RNN
- LSTM
- GRU
- Autoencoder

The same runtime also supports compatible user-created graphs built from the public ML/DL/Math components.

## Training modes

Studio now distinguishes three training/fitting lifecycles:

- `gradient` — causal language-model training
- `supervised` — generic non-text gradient training
- `classical_fit` — KNN, Decision Tree, K-Means, PCA

Non-text supervised graphs no longer load or require a tokenizer.

## Generic supervised data adapters

The runtime currently supports one primary supervised input source per graph:

- `Feature Input` for tabular and sequence data
- `Image Input` for image tensors
- `Signal Input` for signal tensors

It can consume the demo-data field conventions used by the Data Gallery, including tabular features, labels/targets, sequences, images, and reconstruction targets.

## Task and loss handling

For Step 3, the training task is inferred from the graph/model metadata and Studio applies the corresponding runtime loss:

- Multiclass classification → Cross Entropy
- Binary classification → Binary Cross Entropy
- Regression → Mean Squared Error
- Reconstruction / Autoencoder → Mean Squared Error

Explicit visual Loss nodes remain a later roadmap step. Step 3 focuses on making the existing educational model graphs genuinely trainable first.

## Metrics and telemetry

Supervised training reports modality-appropriate metrics instead of language-model token metrics:

- Step
- Samples processed
- Samples/sec
- Average samples/sec
- Training loss
- Validation loss
- Accuracy / validation accuracy for classification
- MAE / validation MAE for regression and reconstruction
- GPU memory and peak memory where available
- Elapsed time

The Training UI hides language-only sample-generation controls for supervised models.

## Training configuration

Supervised models support:

- Step, sample, or epoch budgets
- Batch size
- Gradient accumulation
- Optimizer selection
- Learning rate and weight decay
- Warmup
- Mixed precision / AMP where supported
- Gradient clipping
- Validation interval and validation steps
- Checkpoint interval
- Resume from saved weights/checkpoints

## Artifacts

Trained supervised models use the normal MLBricks Studio model lifecycle/artifact path. Training metadata and supervised metrics are saved with the model, and supplemental training state is stored for resumable runs.

## Validation

The Step 3 regression suite verifies:

- Supervised compilation does not require a tokenizer
- Single Neuron regression training
- ANN classification training and accuracy reporting
- RNN, LSTM, and GRU supervised training
- CNN end-to-end supervised training
- Autoencoder end-to-end reconstruction training
- CNN/Autoencoder tensor contracts
- Step 3 supervised-training UI behavior

Full suite result for this patch: **447 passed, 2 skipped**.
