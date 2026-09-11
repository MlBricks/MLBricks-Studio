# MLBricks Studio — Step 8 Multimodal Models

## Scope

Step 8 adds the first fully trainable multimodal Gallery models and extends Studio runtime input routing so one editable graph can consume aligned inputs from more than one modality.

## Gallery models

- **Multimodal JEPA** — aligned image + text with symmetric image→text and text→image latent prediction. Both encoders learn because each branch is a predictor once and a stop-gradient target once.
- **Sensor + Vision Fusion** — parallel CNN and temporal-sensor encoders, explicit visible fusion, and a shared classifier.

## Data

Step 8 uses the existing categorized Multimodal Data Gallery presets:

- Aligned Image + Text Demo
- Sensor + Vision Demo

Both are deterministic offline Studio demo datasets and preserve alignment sample-by-sample.

## Glass-box design

No opaque multimodal model component is introduced. The Gallery templates use existing public Studio building blocks: modality inputs, Conv1D/Conv2D, pooling, flatten, concatenate, Linear, classifier, JEPA Encoder, JEPA Predictor, and JEPA Latent Loss. Image, Video and Signal inputs now expose the same optional Runtime Input Key used by Text/Audio inputs so researchers can create their own named multi-input graphs.

## Training/runtime

A `multimodal` training lifecycle supports aligned named inputs, validation, checkpoints, resume, artifact persistence, sample throughput and task metrics. Universal inference accepts multimodal dictionaries for the Step 8 image+text and image+sensor reference models.

## Design boundary

Multimodal JEPA is an MLBricks educational/research cross-modal JEPA-style architecture, not a claim of reproducing a proprietary or exact external recipe. Future work can add more modalities, shared recurrent SOUP/ESA world state, cross-attention/fusion alternatives, and larger real-world datasets.
