# Step 5 — JEPA Foundation

This patch adds a universal, glass-box JEPA learning path to MLBricks Studio.

## Gallery models

- Image JEPA
- Video JEPA
- Text JEPA
- Audio JEPA
- Signal JEPA

Each Gallery preset is an editable graph built from public Studio components:

`Input -> JEPA Mask -> Context Encoder -> JEPA Predictor -> JEPA Latent Loss`

The same mask also feeds a matching `Target Encoder · EMA`, whose stop-gradient latent representation is the prediction target.

## Public components

- `JEPA Preparation` — visible Data Graph preprocessing for image/video/text/audio/signal samples.
- `JEPA Mask` — exposes context and target ports.
- `JEPA Encoder` — modality-aware context/target encoder.
- `JEPA Predictor` — predicts the target embedding from the context embedding.
- `JEPA Latent Loss` — MSE, Smooth-L1, or cosine latent loss.

## Data path

The existing JEPA demo datasets now insert `JEPA Preparation` before Train/Validation/Test split. Random masking remains in the Model Graph so the learning objective is not hidden in data preprocessing.

Text JEPA uses a transparent byte vocabulary for the educational default (`0 = pad/mask`, `1..256 = byte + 1`) and therefore does not require a Hugging Face tokenizer. Image/video are prepared as normalized grayscale tensors; audio/signal are fixed-length normalized windows.

## Training runtime

- Dedicated `jepa` training mode.
- Context encoder receives gradients.
- Target encoder is stop-gradient/frozen.
- Target parameters are initialized from the context encoder and updated using EMA after optimizer steps.
- Configurable `Target EMA Momentum` (default `0.996`).
- Sample-based training budgets and samples/sec metrics.
- Validation latent loss.
- Checkpoint/resume and final MLBricks artifact lifecycle.
- MLBricks Adam/AdamW is preferred; PyTorch Adam/AdamW is a standalone/dev fallback when mlbricks-kit is unavailable.

## Scope

This is a universal educational JEPA implementation intended for Studio composition and research iteration. It implements the core joint-embedding predictive principle, but it is **not a byte-for-byte reproduction of Meta I-JEPA or V-JEPA**. The initial target is a pooled latent embedding and masking is deliberately compact/visible.

Future research upgrades can add multi-block target masks, explicit positional target tokens, multiple target blocks, more exact I-JEPA/V-JEPA recipes, collapse regularization/monitoring, and multimodal shared-latent JEPA.

## Validation

- All five modalities compile and backpropagate.
- Signal JEPA performs an end-to-end training smoke test with EMA target updates and artifact save.
- Data preparation is tested for text, signal, and image.
- Gallery/templates and public component contracts are regression tested.
- Full project suite: **473 passed, 2 skipped**.
