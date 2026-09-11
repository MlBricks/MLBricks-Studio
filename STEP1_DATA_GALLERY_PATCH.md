# Step 1 — Categorized Data Gallery Foundation

This cumulative patch is based on the ML/DL/Math foundations build.

## Added

- Data Gallery category selector:
  - All Data
  - Machine Learning
  - Deep Learning
  - Language
  - JEPA
  - Vision
  - Audio
  - Signal
  - Multimodal
- 40 curated data templates, grouped by category and annotated with task, modality, license/source type, and compatible Gallery models.
- `Studio Demo Dataset` source component for deterministic offline educational/research datasets.
- Offline generators for regression, classification, clustering, PCA/high-dimensional features, sequences, images, object detection, text, JEPA data, audio, time-series, anomaly detection, denoising, sensor fusion, spectral signals, RF/IQ, and multimodal samples.
- Existing TinyStories/Wikipedia/Cosmopedia/FineWeb-Edu/OpenWebMath/UltraChat quickstarts retained under Language.
- Data presets now open as editable Data Graphs through the same data runtime.

## Validation

- Frontend JavaScript syntax check passes.
- Python modules compile.
- Test suite: 416 passed, 2 skipped.

## Next step

Step 2 is the categorized Model Gallery. Start with Machine Learning and Deep Learning models that can already be composed from the new ML/DL/Math foundations, then reorganize the existing Language models. JEPA, Vision, Audio, Signal, and Multimodal model cards should be added only as their required public components/runtime become functional.
