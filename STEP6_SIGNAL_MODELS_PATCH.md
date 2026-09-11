# MLBricks Studio — Step 6 Signal Models

## Scope

Step 6 adds the first complete Signal research family to MLBricks Studio while preserving the glass-box rule: Gallery models are editable graphs built from public Studio components, and signal datasets are converted through a visible Data Graph rather than hidden preprocessing.

## Signal Gallery models

- Time-Series Predictor
- Signal Classifier
- Anomaly Detector
- Signal Denoiser
- Sensor Fusion
- Spectral Model
- RF/IQ Model
- Signal JEPA (implemented in Step 5 and now also discoverable from the Signal category)
- SOUP Signal

## New public components

### Signal Schema Mapper (`signal_process`)

A Data Processing component that maps arbitrary tabular/signal fields into Studio's canonical signal contract. It supports one or multiple signal columns, optional target mapping, normalization, and pad/trim length.

Examples:

- `context -> signal`, `target -> target` for forecasting
- `noisy_signal -> signal`, `clean_signal -> target` for denoising
- `sensor_a,sensor_b,sensor_c -> signal` for sensor fusion
- `i,q -> signal` for RF/IQ data

### FFT Magnitude (`signal_fft`)

A differentiable model component based on `torch.fft.rfft`. It supports optional DC removal, bin cropping, logarithmic magnitude scaling, and channel flattening.

## Data Gallery signal contracts

Step 6 wires existing signal demo presets through Signal Schema Mapper so the model runtime receives consistent tensors:

- Time-Series Forecast Demo
- Signal Classification Demo
- Anomaly Detection Demo
- Clean + Noisy Signal Demo
- Multi-Sensor Demo
- Spectral Signal Demo
- RF/IQ Demo
- Unlabeled Signal / Signal JEPA Demo
- Long Sequential Signal Demo

The long-signal demo now exposes a context signal and future target for supervised long-sequence prediction. The spectral demo exposes the target spectrum for the Spectral Model.

## Editable model graphs

The models use visible Studio nodes rather than opaque task implementations:

- Forecasting: Signal Input -> Conv1D stack -> pooling -> Linear forecast head
- Classification / anomaly: Signal Input -> Conv1D + normalization + activations -> classification head
- Denoising: Signal Input -> Conv1D reconstruction stack -> Tensor Output
- Sensor fusion: multi-channel Signal Input -> Conv1D fusion stack -> classifier
- Spectral: Signal Input -> FFT Magnitude -> MLP -> spectrum output
- RF/IQ: two-channel I/Q input -> Conv1D feature extractor -> classifier
- SOUP Signal: Signal Input -> projection -> SOUP -> temporal reduction -> forecast head

Signal JEPA remains the Step 5 JEPA graph but is additionally exposed when filtering the Model Gallery by Signal.

## Training

The differentiable Step 6 models reuse the generic supervised runtime introduced in Step 3. Classification models use classification targets; forecasting, denoising, and spectral models use regression/reconstruction targets. Signal JEPA uses the JEPA training mode from Step 5. SOUP Signal uses the installed MLBricks SOUP runtime.

## UI changes

- Signal added to the preferred Component Library category order.
- Signal Gallery contains the complete Step 6 model family.
- Signal JEPA is discoverable from both its canonical JEPA family and the Signal category filter without being duplicated in All Models.
- Data Gallery automatically inserts a visible Signal Schema Mapper for Signal presets.
- Data pipeline snapshots and generated preview code include signal-processing configuration.
- Prepared-data inspector surfaces signal-processing information.

## Validation

- Full Python regression suite: **483 passed, 2 skipped**
- Frontend source JavaScript syntax check: passed
- Built Studio JavaScript syntax check: passed

## Scope note

These are educational and research-ready signal building blocks. They are not presented as domain-certified production DSP, medical-device, communications, or safety-critical signal systems. Researchers can clone the templates, replace components, alter preprocessing, and build custom signal architectures through the same Studio compiler path.
