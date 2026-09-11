# MLBricks Studio — Step 7 Audio Models

## Scope

Step 7 adds a glass-box audio-generation family to the Studio Model Gallery and connects it to the existing Audio Data Gallery.

## Gallery models

- Neural TTS
- Voice-conditioned TTS
- Voice Clone educational template
- Sound Generator
- Music Generator
- Audio JEPA remains available from the JEPA work completed in Step 5

## Public reusable components

- Speaker Embedding
- Audio Codec Encoder
- Audio Token Predictor
- Audio Codec Decoder
- Audio Output

The audio Gallery templates are constructed from public Studio graph components. They can be opened, edited, cloned as custom models, and rebuilt by users rather than relying on a hidden model-specific architecture.

## Data integration

Step 7 uses the existing categorized Audio Data Gallery:

- Speech + Transcript Demo
- Multi-Speaker Speech Demo
- Music + Caption Demo
- Sound + Caption Demo
- Unlabeled Audio Demo

The multi-speaker demo now includes a reference-audio field so reference-conditioned voice experiments can be represented in the same visible data/model workflow.

## Training runtime

A dedicated `audio_generation` training mode now supports text/caption-conditioned waveform training with:

- byte-token text conditioning for the compact educational models
- optional speaker-ID conditioning
- optional reference-audio conditioning
- waveform MSE loss
- STFT magnitude spectral loss
- validation
- checkpoint/final-artifact persistence through the normal Studio lifecycle

## Inference

Text-conditioned audio models route through the universal inference runtime rather than the causal-language-model generation path. Audio outputs are serialized as WAV data URIs for Studio playback/output handling.

## Design boundary

These are compact educational/research reference architectures. They demonstrate the data flow and building blocks behind TTS, speaker/reference conditioning, sound generation, and music generation. They are not exact reproductions or production-scale clones of ElevenLabs, Suno, or other proprietary systems.

## Validation

- Python module compile checks: passed
- Frontend JavaScript syntax checks: passed
- Frontend distribution rebuild: passed
- Full test suite: **496 passed, 2 skipped**
