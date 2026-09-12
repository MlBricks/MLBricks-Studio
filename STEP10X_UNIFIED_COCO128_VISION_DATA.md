# Step 10X — Unified COCO128 Vision Data

## Goal
Use one reusable COCO128 Cloud pipeline for every current Studio image model that can consume COCO128 without inventing missing modalities, while removing redundant image demo datasets from the Data Gallery.

## Models now configured for COCO128 Cloud
- CNN
- Autoencoder
- Image JEPA
- Image Classifier
- YOLO-style Detector
- VESA-YOLO experimental

COCO128 remains a cloud-only source. The ZIP/extracted files use temporary session storage; no COCO images are packaged into the repository, wheel, or PyPI distribution.

## Shared prepared-data contract
The COCO128 Data Builder pipeline prepares RGB 128×128 images and preserves:
- `image`
- `boxes`
- `class_ids`
- `image_id`
- `label` — deterministic image-level class chosen from the largest annotated object

This lets one prepared dataset serve different training tasks:
- detection models use `image + boxes + class_ids`
- classifiers use `image + label`
- autoencoders use `image` as reconstruction input/target
- Image JEPA uses `image` and ignores supervised labels

## Gallery cleanup
Removed redundant dataset cards that are no longer needed:
- Image Classification Demo
- Image Reconstruction Demo
- Image JEPA Demo
- Vision Classification Demo
- Object Detection Demo
- Unlabeled Audio Demo (duplicate of Audio JEPA Demo)
- Unlabeled Signal Demo (duplicate of Signal JEPA Demo)

Task-specific tabular, language, audio, signal, video, and multimodal datasets remain because COCO128 cannot replace those modalities.

## Model defaults
Gallery image models now open already aligned to COCO128 where appropriate:
- RGB: 3 channels
- image size: 128×128
- supervised COCO heads: 80 classes

CNN now uses global average pooling so its classifier input stays compact at 128×128.
Autoencoder reconstruction geometry is 3×128×128.
Image JEPA encoders use 3 input channels.
YOLO-style Detector and VESA-YOLO use 80-class detection heads.

## Compatibility / auto-configuration
The old detector-only configuration action is generalized to image models. Older saved image graphs can be configured against the selected COCO128 prepared dataset for:
- image channels
- image size
- classifier/detection class count
- VESA image geometry
- JEPA image encoder channels
- autoencoder reconstruction dimensions

Changing model geometry invalidates existing weights and marks the model for rebuild.

## Validation
- Python compile: passed
- Frontend JavaScript syntax: passed
- Full pytest suite: **590 passed, 4 skipped, 0 failed**
