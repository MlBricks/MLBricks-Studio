# MLBricks Studio — Step 4 Vision Foundation

This patch builds on Step 3 and adds the first complete Vision family foundation while keeping Gallery models glass-box and rebuildable from public Studio components.

## Data Gallery

- Vision category keeps Image Classification Demo and Object Detection Demo.
- Image Classification Demo now has explicit image metadata for the offline 16×16 grayscale demo.
- Object Detection Demo now carries image, `boxes`, and `class_ids` fields using xywh pixel coordinates.
- `Image Processing` can emit tensor-ready normalized image data.
- New public `Detection Processing` component resizes images and bounding boxes together and preserves detection labels.
- Data pipeline snapshots, code preview, and the prepared-data inspector understand the new vision processors.

## Public Vision components

- `FPN Fusion` — two named inputs (`high`, `lateral`) and an editable feature-pyramid fusion block.
- `PAN Fusion` — two named inputs (`fine`, `coarse`) and an editable path-aggregation fusion block.
- `Detection Head` — configurable channels, classes, and anchors.
- Existing public Conv2D, BatchNorm2D, SiLU, pooling, classifier, reshape, Image Input, and VESA components are reused rather than hidden inside a special model runtime.

## Vision Model Gallery

- **Image Classifier** — Conv/BN/SiLU/Pool stack with global pooling and a public classifier head.
- **YOLO-style Detector** — visible CNN backbone + P3/P4 feature paths + FPN + PAN + Detection Head.
- **VESA-YOLO experimental** — VESA high-level visual projection + CNN lateral feature path + FPN/PAN + Detection Head. This model uses the MLBricks VESA component and therefore requires the corresponding `mlbricks-kit` runtime to execute.

All three are ordinary editable graph templates and can be cloned/modified with the same Studio component system.

## Training runtime

- Generic supervised training now recognizes `object_detection`.
- Detection batches accept image + boxes + class IDs.
- The educational detector target is normalized `[cx, cy, w, h, class]` for one-object-per-demo-image training.
- Detection loss combines box regression, objectness, and class loss.
- Training UI reports class accuracy and normalized bounding-box MAE.
- Existing classification/regression/reconstruction training behavior is preserved.

## Validation

- Added Step 4 vision regression tests for catalog exposure, detection preprocessing, FPN/PAN graph execution, detection loss/targets, one-step training, and Gallery/Data wiring.
- Updated the component release gate from 99 to 103 public Studio components.
- Frontend distribution rebuilt.
- Full suite: **458 passed, 2 skipped, 0 failed**.

## Scope note

The YOLO-style detector in this Step 4 patch is intentionally an educational, glass-box detector rather than a claim of full production YOLO parity. Multi-object assignment, anchors/multi-scale heads, IoU-family losses, NMS, mAP evaluation, richer detection augmentation, and production inference/export are suitable follow-on Vision steps.
