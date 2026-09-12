# Step 10W — Detection Dataset Auto-Configuration

Step 10W connects the reusable COCO128 Cloud Data Builder contract to editable detector graphs without hiding adaptation inside the runtime.

## Fixes

- Prepared detection pipelines now preserve `detection_processing` in Builder metadata.
- Detection datasets are registered as `modality=image`, `task=object_detection` when their pipeline/schema exposes `image + boxes + class_ids`.
- Frontend modality inference recognizes detection schemas before stale/legacy metadata, so existing COCO128 session datasets no longer appear as text.
- Built object-detection models expose **Configure Model for <Dataset>** when the selected dataset contract does not match the visible graph.
- Auto-configuration updates the visible graph:
  - Image Input channels
  - Image Input size
  - direct image-consuming layer input channels / image size where exposed
  - Detection Head class count
- Changing the graph invalidates old weights and marks the model `needs_rebuild`; the user explicitly clicks **Build** before training.
- Auto-configuration is intentionally disabled for VESA detector geometry until its spatial reshape can be safely recomputed.

## COCO128 example

The existing educational YOLO-style detector can be changed from:

- 1 channel
- 16×16 input
- 3 classes

to the COCO128 prepared-data contract:

- 3 RGB channels
- 128×128 input
- 80 classes

The change is visible in Model Builder and therefore preserves the Glass-Box Model Rule.

## Validation

- Full suite: **585 passed, 3 skipped**
- JavaScript syntax check passed.
- Python compile check passed.
