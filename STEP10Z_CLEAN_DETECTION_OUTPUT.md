# Step 10Z — Clean detection output

This patch reduces duplicate rectangles in the Studio detector preview without changing detector training or mAP evaluation semantics.

## Runtime preview defaults

- confidence threshold: `0.40`
- class-aware NMS IoU: `0.45`
- maximum displayed detections: `20`
- display-only class-agnostic duplicate suppression: enabled
- display duplicate IoU threshold: `0.60`

The second NMS pass is used only for the visual runtime result. It removes near-identical boxes that survived class-aware NMS because they were assigned different class IDs. It can be disabled with runtime metadata `display_class_agnostic_nms=false`.

The low-level `decode_detection_predictions()` defaults are intentionally unchanged so training/evaluation behavior remains backward compatible.
