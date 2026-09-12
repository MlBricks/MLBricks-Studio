# Step 10Y — Runtime Semantics & CUDA Label Guard

This patch fixes three issues exposed after the unified COCO128 vision pipeline:

1. **COCO classification CUDA device-side asserts**
   - COCO images without a usable annotation use the documented `-1` sentinel.
   - Classification preparation now filters only those unlabeled samples before batching.
   - Remaining labels are range-checked on CPU before CUDA `CrossEntropyLoss` can run.
   - A second defensive loss-side range check surfaces a readable Studio error rather than poisoning the CUDA context.

2. **Image JEPA raw scalar output**
   - Image JEPA runtime now returns a semantic `jepa` output envelope.
   - Studio renders the input image together with `Latent Prediction Loss` and an explicit `Lower is better` explanation.
   - The raw scalar is still available in the analysis details.

3. **COCO detector class names missing at runtime**
   - Universal input runtime now carries `class_names`, training mode/task and model name from the model entry into inference metadata.
   - Loaded MLBricks model artifacts restore `class_names`/`num_classes` from artifact or dataset metadata.
   - Detection output can therefore render `dog`, `person`, etc. rather than generic `Class 16` when the model carries COCO metadata.

After a CUDA device-side assertion, restart the Python process before retrying training because the CUDA context may remain in an error state.
