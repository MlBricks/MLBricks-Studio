# Step 10U — Detection Preview Image Fix

- Preserves a higher-resolution pre-resize image preview for image runtime inputs.
- Detection and image-classification output prefer that preview instead of the tiny model tensor.
- Detection overlay stage expands up to 640 px and keeps the preview aspect ratio.
- Normalized detection boxes remain aligned with the displayed source preview.
- Raw detection JSON remains available only as an optional disclosure.
