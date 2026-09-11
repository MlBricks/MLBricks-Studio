# Step 10S — Visual Detection Output

- Object-detection inference now returns a dedicated `detection` output envelope instead of presenting detections as generic JSON.
- The processed model input image is embedded as a PNG preview so local paths and remote URLs render consistently.
- Studio overlays normalized XYXY boxes directly on the image with class ID/name and confidence.
- Detection count and compact result rows are shown under the image.
- Raw JSON remains available in a collapsed **Raw detection data** disclosure for debugging/research.
- React runtime and legacy fallback renderer both support the visual detection contract.
