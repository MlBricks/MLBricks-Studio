# Step 10Y — Visual Autoencoder Reconstruction Output

## Problem
Autoencoder inference returned the reconstructed image tensor as thousands of scalar values in the runtime output panel.

## Fix
- Detect image-shaped Tensor Output results that exactly match the image model input shape.
- Return a semantic `reconstruction` runtime envelope instead of a raw tensor dump.
- Encode the model input and reconstructed output as PNG previews.
- Render Model Input and Reconstruction side by side in both React and legacy runtime renderers.
- Show MSE, MAE, and PSNR for the reconstruction.
- Keep non-image Tensor Output behavior unchanged.

## Validation
- Full pytest suite: 593 passed, 4 skipped.
- Frontend bundle rebuilt successfully.
