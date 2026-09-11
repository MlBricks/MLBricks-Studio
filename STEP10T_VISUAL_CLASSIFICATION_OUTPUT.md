# Step 10T — Visual Classification Output

Image classifiers such as CNN now render a visual classification result instead of exposing raw logits as the primary output.

- Converts logits to probabilities in the runtime contract.
- Reports predicted class and confidence.
- Image classifiers include a preview of the processed model input.
- UI shows the image, prediction, confidence and probability bars.
- Raw logits remain available in a collapsed diagnostic section.
- Binary one-logit classifiers use sigmoid probabilities.
- Non-image classifiers receive the same semantic class/probability output without an image preview.
