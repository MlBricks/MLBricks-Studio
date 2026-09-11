# Step 10H — Image Runtime Input Hotfix

This cumulative hotfix fixes the CNN/image runtime path exposed by the Studio Runtime panel.

## Fixes

- Image-search result URLs such as Google `imgres?...&imgurl=...` are unwrapped to the embedded direct image URL before download.
- HTTP image loading sends an image-aware request header and rejects HTML/search pages with an actionable error instead of a raw Pillow `UnidentifiedImageError`.
- Image decoding now converts to the model's declared/inferred channel contract (1-channel grayscale or 3-channel RGB).
- The runtime derives image size + channels from the built graph. Older Gallery CNN/Autoencoder builds whose saved Image Input still says `3 × 224` are repaired from downstream Conv/Flatten/head constraints and run as `1 × 16 × 16` when that is what the graph requires.
- New CNN and Autoencoder Gallery templates explicitly declare `channels=1, image_size=16`.
- Prompt/Instruction is no longer shown or sent for ordinary CNN image analyze/classify/detect tasks. Prompts remain available where meaningful (text, image edit/caption, video caption, multimodal generation).
- React runtime status also suppresses stale prompt text for non-prompt tasks.
- Image runtime setup now explains that local image paths/direct image URLs are accepted and ordinary web/search pages are not image files.

## Validation

- Full test suite: `540 passed, 2 skipped`.
- Python compile check passes.
- `node --check frontend/src/legacy-builder.js` passes.
- `node --check frontend/src/react-runtime.js` passes.
- `node --check src/mlb_studio/static/builder.js` passes.
