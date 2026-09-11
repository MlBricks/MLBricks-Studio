# Step 10G — Frontend Asset Refresh Hotfix

This hotfix fixes notebook/local Studio sessions continuing to render an older `builder.js` after an in-place package patch.

## Root cause

MLBricks Studio cached the compressed frontend bundle for the lifetime of the Python process and also recorded that frontend assets had already been emitted once. Updating files on disk while the same notebook kernel remained alive therefore did not replace the JavaScript already loaded in the page.

## Fix

- Track a CSS/JS asset signature from file modification time and size.
- Rebuild the compressed frontend bundle when installed assets change.
- Re-emit notebook frontend assets when their signature changes.
- Add `refresh_frontend_assets()` for an explicit one-call reset.

This is especially important for Studio development and editable installs where patches are applied while a Jupyter/Kaggle/local notebook kernel is still running.
