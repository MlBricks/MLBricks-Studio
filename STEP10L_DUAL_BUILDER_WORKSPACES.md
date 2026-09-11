# Step 10L — Dual Builder Workspaces

- Removes the Learn / Build / Research mode switch from the canvas toolbar.
- Keeps one direct Builder workflow with diagnostics, Explain Graph, experiments and bundles still available.
- Replaces the workspace dropdown with two persistent visible workspaces: **Model Builder** and **Data Builder**.
- Switching workspaces preserves both canvases. Data fetching can continue while the user works on the model canvas.
- Renames the old Data Processing workspace to Data Builder, including migration of saved Studio state.
- Model config Load now replaces the current Model Builder canvas directly.
- Data workspace exports Load directly into Data Builder.
- Export remains context-aware; Bundle exports the whole project.
- A model build that finishes after the user switches to Data Builder no longer steals workspace focus.
