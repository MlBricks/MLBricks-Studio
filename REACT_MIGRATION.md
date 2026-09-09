# React migration status

This build introduces the React frontend foundation without replacing the
working Python model/runtime stack.

## Migrated now

- Training Status is rendered by React.
- Generation Status is rendered by React.
- Runtime progress is stored per Studio/model/mode and pushed to subscribed
  components only.
- Generated text updates in place, preserving the output DOM node and scroll
  container while tokens stream.
- Training metrics update in place instead of rebuilding the Studio root.
- Training/generation logs reconcile incrementally.
- Runtime control buttons are React-owned on status pages.
- Full Window/about:blank receives the same React bootstrap, so it does not
  silently fall back to the old status renderer.
- The compiled `builder.js` is self-contained and ships inside the Python wheel.

## Still legacy during the staged migration

- Visual model/data graph editor.
- Workshop / local repository screens.
- Cloud screens.
- Runtime setup forms and API-server pages.

These areas are intentionally kept in the existing shell for feature parity
while they are moved component-by-component. The hot training/generation loops
no longer depend on shell redraws.

## Build

From the repository root:

```powershell
cd frontend
npm run build
```

The build has no external npm dependencies. React and ReactDOM production
browser bundles are vendored under `frontend/vendor` and the output is written
to `src/mlb_studio/static/builder.js` / `builder.css`.

## Distribution policy

`frontend/` is development-only source. It is intentionally excluded from Python
source/wheel distributions. GitHub installs use the already-built files under
`src/mlb_studio/static/`; installing MLBricks Studio never runs Node/npm and does
not install the frontend source tree.
