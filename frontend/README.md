# MLBricks Studio frontend

The shipped Python package uses a **compiled browser bundle** at
`src/mlb_studio/static/builder.js` and `builder.css`. Users do not need Node.js,
npm, React, or a CDN when they run `pip install mlbricks-studio`.

## Architecture

MLBricks Studio is being migrated from the original procedural DOM renderer to
React without rewriting the Python runtime. The first migration boundary owns
the hot **Training Status** and **Generation Status** pages with React runtime
islands and a scoped external store. Runtime events update subscribed metric,
text, log, and control components only; they do not redraw the Studio shell,
workspace canvas, sidebar, or focused editor controls.

The visual graph/editor and low-frequency setup screens remain in the legacy
shell during the staged migration. They are compiled into the same browser file,
so there is still one frontend asset loaded by `Builder`.

## Build

The build is intentionally offline/deterministic:

```powershell
cd frontend
npm run build
```

There are no npm runtime dependencies. The repository vendors the MIT-licensed
React 16 browser production bundles under `frontend/vendor/`, then
`frontend/build.mjs` combines React, the runtime islands, and the existing graph
shell into `src/mlb_studio/static/builder.js` and copies the stylesheet.

This keeps Python package installation independent of npm while allowing the
frontend source to evolve separately from the shipped artifact.
