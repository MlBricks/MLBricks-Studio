# Step 10N — Flat Gallery Navigation

This patch removes the redundant nested Gallery hierarchy.

## New Gallery navigation

The Gallery now has one peer-level navigation row:

- Core
- Models
- My Models
- Components
- Data
- Drafts

There is no longer a parent `Models` tab containing a second `Core / Models / My Models` row.

## Behavior

- Workshop opens directly to `Core` from Model Builder and `Data` from Data Builder.
- Core keeps its `All Core` / ML / DL / Signal Processing filter.
- Models keeps its model-family filter.
- My Models directly shows user-saved models.
- Load, Export, Bundle, and Save Current actions move into the Gallery header so the navigation row stays clean.
- Opening a model/data preset still loads it into the corresponding persistent builder workspace.

## Validation

- Full pytest suite: 557 passed, 2 skipped.
- `node --check frontend/src/legacy-builder.js`: passed.
- `node --check src/mlb_studio/static/builder.js`: passed.
