# MLBricks Studio — Step 9 Advanced Studio Tooling

## Implemented

- Learn / Build / Research workspace modes.
- Graph contract validation for model and data workspaces.
- Transparent graph profiling with visible parameter estimates.
- Explain Model / Explain Graph flow tracing.
- Data Gallery Inspector with source, schema, pipeline and compatible-model contracts.
- Research experiment snapshots and last-two comparison.
- Reproducible Project Bundle export/import containing model graph, data graph, training recipes/configs, custom components, experiments and Gallery metadata. Raw dataset/model bytes remain external and are referenced by metadata.
- Python diagnostics API (`analyze_graph_contract`, `profile_graph`, `compare_experiments`, `make_project_bundle`, `load_project_bundle`).

## Design boundary

Step 9 profiling is a graph/configuration estimate before runtime execution. Actual runtime throughput, VRAM and task metrics remain authoritative once a model has been built/trained. The Project Bundle intentionally stores reproducible configuration/metadata rather than copying potentially huge dataset/checkpoint binaries into the browser JSON.
