# Step 10F — Tabular Schema Precedence Hotfix

Fixes a remaining compatibility issue where an older prepared dataset could retain a stale top-level `modality: signal` value even though the prepared train split exposes tabular `feature_0`, `feature_1`, ... columns.

`datasetModality()` now resolves modality in this order:

1. Canonical demo/source contract (`demo_type`)
2. Concrete prepared-data schema (`feature_N` => `tabular`)
3. Declared legacy modality metadata
4. Processing-node fallbacks

This makes the actual prepared dataset schema authoritative over stale autosaved modality labels while preserving real signal datasets.
