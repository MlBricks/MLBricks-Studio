# Step 10I — Recurrent Sequence Modality Hotfix

Fixes training-data compatibility for the RNN, LSTM, and GRU Gallery models.

- Recurrent graphs backed by Feature Input are classified as `sequence`, not generic `tabular`.
- `Sequence Classification Demo` resolves to the `sequence` modality instead of `signal`.
- Existing built/autosaved RNN/LSTM/GRU entries whose stored requirement says `tabular` are automatically repaired from the saved recurrent architecture.
- Signal datasets remain `signal`; ordinary feature models remain `tabular`.
- The fix is applied to both the editable frontend source and the compiled Studio frontend asset.
