from pathlib import Path


def _read(rel):
    root = Path(__file__).resolve().parents[1]
    return (root / rel).read_text(encoding="utf-8")


def test_rnn_lstm_gru_builds_are_sequence_models():
    for rel in ["frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"]:
        js = _read(rel)
        start = js.index("function inferModelRequirements(model)")
        end = js.index("function validateModelBuild", start)
        block = js[start:end]
        assert 'types.has("rnn")||types.has("lstm")||types.has("gru")' in block
        assert 'modality="sequence"' in block


def test_sequence_demo_resolves_to_sequence_not_signal():
    for rel in ["frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"]:
        js = _read(rel)
        start = js.index("function datasetModality(meta)")
        end = js.index("function datasetTrainingCapabilities", start)
        block = js[start:end]
        assert '["sequence_classification"].includes(demo))return "sequence";' in block
        assert '["tabular","sequence","image","audio","video","signal","text","multimodal"]' in block


def test_legacy_recurrent_builds_auto_heal_for_compatibility():
    for rel in ["frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"]:
        js = _read(rel)
        start = js.index("function modelCompatibilityModality(modelEntry)")
        end = js.index("function modelDatasetCompatibility", start)
        block = js[start:end]
        assert 'modelEntry?.architecture?.nodes' in block
        assert '(modality==="tabular"||modality==="unknown")' in block
        assert 'return "sequence";' in block
        compat_start = js.index("function modelDatasetCompatibility", end)
        compat_end = js.index("function setBuiltModelDataset", compat_start)
        compat = js[compat_start:compat_end]
        assert 'const modelModality=modelCompatibilityModality(modelEntry);' in compat
        assert 'modelModality===modality' in compat
