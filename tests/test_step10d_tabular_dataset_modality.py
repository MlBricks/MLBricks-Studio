from pathlib import Path


def _js(path):
    return Path(path).read_text(encoding="utf-8")


def test_tabular_demo_datasets_resolve_to_tabular_modality():
    root = Path(__file__).resolve().parents[1]
    for rel in ["frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"]:
        js = _js(root / rel)
        start = js.index("function datasetModality(meta)")
        end = js.index("function datasetTrainingCapabilities", start)
        block = js[start:end]
        assert '"tabular_regression","neuron_regression","binary_classification","multiclass_classification"' in block
        assert '"tabular_classification","high_dimensional","clustering"' in block
        assert '].includes(demo))return "tabular";' in block
        # Sequence/signal demos must remain signal, not be folded into tabular.
        assert '"sequence_classification","signal_jepa","signal_classification","anomaly_detection"' in block
        assert '].includes(demo))return "signal";' in block


def test_linear_gallery_and_tabular_regression_can_pass_modality_check():
    root = Path(__file__).resolve().parents[1]
    js = _js(root / "src/mlb_studio/static/builder.js")
    assert 'name:"Tabular Regression Demo",category:"Machine Learning",modality:"Tabular"' in js
    assert 'name:"Linear Regression"' in js
    assert 'else if(types.has("feature_input"))modality="tabular";' in js
