from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step10j_model_gallery_splits_core_models_and_saved_models():
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert '[["core","Core"],["models","Models"],["mine","My Models"],["components","Components"],["data","Data"],["drafts","Drafts"]]' in text
        assert 'const mlbricksCoreCategories=["All Core","Machine Learning","Deep Learning","Signal Processing"]' in text
        assert 'const mlbricksModelCategories=["All Models","Language","Vision","Audio","JEPA","Multimodal","State & Memory"]' in text
        assert 'mlb-model-gallery-filter-select' in text
        assert 'mlb-gallery-flat-nav-tools' in text
        assert 'mlbricksCorePresetCategory=(preset)=>preset?.category==="Signal"?"Signal Processing"' in text


def test_step10j_core_contains_fundamentals_but_not_research_signal_models():
    text = (ROOT / "frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    for name in ("Linear Regression", "Logistic Regression", "KNN", "Decision Tree", "K-Means", "PCA",
                 "Single Neuron", "ANN", "CNN", "RNN", "LSTM", "GRU", "Autoencoder",
                 "Time-Series Predictor", "Signal Classifier", "Anomaly Detector", "Signal Denoiser",
                 "Sensor Fusion", "Spectral Model", "RF/IQ Model"):
        assert f'name:"{name}"' in text
    soup_line = next(line for line in text.splitlines() if 'name:"SOUP Signal"' in line)
    assert 'category:"State & Memory"' in soup_line
    jepa_line = next(line for line in text.splitlines() if 'name:"Signal JEPA"' in line)
    assert 'category:"JEPA"' in jepa_line
    assert 'also_categories:["Signal"]' not in jepa_line


def test_step10j_gallery_css_retains_core_model_styling_compatibility():
    for rel in ("frontend/src/builder.css", "src/mlb_studio/static/builder.css"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert ".mlb-model-gallery-filter-select" in text
        assert ".mlb-gallery-flat-tabs" in text
