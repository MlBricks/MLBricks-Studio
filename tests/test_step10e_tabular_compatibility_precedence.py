from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "frontend" / "src" / "legacy-builder.js",
    ROOT / "src" / "mlb_studio" / "static" / "builder.js",
]


def _modality_block(text: str) -> str:
    start = text.index("function datasetModality(meta)")
    end = text.index("function datasetTrainingCapabilities", start)
    return text[start:end]


def test_demo_contract_precedes_signal_processing_fallback():
    for path in FILES:
        block = _modality_block(path.read_text(encoding="utf-8"))
        demo_pos = block.index('"tabular_regression"')
        signal_fallback_pos = block.index('if(p.signal_processing)return "signal"')
        assert demo_pos < signal_fallback_pos, path


def test_feature_columns_can_recover_tabular_modality():
    for path in FILES:
        block = _modality_block(path.read_text(encoding="utf-8"))
        assert '/^feature_\\d+$/' in block
        assert 'return "tabular"' in block


def test_detection_processing_is_image_fallback():
    for path in FILES:
        block = _modality_block(path.read_text(encoding="utf-8"))
        assert 'p.image_processing||p.detection_processing' in block
