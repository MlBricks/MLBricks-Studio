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


def test_feature_schema_precedes_stale_declared_modality():
    for path in FILES:
        block = _modality_block(path.read_text(encoding="utf-8"))
        feature_pos = block.index('/^feature_\\d+$/')
        declared_pos = block.index('const declared=String(meta?.modality||meta?.data_modality||"")')
        assert feature_pos < declared_pos, path


def test_tabular_schema_beats_signal_processing_fallback():
    for path in FILES:
        block = _modality_block(path.read_text(encoding="utf-8"))
        feature_pos = block.index('/^feature_\\d+$/')
        signal_fallback_pos = block.index('if(p.signal_processing)return "signal"')
        assert feature_pos < signal_fallback_pos, path


def test_demo_type_still_has_highest_priority():
    for path in FILES:
        block = _modality_block(path.read_text(encoding="utf-8"))
        demo_pos = block.index('"tabular_regression"')
        feature_pos = block.index('/^feature_\\d+$/')
        assert demo_pos < feature_pos, path
