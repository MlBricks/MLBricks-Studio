from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REACT = ROOT / "frontend" / "src" / "react-runtime.js"
LEGACY = ROOT / "frontend" / "src" / "legacy-builder.js"
CSS = ROOT / "frontend" / "src" / "builder.css"
STATIC = ROOT / "src" / "mlb_studio" / "static" / "builder.js"


def test_training_log_uses_compact_metric_columns():
    react = REACT.read_text(encoding="utf-8")
    assert "TrainingEventLog" in react
    assert "Tok/s" in react
    assert "E2E Tok/s" in react
    assert "mlb-training-log-metric-row" in react
    assert "h(TrainingEventLog,{store:store" in react


def test_training_log_fallback_uses_same_columns():
    legacy = LEGACY.read_text(encoding="utf-8")
    assert "renderTrainingEventLog" in legacy
    assert '["Status","Tok/s","E2E Tok/s","Loss","PPL"]' in legacy
    assert "renderTrainingEventLog(logs,history" in legacy


def test_training_log_has_stable_table_layout_and_built_asset():
    css = CSS.read_text(encoding="utf-8")
    assert ".mlb-training-log-head,.mlb-training-log-metric-row" in css
    assert "font-variant-numeric:tabular-nums" in css
    built = STATIC.read_text(encoding="utf-8")
    assert "MLBricks Studio compiled frontend" in built
