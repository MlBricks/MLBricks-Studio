from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def test_busy_runtime_header_reopens_status_page():
    source = _read("frontend/src/legacy-builder.js")
    assert 'run.disabled=dataFetchBusy;' in source
    assert 'Open live Training Status' in source
    assert 'Open live Generation Status' in source
    assert 'runtimePanel={mode,modelId:entry.id,tab:"status"};' in source
    assert 'builtModelById(execution.model_id)||builtModelById(outputDirectorySelection)||null' in source


def test_compiled_bundle_contains_runtime_navigation_fix():
    built = _read("src/mlb_studio/static/builder.js")
    assert 'Open live Training Status' in built
    assert 'Open live Generation Status' in built
