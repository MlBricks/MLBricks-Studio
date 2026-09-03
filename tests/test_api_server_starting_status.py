from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_api_server_status_renderer_has_starting_state():
    js = (_root() / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")
    assert 'entry.serve_status==="starting"' in js
    assert '"◌ STARTING"' in js
    assert 'starting?"Starting"' in js
    assert 'action==="serve_start"' in js


def test_api_server_starting_state_has_distinct_style():
    css = (_root() / "src/mlb_studio/static/builder.css").read_text(encoding="utf-8")
    assert ".mlb-serve-status.starting" in css


def test_server_start_progress_publishes_starting_model_update():
    source = (_root() / "src/mlb_studio/builder.py").read_text(encoding="utf-8")
    assert 'entry["serve_status"] = "starting"' in source
    assert '"model_update":{"serve_status":"starting","serve_urls":{}}' in source.replace(" ", "")
    assert 'runtime_kind = "serve" if str(action).startswith("serve_") else action' in source
