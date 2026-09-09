from pathlib import Path
import json

from mlb_studio.builder import Builder


def _builder_js() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")


def test_fetch_data_prefers_atomic_state_and_command_bridge():
    js = _builder_js()
    start = js.index("function requestRun()")
    end = js.index("function requestStop()", start)
    block = js[start:end]
    assert 'const command={action:"data",ts:Date.now()}' in block
    assert "dispatchAtomicDataRequest(command)" in block
    assert "state + command travel through ONE observed widget" in block
    # Keep an explicit compatibility fallback for older Studio outputs.
    assert "setBridgeCommand(command)" in block


def test_full_window_forwards_data_through_atomic_request_bridge():
    js = _builder_js()
    assert 'type:"atomic_data"' in js
    assert 'bridge.request,"textarea"' in js
    assert 'request:"__popout_request__"' in js


def test_python_atomic_data_envelope_keeps_state_and_command_together(tmp_path, monkeypatch):
    monkeypatch.setenv("MLBRICKS_STUDIO_HOME", str(tmp_path))
    builder = Builder()

    class Dummy:
        def __init__(self, value=""):
            self.value = value

    state_widget = Dummy(json.dumps(builder.to_dict()))
    command_widget = Dummy("{}")
    progress_widget = Dummy("")
    builder._bridge_widgets = {
        "state": state_widget,
        "command": command_widget,
        "progress": progress_widget,
    }
    captured = []
    builder._start_bridge_run = lambda: captured.append(
        (json.loads(state_widget.value), json.loads(command_widget.value))
    )

    incoming = builder.to_dict()
    incoming["project"]["name"] = "Atomic Data Fetch"
    raw = json.dumps({
        "request_id": "req-1",
        "state": incoming,
        "command": {"action": "data", "ts": 123},
    })

    assert builder._dispatch_bridge_request_envelope(raw) is True
    assert captured[0][0]["project"]["name"] == "Atomic Data Fetch"
    assert captured[0][1]["action"] == "data"
    progress = json.loads(progress_widget.value)
    assert progress["runtime_kind"] == "data"
    assert progress["phase"] == "dispatch"


def test_background_component_imports_do_not_overwrite_active_runtime_command():
    js = _builder_js()
    start = js.index("function pumpComponentImportQueue()")
    end = js.index("function requestRuntimeCommand", start)
    block = js[start:end]
    assert 'execution.status==="running"' in block
