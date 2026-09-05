from pathlib import Path


def _builder_js() -> str:
    root = Path(__file__).resolve().parents[1]
    return (root / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")


def test_fetch_data_explicitly_selects_data_bridge_command():
    js = _builder_js()
    start = js.index("function requestRun()")
    end = js.index("function requestStop()", start)
    block = js[start:end]
    assert 'setBridgeCommand({action:"data",ts:Date.now()})' in block
    assert "explicitly select the data command" in block


def test_background_component_imports_do_not_overwrite_active_runtime_command():
    js = _builder_js()
    start = js.index("function pumpComponentImportQueue()")
    end = js.index("function requestRuntimeCommand", start)
    block = js[start:end]
    assert 'execution.status==="running"' in block
