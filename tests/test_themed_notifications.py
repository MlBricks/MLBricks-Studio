from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_native_browser_dialogs_are_replaced_with_studio_modals():
    source = (ROOT / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    assert "function studioAlert" in source
    assert "function studioConfirm" in source
    assert "function studioPrompt" in source
    assert "function studioChoice" in source
    assert "win.confirm(" not in source
    assert "win.prompt(" not in source
    assert "win.alert(" not in source


def test_notebook_keeps_artifact_confirmation_when_it_has_focus():
    source = (ROOT / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    assert "function studioSurfaceHasFocus" in source
    assert "popoutPeerConnected&&!studioSurfaceHasFocus()" in source
    assert "sendPopoutMessage({type:\"progress\"" in source


def test_modal_is_scoped_to_studio_bounds_without_background_blur():
    source = (ROOT / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8")
    css = (ROOT / "frontend" / "src" / "builder.css").read_text(encoding="utf-8")
    assert "positionStudioModalOverlay" in source
    assert "root.getBoundingClientRect()" in source
    assert "backdrop-filter" not in css
    assert ".mlb-modal-backdrop{position:fixed" in css
    assert "background:rgba(2,7,13,.42)" in css
