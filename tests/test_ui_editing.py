from __future__ import annotations

from pathlib import Path


def _builder_js() -> str:
    return (Path(__file__).resolve().parents[1] / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")


def test_redraw_is_deferred_while_an_editor_has_focus():
    text = _builder_js()
    assert "let focusedEditorActive=false;" in text
    assert 'root.addEventListener("focusin"' in text
    assert 'root.addEventListener("focusout"' in text
    assert "if(pointerInteractionActive || focusedEditorActive)" in text


def test_model_settings_do_not_reject_intermediate_width_head_values():
    text = _builder_js()
    assert 'if(key==="embedding_size" && next.embedding_size%next.heads!==0)' not in text
    assert 'if(key==="heads" && next.embedding_size%next.heads!==0)' not in text
    assert "must be divisible by Heads before Build" in text


def test_build_validation_catches_incompatible_width_and_heads():
    text = _builder_js()
    assert "Number(settings.embedding_size)%Number(settings.heads)!==0" in text
    assert 'message:"Embedding Size ("+settings.embedding_size+") must be divisible by Heads' in text
