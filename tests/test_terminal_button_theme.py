from pathlib import Path


def test_custom_terminal_buttons_use_dark_studio_theme():
    css = (Path(__file__).parents[1] / "src" / "mlb_studio" / "static" / "builder.css").read_text(encoding="utf-8")

    assert ".mlb-terminal-move{" in css
    assert ".mlb-terminal-move:disabled{" in css
    assert "background:#101821!important;" in css
    assert "opacity:1!important;" in css

    assert ".mlb-custom-arg-remove{" in css
    assert "background:#26161b!important;" in css
    assert "border:1px solid #6a3b45!important;" in css

    assert ".mlb-custom-add-arg{" in css
    assert "background:#121824!important;" in css
    assert "border:1px dashed #6d58cf!important;" in css
