from pathlib import Path


def test_studio_scrollbars_use_dark_theme_globally():
    css = (Path(__file__).parents[1] / "src" / "mlb_studio" / "static" / "builder.css").read_text(encoding="utf-8")

    assert ".mlb-root *::-webkit-scrollbar{" in css
    assert ".mlb-root *::-webkit-scrollbar-thumb{" in css
    assert ".mlb-root *::-webkit-scrollbar-track{" in css
    assert "scrollbar-color:#435667 #0a1117;" in css
    assert "background:#435667;" in css
    assert "background:#596f83;" in css
    assert "background:#6d59bf;" in css
