from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_zoom_cluster_is_centered_beneath_top_switcher():
    for rel in ("frontend/src/builder.css", "src/mlb_studio/static/builder.css"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "Step 10R — center canvas zoom controls" in text
        assert ".mlb-toolbar>.mlb-zoom{" in text
        assert "position:absolute!important;" in text
        assert "left:50%!important;" in text
        assert "top:50%!important;" in text
        assert "transform:translate(-50%,-50%)!important;" in text
