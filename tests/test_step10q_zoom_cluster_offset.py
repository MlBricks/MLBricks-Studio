from pathlib import Path


def test_zoom_cluster_moves_left_independently_of_file_actions():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "frontend/src/builder.css",
        "src/mlb_studio/static/builder.css",
    ]:
        text = (root / rel).read_text(encoding="utf-8")
        assert ".mlb-toolbar>.mlb-zoom{" in text
        assert "transform:translateX(-12px)!important;" in text
        assert "margin-right:-11px!important;" not in text


def test_save_action_remains_compact():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "frontend/src/legacy-builder.js",
        "src/mlb_studio/static/builder.js",
    ]:
        text = (root / rel).read_text(encoding="utf-8")
        assert 'const saveLabel="Save";' in text
