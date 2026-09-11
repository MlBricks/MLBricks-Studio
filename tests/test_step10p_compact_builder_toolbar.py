from pathlib import Path


def _texts():
    root = Path(__file__).resolve().parents[1]
    return [
        (root / "frontend" / "src" / "legacy-builder.js").read_text(encoding="utf-8"),
        (root / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8"),
    ], [
        (root / "frontend" / "src" / "builder.css").read_text(encoding="utf-8"),
        (root / "src" / "mlb_studio" / "static" / "builder.css").read_text(encoding="utf-8"),
    ]


def test_builder_save_label_is_compact():
    js_files, _ = _texts()
    for text in js_files:
        assert 'const saveLabel="Save";' in text
        assert '+ Save Current Model' not in text
        assert '+ Save Current Data' not in text
        assert 'saveCurrentToGallery' in text


def test_zoom_cluster_is_nudged_left_without_reordering_actions():
    js_files, css_files = _texts()
    for text in js_files:
        assert text.index('z.append(fit,zm,zs,zp);toolbar.appendChild(z);') < text.index('const loadAction=btn("⇧ Load"')
    for text in css_files:
        assert '.mlb-toolbar>.mlb-zoom{' in text
        assert 'transform:translateX(-12px)!important;' in text
        assert 'margin-right:-11px!important;' not in text
