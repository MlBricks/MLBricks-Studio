from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC_JS = ROOT / "src" / "mlb_studio" / "static" / "builder.js"
STATIC_CSS = ROOT / "src" / "mlb_studio" / "static" / "builder.css"
LEGACY_JS = ROOT / "frontend" / "src" / "legacy-builder.js"


def test_footer_never_renders_raw_runtime_errors():
    source = LEGACY_JS.read_text(encoding="utf-8")
    built = STATIC_JS.read_text(encoding="utf-8")
    for text in (source, built):
        assert "function footerLooksLikeError(value)" in text
        assert "function footerStatusText(value,run=execution)" in text
        assert 'String(run?.status||"").toLowerCase()==="error"' in text
        assert "out of memory|cuda error|cudnn error" in text
        assert 'stat.textContent="● "+footerStatusText(footerRaw,execution);' in text
        assert 'footerStatusText(status,execution)' in text
        assert 'stat.textContent="● "+footerStatus;' not in text


def test_statusbar_clamps_all_messages_to_single_line():
    css = STATIC_CSS.read_text(encoding="utf-8")
    assert ".mlb-statusbar .right{margin-left:auto;min-width:0;max-width:52%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}" in css
    assert ".mlb-statusbar>span{min-width:0;white-space:nowrap}" in css


def test_workshop_hides_build_and_fetch_actions():
    text = LEGACY_JS.read_text(encoding="utf-8")
    start = text.index("Workshop is a browsing surface")
    block = text[start:start + 5000]
    assert "if(!galleryWorkspace.open){" in block
    assert 'state.active_workspace==="model"' in block
    assert ':"Build"' in block
    assert ':"Fetch Data"' in block
    assert 'const galleryBtn=actionBtn("Workshop"' in block
