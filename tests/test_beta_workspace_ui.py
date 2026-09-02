from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / 'src/mlb_studio/static/builder.js').read_text(encoding='utf-8')
CSS = (ROOT / 'src/mlb_studio/static/builder.css').read_text(encoding='utf-8')


def test_visible_studio_brand_is_beta():
    assert 'versionBadge.textContent="BETA"' not in JS
    assert 'MLB Studio Beta' not in JS
    assert '<title>MLB Studio</title>' in JS


def test_build_workspace_uses_themed_picker():
    assert 'mlb-workspace-trigger' in JS
    assert 'mlb-workspace-menu' in JS
    assert 'mlb-workspace-option' in JS
    assert '.mlb-workspace-option.active' in CSS
    assert 'background:#2878d4' in CSS
