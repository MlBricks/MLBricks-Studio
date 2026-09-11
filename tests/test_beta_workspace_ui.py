from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / 'src/mlb_studio/static/builder.js').read_text(encoding='utf-8')
CSS = (ROOT / 'src/mlb_studio/static/builder.css').read_text(encoding='utf-8')


def test_visible_studio_brand_is_beta():
    assert 'versionBadge.textContent="BETA"' not in JS
    assert 'MLB Studio Beta' not in JS
    assert '<title>MLB Studio</title>' in JS


def test_build_workspace_uses_persistent_dual_builder_tabs():
    assert 'mlb-workspace-buttons' in JS
    assert 'mlb-workspace-tab' in JS
    assert 'Model Builder' in JS
    assert 'Data Builder' in JS
    assert '.mlb-workspace-tab.active' in CSS


def test_build_workspace_remains_visible_while_gallery_is_open():
    assert 'current(state)?.kind!=="custom_edit" && !galleryWorkspace.open' not in JS
    assert 'if(current(state)?.kind!=="custom_edit"){' in JS
    assert 'function closeGallery(){' in JS
    assert 'galleryWorkspace.open=false;' in JS


def test_build_workspace_switcher_has_no_dropdown_or_caret():
    assert 'workspaceTrigger.textContent=activeWorkspaceLabel;' not in JS
    assert 'workspaceMenu.hidden' not in JS[JS.index('// Model Builder and Data Builder are persistent sibling workspaces.'):JS.index('if(!galleryWorkspace.open){', JS.index('// Model Builder and Data Builder are persistent sibling workspaces.'))]
    assert "mlb-workspace-caret'>⌄" not in JS
