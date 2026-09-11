from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS_FILES = ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js")
CSS_FILES = ("frontend/src/builder.css", "src/mlb_studio/static/builder.css")


def test_workspace_switcher_is_visible_even_when_gallery_is_open():
    for rel in JS_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'if(current(state)?.kind!=="custom_edit"){' in text
        assert 'if(current(state)?.kind!=="custom_edit" && !galleryWorkspace.open){' not in text
        assert 'workspaceLabel.textContent="BUILD WORKSPACE"' in text
        assert '[["model","Model Builder"],["data","Data Builder"]]' in text


def test_workspace_switcher_is_compact_and_has_busy_indicator():
    for rel in JS_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'option.textContent=label' in text
        assert 'option.setAttribute("data-busy","true")' in text
        assert 'modelCaption' not in text[text.index('// Model Builder and Data Builder are persistent sibling workspaces.'):text.index('if(!galleryWorkspace.open){', text.index('// Model Builder and Data Builder are persistent sibling workspaces.'))]

    for rel in CSS_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert '.mlb-workspace-tab{position:relative;width:100%;min-height:36px' in text
        assert '.mlb-workspace-tab.busy:after' in text
        assert '.mlb-studio-mode-switch{' not in text
        assert '.mlb-studio-mode-btn{' not in text
