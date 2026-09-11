from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS_FILES = ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js")
CSS_FILES = ("frontend/src/builder.css", "src/mlb_studio/static/builder.css")


def test_step10l_exposes_two_persistent_builder_workspaces_and_removes_mode_switch():
    for rel in JS_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert '[["model","Model Builder"],["data","Data Builder"]]' in text
        assert 'workspaceButtons.className="mlb-workspace-buttons"' in text
        assert 'option.className="mlb-workspace-tab"' in text
        assert 'modeSwitch.className="mlb-studio-mode-switch"' not in text
        assert '[["learn","Learn"],["build","Build"],["research","Research"]]' not in text
        assert 'state.studio_mode="build"' in text


def test_step10l_load_and_export_are_workspace_direct():
    for rel in JS_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'function loadModelConfigIntoBuilder' in text
        assert 'function loadWorkspaceExportIntoBuilder' in text
        assert 'parsed?.format==="mlbricks-model-config"' in text
        assert 'parsed?.format==="mlbricks-export"' in text
        assert 'custom_components:cp(state.custom_components||{})' in text
        assert 'component_cache:cp(state.component_cache||{})' in text
        assert 'exportAction.addEventListener("click",exportWorkspace)' in text


def test_step10l_model_build_completion_does_not_steal_data_builder_focus():
    text = (ROOT / "frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    assert 'const modelWasVisible=state.active_workspace==="model"' in text
    assert 'Model build complete in Model Builder. Data Builder stayed open.' in text
    # The old forced navigation back to model at completion is gone from this block.
    start = text.index('      const finish=()=>{')
    end = text.index('      modelBuildTimer=setInterval', start)
    block = text[start:end]
    assert 'state.active_workspace="model"' not in block


def test_step10l_workspace_css_uses_visible_stacked_switcher():
    for rel in CSS_FILES:
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert '.mlb-workspace-buttons{' in text
        assert '.mlb-workspace-tab{' in text
        assert '.mlb-workspace-tab.active{' in text
        assert '.mlb-workspace-tab.busy' in text
