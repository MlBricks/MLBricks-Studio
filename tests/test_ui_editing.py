from __future__ import annotations

from pathlib import Path


def _builder_js() -> str:
    return (Path(__file__).resolve().parents[1] / "src/mlb_studio/static/builder.js").read_text(encoding="utf-8")


def test_redraw_guard_only_blocks_free_form_editors():
    text = _builder_js()
    assert "let focusedEditorActive=false;" in text
    assert 'root.addEventListener("focusin"' in text
    assert 'root.addEventListener("focusout"' in text
    assert 'if(!force && (pointerInteractionActive || focusedEditorActive))' in text
    assert "if(!el.matches('input'))return false;" in text
    assert '["checkbox","radio","range","button","submit","reset","file","color"]' in text
    # Selects must not be treated as long-lived editors; otherwise workspace and
    # runtime dropdowns redraw only after another click.
    assert "input,textarea,select" not in text


def test_workspace_switch_forces_immediate_redraw():
    text = _builder_js()
    start = text.index("function switchWorkspace(next)")
    block = text[start : start + 2200]
    assert "state.active_workspace=next;" in block
    assert "draw(true);" in block


def test_live_search_can_force_redraw_and_restore_focus():
    text = _builder_js()
    assert "searchFocusRestore={start:searchInput.selectionStart" in text
    assert "draw(true);" in text


def test_model_settings_do_not_reject_intermediate_width_head_values():
    text = _builder_js()
    assert 'if(key==="embedding_size" && next.embedding_size%next.heads!==0)' not in text
    assert 'if(key==="heads" && next.embedding_size%next.heads!==0)' not in text
    assert "must be divisible by Heads before Build" in text


def test_build_validation_catches_incompatible_width_and_heads():
    text = _builder_js()
    assert "Number(settings.embedding_size)%Number(settings.heads)!==0" in text
    assert 'message:"Embedding Size ("+settings.embedding_size+") must be divisible by Heads' in text


def test_runtime_configs_preserve_object_identity_during_redraws():
    text = _builder_js()
    assert "function mergeRuntimeDefaultsInPlace(defaults,saved)" in text
    assert "Object.assign(saved,merged)" in text
    assert "entry.training_config=mergeRuntimeDefaultsInPlace" in text
    assert "entry.generation_config=mergeRuntimeDefaultsInPlace" in text
    assert "entry.serve_config=mergeRuntimeDefaultsInPlace" in text


def test_runtime_fields_persist_drafts_before_blur():
    text = _builder_js()
    assert 'input.addEventListener("input",()=>commit(false))' in text
    assert 'input.addEventListener("change",()=>commit(true))' in text


def test_runtime_updates_resolve_latest_model_entry():
    text = _builder_js()
    assert "const latest=builtModelById(entry.id)||entry;" in text
    assert 'const target=mode==="train"?latest.training_config:latest.generation_config;' in text
    assert "latest.serve_config[key]=value;" in text


def test_model_setting_updates_resolve_latest_model_entry():
    text = _builder_js()
    start = text.index("function updateBuiltModelSetting(entry,key,value)")
    block = text[start : text.index("function modelSettingField", start)]
    assert "const latest=builtModelById(entry.id)||entry;" in block
    assert "latest.model_settings={...next};" in block
    assert "latest.training_config.batch_size=next.default_batch;" in block


def test_component_and_data_fields_resolve_latest_node_by_id():
    text = _builder_js()
    assert "function liveNodeById(nodeId)" in text
    start = text.index("function renderField(body,node,f)")
    block = text[start : text.index("function fieldCurrentValue", start)]
    assert "const target=liveNodeById(node.id)||node;" in block
    assert 'input.addEventListener("input",()=>commit(input.value,false))' in block
    assert 'input.addEventListener("change",()=>commit(input.value,true))' in block


def test_percent_fields_persist_while_dragging_and_commit_once():
    text = _builder_js()
    start = text.index("function renderField(body,node,f)")
    block = text[start : text.index("function fieldCurrentValue", start)]
    assert "commit(range.value,false)" in block
    assert "commit(range.value,true)" in block
    assert "checkpoint(\"Edit \"+target.name+\".\"+f.key,editBefore);" in block


def test_device_cards_use_runtime_update_path():
    text = _builder_js()
    assert "function deviceCards(config,onSelect=null)" in text
    assert 'deviceCards(config,v=>update("device",v))' in text


def test_all_number_inputs_get_shared_themed_steppers():
    text = _builder_js()
    assert "function installNumberSteppers(scope)" in text
    assert 'input[type="number"]:not([data-mlb-number-ready="1"])' in text
    assert 'makeStep(-1,"−","Decrement value")' in text
    assert 'makeStep(1,"+","Increment value")' in text
    assert "installNumberSteppers(root);" in text


def _builder_css() -> str:
    return (Path(__file__).resolve().parents[1] / "src/mlb_studio/static/builder.css").read_text(encoding="utf-8")


def test_notebook_layout_stays_desktop_and_uses_horizontal_scroll():
    text = _builder_css()
    assert "V1.0 stable desktop notebook / Kaggle layout" in text
    assert "height:720px!important" in text
    assert "min-width:1360px!important" in text
    assert "overflow-x:auto!important" in text
    assert "grid-template-columns:230px minmax(850px,1fr) 280px!important" in text
    assert "grid-template-columns:1fr;grid-template-rows:auto minmax(420px,1fr) auto" not in text
    assert "@media" not in text


def test_remove_all_links_action_is_not_rendered():
    text = _builder_js()
    assert 'btn("Remove All Links")' not in text
    assert 'checkpoint("Remove all links from "+n.name)' not in text


def test_desktop_control_geometry_does_not_shrink_in_kaggle():
    text = _builder_css()
    assert "V1.0 desktop control geometry lock" in text
    assert "grid-template-columns:0 minmax(1040px,1fr) 300px!important" in text
    assert "grid-template-columns:repeat(3,minmax(210px,1fr))!important" in text
    assert "V1.0 compact horizontal number steppers" in text
    assert "grid-template-columns:minmax(0,1fr) 24px 24px!important" in text
    assert "display:contents!important" in text
    assert "height:32px!important" in text
    assert "height:34px!important" in text
    assert "min-width:112px!important" in text
