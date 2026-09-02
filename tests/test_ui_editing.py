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


def test_all_number_inputs_are_typing_only():
    text = _builder_js()
    assert "function installNumberSteppers(scope)" not in text
    assert 'input[type="number"]:not([data-mlb-number-ready="1"])' not in text
    assert 'makeStep(1,"▲","Increment value")' not in text
    assert 'makeStep(-1,"▼","Decrement value")' not in text
    assert "installNumberSteppers(root);" not in text


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
    assert "V1.0 typing-only numeric inputs" in text
    assert '.mlb-number-stepper,.mlb-number-step{display:none!important}' in text
    assert "height:32px!important" in text
    assert "height:34px!important" in text
    assert "min-width:112px!important" in text


def test_numeric_inputs_are_typing_only_without_custom_steppers():
    text = _builder_js()
    css = _builder_css()
    assert "installNumberSteppers(root)" not in text
    assert 'makeStep(1,"▲"' not in text
    assert 'makeStep(-1,"▼"' not in text
    assert "V1.0 typing-only numeric inputs" in css
    assert '.mlb-number-stepper,.mlb-number-step{display:none!important}' in css


def test_port_layout_labels_are_simple_and_clear():
    js = (Path(__file__).resolve().parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    assert 'editorRow("Port Layout"' in js
    assert '{value:"standard",label:"Standard"}' in js
    assert '{value:"named",label:"Custom"}' in js
    assert 'editorRow("Visual Port Mode"' not in js
    assert 'label:"Main / Skip / Extra"' not in js
    assert 'label:"Custom Named Ports"' not in js


def test_blueprints_and_loaded_designs_start_with_workspace_collapsed():
    text = _builder_js()
    assert 'function collapseArtifactWorkspace()' in text
    assert 'bottomView="outputs";' in text
    assert 'bottomExpanded=false;' in text
    assert 'collapseArtifactWorkspace();setStatus(entry.name+" loaded from Gallery.")' in text
    assert 'collapseArtifactWorkspace();setStatus("TinyStories starter loaded.")' in text
    assert 'collapseArtifactWorkspace();setStatus(spec.name+" loaded.")' in text
    assert 'selected=null;pendingPort=null;collapseArtifactWorkspace();switchingWorkspace=true;' in text


def test_successful_data_and_model_results_expand_output_workspace_without_stealing_navigation():
    text = _builder_js()
    assert 'function revealArtifactWorkspace(kind,artifactId=null)' in text
    assert 'bottomExpanded=true;' in text
    prepared = text[text.index('if(next.prepared_dataset){'):text.index('root.querySelectorAll(".mlb-node")', text.index('if(next.prepared_dataset){'))]
    assert 'if(state.active_workspace==="data"){' in prepared
    assert 'revealArtifactWorkspace("data",next.prepared_dataset.id);' in prepared
    assert 'Finishing a data job must never steal navigation from Model Builder.' in prepared
    assert 'bottomExpanded=true;\n        outputDirectorySelection=entry.id;' in text


def test_model_build_completion_is_anchored_to_model_workspace():
    text = _builder_js()
    start = text.index('function requestModelBuild()')
    end = text.index('function datasetModality', start)
    block = text[start:end]
    assert 'state.active_workspace="model";' in block
    assert 'const modelWs=state.workspaces?.model;' in block
    assert 'galleryWorkspace.open=false;' in block
    assert 'cloudWorkspace.open=false;' in block
    assert 'runtimePanel=null;' in block


def test_imported_or_searched_artifacts_stay_visible_after_completion():
    text = _builder_js()
    assert 'revealArtifactWorkspace(type==="data"?"data":"model",imported.length?imported[imported.length-1].id:null);' in text
    assert 'if(loadedDataset)revealArtifactWorkspace("data",loadedDataset.id||null);' in text
    assert 'else if(loadedModel)revealArtifactWorkspace("model",loadedModel.id||null);' in text
    assert 'if(contentType==="dataset")revealArtifactWorkspace("data",restored.dataset?.id||null);' in text
    assert 'else if(contentType==="model")revealArtifactWorkspace("model",restored.model?.id||null);' in text


def test_all_buttons_have_global_press_and_acknowledgement_feedback():
    js = _builder_js()
    css = _builder_css()
    assert 'function actionButton(target)' in js
    assert 'button.classList.add("mlb-button-pressed")' in js
    assert 'queueMicrotask(()=>showActionAck(button))' in js
    assert 'ack.textContent="✓ "+actionLabel(button);' in js
    assert '.mlb-root button:not(:disabled):not(.mlb-port):active' in css
    assert '.mlb-root button.mlb-button-pressed:not(:disabled):not(.mlb-port)' in css
    assert '.mlb-action-ack.show' in css


def test_connection_ports_are_excluded_from_generic_button_animation():
    js = _builder_js()
    css = _builder_css()
    assert 'b.classList.contains("mlb-port")' in js
    assert ':not(.mlb-port)' in css


def test_kaggle_bridge_lookup_never_deep_scans_during_startup_polling():
    text = _builder_js()
    assert "const bridgeHostCache=new Map();" in text
    assert "const bridgeControlCache=new Map();" in text
    assert "function ensureBridgeForAction()" in text
    assert "bridgeDocuments(force,false)" in text
    assert "if(!allowDeep)return null;" in text
    assert "bridgeDocuments(true,true)" in text
    start = text.index("function startBridgePolling()")
    end = text.index("function handlePopoutMessage", start)
    block = text[start:end]
    assert "updateKernelBadge(false,false)" in block
    assert "updateKernelBadge(true,true)" not in block
    assert "deepQuery(" not in block


def test_initial_studio_render_yields_for_first_paint_and_defers_bridge_work():
    text = _builder_js()
    assert 'class="mlb-startup-shell"' in text
    assert "requestAnimationFrame(()=>" in text
    assert "draw();\n      const beginBackgroundBridge=()=>{" in text
    assert 'if(typeof requestIdleCallback==="function")requestIdleCallback(beginBackgroundBridge,{timeout:1500});' in text


def test_full_window_source_serialization_is_lazy():
    text = _builder_js()
    assert "window.__MLB_STUDIO_GET_JS_SOURCE__=function()" in text
    tail = text[text.index("window.__MLB_STUDIO_FACTORY__=__MLB_STUDIO_FACTORY__;"):]
    assert 'window.__MLB_STUDIO_JS_SOURCE__="("+__MLB_STUDIO_FACTORY__.toString()+")();";' in tail
    assert tail.index("window.__MLB_STUDIO_GET_JS_SOURCE__=function()") < tail.index("__MLB_STUDIO_FACTORY__();")




def test_full_window_invalidates_stale_frontend_source_snapshot():
    text = _builder_js()
    assert 'window.__MLB_STUDIO_JS_SOURCE__=null;' in text
    assert 'Full Window must serialize this exact factory version.' in text
    start = text.index('function fullWindowPage()')
    end = text.index('function openFullWindow()', start)
    block = text[start:end]
    assert 'window.__MLB_STUDIO_GET_JS_SOURCE__?window.__MLB_STUDIO_GET_JS_SOURCE__()' in block
    assert 'window.__MLB_STUDIO_JS_SOURCE__||(window.__MLB_STUDIO_GET_JS_SOURCE__' not in block
    assert 'COMPILER TEST MODELS' in text


def test_graph_resize_observer_does_not_observe_self_resized_wrapper():
    text = _builder_js()
    assert "Never observe `wrap`" in text
    assert "[flow,canvas].forEach" in text
    assert "[flow,wrap,canvas,root].forEach" not in text
    assert "if(wrap.style.width!==nextW)wrap.style.width=nextW;" in text
    assert "setTimeout(renderConnections,180);" in text


def test_library_subtext_is_descriptive_and_backend_neutral():
    from mlb_studio.graph import primitive_catalog

    catalog = {item["type"]: item for item in primitive_catalog()}
    assert catalog["embedding"]["description"] == "Token embedding layer that maps token IDs into dense vector representations."
    assert catalog["esa"]["description"] == "Entangled State Attention sequence-mixing layer."
    assert catalog["ffn"]["description"] == "Feed-Forward Network for transforming features within each layer."
    assert catalog["rmsnorm"]["description"] == "Root Mean Square Normalization layer for stabilizing activations."
    assert catalog["hf_dataset"]["description"].startswith("Dataset source")
    assert catalog["tokenize_text"]["description"].startswith("Tokenization step")

    visible_descriptions = " ".join(item.get("description", "") for item in catalog.values()).lower()
    assert "pytorch" not in visible_descriptions
    assert "native" not in visible_descriptions
    assert "builder utility node" not in visible_descriptions


def test_elasticbit_card_uses_simple_name():
    from mlb_studio.graph import primitive_catalog

    catalog = {item["type"]: item for item in primitive_catalog()}
    elastic = catalog["elasticbit_runtime"]
    assert elastic["name"] == "ElasticBit"
    assert "4–32" not in elastic["name"]
    assert "4-32" not in elastic["name"]


def test_library_subtext_is_clamped_to_two_lines():
    css = _builder_css()
    assert "-webkit-line-clamp:2;" in css
    assert "line-clamp:2;" in css


def test_elasticbit_old_saved_label_is_normalized_at_display_time():
    js = _builder_js()
    assert 'node.type==="elasticbit_runtime"' in js
    assert '/^ElasticBit(?:\\s+4[-–—]32)?$/i.test(saved)' in js
    assert 'return "ElasticBit";' in js


def test_logo_is_centered_over_left_sidebar():
    css = _builder_css()
    assert '.mlb-top-left>.mlb-logo{' in css
    assert 'width:182px!important;' in css
    assert 'justify-content:center!important;' in css
    assert '.mlb-top-left>.mlb-logo .mlb-logo-brand{' in css
    assert 'object-position:center center!important;' in css


def test_gallery_contains_tiny_compiler_test_models():
    js = _builder_js()
    assert 'COMPILER TEST MODELS' in js
    assert 'TEST · BOLT Direct API' in js
    assert 'TEST · ESA → BOLT Pipeline' in js
    assert 'TEST · ResController Multi-Input' in js
    assert 'TEST · SAFFN Stateful API' in js
    assert 'loadCompilerTestBOLT' in js
    assert 'loadCompilerTestESABOLT' in js
    assert 'loadCompilerTestResController' in js
    assert 'loadCompilerTestSAFFN' in js
    assert 'named_out:state' in js
    assert 'toNamed(saffn1,saffn2,"previous_state","named_out:state")' in js
    assert 'dim:64,heads:4,block:64,batch:2,vocab:2048' in js
    assert 'Object.assign(edge(ctx.emb.id,rc.id,"residual"),{source_port:"skip_out",target_port:"skip_in"})' in js


def test_saffn_catalog_exposes_original_api_named_runtime_ports():
    from mlb_studio.graph import primitive_catalog

    catalog = {item["type"]: item for item in primitive_catalog()}
    ports = catalog["saffn"]["runtime_ports"]
    assert [p["id"] for p in ports["inputs"]] == ["x", "esa_update", "previous_esa", "previous_state"]
    assert [p["id"] for p in ports["outputs"]] == ["main", "state"]


def test_model_build_validation_counts_named_and_skip_edges_as_real_connectivity():
    js = _builder_js()
    start = js.index("function validateModelBuild()")
    end = js.index("function buildModel", start) if "function buildModel" in js[start:] else start + 7000
    block = js[start:end]
    assert 'const executionEdges=(model.edges||[]).filter(e=>byId.has(e.source)&&byId.has(e.target));' in block
    assert 'executionEdges.forEach(e=>' in block
    assert 'const mainEdges=(model.edges||[]).filter(e=>(e.kind||"main")==="main");' not in block
    assert 'Model execution graph contains a cycle.' in block
