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


def test_port_layout_exposes_universal_plus_custom_terminals():
    js = (Path(__file__).resolve().parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    assert 'editorRow("Port Layout"' in js
    assert '{value:"standard",label:"Universal · 3 In / 3 Out"}' in js
    assert '{value:"extended",label:"Universal + Custom Terminals"}' in js
    assert 'terminalTitle.textContent="CUSTOM TERMINALS"' in js
    assert 'btn("+ Add Input Terminal"' in js
    assert 'btn("+ Add Output Terminal"' in js
    assert '{value:"top",label:"Top"}' in js
    assert '{value:"right",label:"Right"}' in js
    assert '{value:"bottom",label:"Bottom"}' in js
    assert '{value:"left",label:"Left"}' in js
    assert '?"← Left":"↑ Up"' in js
    assert '?"Right →":"Down ↓"' in js
    assert 'editorRow("Visual Port Mode"' not in js
    assert 'function redrawCustomTerminalLayout()' in js
    assert 'draw(true);' in js[js.index('function redrawCustomTerminalLayout()'):js.index('function ensureAPIStepObjectIds', js.index('function redrawCustomTerminalLayout()'))]
    assert 'changeCustomTerminalSide(binding,port,v);redrawCustomTerminalLayout();' in js
    assert 'moveCustomTerminal(binding,port,-1);redrawCustomTerminalLayout();' in js
    assert 'moveCustomTerminal(binding,port,1);redrawCustomTerminalLayout();' in js


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
    assert 'COMPONENT TEST MODELS' in text


def test_graph_resize_observer_does_not_observe_self_resized_wrapper():
    text = _builder_js()
    assert "Never observe `wrap`" in text
    assert "try{edgeObserver.observe(flow);}catch(_){}" in text
    assert "[flow,canvas].forEach" not in text
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
    assert 'COMPONENT TEST MODELS' in js
    assert 'SPECIALIZED API PROBES' in js
    assert 'TEST · Core Trainable Stack' in js
    assert 'TEST · ESA Direct' in js
    assert 'TEST · BOLT Direct API' in js
    assert 'TEST · ESA → BOLT Pipeline' in js
    assert 'TEST · Residual Add Multi-Input' in js
    assert 'TEST · ResController Multi-Input' in js
    assert 'TEST · MicroVirtualFFN API' in js
    assert 'TEST · SAFFN Stateful API' in js
    assert 'TEST · VirtualStateAwareFFN Stateful API' in js
    assert 'TEST · SOUP Direct' in js
    assert 'TEST · StateAware ESA Stack' in js
    assert 'TEST · Previous Value Buffer' in js
    assert 'PROBE · RoPE 4D API' in js
    assert 'PROBE · ElasticBit Post-Training' in js
    assert 'loadCompilerTestCoreStack' in js
    assert 'loadCompilerTestESA' in js
    assert 'loadCompilerTestBOLT' in js
    assert 'loadCompilerTestESABOLT' in js
    assert 'loadCompilerTestResidual' in js
    assert 'loadCompilerTestResController' in js
    assert 'loadCompilerTestMicroFFN' in js
    assert 'loadCompilerTestSAFFN' in js
    assert 'loadCompilerTestVirtualSAFFN' in js
    assert 'loadCompilerTestSOUP' in js
    assert 'loadCompilerTestStateAwareESAStack' in js
    assert 'loadCompilerTestValueBuffer' in js
    assert 'Previous Signal · Zero Init' not in js
    assert 'Previous State · Zero Init' not in js
    assert 'Layer 1 auto-initializes unconnected previous signal/state' in js
    assert 'const esa1=makeNode(cat(catalog,"esa"))' in js
    assert 'const esa2=makeNode(cat(catalog,"esa"))' in js
    assert 'const prevEsa=makeNode(cat(catalog,"esa"))' not in js
    assert 'Layer 2 receives the real previous signal and state from physical depth 1' in js
    assert 'toNamed(esa1,saffn2,"previous_esa","main_out","","top_aux")' in js
    assert 'toNamed(saffn1,saffn2,"previous_state","named_out:state","bottom_aux","bottom_aux")' in js
    assert 'dim:64,heads:4,block:64,batch:2,vocab:2048' in js
    assert 'Object.assign(edge(ctx.emb.id,rc.id,"residual"),{source_port:"skip_out",target_port:"skip_in"})' in js


def test_previous_value_buffer_is_available_as_builder_utility():
    from mlb_studio.graph import primitive_catalog

    catalog = {item["type"]: item for item in primitive_catalog()}
    item = catalog["value_buffer"]
    assert item["builder_utility"] is True
    assert item["name"] == "Previous Value Buffer"
    params = {spec["key"]: spec for spec in item["api"]}
    assert params["mode"]["value"] == "hold"
    assert set(params["mode"]["options"]) == {"hold", "zero_init"}


def test_saffn_catalog_exposes_original_api_named_runtime_ports():
    from mlb_studio.graph import primitive_catalog

    catalog = {item["type"]: item for item in primitive_catalog()}
    ports = catalog["saffn"]["runtime_ports"]
    assert [p["id"] for p in ports["inputs"]] == ["x", "esa_update", "previous_esa", "previous_state"]
    assert [p["id"] for p in ports["outputs"]] == ["main", "state"]
    assert [p["name"] for p in ports["inputs"]] == ["Input Signal", "Current Signal", "Previous Signal", "Previous State"]
    assert [p["name"] for p in ports["outputs"]] == ["Main Output", "State Out"]
    assert [p["socket"] for p in ports["inputs"]] == ["top", "back", "top_aux", "bottom_aux"]
    assert ports["outputs"][0]["socket"] == "front"
    assert ports["outputs"][1]["sockets"] == ["top_aux", "bottom_aux"]


    virtual_ports = catalog["virtual_saffn"]["runtime_ports"]
    assert virtual_ports == ports


def test_model_build_validation_counts_named_and_skip_edges_as_real_connectivity():
    js = _builder_js()
    start = js.index("function validateModelBuild()")
    end = js.index("function buildModel", start) if "function buildModel" in js[start:] else start + 7000
    block = js[start:end]
    assert 'const executionEdges=(model.edges||[]).filter(e=>byId.has(e.source)&&byId.has(e.target));' in block
    assert 'executionEdges.forEach(e=>' in block
    assert 'const mainEdges=(model.edges||[]).filter(e=>(e.kind||"main")==="main");' not in block
    assert 'Model execution graph contains a cycle.' in block


def test_complex_named_api_nodes_have_readable_graph_ui():
    js = _builder_js()
    css = _builder_css()
    assert 'function namedPortColor(key,index=0)' in js
    assert 'card.classList.add("mlb-complex-api-node")' in js
    assert 'mlb-edge-side-aware' in js
    assert 'p.style.stroke=namedPortColor(targetKey||sourceKey,targetIndex);' in js
    assert 'p.classList.add("mlb-edge-focus")' in js
    assert 'p.classList.add("mlb-edge-dim")' in js
    assert 'btn("Fit","mlb-zoom-fit")' in js
    assert '.mlb-node.mlb-complex-api-node{' in css
    assert 'width:196px!important;' in css
    assert 'function namedPortVisualSide(node,port,ioSide,key)' in js
    assert 'data-visual-side="' in js
    assert 'function namedSocketOrder(side)' in js
    assert 'function namedSocketGroups(node,side)' in js
    assert 'namedNodeAvoidingPath(a,b,x1,y1,sourceVisual,x2,y2,targetVisual)' in js
    assert 'openNamedPortHubPicker(portEl,items,item=>' in js
    assert 'data-port-keys="' in js
    assert 'data-tooltip="' in js
    assert 'card.classList.add("mlb-universal-six-socket");' in js
    assert 'const inputNames=["Top Input","Input Signal","Bottom Input"]' in js
    assert 'const outputNames=["Top Output","Main Output","Bottom Output"]' in js
    assert '.mlb-node.mlb-universal-six-socket{' in css
    assert 'mlb-input-socket' in js
    assert 'mlb-output-socket' in js
    assert 'data-io-role=' in js
    assert 'data-physical-slot=' in js
    assert 'function physicalSocketName(side,socket)' in js
    assert 'top_aux' in js
    assert 'bottom_aux' in js
    assert 'dedicated-signal-socket' in js
    assert 'const left=side==="in"?14:86;' in js
    assert 'const left=side==="in"?38:62;' in js
    assert '.mlb-port.mlb-input-socket{' in css
    assert '.mlb-port.mlb-output-socket{' in css
    assert 'width:210px!important;' in css
    assert '.mlb-port-hub-picker{' in css
    assert '.mlb-edge-dim{opacity:.13!important}' in css
    assert '.mlb-edge-focus{' in css
    assert '.mlb-zoom-fit{' in css


def test_universal_graph_renderer_uses_cached_geometry_and_no_canvas_resize_loop():
    from pathlib import Path
    js = (Path(__file__).parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    assert 'const nodeById=new Map(nodeEls.map(el=>[el.dataset.nodeId,el]))' in js
    assert 'const nodeRects=new Map(nodeEls.map(el=>[el,el.getBoundingClientRect()]))' in js
    assert 'try{edgeObserver.observe(flow);}catch(_){}' in js
    assert '[flow,canvas].forEach' not in js


def test_custom_terminal_normalization_preserves_live_object_identity():
    js = (Path(__file__).resolve().parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    start = js.index("function normalizeAPIBinding")
    end = js.index("function apiBindingImportPath", start)
    block = js[start:end]
    assert "normalize custom terminals in place" in block
    assert 'const port=(rawPort&&typeof rawPort==="object")?rawPort:{};' in block
    assert "return port;" in block
    # Regression: cloning each terminal object here detached Inspector handlers
    # from the graph state and made Side/Move/Mapping controls snap back.
    assert "binding.input_ports=binding.input_ports.map((port,i)=>({" not in block
    assert "binding.output_ports=binding.output_ports.map((port,i)=>({" not in block
    assert "function resolveCustomTerminal(binding,port)" in js


def test_custom_terminals_are_capped_at_four_per_side():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    assert 'const CUSTOM_TERMINAL_LIMIT_PER_SIDE=4;' in js
    assert 'const CUSTOM_TERMINAL_SIDES=["top","right","bottom","left"];' in js
    assert 'function customTerminalSideCount(binding,side,excludePort=null)' in js
    assert 'function customTerminalSideHasRoom(binding,side,excludePort=null)' in js
    assert 'function firstCustomTerminalSideWithRoom(binding,preferred="top",excludePort=null)' in js
    assert 'function enforceCustomTerminalSideCapacity(binding)' in js
    assert 'enforceCustomTerminalSideCapacity(binding);' in js
    assert 'if(!customTerminalSideHasRoom(binding,next,live)){' in js
    assert 'Maximum "+CUSTOM_TERMINAL_LIMIT_PER_SIDE+" extra terminals allowed on ' in js
    assert 'addIn.disabled=!firstCustomTerminalSideWithRoom(binding,"top");' in js
    assert 'addOut.disabled=!firstCustomTerminalSideWithRoom(binding,"top");' in js
    assert 'Maximum reached: 4 custom terminals on each side.' in js
    assert 'Add up to 4 extra terminals on each side' in js


def test_custom_terminal_surface_spacing_reserves_fixed_center_socket():
    from pathlib import Path
    js = (Path(__file__).resolve().parents[1] / "src" / "mlb_studio" / "static" / "builder.js").read_text(encoding="utf-8")
    assert 'const CUSTOM_TERMINAL_REGION_MIN=30;' in js
    assert 'const CUSTOM_TERMINAL_REGION_MAX=70;' in js
    assert 'function evenlySpacedCustomTerminalPercent(index,count)' in js
    assert 'const CUSTOM_SIDE_TERMINAL_SLOTS={' in js
    assert '1:[40]' in js
    assert '2:[40,60]' in js
    assert '3:[30,40,60]' in js
    assert '4:[30,40,60,70]' in js
    assert 'function centeredSideCustomTerminalPercent(index,count)' in js
    assert '?centeredSideCustomTerminalPercent(index,group.length)' in js
    assert "Math.abs(percent-50)<0.01?'-18px':'-6px'" not in js
    assert 'never overlap it' in js
    # Old regressions either stacked custom terminals below center or created a
    # second column beside the fixed 50% socket.
    assert 'percent=group.length<=1?64:(62+(26*index/(group.length-1)));' not in js


def test_user_function_has_central_visual_to_function_mapping_editor():
    js = _builder_js()
    css = _builder_css()
    assert 'title.textContent="VISUAL ↔ FUNCTION MAPPING"' in js
    assert 'inputTitle.textContent="VISUAL INPUTS → FUNCTION ARGUMENTS"' in js
    assert 'outputTitle.textContent="FUNCTION RETURNS → VISUAL OUTPUTS"' in js
    assert '{lane:"skip",source:"skip",input:"Top Input",output:"Top Output"}' in js
    assert '{lane:"main",source:"main",input:"Main Input",output:"Main Output"}' in js
    assert '{lane:"extra",source:"extra",input:"Bottom Input",output:"Bottom Output"}' in js
    assert 'renderVisualFunctionMapping(body,def,step,binding)' in js
    assert 'Function argument mapping is configured in Visual ↔ Function Mapping below.' in js
    assert 'Return mapping is configured in Visual ↔ Function Mapping below.' in js
    assert '.mlb-visual-map-card' in css
    assert '.mlb-visual-map-row' in css
