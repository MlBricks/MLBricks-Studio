function __MLB_STUDIO_FACTORY__(){
  // Always overwrite any renderer left by an older notebook output.
  // Kaggle keeps browser globals even when Python modules are reinstalled.

  function cp(v){return JSON.parse(JSON.stringify(v));}
  function uid(p){return p+"_"+Math.random().toString(36).slice(2,10);}
  function current(state){return state.components[state.view_component_id];}
  function cat(catalog,type){return catalog.find(x=>x.type===type)||{};}
  function edge(a,b,kind="main"){return{id:uid("edge"),source:a,target:b,source_port:"out",target_port:"in",kind};}

  function makeNode(item){
    const params={};
    (item.api||[]).forEach(f=>{
      if(f.value!==undefined) params[f.key]=f.value;
    });
    return {
      id:uid("node"),
      type:item.type,
      name:item.name,
      definition_id:null,
      repeat:1,
      params,
      input_count:3,
      output_count:3,
      position:{x:0,y:0}
    };
  }


  function mount(root,payload){
    if(!root || root.dataset.mounted==="1") return;
    root.dataset.mounted="1";

    let state=cp(payload.state);
    const catalog=cp(payload.catalog);
    const mlapi=cp(payload.mlbricks_api||{});
    let selected=null,pendingPort=null,filter="All",search="",inspectorTab="settings",zoom=1,status="Ready";
    let searchFocusRestore=null;
    const inspectorScrollPositions={};
    let lastInspectorRenderKey=null;
    let scrollBuiltModelActionsOnce=false;
    const bridge=payload.bridge||null;
    const isPopout=!!(bridge&&(bridge.mode==="broadcast"||bridge.mode==="popout"));
    const popoutChannelName=(bridge&&bridge.channel)||("mlb-studio-"+(payload.instance_id||root.id||"session"));
    let popoutChannel=null;
    let popoutHostConnected=!isPopout;
    let popoutPeerWindow=null;
    let popoutMessagePort=null;
    let popoutPeerConnected=false;
    let popoutHelloTimer=null;
    let pendingBroadcastState=null;
    let pendingBroadcastCommand=null;
    let popoutSyncTimer=null;
    const runtimeCaps=cp(payload.runtime_capabilities||{devices:[{id:"auto",label:"Auto"},{id:"cpu",label:"CPU"}]});
    const localEnvironment=cp(payload.local_environment||{kind:"python",name:"Python / Jupyter Environment",roots:["."],default_root:"."});
    const localDefaultRoot=localEnvironment.workspace_root||localEnvironment.default_root||(localEnvironment.roots||[])[0]||".";
    const localPaths=cp(localEnvironment.paths||{});
    let runtimePanel=null;
    let galleryWorkspace={open:false,tab:"models"};
    let galleryPreviousBottomExpanded=true;
    let componentInsertPicker={open:false,afterNodeId:null};
    let cloudWorkspace={open:false};
    let cloudPreviousBottomExpanded=true;
    let execution={status:"idle",overall:0,message:"Ready",nodes:{}};
    let localFiles={roots:[],entries:[],truncated:false};
    let localImportReports={model:null,data:null};
    let localForm={model_path:localDefaultRoot,data_path:localDefaultRoot};
    let serveSecrets={};
    let cloudStatus={};
    let cloudSecrets={
      huggingface:{token:""},
      github:{token:""},
      aws:{access_key:"",secret_key:"",session_token:""},
      gcp:{service_account_json:""},
      azure:{connection_string:""}
    };
    let cloudForm={
      provider:"huggingface",
      push_type:state.active_workspace==="data"?"dataset":"model",
      push_artifact:"",
      load_type:state.active_workspace==="data"?"dataset":"model",
      repo:"",
      branch:"main",
      revision:"main",
      bucket:"",
      container:"",
      object_path:"",
      private:true,
      region:""
    };
    let lastProgressRaw="";
    let bridgePollTimer=null;
    let bridgeAwaitTimer=null;
    let bridgeLastReady=false;
    let runtimeStatusRedrawTimer=null;
    let modelBuildTimer=null;
    const workspaceScroll={model:{left:0,top:0},data:{left:0,top:0}};
    const sidebarScroll={model:{left:0,top:0},data:{left:0,top:0}};
    let switchingWorkspace=false;
    const undoStack=[],redoStack=[];
    const historyLimit=60;
    // Focused Module/API editors are isolated transactions. Undo/Redo must
    // never cross an editor boundary and accidentally behave like Cancel.
    const customEditorTransactions=[];
    const componentImportQueue=[];
    let componentImportBusy=false;
    const customImportStatus={};

    // Single-click reliability guard. A focused editor can emit change/blur on
    // pointerdown when the user clicks another control. Those handlers may call
    // draw(), which replaces the DOM before the target control receives click.
    // Defer only that redraw until the click completes.
    let pointerInteractionActive=false;
    let deferredInteractionDraw=false;
    let interactionReleaseQueued=false;

    function isCommitEditor(el){
      return !!(el && el.matches && el.matches('input,textarea,select,[contenteditable="true"]'));
    }

    function releasePointerInteraction(){
      interactionReleaseQueued=false;
      if(!pointerInteractionActive && !deferredInteractionDraw)return;
      pointerInteractionActive=false;
      if(deferredInteractionDraw){
        deferredInteractionDraw=false;
        draw();
      }
    }

    root.addEventListener("pointerdown",ev=>{
      const active=root.ownerDocument?.activeElement;
      pointerInteractionActive=!!(active && root.contains(active) && isCommitEditor(active) && active!==ev.target && !active.contains?.(ev.target));
      interactionReleaseQueued=false;
    },true);
    root.addEventListener("click",()=>{
      if(!pointerInteractionActive||interactionReleaseQueued)return;
      interactionReleaseQueued=true;
      queueMicrotask(releasePointerInteraction);
    },true);
    root.addEventListener("pointerup",()=>{
      if(!pointerInteractionActive)return;
      // click normally follows pointerup; this is only a fallback for a
      // pointer interaction that does not produce click.
      setTimeout(()=>{if(pointerInteractionActive)releasePointerInteraction();},0);
    },true);
    root.addEventListener("pointercancel",releasePointerInteraction,true);

    function snapshot(){ return cp(state); }
    function checkpoint(label){
      undoStack.push({state:snapshot(),label:label||"Edit"});
      if(undoStack.length>historyLimit) undoStack.shift();
      redoStack.length=0;
    }
    function undo(){
      if(!undoStack.length){setStatus("Nothing to undo.");draw();return;}
      redoStack.push({state:snapshot(),label:"Redo"});
      const item=undoStack.pop();
      state=cp(item.state);
      selected=null;pendingPort=null;
      setStatus("Undo: "+item.label);
      draw();
    }
    function redo(){
      if(!redoStack.length){setStatus("Nothing to redo.");draw();return;}
      undoStack.push({state:snapshot(),label:"Undo"});
      const item=redoStack.pop();
      state=cp(item.state);
      selected=null;pendingPort=null;
      setStatus("Redo");
      draw();
    }

    function cloneHistory(stack){
      return stack.map(item=>({state:cp(item.state),label:item.label}));
    }

    function beginCustomEditorTransaction(){
      customEditorTransactions.push({
        before_state:snapshot(),
        undo_before:cloneHistory(undoStack),
        redo_before:cloneHistory(redoStack),
        selected_before:selected,
        pending_before:pendingPort?cp(pendingPort):null,
        gallery_before:cp(galleryWorkspace),
        bottom_before:bottomExpanded,
        gallery_bottom_before:galleryPreviousBottomExpanded
      });
      // The editor starts with its own history. There is deliberately no
      // initial checkpoint, so Undo cannot leave the editor.
      undoStack.length=0;
      redoStack.length=0;
    }

    function restoreHistory(frame){
      undoStack.length=0;
      redoStack.length=0;
      (frame?.undo_before||[]).forEach(item=>undoStack.push({state:cp(item.state),label:item.label}));
      (frame?.redo_before||[]).forEach(item=>redoStack.push({state:cp(item.state),label:item.label}));
    }

    function cancelCustomEditor(){
      const frame=customEditorTransactions.pop();
      if(!frame){
        setStatus("No editor transaction is active.");
        draw();
        return;
      }
      state=cp(frame.before_state);
      ensureWorkspaces();
      restoreHistory(frame);
      selected=frame.selected_before||null;
      pendingPort=frame.pending_before?cp(frame.pending_before):null;
      galleryWorkspace=cp(frame.gallery_before||{open:false,tab:"components"});
      bottomExpanded=frame.bottom_before;
      galleryPreviousBottomExpanded=frame.gallery_bottom_before;
      componentInsertPicker={open:false,afterNodeId:null};
      runtimePanel=null;
      cloudWorkspace.open=false;
      setStatus(customEditorTransactions.length?"Nested Module cancelled. Returned to the parent editor.":"Editor cancelled. No Module/API Component changes were saved.");
      draw();
    }

    // Compact notebook defaults: keep the most-used sections open and the rest collapsed.
    const collapsedCategories=new Set(["Advanced","Position","Heads","Outputs","Image","Audio"]);
    const collapsedInspectorGroups=new Set();
    let myBricksCollapsed=false;
    let customActionMenuId=null;
    let bottomExpanded=false;
    let bottomView="details";
    let outputDirectorySelection=null;
    let filesFilter="all";

    // Full Window must open on the exact page the notebook is currently showing
    // (Training Setup/Status, Generation, Gallery, etc.), not reset to Model Builder.
    const initialView=(payload.initial_view&&typeof payload.initial_view==="object")?cp(payload.initial_view):null;
    if(initialView){
      if(initialView.runtime_panel&&typeof initialView.runtime_panel==="object")runtimePanel=cp(initialView.runtime_panel);
      if(initialView.gallery_workspace&&typeof initialView.gallery_workspace==="object")galleryWorkspace=cp(initialView.gallery_workspace);
      if(initialView.cloud_workspace&&typeof initialView.cloud_workspace==="object")cloudWorkspace=cp(initialView.cloud_workspace);
      if(typeof initialView.bottom_expanded==="boolean")bottomExpanded=initialView.bottom_expanded;
      if(typeof initialView.bottom_view==="string")bottomView=initialView.bottom_view;
      if(typeof initialView.selected==="string"||initialView.selected===null)selected=initialView.selected;
      if(typeof initialView.inspector_tab==="string")inspectorTab=initialView.inspector_tab;
      if(Number.isFinite(Number(initialView.zoom)))zoom=Math.max(.65,Math.min(1.5,Number(initialView.zoom)));
    }

    Object.values(state.components||{}).forEach(c=>{if(!c.edges)c.edges=[];});
    if(state.auto_connect===undefined) state.auto_connect=true;

    function ensureWorkspaces(){
      if(!Array.isArray(state.prepared_datasets))state.prepared_datasets=[];
      if(!Array.isArray(state.model_outputs))state.model_outputs=[];
      if(!Array.isArray(state.project_files))state.project_files=[];
      if(!state.gallery||typeof state.gallery!=="object")state.gallery={components:[],models:[],data:[]};
      if(!Array.isArray(state.gallery.components))state.gallery.components=[];
      if(!Array.isArray(state.gallery.models))state.gallery.models=[];
      if(!Array.isArray(state.gallery.data))state.gallery.data=[];
      if(!state.component_cache||typeof state.component_cache!=="object")state.component_cache={};
      if(!state.layout_locks||typeof state.layout_locks!=="object")state.layout_locks={};
      if(!state.workspaces){
        const modelRoot=state.root_component_id;
        const dataRoot=uid("component");
        const starter=defaultDataNodes();
        state.components[dataRoot]={
          id:dataRoot,name:"Data Processing",kind:"data",revision:1,nodes:starter.nodes,edges:starter.edges
        };
        state.workspaces={
          model:{
            name:"Model Builder",
            root_component_id:modelRoot,
            view_component_id:state.view_component_id||modelRoot,
            breadcrumbs:cp(state.breadcrumbs||[{id:modelRoot,name:state.project?.name||"Model"}])
          },
          data:{
            name:"Data Processing",
            root_component_id:dataRoot,
            view_component_id:dataRoot,
            breadcrumbs:[{id:dataRoot,name:"Data Processing"}]
          }
        };
        state.active_workspace="model";
      }
      if(!state.active_workspace || !state.workspaces[state.active_workspace]){
        state.active_workspace="model";
      }
      const ws=state.workspaces[state.active_workspace];
      if(!ws.view_component_id || !state.components[ws.view_component_id]){
        ws.view_component_id=ws.root_component_id;
      }
      if(!Array.isArray(ws.breadcrumbs)||!ws.breadcrumbs.length){
        ws.breadcrumbs=[{id:ws.root_component_id,name:ws.name}];
      }
      state.view_component_id=ws.view_component_id;
      state.breadcrumbs=cp(ws.breadcrumbs);
    }

    function rememberWorkspaceView(){
      const ws=state.workspaces?.[state.active_workspace];
      if(!ws)return;
      // Custom Module/API editors are focused transient workspaces. Never let
      // them replace the saved Model Builder view or breadcrumbs.
      if(current(state)?.kind==="custom_edit")return;
      ws.view_component_id=state.view_component_id;
      ws.breadcrumbs=cp(state.breadcrumbs||[]);
    }

    function workspaceName(){
      const c=current(state);
      if(c?.kind==="custom_edit"){
        const def=state.custom_components?.[c.definition_id];
        return String(def?.implementation||"graph")==="api" ? "API Component Editor" : "Module Editor";
      }
      return state.active_workspace==="data" ? "Data Processing" : "Model Builder";
    }

    function inspectorRenderKey(){
      const target=outputDirectorySelection
        ?("output:"+outputDirectorySelection)
        :(selected?("node:"+selected):"empty");
      return (state.active_workspace||"model")+"|"+inspectorTab+"|"+target;
    }

    function renameProjectInline(rawName){
      const name=String(rawName||"").trim().replace(/\s+/g," ");
      const modelRootId=state.workspaces?.model?.root_component_id||state.root_component_id;
      const modelRoot=state.components?.[modelRootId];
      const oldName=String(state.project?.name||modelRoot?.name||"Untitled Model");
      if(!name){setStatus("Model name cannot be empty.");draw();return false;}
      if(name===oldName)return true;
      if(modelRoot && layoutNameExists(name,modelRoot.id)){
        setStatus('Another layout is already named "'+name+'".');draw();return false;
      }
      checkpoint("Rename model");
      state.project=state.project||{};
      state.project.name=name;
      if(modelRoot)modelRoot.name=name;
      const modelWs=state.workspaces?.model;
      if(modelWs?.breadcrumbs?.length)modelWs.breadcrumbs[0].name=name;
      if(state.active_workspace==="model" && state.breadcrumbs?.length && state.breadcrumbs[0]?.id===modelRootId){
        state.breadcrumbs[0].name=name;
      }
      setStatus('Model renamed to "'+name+'".');
      draw();
      return true;
    }

    function switchWorkspace(next){
      if(next===state.active_workspace)return;
      const oldKey=state.active_workspace||"model";
      const oldCanvas=root.querySelector(".mlb-canvas");
      if(oldCanvas){
        workspaceScroll[oldKey]={left:oldCanvas.scrollLeft,top:oldCanvas.scrollTop};
      }
      const oldSidebar=root.querySelector(".mlb-sidebar");
      if(oldSidebar){
        sidebarScroll[oldKey]={left:oldSidebar.scrollLeft,top:oldSidebar.scrollTop};
      }
      rememberWorkspaceView();
      runtimePanel=null;
      state.active_workspace=next;
      const ws=state.workspaces[next];
      state.view_component_id=ws.view_component_id||ws.root_component_id;
      state.breadcrumbs=cp(ws.breadcrumbs||[{id:ws.root_component_id,name:ws.name}]);
      selected=null;pendingPort=null;search="";
      switchingWorkspace=true;
      setStatus(workspaceName()+" opened.");
      draw();
    }

    const dataNodeTypes=new Set([
      "manual_dataset","hf_dataset","kaggle_dataset","url_dataset","local_dataset",
      "text_process","train_test_split","tokenize_text","image_process","audio_process",
      "batch_data","prepared_dataset"
    ]);
    function itemWorkspace(item){
      return dataNodeTypes.has(item.type) ? "data" : "model";
    }
    function defaultDataNodes(){
      const nodes=[
        makeNode(cat(catalog,"hf_dataset")),
        makeNode(cat(catalog,"text_process")),
        makeNode(cat(catalog,"train_test_split")),
        makeNode(cat(catalog,"tokenize_text")),
        makeNode(cat(catalog,"prepared_dataset"))
      ];
      nodes[0].params.dataset_id="roneneldan/TinyStories";
      nodes[0].params.split="train";
      nodes[0].params.max_rows=10000;
      nodes[2].params.train_size=90;
      nodes[2].params.validation_size=5;
      nodes[2].params.test_size=5;
      const edges=[];
      for(let i=0;i<nodes.length-1;i++){
        const e=edge(nodes[i].id,nodes[i+1].id,"main");
        e.source_port="main_out";e.target_port="main_in";edges.push(e);
      }
      return {nodes,edges};
    }


    ensureWorkspaces();
    loadGalleryStorage();
    loadComponentCacheStorage();

    function normalizedUserName(value){return String(value||"").trim().replace(/\s+/g," ").toLowerCase();}

    function layoutIsLocked(componentId=state.view_component_id){
      return !!state.layout_locks?.[componentId];
    }

    function requireEditableLayout(action="edit this layout"){
      if(!layoutIsLocked())return true;
      setStatus("Layout is locked. Click Edit Layout before you "+action+".");
      return false;
    }

    function toggleLayoutLock(){
      const id=state.view_component_id;
      if(!id)return;
      checkpoint(layoutIsLocked(id)?"Edit layout":"Lock layout");
      state.layout_locks[id]=!layoutIsLocked(id);
      pendingPort=null;
      setStatus(state.layout_locks[id]?"Layout locked. Structure is protected.":"Edit Layout enabled.");
      draw();
    }

    function nodeNameExists(name,component=current(state),exceptId=null){
      const wanted=normalizedUserName(name);
      return !!wanted && (component?.nodes||[]).some(n=>n.id!==exceptId&&normalizedUserName(n.name)===wanted);
    }

    function uniqueNodeName(base,component=current(state),exceptId=null){
      const clean=String(base||"Component").trim().replace(/\s+/g," ")||"Component";
      if(!nodeNameExists(clean,component,exceptId))return clean;
      let i=2;
      while(nodeNameExists(clean+" "+i,component,exceptId))i++;
      return clean+" "+i;
    }

    function nodeDisplayName(node){
      if(!node)return "Component";
      if(node.display_name)return String(node.display_name);
      let base="";
      if(node.definition_id&&state.custom_components?.[node.definition_id]){
        base=String(state.custom_components[node.definition_id].name||"").trim();
      }else{
        base=String(cat(catalog,node.type)?.name||"").trim();
      }
      const actual=String(node.name||base||"Component").trim();
      if(!base)return actual;
      if(actual===base)return base;
      if(actual.startsWith(base+" ")){
        const suffix=actual.slice(base.length+1);
        if(/^\d+$/.test(suffix) || /^Copy(?: \d+)?$/i.test(suffix))return base;
      }
      return actual;
    }

    function compactIconLabel(label){
      const raw=String(label||"ML").trim().toUpperCase();
      if(!raw) return "ML";
      if(raw==="SPLIT") return "SPLT";
      return raw.length>4 ? raw.slice(0,4) : raw;
    }

    function layoutNameExists(name,exceptId=null){
      const wanted=normalizedUserName(name);
      if(!wanted)return false;
      return Object.values(state.components||{}).some(c=>c.id!==exceptId&&normalizedUserName(c.name)===wanted);
    }

    function uniqueCustomDefinitionName(base){
      const clean=String(base||"My Module").trim().replace(/\s+/g," ")||"My Module";
      if(!customNameExists(clean))return clean;
      let i=2;
      while(customNameExists(clean+" "+i))i++;
      return clean+" "+i;
    }

    function renameCurrentLayout(){
      const c=current(state);if(!c)return;
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const proposed=win&&typeof win.prompt==="function"?win.prompt("Rename layout:",c.name||state.project?.name||"Layout"):null;
      if(proposed===null)return;
      const name=String(proposed||"").trim().replace(/\s+/g," ");
      if(!name){setStatus("Layout name cannot be empty.");return;}
      if(layoutNameExists(name,c.id)){setStatus('Another layout is already named "'+name+'".');return;}
      checkpoint("Rename layout");
      const oldName=c.name;
      c.name=name;
      const crumbs=state.breadcrumbs||[];
      if(crumbs.length)crumbs[crumbs.length-1].name=name;
      const ws=state.workspaces?.[state.active_workspace];
      if(ws?.breadcrumbs?.length)ws.breadcrumbs[ws.breadcrumbs.length-1].name=name;
      if(c.id===state.workspaces?.model?.root_component_id){
        state.project=state.project||{};state.project.name=name;
        if(state.workspaces.model?.breadcrumbs?.length)state.workspaces.model.breadcrumbs[0].name=name;
        if(state.breadcrumbs?.length===1)state.breadcrumbs[0].name=name;
      }
      if(c.kind==="custom_edit"&&c.definition_id&&state.custom_components?.[c.definition_id]){
        if(customNameExists(name,c.definition_id,c.definition_id)){
          c.name=oldName;
          setStatus('An unrelated Module/API Component named "'+name+'" already exists.');
          return;
        }
        const def=state.custom_components[c.definition_id];
        def.name=name;
        Object.values(state.components||{}).forEach(comp=>(comp.nodes||[]).forEach(n=>{if(n.definition_id===def.id){n.name=uniqueNodeName(name,comp,n.id);n.display_name=name;}}));
      }
      setStatus('Layout renamed to "'+name+'".');draw();
    }

    function renameSelectedComponent(){
      const n=selectedNode();if(!n)return;
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const proposed=win&&typeof win.prompt==="function"?win.prompt("Rename component:",n.name||"Component"):null;
      if(proposed===null)return;
      const name=String(proposed||"").trim().replace(/\s+/g," ");
      if(!name){setStatus(kind+" name cannot be empty.");return;}
      if(nodeNameExists(name,current(state),n.id)){setStatus('Another component in this layout is already named "'+name+'".');return;}
      checkpoint("Rename component");n.name=name;n.display_name=name;setStatus('Component renamed to "'+name+'".');draw();
    }

    const galleryStorageKey="mlb-studio-gallery-v1";
    function loadGalleryStorage(){
      try{
        const store=(root.ownerDocument?.defaultView||window).localStorage;
        const parsed=JSON.parse(store.getItem(galleryStorageKey)||"null");
        if(!parsed)return;
        ["components","models","data"].forEach(kind=>{
          const existing=new Set((state.gallery[kind]||[]).map(x=>x.id));
          (parsed[kind]||[]).forEach(item=>{if(item?.id&&!existing.has(item.id))state.gallery[kind].push(cp(item));});
        });
      }catch(_){/* browser storage is optional in notebook iframes */}
    }

    function persistGallery(){
      try{(root.ownerDocument?.defaultView||window).localStorage.setItem(galleryStorageKey,JSON.stringify(state.gallery));}catch(_){ }
    }

    const componentCacheStorageKey="mlb-studio-component-cache-v1";
    function loadComponentCacheStorage(){
      try{
        const store=(root.ownerDocument?.defaultView||window).localStorage;
        const parsed=JSON.parse(store.getItem(componentCacheStorageKey)||"null");
        if(parsed&&typeof parsed==="object")Object.entries(parsed).forEach(([id,item])=>{if(id&&item&&!state.component_cache[id])state.component_cache[id]=cp(item);});
      }catch(_){/* optional in sandboxed notebook frames */}
    }
    function persistComponentCache(){
      try{(root.ownerDocument?.defaultView||window).localStorage.setItem(componentCacheStorageKey,JSON.stringify(state.component_cache||{}));}catch(_){ }
    }
    function userSourceHash(source){
      const s=String(source||"");let h=0x811c9dc5;
      for(let i=0;i<s.length;i++){h^=s.charCodeAt(i);h=Math.imul(h,0x01000193);}
      return "fnv1a-"+(h>>>0).toString(16).padStart(8,"0");
    }
    function extractPythonDependencies(source){
      const deps=new Set();
      String(source||"").split(/\r?\n/).forEach(line=>{
        const clean=line.replace(/#.*/,"").trim();
        let m=clean.match(/^import\s+(.+)$/);
        if(m){m[1].split(",").forEach(part=>{const rootName=String(part||"").trim().split(/\s+as\s+/i)[0].split(".")[0];if(rootName)deps.add(rootName);});return;}
        m=clean.match(/^from\s+([A-Za-z_][\w.]*)\s+import\s+/);
        if(m){const rootName=m[1].split(".")[0];if(rootName)deps.add(rootName);}
      });
      return [...deps].sort();
    }
    function userSourcePayload(step,definitionId=""){
      const b=ensureAPIStepObjectIds(step);
      const kind=b.call_type;
      if(!["user_function","user_class"].includes(kind))return null;
      const source=kind==="user_class"?b.user_class_code:b.user_code;
      const entryName=kind==="user_class"?b.user_class_name:b.user_function_name;
      const hash=userSourceHash(source);
      const deps=extractPythonDependencies(source);
      b.dependencies=deps;b.source_hash=hash;
      const expectedPrefix="source::"+String(definitionId||"component")+"::";
      if(!b.source_cache_id||!String(b.source_cache_id).startsWith(expectedPrefix))b.source_cache_id=expectedPrefix+step.id;
      const previous=state.component_cache?.[b.source_cache_id];
      const revision=previous&&previous.source_hash===hash?Number(previous.revision||1):Number(previous?.revision||0)+1;
      b.source_revision=Math.max(1,revision);
      return {id:b.source_cache_id,kind,entry_point:entryName,source_code:String(source||""),dependencies:deps,source_hash:hash,revision:b.source_revision,updated_at:new Date().toISOString(),node_id:step.id,node_name:step.name||entryName};
    }
    function cacheUserSourcesForDefinition(def,c=null){
      if(!def)return;
      const sourceNodes=c?.nodes||def.nodes||[];
      sourceNodes.forEach(step=>{if(step?.type!=="api_step")return;const payload=userSourcePayload(step,def.id);if(payload)state.component_cache[payload.id]=payload;});
      persistComponentCache();
    }
    function hydrateCachedUserSources(def){
      if(!def)return def;
      (def.nodes||[]).forEach(step=>{
        if(step?.type!=="api_step")return;
        const b=ensureAPIStepObjectIds(step);const cached=b.source_cache_id?state.component_cache?.[b.source_cache_id]:null;
        if(!cached)return;
        if(b.call_type==="user_function"&&!String(b.user_code||"").trim())b.user_code=String(cached.source_code||"");
        if(b.call_type==="user_class"&&!String(b.user_class_code||"").trim())b.user_class_code=String(cached.source_code||"");
        if(!b.dependencies?.length)b.dependencies=cp(cached.dependencies||[]);
        b.source_hash=b.source_hash||cached.source_hash||"";b.source_revision=b.source_revision||cached.revision||1;
      });
      return def;
    }

    function galleryNameExists(kind,name,exceptId=null){
      const wanted=normalizedUserName(name);
      return (state.gallery?.[kind]||[]).some(x=>x.id!==exceptId&&normalizedUserName(x.name)===wanted);
    }

    function askGalleryName(kind,defaultName,label){
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const proposed=win&&typeof win.prompt==="function"?win.prompt(label,defaultName||""):null;
      if(proposed===null)return null;
      const name=String(proposed||"").trim().replace(/\s+/g," ");
      if(!name){setStatus("Gallery name cannot be empty.");return null;}
      if(galleryNameExists(kind,name)){setStatus('Gallery already contains "'+name+'".');return null;}
      return name;
    }

    function saveCurrentToGallery(){
      const c=current(state);if(!c)return;
      if(c.kind==="custom_edit"){
        const def=state.custom_components?.[c.definition_id];
        const name=askGalleryName("components",def?.name||c.name,"Save Module / API Component to Gallery as:");
        if(!name)return;
        const snapshot=customGallerySnapshot(def,c);snapshot.name=name;
        state.gallery.components.push({
          id:uid("gallery_component"),name,kind:"component",saved_at:new Date().toISOString(),
          source_definition_id:def?.id||snapshot.id,definition:snapshot
        });
        persistGallery();persistComponentCache();setStatus(name+" saved to Gallery with cached user source.");draw();return;
      }
      if(state.active_workspace==="data"){
        const pipeline=current(state);if(!pipeline)return;
        const name=askGalleryName("data",pipeline.name||"Data Pipeline","Save data pipeline to Gallery as:");
        if(!name)return;
        state.gallery.data.push({
          id:uid("gallery_data"),name,kind:"data",saved_at:new Date().toISOString(),
          architecture:cp(pipeline)
        });
        persistGallery();setStatus(name+" saved to Data Gallery.");draw();return;
      }
      if(state.active_workspace!=="model"){
        setStatus("Open Model Builder or Data Processing to save the current design to Gallery.");return;
      }
      const model=modelRootComponent();if(!model)return;
      const name=askGalleryName("models",state.project?.name||model.name||"My Model","Save model to Gallery as:");
      if(!name)return;
      state.gallery.models.push({
        id:uid("gallery_model"),name,kind:"model",saved_at:new Date().toISOString(),
        project:cp(state.project||{}),architecture:cp(model),custom_components:cp(state.custom_components||{}),component_cache:cp(state.component_cache||{})
      });
      persistGallery();setStatus(name+" saved to Model Gallery.");draw();
    }

    function restoreGalleryComponentDefinition(entry,{installed=false,allowNameConflictWith=null}={}){
      if(!entry?.definition)return null;
      let existing=Object.values(state.custom_components||{}).find(d=>d.gallery_entry_id===entry.id||d.id===entry.source_definition_id);
      if(existing){
        if(installed){existing.palette_hidden=false;existing.palette_installed=true;}
        return existing;
      }

      const rootSource=cp(entry.definition);
      const embeddedCache=cp(rootSource.component_cache||{});
      Object.entries(embeddedCache).forEach(([id,item])=>{if(id&&item)state.component_cache[id]=cp(item);});
      persistComponentCache();
      const dependencyDefs=cp(rootSource.dependency_definitions||{});
      delete rootSource.dependency_definitions;
      delete rootSource.component_cache;
      const oldRootId=rootSource.id||entry.source_definition_id||uid("gallery_root");
      const remap={};
      const created=[];

      Object.entries(dependencyDefs).forEach(([oldId,snapshot])=>{
        if(!snapshot)return;
        let dep=state.custom_components?.[oldId]||null;
        if(dep){remap[oldId]=dep.id;return;}
        dep=cp(snapshot);delete dep.dependency_definitions;
        const newId=uid("custom");remap[oldId]=newId;dep.id=newId;
        // Names are display labels. Nested Modules are identified by stable IDs,
        // so parent and child are allowed to use the same visible name.
        dep.name=String(dep.name||"Module").trim()||"Module";
        dep.palette_hidden=true;dep.palette_installed=false;dep.gallery_entry_id=dep.gallery_entry_id||null;
        hydrateCachedUserSources(dep);created.push(dep);
      });

      const root=cp(rootSource);
      const rootId=uid("custom");remap[oldRootId]=rootId;root.id=rootId;
      const requestedRootName=String(entry.name||root.name||"Module").trim().replace(/\s+/g," ")||"Module";
      root.name=allowNameConflictWith
        ?requestedRootName
        :uniqueCustomDefinitionName(requestedRootName);
      root.gallery_entry_id=entry.id;root.palette_hidden=!installed;root.palette_installed=!!installed;hydrateCachedUserSources(root);

      const remapDefinitionRefs=def=>{
        (def.nodes||[]).forEach(n=>{if(n?.definition_id&&remap[n.definition_id])n.definition_id=remap[n.definition_id];});
      };
      created.forEach(remapDefinitionRefs);remapDefinitionRefs(root);
      created.forEach(dep=>{state.custom_components[dep.id]=dep;cacheUserSourcesForDefinition(dep);});
      state.custom_components[root.id]=root;cacheUserSourcesForDefinition(root);
      return root;
    }

    function addGalleryComponent(entry){
      const source=restoreGalleryComponentDefinition(entry,{installed:true});
      if(!source)return;
      persistGallery();setStatus(source.name+" added to the Component Library.");draw();
    }

    function loadGalleryModel(entry){
      if(!entry?.architecture)return;
      checkpoint("Load model from Gallery");
      rememberWorkspaceView();
      state.active_workspace="model";
      const rootId=state.workspaces.model.root_component_id;
      const architecture=cp(entry.architecture);
      Object.entries(entry.component_cache||{}).forEach(([id,item])=>{if(id&&item)state.component_cache[id]=cp(item);});
      persistComponentCache();
      const remap={};
      const importedDefs=[];
      // Custom definitions are identified by IDs, never by their display names.
      // Clone the saved dependency graph as one self-contained namespace so two
      // Modules may legitimately share the same visible label.
      Object.entries(entry.custom_components||{}).forEach(([oldId,def])=>{
        const newId=uid("custom");
        remap[oldId]=newId;
        const copyDef=cp(def);copyDef.id=newId;
        copyDef.name=String(copyDef.name||"Module").trim()||"Module";
        copyDef.gallery_entry_id=null;hydrateCachedUserSources(copyDef);
        importedDefs.push(copyDef);
      });
      importedDefs.forEach(copyDef=>{
        (copyDef.nodes||[]).forEach(n=>{if(n?.definition_id&&remap[n.definition_id])n.definition_id=remap[n.definition_id];});
        state.custom_components[copyDef.id]=copyDef;
      });
      (architecture.nodes||[]).forEach(n=>{if(n.definition_id&&remap[n.definition_id])n.definition_id=remap[n.definition_id];n.name=uniqueNodeName(n.name,{nodes:(architecture.nodes||[]).filter(x=>x.id!==n.id)},n.id);});
      architecture.id=rootId;architecture.name=entry.name;
      state.components[rootId]=architecture;
      state.root_component_id=rootId;state.view_component_id=rootId;
      state.project={...(entry.project||{}),name:entry.name};
      state.breadcrumbs=[{id:rootId,name:entry.name}];
      state.workspaces.model.view_component_id=rootId;state.workspaces.model.breadcrumbs=cp(state.breadcrumbs);
      selected=null;pendingPort=null;setStatus(entry.name+" loaded from Gallery.");draw();
    }

    function loadGalleryData(entry){
      if(!entry?.architecture)return;
      checkpoint("Load data pipeline from Gallery");
      rememberWorkspaceView();
      state.active_workspace="data";
      const ws=state.workspaces.data;
      const rootId=ws.root_component_id;
      const architecture=cp(entry.architecture);
      architecture.id=rootId;architecture.name=entry.name;architecture.kind="data";
      state.components[rootId]=architecture;
      state.view_component_id=rootId;
      state.breadcrumbs=[{id:rootId,name:entry.name}];
      ws.view_component_id=rootId;ws.breadcrumbs=cp(state.breadcrumbs);
      selected=null;pendingPort=null;
      execution={status:"idle",overall:0,message:"Ready",nodes:{}};
      switchingWorkspace=true;
      setStatus(entry.name+" loaded from Gallery.");draw();
    }

    function editGalleryComponent(entry){
      const parentDef=activeCustomDefinition();
      const def=restoreGalleryComponentDefinition(entry,{installed:false,allowNameConflictWith:parentDef?.id||null});
      if(def)editCustomDefinition(def);
    }

    function removeGalleryEntry(kind,id){
      const entry=(state.gallery?.[kind]||[]).find(x=>x.id===id);
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const label=entry?.name||"this Gallery item";
      if(win&&typeof win.confirm==="function"&&!win.confirm('Remove "'+label+'" from Gallery?'))return;
      checkpoint("Remove Gallery item");
      state.gallery[kind]=(state.gallery[kind]||[]).filter(x=>x.id!==id);
      persistGallery();setStatus(label+" removed from Gallery.");draw();
    }

    function openGallery(tab){
      if(current(state)?.kind==="custom_edit"){
        setStatus("Finish the component in the editor. Save returns you to Gallery automatically.");
        draw();
        return;
      }
      if(!galleryWorkspace.open&&!cloudWorkspace.open)galleryPreviousBottomExpanded=bottomExpanded;
      else if(cloudWorkspace.open)galleryPreviousBottomExpanded=cloudPreviousBottomExpanded;
      cloudWorkspace.open=false;
      bottomExpanded=false;
      galleryWorkspace={open:true,tab:["models","components","data"].includes(tab)?tab:"models"};
      outputDirectorySelection=null;
      selected=null;
      setStatus("Gallery opened.");
      draw();
    }

    function closeGallery(){
      galleryWorkspace.open=false;
      bottomExpanded=galleryPreviousBottomExpanded;
      setStatus("Gallery closed.");
      draw();
    }

    function openCloudWorkspace(){
      if(!cloudWorkspace.open&&!galleryWorkspace.open)cloudPreviousBottomExpanded=bottomExpanded;
      else if(galleryWorkspace.open)cloudPreviousBottomExpanded=galleryPreviousBottomExpanded;
      galleryWorkspace.open=false;
      cloudWorkspace.open=true;
      bottomExpanded=false;
      outputDirectorySelection=null;
      selected=null;
      setStatus("Cloud & Repositories opened.");
      draw();
    }

    function closeCloudWorkspace(){
      cloudWorkspace.open=false;
      bottomExpanded=cloudPreviousBottomExpanded;
      setStatus("Cloud & Repositories closed.");
      draw();
    }

    function exitRuntimePanel(){
      if(!runtimePanel)return;
      const mode=runtimePanel.mode;
      runtimePanel=null;
      setStatus((mode==="train"?"Training":mode==="generate"?"Generation":"Runtime")+" view closed.");
      draw();
    }

    function cancelRuntimeToModelEditor(entry,mode){
      const kind=String(mode||runtimePanel?.mode||"");
      const activeRun=execution.status==="running"&&execution.runtime_kind===kind;
      if((kind==="train"||kind==="generate")&&activeRun)requestStop();
      if(kind==="serve"&&(entry?.serve_status==="running"||entry?.serve_live?.local_url))requestServeCommand("serve_stop",entry);
      if(entry&&kind==="train"){
        entry.training_status=activeRun?"stopping":entry.training_status;
        if(entry.training_live&&activeRun)entry.training_live.message="Cancellation requested. Returning to Model Builder…";
      }
      if(entry&&kind==="generate"&&entry.generation_live&&activeRun){
        entry.generation_live.message="Cancellation requested. Returning to Model Builder…";
      }
      runtimePanel=null;
      galleryWorkspace.open=false;
      cloudWorkspace.open=false;
      const label=kind==="train"?"Training":kind==="generate"?"Generation":"Runtime";
      setStatus(activeRun?(label+" cancellation requested. Returned to Model Builder."):"Returned to Model Builder.");
      draw();
    }

    function cancelTrainingToModelEditor(entry){cancelRuntimeToModelEditor(entry,"train");}

    function trainingIsRunning(){
      return execution.status==="running" && execution.runtime_kind==="train";
    }

    function startTrainingFromRuntime(entry){
      if(!entry || trainingIsRunning())return;
      entry.training_status="starting";
      entry.training_history=[];
      entry.training_live={
        status:"running",phase:"starting",overall:0,step:0,max_steps:entry.training_config?.max_steps??null,
        tokens_seen:0,tokens_per_sec:null,avg_tokens_per_sec:null,end_to_end_tokens_per_sec:null,avg_end_to_end_tokens_per_sec:null,loss:null,ppl:null,val_loss:null,val_ppl:null,
        memory_allocated_gb:null,memory_reserved_gb:null,memory_peak_gb:null,memory_total_gb:null,
        elapsed_seconds:null,compile_seconds:null,message:"Starting training in Python…"
      };
      runtimePanel={mode:"train",modelId:entry.id,tab:"status"};
      draw();
      setTimeout(()=>requestRuntimeCommand("train",entry),80);
    }

    function trainingActionButton(entry,valid){
      if(trainingIsRunning()){
        const stop=btn("Stop Training","mlb-runtime-stop");
        stop.addEventListener("click",requestStop);
        return stop;
      }
      const start=btn("Start Training","mlb-runtime-start");
      start.disabled=valid ? !valid.ok : false;
      start.title=(valid && !valid.ok)?"Fix training compatibility/settings before starting":"Start training";
      start.addEventListener("click",()=>startTrainingFromRuntime(entry));
      return start;
    }

    function generationIsRunning(){
      return execution.status==="running" && execution.runtime_kind==="generate";
    }

    function startGenerationFromRuntime(entry){
      if(!entry || generationIsRunning() || !entry.weights_ready)return;
      entry.generation_history=[];
      entry.generation_live={status:"running",phase:"starting",overall:0,generated_tokens:0,message:"Starting generation in Python…",generated_text:""};
      runtimePanel={mode:"generate",modelId:entry.id,tab:"status"};
      draw();
      setTimeout(()=>requestRuntimeCommand("generate",entry),80);
    }

    function generationActionButton(entry){
      if(generationIsRunning()){
        const stop=btn("Stop Generation","mlb-runtime-stop");
        stop.addEventListener("click",requestStop);
        return stop;
      }
      const start=btn("Generate Tokens","mlb-runtime-start");
      start.disabled=!entry?.weights_ready || execution.status==="running";
      start.title=!entry?.weights_ready?"Train or load model weights before generation":"Generate tokens";
      start.addEventListener("click",()=>startGenerationFromRuntime(entry));
      return start;
    }

    function bridgeDocuments(){
      const docs=[];
      const add=doc=>{if(doc && !docs.includes(doc))docs.push(doc);};
      add(document);
      try{add(window.parent && window.parent.document);}catch(_){}
      try{add(window.top && window.top.document);}catch(_){}

      // Kaggle/Jupyter may render output and standard widgets in neighboring
      // same-origin frames. Search accessible frame documents as a fallback.
      const parents=[...docs];
      parents.forEach(doc=>{
        try{
          doc.querySelectorAll("iframe").forEach(frame=>{
            try{add(frame.contentDocument);}catch(_){}
          });
        }catch(_){}
      });
      return docs;
    }

    function deepQuery(rootNode,selector){
      if(!rootNode)return null;
      try{
        const direct=rootNode.querySelector(selector);
        if(direct)return direct;
        const all=rootNode.querySelectorAll("*");
        for(const el of all){
          if(el.shadowRoot){
            const found=deepQuery(el.shadowRoot,selector);
            if(found)return found;
          }
        }
      }catch(_){}
      return null;
    }

    function bridgeRoot(cls){
      if(!cls)return null;
      const selector="."+cls;
      for(const doc of bridgeDocuments()){
        const found=deepQuery(doc,selector);
        if(found)return found;
      }
      return null;
    }

    function bridgeControl(cls,selector){
      if(isPopout){
        if(selector==="button")return {__mlbBroadcastKind:cls===bridge?.stop?"stop":"run",click(){return true;}};
        if(selector==="textarea")return {value:lastProgressRaw||""};
      }
      const host=bridgeRoot(cls);
      if(!host)return null;
      if(host.matches && host.matches(selector))return host;
      return deepQuery(host,selector);
    }

    function setNativeValue(input,value){
      if(!input)return false;
      try{
        const view=input.ownerDocument?.defaultView||window;
        const proto=view.HTMLTextAreaElement && input instanceof view.HTMLTextAreaElement
          ? view.HTMLTextAreaElement.prototype
          : view.HTMLInputElement?.prototype;
        const descriptor=proto ? Object.getOwnPropertyDescriptor(proto,"value") : null;
        if(descriptor?.set)descriptor.set.call(input,value);
        else input.value=value;

        input.dispatchEvent(new view.Event("input",{bubbles:true,composed:true}));
        input.dispatchEvent(new view.Event("change",{bubbles:true,composed:true}));
        return true;
      }catch(_){
        try{
          input.value=value;
          input.dispatchEvent(new Event("input",{bubbles:true}));
          input.dispatchEvent(new Event("change",{bubbles:true}));
          return true;
        }catch(__){return false;}
      }
    }

    function popoutPacket(message){
      return Object.assign({__mlb_studio_popout__:true,channel:popoutChannelName},message||{});
    }

    function attachPopoutMessagePort(port){
      if(!port)return false;
      try{
        if(popoutMessagePort && popoutMessagePort!==port){
          try{popoutMessagePort.close();}catch(_){}
        }
        popoutMessagePort=port;
        popoutMessagePort.onmessage=event=>{
          try{handlePopoutMessage(event.data||{},null);}catch(_){}
        };
        try{popoutMessagePort.start();}catch(_){}
        return true;
      }catch(_){return false;}
    }

    function sendPopoutMessage(message){
      const packet=popoutPacket(message);
      let sent=false;
      // Prefer the dedicated transferred MessagePort, but also send through the
      // browser-window fallbacks. Duplicates are harmless and this prevents a
      // stale port from hiding a working opener/BroadcastChannel route.
      try{if(popoutMessagePort){popoutMessagePort.postMessage(packet);sent=true;}}catch(_){popoutMessagePort=null;}

      if(isPopout){
        try{
          if(window.opener && !window.opener.closed){
            window.opener.postMessage(packet,"*");
            sent=true;
          }
        }catch(_){}
        try{if(popoutChannel){popoutChannel.postMessage(packet);sent=true;}}catch(_){}
        return sent;
      }

      try{
        if(popoutPeerWindow && !popoutPeerWindow.closed){
          popoutPeerWindow.postMessage(packet,"*");
          sent=true;
        }
      }catch(_){}
      try{if(popoutChannel){popoutChannel.postMessage(packet);sent=true;}}catch(_){}
      return sent;
    }

    function sendHostReply(targetWindow,message){
      const packet=popoutPacket(Object.assign({source:"host"},message||{}));
      let sent=false;
      try{if(popoutMessagePort){popoutMessagePort.postMessage(packet);sent=true;}}catch(_){popoutMessagePort=null;}
      try{
        if(targetWindow && !targetWindow.closed){
          targetWindow.postMessage(packet,"*");
          sent=true;
        }
      }catch(_){}
      try{
        if(popoutPeerWindow && popoutPeerWindow!==targetWindow && !popoutPeerWindow.closed){
          popoutPeerWindow.postMessage(packet,"*");
          sent=true;
        }
      }catch(_){}
      try{if(popoutChannel){popoutChannel.postMessage(packet);sent=true;}}catch(_){}
      return sent;
    }

    function clickBridgeButton(button){
      if(!button)return false;
      if(isPopout&&button.__mlbBroadcastKind){
        if(!popoutHostConnected)return false;
        if(button.__mlbBroadcastKind==="stop") return sendPopoutMessage({type:"stop",source:"popout",ts:Date.now()});
        const ok=sendPopoutMessage({type:"command",source:"popout",ts:Date.now(),state:pendingBroadcastState||bridgeStatePayload(),command:pendingBroadcastCommand||{action:"data",ts:Date.now()}});
        pendingBroadcastCommand=null;
        return ok;
      }
      try{
        button.click();
        return true;
      }catch(_){
        try{
          const view=button.ownerDocument?.defaultView||window;
          button.dispatchEvent(new view.MouseEvent("click",{
            bubbles:true,cancelable:true,view
          }));
          return true;
        }catch(__){return false;}
      }
    }

    function bridgeReady(){
      if(!bridge)return false;
      if(isPopout)return !!popoutHostConnected;
      return !!(
        bridgeControl(bridge.state,"textarea") &&
        (!bridge.command||bridgeControl(bridge.command,"textarea")) &&
        bridgeControl(bridge.run,"button") &&
        bridgeControl(bridge.stop,"button") &&
        bridgeControl(bridge.progress,"textarea")
      );
    }

    function updateKernelBadge(){
      const badge=root.querySelector(".mlb-kernel-badge");
      if(!badge)return;
      const ready=bridgeReady();
      bridgeLastReady=ready;
      badge.className="mlb-kernel-badge "+(ready?"connected":"offline");
      badge.innerHTML=ready
        ?"<i></i><span>Kernel Connected</span>"
        :"<i></i><span>Kernel Offline</span>";
      badge.title=ready
        ?"Run can execute this data pipeline in the Python kernel."
        :"Builder cannot currently reach the Python widget bridge.";
      if(ready) pumpComponentImportQueue();
    }

    function bridgeStatePayload(){
      const clean=cp(state);
      delete clean._runtime_command;
      delete clean._session_secrets;
      // API/ngrok secrets are browser-session values, never project state.
      (clean.model_outputs||[]).forEach(entry=>{
        if(entry.serve_live&&typeof entry.serve_live==="object"){
          delete entry.serve_live.api_key;
        }
      });
      return clean;
    }

    function setBridgeState(){
      if(!bridge)return false;
      if(isPopout){pendingBroadcastState=bridgeStatePayload();return true;}
      const input=bridgeControl(bridge.state,"textarea");
      if(!input)return false;
      return setNativeValue(input,JSON.stringify(bridgeStatePayload()));
    }

    function setBridgeCommand(command){
      if(!bridge)return false;
      if(isPopout){pendingBroadcastCommand=cp(command||{});return true;}
      if(!bridge.command){
        // Backward compatibility with an older Python bridge.
        state._runtime_command=cp(command||{});
        const ok=setBridgeState();
        delete state._runtime_command;
        return ok;
      }
      const input=bridgeControl(bridge.command,"textarea");
      if(!input)return false;
      return setNativeValue(input,JSON.stringify(command||{}));
    }

    function markExecutionLocally(kind,message){
      const nodes={};
      (current(state).nodes||[]).forEach(n=>{
        nodes[n.id]={status:kind,message:message||kind};
      });
      execution={status:kind,overall:0,message:message||kind,nodes};
      applyExecutionProgress(execution);
    }

    function clientDataValidation(){
      if(state.active_workspace!=="data")return [];
      const comp=current(state);
      const nodes=comp.nodes||[];
      const edges=(comp.edges||[]).filter(e=>(e.kind||"main")==="main");
      const sources=new Set(["manual_dataset","hf_dataset","kaggle_dataset","url_dataset","local_dataset"]);
      const sourceNodes=nodes.filter(n=>sources.has(n.type));
      const outputs=nodes.filter(n=>n.type==="prepared_dataset");
      const outgoing={};nodes.forEach(n=>outgoing[n.id]=[]);
      edges.forEach(e=>{if(outgoing[e.source])outgoing[e.source].push(e.target);});
      const errors=[];

      if(sourceNodes.length!==1){
        errors.push({
          node_ids:sourceNodes.map(n=>n.id),
          message:"Use exactly one Data Source. Found "+sourceNodes.length+"."
        });
      }
      if(outputs.length!==1){
        errors.push({
          node_ids:outputs.map(n=>n.id),
          message:"Use exactly one Prepared Dataset output. Found "+outputs.length+"."
        });
      }else if((outgoing[outputs[0].id]||[]).length){
        errors.push({
          node_ids:[outputs[0].id],
          message:"Prepared Dataset must be the final step."
        });
      }
      nodes.filter(n=>n.type==="train_test_split").forEach(n=>{
        const total=splitTotal(n);
        if(!splitIsValid(n)){
          errors.push({
            node_ids:[n.id],
            message:"Train + Validation + Test must equal 100%. Current total: "+total+"%."
          });
        }
      });
      return errors;
    }

    function showClientErrors(errors){
      const nodeStates={};
      (current(state).nodes||[]).forEach(n=>nodeStates[n.id]={status:"queued",message:"Waiting"});
      errors.forEach(err=>(err.node_ids||[]).forEach(id=>{
        nodeStates[id]={status:"error",message:err.message};
      }));
      execution={
        status:"error",
        overall:0,
        message:errors[0]?.message||"Pipeline needs attention.",
        nodes:nodeStates
      };
      applyExecutionProgress(execution);
      setStatus(execution.message);
    }

    function requestServeCommand(action,entry){
      if(!entry)return;
      updateKernelBadge();
      if(!bridgeReady()){
        execution={status:"error",runtime_kind:"serve",overall:0,message:"Kernel bridge is offline. Re-run the Builder cell, then try again.",nodes:{}};
        applyExecutionProgress(execution);setStatus(execution.message);return;
      }
      const secret=serveSecrets[entry.id]||{api_key:"",ngrok_token:""};
      const command={action,model_id:entry.id,serve:{
        config:cp(entry.serve_config||{}),
        credentials:{api_key:secret.api_key||"",ngrok_token:secret.ngrok_token||""}
      },ts:Date.now()};
      if(!setBridgeState()||!setBridgeCommand(command)){setStatus("Could not send API server configuration to Python.");return;}
      const button=bridgeControl(bridge.run,"button");
      if(!button){setStatus("Python API server control was not found.");return;}
      const progressInput=bridgeControl(bridge.progress,"textarea");lastProgressRaw=progressInput?.value||lastProgressRaw;
      execution={status:"running",runtime_kind:"serve",phase:action,overall:0,message:
        action==="serve_start"?"Starting model API server…":action==="serve_stop"?"Stopping model API server…":"Checking model API server…",nodes:{}};
      applyExecutionProgress(execution);setStatus(execution.message);
      setTimeout(()=>{clickBridgeButton(button);},250);
    }

    function queueComponentImport(componentType){
      const type=String(componentType||"").trim();
      const api=mlapi[type];
      if(!type||!api||api.builder_utility||!api.import_path||api.loaded)return;
      if(componentImportQueue.includes(type))return;
      componentImportQueue.push(type);
      pumpComponentImportQueue();
    }

    function pumpComponentImportQueue(){
      if(componentImportBusy||!componentImportQueue.length||!bridgeReady())return;
      const type=componentImportQueue.shift();
      const api=mlapi[type];
      if(!api||api.loaded){pumpComponentImportQueue();return;}
      const command={action:"ensure_component_import",component_type:type,ts:Date.now()};
      if(!setBridgeState()||!setBridgeCommand(command)){
        componentImportQueue.unshift(type);
        return;
      }
      const button=bridgeControl(bridge.run,"button");
      if(!button){componentImportQueue.unshift(type);return;}
      componentImportBusy=true;
      setTimeout(()=>{
        const ok=clickBridgeButton(button);
        if(!ok){
          componentImportBusy=false;
          componentImportQueue.unshift(type);
        }
      },40);
    }

    function requestRuntimeCommand(action,entry){
      if(!entry)return;
      updateKernelBadge();
      if(!bridgeReady()){
        execution={status:"error",runtime_kind:action,overall:0,message:"Kernel bridge is offline. Re-run the Builder cell, then try again.",nodes:{}};
        applyExecutionProgress(execution);setStatus(execution.message);return;
      }
      const command={action,model_id:entry.id,ts:Date.now()};
      if(!setBridgeState()||!setBridgeCommand(command)){
        setStatus("Could not send runtime configuration to Python.");return;
      }
      const button=bridgeControl(bridge.run,"button");
      if(!button){setStatus("Python runtime control was not found.");return;}
      const progressInput=bridgeControl(bridge.progress,"textarea");
      lastProgressRaw=progressInput?.value||lastProgressRaw;
      execution={status:"running",runtime_kind:action,phase:"starting",overall:0,message:action==="train"?"Starting training in Python…":"Starting generation in Python…",nodes:{}};
      applyExecutionProgress(execution);setStatus(execution.message);
      setTimeout(()=>{clickBridgeButton(button);},350);
    }

    function requestLocalCommand(action,config={}){
      updateKernelBadge();
      if(!bridgeReady()){
        execution={status:"error",runtime_kind:"local",overall:0,message:"Kernel bridge is offline. Re-run the Builder cell, then try again.",nodes:{}};
        applyExecutionProgress(execution);setStatus(execution.message);return;
      }
      const command={action,local:cp(config),ts:Date.now()};
      if(!setBridgeState()||!setBridgeCommand(command)){setStatus("Could not send local filesystem command to Python.");return;}
      const button=bridgeControl(bridge.run,"button");
      if(!button){setStatus("Python local filesystem control was not found.");return;}
      const progressInput=bridgeControl(bridge.progress,"textarea");
      lastProgressRaw=progressInput?.value||lastProgressRaw;
      execution={status:"running",runtime_kind:"local",phase:action,overall:0,message:
        action==="local_import_models"?"Scanning and importing models…":
        action==="local_import_data"?"Scanning and importing datasets…":
        action==="local_scan"?"Scanning local environment…":"Loading local content…",nodes:{}};
      applyExecutionProgress(execution);setStatus(execution.message);
      setTimeout(()=>{clickBridgeButton(button);},250);
    }

    function requestCloudCommand(action,config={}){
      updateKernelBadge();
      if(!bridgeReady()){
        execution={status:"error",runtime_kind:"cloud",overall:0,message:"Kernel bridge is offline. Re-run the Builder cell, then try again.",nodes:{}};
        applyExecutionProgress(execution);setStatus(execution.message);return;
      }
      const command={action,cloud:cp(config),ts:Date.now()};
      if(!setBridgeState()||!setBridgeCommand(command)){
        setStatus("Could not send cloud command to Python.");return;
      }
      const button=bridgeControl(bridge.run,"button");
      if(!button){setStatus("Python cloud control was not found.");return;}
      const progressInput=bridgeControl(bridge.progress,"textarea");
      lastProgressRaw=progressInput?.value||lastProgressRaw;
      execution={status:"running",runtime_kind:"cloud",phase:action,overall:0,message:"Connecting to cloud provider…",nodes:{}};
      applyExecutionProgress(execution);setStatus(execution.message);
      setTimeout(()=>{clickBridgeButton(button);},250);
    }

    function requestHubCommand(action,config={}){
      updateKernelBadge();
      if(!bridgeReady()){
        execution={status:"error",runtime_kind:"hub",overall:0,message:"Kernel bridge is offline. Re-run the Builder cell, then try again.",nodes:{}};
        applyExecutionProgress(execution);setStatus(execution.message);return;
      }
      const command={action,hub:cp(config),ts:Date.now()};
      if(!setBridgeState()||!setBridgeCommand(command)){
        setStatus("Could not send Hugging Face command to Python.");return;
      }
      const button=bridgeControl(bridge.run,"button");
      if(!button){setStatus("Python Hub control was not found.");return;}
      const progressInput=bridgeControl(bridge.progress,"textarea");
      lastProgressRaw=progressInput?.value||lastProgressRaw;
      execution={status:"running",runtime_kind:"hub",phase:action,overall:0,message:"Connecting to Hugging Face Hub…",nodes:{}};
      applyExecutionProgress(execution);setStatus(execution.message);
      setTimeout(()=>{clickBridgeButton(button);},300);
    }

    function requestRun(){
      if(state.active_workspace!=="data"){
        setStatus("Model execution is not compiled yet. Run is currently available for Data Processing.");
        draw();
        return;
      }

      const errors=clientDataValidation();
      if(errors.length){
        showClientErrors(errors);
        return;
      }

      updateKernelBadge();
      if(!bridgeReady()){
        execution={
          status:"error",
          overall:0,
          message:"Kernel bridge is offline. Re-run the Builder cell, then click Run.",
          nodes:{}
        };
        applyExecutionProgress(execution);
        setStatus(execution.message);
        return;
      }

      if(!setBridgeState()){
        execution={
          status:"error",overall:0,
          message:"Could not send the current design to Python.",
          nodes:{}
        };
        applyExecutionProgress(execution);
        setStatus(execution.message);
        return;
      }

      const runButton=bridgeControl(bridge.run,"button");
      if(!runButton){
        setStatus("Python Run control was not found. Re-run the Builder cell.");
        return;
      }

      // Ignore the bridge's old idle payload. The next changed payload must
      // come from Python after this click.
      const progressInput=bridgeControl(bridge.progress,"textarea");
      lastProgressRaw=progressInput?.value||lastProgressRaw;

      const queued={};
      (current(state).nodes||[]).forEach(n=>{
        queued[n.id]={status:"queued",message:"Waiting"};
      });
      execution={
        status:"running",
        runtime_kind:"data",
        overall:0,
        message:"Fetching data with Python pipeline…",
        nodes:queued
      };
      applyExecutionProgress(execution);
      setStatus(execution.message);

      if(bridgeAwaitTimer)clearTimeout(bridgeAwaitTimer);

      // Let the standard textarea comm flush first, then activate the standard
      // ipywidgets button in whichever notebook document contains it.
      setTimeout(()=>{
        const ok=clickBridgeButton(runButton);
        if(!ok){
          execution={
            status:"error",runtime_kind:"data",overall:0,
            message:"Could not activate the Python data control.",
            nodes:queued
          };
          applyExecutionProgress(execution);
          return;
        }

        bridgeAwaitTimer=setTimeout(()=>{
          if(
            execution.status==="running" &&
            (execution.message==="Starting Python pipeline…" ||
             execution.message==="Sending pipeline to Python…")
          ){
            execution={
              status:"error",
              overall:0,
              message:"Python kernel did not acknowledge Run. Re-run the Builder cell and confirm Kernel Connected.",
              nodes:queued
            };
            applyExecutionProgress(execution);
            setStatus(execution.message);
            updateKernelBadge();
          }
        },3000);
      },350);
    }

    function requestStop(){
      if(execution.status!=="running"){
        setStatus("Nothing is running.");
        return;
      }
      if(isPopout){
        if(popoutHostConnected&&sendPopoutMessage({type:"stop",source:"popout",ts:Date.now()}))setStatus("Stop requested in notebook kernel.");
        else setStatus("Notebook bridge is disconnected.");
        return;
      }
      if(!bridge){
        setStatus("Stop bridge unavailable.");
        return;
      }
      const stopButton=bridgeControl(bridge.stop,"button");
      if(stopButton && clickBridgeButton(stopButton)){
        setStatus("Stop requested. The active step will finish, then the pipeline will stop.");
      }else{
        setStatus("Python Stop control is unavailable.");
      }
    }

    function runLabel(s){
      return s==="running"?"RUNNING":s==="done"?"DONE":s==="error"?"ERROR":
             s==="stopped"?"STOPPED":s==="queued"?"QUEUED":"";
    }

    function applyExecutionProgress(next){
      if(!next||typeof next!=="object")return;
      if(next.runtime_kind==="import"){
        if(!isPopout)sendPopoutMessage({type:"progress",source:"host",payload:cp(next),ts:Date.now()});
        const type=String(next.component_type||next.component_import?.component_type||"");
        if(type&&next.component_api){
          mlapi[type]=cp(next.component_api);
          const item=catalog.find(entry=>entry.type===type);
          if(item){
            item.real_api=cp(next.component_api);
            item.api=cp(next.component_api.parameters||item.api||[]);
            if(next.component_api.description)item.description=next.component_api.description;
          }
        }
        componentImportBusy=false;
        if(next.status==="error"&&next.message)setStatus(next.message);
        else if(next.status==="done"&&type)setStatus((catalog.find(x=>x.type===type)?.name||type)+" API ready.");
        setTimeout(pumpComponentImportQueue,80);
        if(next.status==="done")setTimeout(draw,20);
        return;
      }
      if(next.runtime_kind==="external_import"){
        const did=String(next.definition_id||"");
        if(did)customImportStatus[did]={status:next.status,message:next.message||"Custom API import checked."};
        setStatus(next.message||"Custom API import checked.");
        setTimeout(draw,20);return;
      }
      if(next.runtime_kind==="user_function_validation"||next.runtime_kind==="user_class_validation"){
        const did=String(next.definition_id||"");
        const details=next.user_function_validation||next.user_class_validation||null;
        if(did)customImportStatus[did]={status:next.status,message:next.message||"User source checked.",details};
        setStatus(next.message||"User source checked.");
        setTimeout(draw,20);return;
      }
      execution=next;
      if(!isPopout)sendPopoutMessage({type:"progress",source:"host",payload:cp(next),state:next.state_replace?cp(state):null,ts:Date.now()});

      if(next.model_id && next.model_update){
        const entry=builtModelById(next.model_id);
        if(entry)Object.assign(entry,next.model_update);
      }
      if(next.runtime_kind==="train"||next.runtime_kind==="generate"){
        recordRuntimeEvent(next);
        updateRuntimeLive(next);
        if(runtimePanel?.tab==="status"&&runtimePanel?.mode===next.runtime_kind)scheduleRuntimeStatusDraw();
        if(next.status==="done"||next.status==="error"||next.status==="stopped"){
          if(next.message)setStatus(next.message);
          setTimeout(draw,80);
        }
      }

      if(next.runtime_kind==="hub"){
        if(next.state_replace){
          state=cp(next.state_replace);
          delete state._runtime_command;
          ensureWorkspaces();
          selected=null;pendingPort=null;outputDirectorySelection=null;
        }
        if(next.message)setStatus(next.message);
        if(next.status==="done"||next.status==="error")setTimeout(draw,80);
      }

      if(next.runtime_kind==="cloud"){
        if(next.cloud_status){cloudStatus[next.cloud_status.provider]=cp(next.cloud_status);}
        if(next.state_replace){
          state=cp(next.state_replace);delete state._runtime_command;delete state._session_secrets;ensureWorkspaces();
          selected=null;pendingPort=null;outputDirectorySelection=null;
        }
        if(next.message)setStatus(next.message);
        if(next.status==="done"||next.status==="error")setTimeout(draw,80);
      }

      if(next.runtime_kind==="local"){
        if(next.local_scan)localFiles=cp(next.local_scan);
        if(next.local_import){
          const type=next.local_import_type||"model";
          localImportReports[type]=cp(next.local_import);
        }
        if(next.state_replace){
          state=cp(next.state_replace);delete state._runtime_command;ensureWorkspaces();
          selected=null;pendingPort=null;
          if(next.local_import){
            const type=next.local_import_type||"model";
            state.active_workspace=type==="data"?"data":"model";
            bottomView="outputs";
            bottomExpanded=false;
            const imported=next.local_import.imported||[];
            outputDirectorySelection=imported.length?imported[imported.length-1].id:null;
          }else{
            outputDirectorySelection=null;
          }
        }
        if(next.message)setStatus(next.message);
        if(next.status==="done"||next.status==="error")setTimeout(draw,80);
      }

      if(next.runtime_kind==="serve"){
        const entry=builtModelById(next.model_id||runtimePanel?.modelId);
        if(entry&&next.serve_info){
          const liveInfo=cp(next.serve_info);
          const returnedApiKey=liveInfo.api_key||"";
          delete liveInfo.api_key;
          entry.serve_live=liveInfo;
          entry.serve_tunnel_error=liveInfo.public_tunnel_error||null;
          if(returnedApiKey){
            serveSecrets[entry.id]=serveSecrets[entry.id]||{};
            serveSecrets[entry.id].api_key=returnedApiKey;
          }
        }
        if(entry&&next.status==="error"){
          entry.serve_status="error";
          entry.serve_live={
            ...(entry.serve_live||{}),
            running:false,
            error:next.message||"API server failed to start."
          };
        }
        if(next.message)setStatus(next.message);
        if(runtimePanel?.mode==="serve")setTimeout(draw,80);
      }

      if(next.prepared_dataset){
        const changed=upsertPreparedDataset(next.prepared_dataset);
        if(changed){
          if(next.prepared_dataset.output_node_id)selected=next.prepared_dataset.output_node_id;
          setStatus("Data ready: "+next.prepared_dataset.name+" · "+compactDatasetSummary(next.prepared_dataset));
          draw();
          return;
        }
      }

      root.querySelectorAll(".mlb-node").forEach(card=>{
        const nodeState=execution.nodes?.[card.dataset.nodeId];
        card.classList.remove("run-queued","run-running","run-done","run-error","run-stopped");
        const old=card.querySelector(".mlb-run-badge");if(old)old.remove();
        const oldTrack=card.querySelector(".mlb-run-track");if(oldTrack)oldTrack.remove();
        if(!nodeState)return;

        card.classList.add("run-"+nodeState.status);
        const badge=document.createElement("div");badge.className="mlb-run-badge";
        badge.textContent=runLabel(nodeState.status);
        badge.title=nodeState.message||"";
        card.appendChild(badge);

        if(nodeState.status==="running"){
          const track=document.createElement("div");track.className="mlb-run-track";
          track.innerHTML="<i></i>";card.appendChild(track);
        }
      });

      const live=root.querySelector(".mlb-run-live");
      if(live){
        live.className="mlb-run-live "+(execution.status||"idle");
        live.innerHTML="<strong>"+Math.max(0,Math.min(100,Number(execution.overall||0)))+"%</strong><span>"+(execution.message||"Ready")+"</span>";
      }

      const selectedLive=root.querySelector(".mlb-ins-run-live");
      const selectedState=selected ? execution.nodes?.[selected] : null;
      if(selectedLive){
        if(selectedState){
          selectedLive.style.display="block";
          selectedLive.className="mlb-ins-run-live "+selectedState.status;
          selectedLive.innerHTML="<strong>"+runLabel(selectedState.status)+"</strong><span>"+(selectedState.message||"")+"</span>";
        }else{
          selectedLive.style.display="none";
        }
      }

      const stat=root.querySelector(".mlb-statusbar .right");
      if(stat)stat.textContent="● "+(execution.message||status);

      const run=root.querySelector(".mlb-run");
      if(run){
        const runtimeBusy=state.active_workspace==="model" && execution.status==="running" &&
          (execution.runtime_kind==="train"||execution.runtime_kind==="generate");
        run.classList.toggle("runtime-busy",runtimeBusy);
        run.classList.toggle("train",runtimeBusy&&execution.runtime_kind==="train");
        run.classList.toggle("generate",runtimeBusy&&execution.runtime_kind==="generate");
        run.disabled=execution.status==="running";
        if(state.active_workspace==="model"){
          const label=runtimeBusy
            ?(execution.runtime_kind==="train"?"Training":"Generating")
            :(execution.status==="running"?"Building":"Build");
          setActionButtonContent(run,runtimeBusy?"activity":"build",label);
        }else{
          const dataBusy=execution.status==="running"&&execution.runtime_kind==="data";
          setActionButtonContent(run,dataBusy?"activity":"fetch",dataBusy?"Fetching":"Fetch Data");
          run.disabled=dataBusy;
        }
      }
      const centerStop=root.querySelector(".mlb-center-stop");
      if(centerStop){
        const dataBusy=state.active_workspace==="data"&&execution.status==="running"&&execution.runtime_kind==="data";
        const modelBusy=state.active_workspace==="model"&&execution.status==="running"&&
          (execution.runtime_kind==="train"||execution.runtime_kind==="generate");
        centerStop.style.display=(dataBusy||modelBusy)?"inline-flex":"none";
      }
    }

    function updateRuntimeLive(next){
      const box=root.querySelector(".mlb-runtime-live");
      if(!box)return;
      box.className="mlb-runtime-live "+(next.status||"idle");
      const pct=Math.max(0,Math.min(100,Number(next.overall||0)));
      let html="<div class='mlb-runtime-live-head'><strong>"+(next.runtime_kind==="train"?"LIVE TRAINING":"LIVE GENERATION")+"</strong><span>"+pct+"%</span></div>";
      html+="<div class='mlb-runtime-live-message'>"+(next.message||"Working…")+"</div>";
      if(next.runtime_kind==="train"){
        const memNow=next.memory_allocated_gb==null?"—":Number(next.memory_allocated_gb).toFixed(2)+" GB";
        html+="<div class='mlb-runtime-live-stats'>"+
          "<div><span>Step</span><strong>"+(next.step??"—")+(next.max_steps?" / "+next.max_steps:"")+"</strong></div>"+
          "<div><span>GPU Tok/s</span><strong>"+(next.tokens_per_sec==null?"—":Math.round(Number(next.tokens_per_sec)).toLocaleString())+"</strong></div>"+
          "<div><span>E2E Tok/s</span><strong>"+(next.end_to_end_tokens_per_sec==null?"—":Math.round(Number(next.end_to_end_tokens_per_sec)).toLocaleString())+"</strong></div>"+
          "<div><span>Loss</span><strong>"+(next.loss==null?"—":Number(next.loss).toFixed(4))+"</strong></div>"+
          "<div><span>PPL</span><strong>"+(next.ppl==null?"—":Number(next.ppl).toFixed(2))+"</strong></div>"+
          "<div><span>Val Loss</span><strong>"+(next.val_loss==null?"—":Number(next.val_loss).toFixed(4))+"</strong></div>"+
          "<div><span>Val PPL</span><strong>"+(next.val_ppl==null?"—":Number(next.val_ppl).toFixed(2))+"</strong></div>"+
          "<div><span>Memory</span><strong>"+memNow+"</strong></div>"+
          "<div><span>Tokens</span><strong>"+Number(next.tokens_seen||0).toLocaleString()+"</strong></div>"+
          "</div>";
        if(next.sample_text)html+="<div class='mlb-runtime-sample'><span>Validation sample</span><pre>"+String(next.sample_text).replace(/&/g,"&amp;").replace(/</g,"&lt;")+"</pre></div>";
      }else if(next.generated_text){
        html+="<div class='mlb-runtime-sample'><span>Generated text</span><pre>"+String(next.generated_text).replace(/&/g,"&amp;").replace(/</g,"&lt;")+"</pre></div>";
      }
      html+="<div class='mlb-runtime-progress'><i style='width:"+pct+"%'></i></div>";
      box.innerHTML=html;
      const start=root.querySelector(".mlb-runtime-start");if(start)start.disabled=next.status==="running";
      const stop=root.querySelector(".mlb-runtime-stop");if(stop)stop.disabled=next.status!=="running";
    }

    function pollBridgeProgress(){
      updateKernelBadge();
      if(!bridge)return;
      const input=bridgeControl(bridge.progress,"textarea");
      if(!input)return;
      const raw=input.value||"";
      if(!raw || raw===lastProgressRaw)return;
      lastProgressRaw=raw;
      if(bridgeAwaitTimer){clearTimeout(bridgeAwaitTimer);bridgeAwaitTimer=null;}
      try{
        const parsed=JSON.parse(raw);
        applyExecutionProgress(parsed);
        if(parsed.message)setStatus(parsed.message);
      }catch(_){}
    }

    function startBridgePolling(){
      updateKernelBadge();
      if(bridgePollTimer)return;
      bridgePollTimer=setInterval(()=>{
        pollBridgeProgress();
        updateKernelBadge();
      },250);
    }

    function handlePopoutMessage(raw,sourceWindow=null){
      const msg=raw||{};
      if(msg.__mlb_studio_popout__!==true || msg.channel!==popoutChannelName)return;

      if(isPopout){
        if(msg.source!=="host")return;
        try{
          if(sourceWindow && window.opener && sourceWindow!==window.opener)return;
        }catch(_){}
        if(msg.type==="hello_ack"){
          popoutHostConnected=true;
          if(popoutHelloTimer){clearInterval(popoutHelloTimer);popoutHelloTimer=null;}
          if(msg.state?.components){state=cp(msg.state);ensureWorkspaces();selected=null;pendingPort=null;draw();}
          updateKernelBadge();
          setStatus("Full Window connected to notebook kernel.");
          return;
        }
        if(msg.type==="progress"){
          if(msg.state?.components){state=cp(msg.state);ensureWorkspaces();}
          applyExecutionProgress(cp(msg.payload||{}));
          if(msg.payload?.message)setStatus(msg.payload.message);
          draw();
        }
        return;
      }

      if(msg.source!=="popout")return;
      if(sourceWindow)popoutPeerWindow=sourceWindow;
      if(msg.type==="hello"){
        popoutPeerConnected=true;
        sendHostReply(sourceWindow,{type:"hello_ack",state:cp(state),ts:Date.now()});
        return;
      }
      if(msg.type==="state_sync"&&msg.state?.components){
        state=cp(msg.state);ensureWorkspaces();selected=null;pendingPort=null;draw();return;
      }
      if(msg.type==="stop"){
        const stopButton=bridgeControl(bridge?.stop,"button");
        if(stopButton)clickBridgeButton(stopButton);
        return;
      }
      if(msg.type==="command"){
        if(msg.state?.components){state=cp(msg.state);ensureWorkspaces();}
        if(!bridgeReady()){
          sendHostReply(sourceWindow,{type:"progress",payload:{status:"error",overall:0,message:"Notebook Python bridge is offline."},ts:Date.now()});
          return;
        }
        const okState=setBridgeState();
        const okCommand=setBridgeCommand(msg.command||{action:"data"});
        const runButton=bridgeControl(bridge.run,"button");
        if(okState&&okCommand&&runButton)clickBridgeButton(runButton);
        else sendHostReply(sourceWindow,{type:"progress",payload:{status:"error",overall:0,message:"Could not forward Full Window command to Python."},ts:Date.now()});
      }
    }

    function setupPopoutBridge(){
      window.addEventListener("message",event=>{
        try{
          const msg=event.data||{};
          if(
            isPopout &&
            msg.__mlb_studio_popout__===true &&
            msg.channel===popoutChannelName &&
            msg.type==="port_offer" &&
            event.ports && event.ports[0]
          ){
            attachPopoutMessagePort(event.ports[0]);
            if(msg.state?.components){state=cp(msg.state);ensureWorkspaces();selected=null;pendingPort=null;}
            popoutHostConnected=true;
            if(popoutHelloTimer){clearInterval(popoutHelloTimer);popoutHelloTimer=null;}
            sendPopoutMessage({type:"hello",source:"popout",ts:Date.now()});
            updateKernelBadge();
            setStatus("Full Window connected to notebook kernel.");
            draw();
            return;
          }
          handlePopoutMessage(msg,event.source||null);
        }catch(_){}
      });

      if(typeof BroadcastChannel!=="undefined"){
        try{
          popoutChannel=new BroadcastChannel(popoutChannelName);
          popoutChannel.onmessage=event=>{
            try{handlePopoutMessage(event.data||{},null);}catch(_){}
          };
        }catch(_){popoutChannel=null;}
      }

      if(isPopout){
        const hello=()=>{
          if(popoutHostConnected)return;
          sendPopoutMessage({type:"hello",source:"popout",ts:Date.now()});
          updateKernelBadge();
        };
        hello();
        let attempts=0;
        popoutHelloTimer=setInterval(()=>{
          attempts+=1;
          hello();
          if(popoutHostConnected||attempts>=12){clearInterval(popoutHelloTimer);popoutHelloTimer=null;}
        },500);
      }
    }

    function schedulePopoutStateSync(){
      if(!isPopout)return;
      if(popoutSyncTimer)clearTimeout(popoutSyncTimer);
      popoutSyncTimer=setTimeout(()=>{
        sendPopoutMessage({type:"state_sync",source:"popout",state:bridgeStatePayload(),ts:Date.now()});
      },180);
    }

    function fullWindowPage(){
      const assets=payload.popout_assets||{css:window.__MLB_STUDIO_CSS__||"",js:window.__MLB_STUDIO_JS_SOURCE__||""};
      if(!assets.css||!assets.js)return null;
      const popPayload=cp(payload);
      delete popPayload.popout_assets;
      popPayload.state=bridgeStatePayload();
      popPayload.initial_view={
        runtime_panel:runtimePanel?cp(runtimePanel):null,
        gallery_workspace:cp(galleryWorkspace),
        cloud_workspace:cp(cloudWorkspace),
        bottom_expanded:!!bottomExpanded,
        bottom_view:bottomView,
        selected:selected,
        inspector_tab:inspectorTab,
        zoom:zoom
      };
      // Use distinct placeholder bridge ids in the popout so Run and Stop remain
      // distinguishable while commands are proxied back to the notebook host.
      popPayload.bridge={
        mode:"popout",
        channel:popoutChannelName,
        state:"__popout_state__",
        command:"__popout_command__",
        run:"__popout_run__",
        stop:"__popout_stop__",
        progress:"__popout_progress__"
      };
      popPayload.instance_id=(payload.instance_id||root.id||"mlbricks")+"-full";
      const targetId="mlbricks-full-"+Date.now();
      const safePayload=JSON.stringify(popPayload).replace(/</g,"\\u003c");
      const cssText=String(assets.css).split("</style").join("<\\/style");
      const jsText=String(assets.js).split("</script").join("<\\/script");
      // Build the closing script tag by concatenation so builder.js itself never
      // contains a raw script end tag while generated HTML receives a real one.
      const closeScript="</"+"script>";
      return '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MLB Studio V1.0</title><style>'+cssText+'</style><style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#0b1118}body{padding:0}.mlb-root{width:100vw!important;height:100vh!important;min-height:0!important;max-height:none!important;min-width:0!important;border-radius:0!important;border:0!important;box-shadow:none!important}</style></head><body><div id="'+targetId+'" class="mlb-root" data-mlb-studio-version="1.0.0"></div><script>'+jsText+closeScript+'<script>window.MLBricksBuilder.mount(document.getElementById('+JSON.stringify(targetId)+'),'+safePayload+');'+closeScript+'</body></html>';
    }

    function openFullWindow(){
      const page=fullWindowPage();
      if(!page){
        setStatus("Full Window assets are unavailable. Re-run the Builder cell.");
        return false;
      }

      const targetName="mlb_studio_full_"+String(popoutChannelName).replace(/[^a-zA-Z0-9_-]/g,"_");
      const launcherUrl="https://builder.mlbricks.io/";
      let popup=null;
      let bootstrapSent=false;
      let launcherProbe=null;
      let launcherUpgraded=false;

      // Open the known-working Builder immediately. Do not briefly navigate to
      // builder.mlbricks.io and then bounce back to about:blank when the hosted
      // launcher has not been deployed yet.
      try{popup=window.open("about:blank",targetName);}catch(_){popup=null;}
      if(!popup){
        setStatus("Could not open the Builder tab. Allow pop-ups for this notebook, then try again.");
        return false;
      }
      try{
        popup.document.open();popup.document.write(page);popup.document.close();
        popup.document.title="MLB Studio V1.0";
        bootstrapSent=true;
      }catch(_){ }

      popoutPeerWindow=popup;
      popoutPeerConnected=false;

      const offerPort=()=>{
        if(popoutPeerConnected || !bootstrapSent || !popup || popup.closed || typeof MessageChannel==="undefined")return;
        try{
          const channel=new MessageChannel();
          attachPopoutMessagePort(channel.port1);
          popup.postMessage(
            popoutPacket({type:"port_offer",source:"host",state:cp(state),ts:Date.now()}),
            "*",
            [channel.port2]
          );
        }catch(_){ }
      };

      // Probe the real launcher in a hidden frame. Only if that exact page
      // announces itself do we upgrade the already-working popout to the real
      // builder.mlbricks.io URL. This removes the visible URL -> about:blank bounce.
      const doc=root.ownerDocument||document;
      const probeToken="mlb_probe_"+Date.now()+"_"+Math.random().toString(36).slice(2);
      const cleanupProbe=()=>{
        try{window.removeEventListener("message",onProbeMessage);}catch(_){ }
        try{launcherProbe?.remove();}catch(_){ }
        launcherProbe=null;
      };
      const onProbeMessage=event=>{
        const msg=event.data||{};
        if(msg.__mlb_studio_launcher__!==true||msg.type!=="ready"||msg.probe_token!==probeToken)return;
        if(event.origin!=="https://builder.mlbricks.io")return;
        cleanupProbe();
        if(!popup||popup.closed)return;
        launcherUpgraded=true;
        bootstrapSent=false;
        const onReady=readyEvent=>{
          const readyMsg=readyEvent.data||{};
          if(readyEvent.source!==popup||readyMsg.__mlb_studio_launcher__!==true||readyMsg.type!=="ready")return;
          if(readyEvent.origin!=="https://builder.mlbricks.io")return;
          window.removeEventListener("message",onReady);
          try{
            popup.postMessage({__mlb_studio_launcher__:true,type:"bootstrap",html:page,title:"MLB Studio V1.0"},"https://builder.mlbricks.io");
            bootstrapSent=true;
            setStatus("MLB Studio V1.0 opened at builder.mlbricks.io. Keep this notebook tab open for Python execution.");
            [150,450,900,1600].forEach(ms=>setTimeout(offerPort,ms));
          }catch(_){ }
        };
        window.addEventListener("message",onReady);
        try{popup.location.href=launcherUrl;}catch(_){window.removeEventListener("message",onReady);launcherUpgraded=false;bootstrapSent=true;}
      };
      window.addEventListener("message",onProbeMessage);
      try{
        launcherProbe=doc.createElement("iframe");
        launcherProbe.setAttribute("aria-hidden","true");
        launcherProbe.style.cssText="position:fixed;width:1px;height:1px;left:-9999px;top:-9999px;border:0;opacity:0;pointer-events:none";
        launcherProbe.src=launcherUrl+"?mlb_probe="+encodeURIComponent(probeToken);
        (doc.body||doc.documentElement).appendChild(launcherProbe);
        setTimeout(cleanupProbe,2200);
      }catch(_){cleanupProbe();}

      [120,350,700,1200,2200,3400].forEach(ms=>setTimeout(()=>{if(!launcherUpgraded)offerPort();},ms));
      setTimeout(()=>sendHostReply(popup,{type:"hello_ack",state:cp(state),ts:Date.now()}),500);
      setStatus("MLB Studio V1.0 opened. Keep this notebook tab open for Python execution.");
      return true;
    }

    function activateFullWindowLink(event){
      event.preventDefault();
      openFullWindow();
    }

    function btn(text,cls){const b=document.createElement("button");b.type="button";b.className=cls||"";b.textContent=text;return b;}

    function uiIcon(name){
      const common='viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"';
      const paths={
        build:'<svg '+common+'><rect x="3.5" y="5" width="7.5" height="5" rx="1"/><rect x="13" y="5" width="7.5" height="5" rx="1"/><rect x="6.5" y="14" width="7.5" height="5" rx="1"/><path d="M16 14h4.5v5H16"/></svg>',
        gallery:'<svg '+common+'><rect x="3.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="3.5" width="7" height="7" rx="1.5"/><rect x="3.5" y="13.5" width="7" height="7" rx="1.5"/><rect x="13.5" y="13.5" width="7" height="7" rx="1.5"/></svg>',
        fetch:'<svg '+common+'><ellipse cx="12" cy="5.5" rx="7.5" ry="3"/><path d="M4.5 5.5v6c0 1.65 3.36 3 7.5 3 1.15 0 2.24-.1 3.2-.3"/><path d="M4.5 11.5v6c0 1.65 3.36 3 7.5 3 1.18 0 2.3-.11 3.28-.32"/><path d="M18 13v7"/><path d="m15.2 17.3 2.8 2.8 2.8-2.8"/></svg>',
        stop:'<svg '+common+'><rect x="6.5" y="6.5" width="11" height="11" rx="1.8"/></svg>',
        cloud:'<svg '+common+'><path d="M7.3 18.5h10.6a4.1 4.1 0 0 0 .4-8.18A6.4 6.4 0 0 0 6.1 8.9a4.8 4.8 0 0 0 1.2 9.6Z"/><path d="M12 10.5v5"/><path d="m9.7 13.2 2.3 2.3 2.3-2.3"/></svg>',
        activity:'<svg '+common+'><path d="M3.5 12h4l2-5 4 10 2-5h5"/></svg>'
      };
      return paths[name]||paths.build;
    }

    function setActionButtonContent(button,icon,label){
      if(!button)return;
      button.innerHTML='<span class="mlb-action-icon">'+uiIcon(icon)+'</span><span class="mlb-action-label"></span>';
      const text=button.querySelector('.mlb-action-label');if(text)text.textContent=label;
      button.setAttribute('aria-label',label);
    }

    function actionBtn(label,cls,icon){
      const b=btn('',cls);setActionButtonContent(b,icon,label);return b;
    }
    function portLabel(side,index){
      const lane=["Skip","Main","Extra"][index] || ("Lane "+(index+1));
      return lane+" "+(side==="in"?"In":"Out");
    }

    function selectedNode(){return current(state).nodes.find(n=>n.id===selected)||null;}
    function setStatus(s){status=s;}

    function apiInfo(node){
      if(node.type==="custom") return {public_name:"Custom Layer",parameters:[],available:true};
      const item=cat(catalog,node.type);
      if(item.builder_utility){
        return {
          available:true,
          runtime_available:null,
          builder_utility:true,
          builder_python_api:!!item.builder_python_api,
          public_name:item.name,
          parameters:item.api||[],
          description:item.description||"",
          source:"MLB Studio"
        };
      }
      return mlapi[node.type] || item.real_api || {};
    }

    function availablePreparedDatasets(){
      return Array.isArray(state.prepared_datasets) ? state.prepared_datasets : [];
    }

    function preparedDatasetById(id){
      return availablePreparedDatasets().find(d=>d.id===id)||null;
    }

    function latestPreparedDataset(){
      const all=availablePreparedDatasets();
      return all.length ? all[all.length-1] : null;
    }

    function splitRows(meta,name){
      const rows=meta?.splits?.[name]?.rows;
      return rows===null||rows===undefined ? "?" : Number(rows).toLocaleString();
    }

    function datasetSplitLabel(name,meta){
      const pretty=name==="validation"?"Validation":name.charAt(0).toUpperCase()+name.slice(1);
      return pretty+" — "+splitRows(meta,name)+" rows";
    }

    function compactDatasetSummary(meta){
      if(!meta)return "No prepared data";
      const parts=[];
      ["train","validation","test"].forEach(name=>{
        if(meta.splits?.[name]){
          const label=name==="validation"?"Val":name.charAt(0).toUpperCase()+name.slice(1);
          parts.push(label+" "+splitRows(meta,name));
        }
      });
      return parts.join(" · ") || ((meta.total_rows??"?")+" rows");
    }

    function configureTextInputForDataset(node,meta){
      if(!node||node.type!=="text_input"||!meta)return;
      node.params=node.params||{};
      node.params.input_mode="prepared_dataset";
      node.params.dataset_id=meta.id;
      node.params.dataset_split=meta.default_split || (meta.splits?.train?"train":Object.keys(meta.splits||{})[0]||"train");
    }

    function configureTextInputForLatest(node){
      const latest=latestPreparedDataset();
      if(latest)configureTextInputForDataset(node,latest);
      return node;
    }

    function autoBindDatasetToModel(meta){
      const modelRoot=state.workspaces?.model?.root_component_id;
      const model=state.components?.[modelRoot];
      if(!model)return;
      (model.nodes||[]).filter(n=>n.type==="text_input").forEach(n=>configureTextInputForDataset(n,meta));
      state.project=state.project||{};
      state.project.dataset=meta.name;
    }

    function upsertPreparedDataset(meta){
      if(!meta||!meta.id)return false;
      state.prepared_datasets=availablePreparedDatasets();
      const idx=state.prepared_datasets.findIndex(d=>d.id===meta.id);
      const raw=JSON.stringify(meta);
      if(idx>=0){
        if(JSON.stringify(state.prepared_datasets[idx])===raw)return false;
        state.prepared_datasets[idx]=cp(meta);
      }else{
        state.prepared_datasets.push(cp(meta));
      }
      autoBindDatasetToModel(meta);
      return true;
    }

    function datasetSummaryCard(meta,titleText){
      const card=document.createElement("div");card.className="mlb-dataset-result";
      const title=document.createElement("div");title.className="mlb-dataset-result-title";
      title.innerHTML="<strong>"+(titleText||"DATA READY")+"</strong><span>"+meta.name+"</span>";
      card.appendChild(title);
      const grid=document.createElement("div");grid.className="mlb-dataset-splits";
      const names=["train","validation","test"].filter(name=>meta.splits?.[name]);
      (names.length?names:Object.keys(meta.splits||{})).forEach(name=>{
        const item=document.createElement("div");
        item.innerHTML="<span>"+(name==="validation"?"Validation":name.charAt(0).toUpperCase()+name.slice(1))+"</span><strong>"+splitRows(meta,name)+"</strong>";
        grid.appendChild(item);
      });
      card.appendChild(grid);
      const foot=document.createElement("div");foot.className="mlb-dataset-result-foot";
      foot.textContent=meta.storage==="disk+memory"
        ?("Saved + in memory · "+(meta.path||""))
        :"Available in Python memory";
      card.appendChild(foot);
      return card;
    }

    function modelRootComponent(){
      const id=state.workspaces?.model?.root_component_id || state.root_component_id;
      return state.components?.[id] || null;
    }

    function selectedModelDataset(){
      const model=modelRootComponent();
      if(!model)return null;
      const textInput=(model.nodes||[]).find(n=>
        n.type==="text_input" &&
        String(n.params?.input_mode||"prompt")==="prepared_dataset"
      );
      return textInput ? preparedDatasetById(textInput.params?.dataset_id) : null;
    }

    function currentModelDirectoryEntry(){
      const model=modelRootComponent();
      if(!model)return null;
      const dataset=selectedModelDataset();
      return {
        id:"current_model_design",
        name:state.project?.name || model.name || "Current Model",
        kind:"design",
        status:"design",
        nodes:(model.nodes||[]).length,
        connections:(model.edges||[]).length,
        dataset:dataset?.name || state.project?.dataset || null,
        context_length:state.project?.context_length ?? null,
        batch_size:state.project?.batch_size ?? null,
      };
    }

    function modelDirectoryEntries(){
      return (state.model_outputs||[]).filter(item=>item.kind==="built_model"||item.kind==="trained_model"||item.kind==="model_artifact");
    }

    function numberOr(value,fallback){
      const n=Number(value);
      return Number.isFinite(n)&&n>0?n:fallback;
    }

    function precisionToDtype(value){
      const p=String(value||"fp16").toLowerCase();
      return p==="fp32"?"float32":p==="bf16"?"bfloat16":"float16";
    }

    function firstModelNode(type){
      const model=modelRootComponent();
      return (model?.nodes||[]).find(n=>n.type===type)||null;
    }

    function referencedCustomDefinitions(){
      const found=[];
      const seen=new Set();
      const visitNodes=(nodes)=>{
        (nodes||[]).forEach(node=>{
          if(node.type!=="custom"||!node.definition_id||seen.has(node.definition_id))return;
          seen.add(node.definition_id);
          const def=state.custom_components?.[node.definition_id];
          if(def){
            found.push(def);
            visitNodes(def.nodes||[]);
          }
        });
      };
      visitNodes(modelRootComponent()?.nodes||[]);
      return found;
    }

    function allModelSettingNodes(){
      const model=modelRootComponent();
      const nodes=[...(model?.nodes||[])];
      referencedCustomDefinitions().forEach(def=>nodes.push(...(def.nodes||[])));
      return nodes;
    }

    function parseJsonish(value,fallback){
      if(value&&typeof value==="object")return cp(value);
      const text=String(value??"").trim();
      if(!text)return cp(fallback);
      try{return JSON.parse(text);}catch(_){return cp(fallback);}
    }

    function soupHeadCount(node){
      if(!node)return null;
      const cfg=parseJsonish(node.params?.mixer_config,{});
      const first=Array.isArray(cfg)?(cfg[0]||{}):cfg;
      return numberOr(first?.head ?? first?.num_heads,null);
    }

    function soupMixerConfigWithHeads(value,heads,depth){
      const parsed=parseJsonish(value,{});
      const apply=obj=>{
        const next=(obj&&typeof obj==="object"&&!Array.isArray(obj))?{...obj}:{};
        next.head=heads;next.num_heads=heads;
        return next;
      };
      if(Array.isArray(parsed)){
        const count=Math.max(1,Number(depth)||parsed.length||1);
        const out=[];
        for(let i=0;i<count;i++)out.push(apply(parsed[i]||parsed[parsed.length-1]||{}));
        return JSON.stringify(out);
      }
      return JSON.stringify(apply(parsed));
    }

    function deriveModelSettings(entry){
      state.project=state.project||{};
      const stored=state.project.model_settings||{};
      const nodes=allModelSettingNodes();
      const embedding=nodes.find(n=>n.type==="embedding");
      const esa=nodes.find(n=>n.type==="esa");
      const soup=nodes.find(n=>n.type==="soup");
      const head=nodes.find(n=>n.type==="lm_head");

      const embeddingSize=numberOr(
        stored.embedding_size,
        numberOr(
          embedding?.params?.embedding_dim ?? embedding?.params?.hidden_size ?? embedding?.params?.dim,
          numberOr(esa?.params?.embd ?? esa?.params?.dim,numberOr(soup?.params?.dim,384))
        )
      );
      const heads=numberOr(
        stored.heads,
        numberOr(esa?.params?.head ?? esa?.params?.heads ?? esa?.params?.num_heads,numberOr(soupHeadCount(soup),6))
      );
      const block=numberOr(
        stored.block,
        numberOr(esa?.params?.block,numberOr(entry?.context_length ?? state.project.context_length,512))
      );
      const defaultBatch=numberOr(
        stored.default_batch,
        numberOr(esa?.params?.batch,numberOr(entry?.batch_size ?? state.project.batch_size,16))
      );
      const vocab=numberOr(
        stored.vocab_size,
        numberOr(embedding?.params?.vocab_size,numberOr(head?.params?.vocab_size,32000))
      );
      const precision=String(
        stored.precision ||
        esa?.params?.precision ||
        soup?.params?.precision ||
        (embedding?.params?.dtype==="float32"?"fp32":embedding?.params?.dtype==="bfloat16"?"bf16":"fp16")
      ).toLowerCase();

      const settings={
        embedding_size:embeddingSize,
        heads,
        block,
        default_batch:defaultBatch,
        vocab_size:vocab,
        precision:["fp32","fp16","bf16"].includes(precision)?precision:"fp16"
      };
      state.project.model_settings={...settings};
      return settings;
    }

    function syncModelSettingsToGraph(settings,oldSettings){
      state.project=state.project||{};
      state.project.context_length=settings.block;
      state.project.batch_size=settings.default_batch;
      state.project.model_settings={...settings};

      const dtype=precisionToDtype(settings.precision);
      allModelSettingNodes().forEach(node=>{
        node.params=node.params||{};
        const p=node.params;
        const t=node.type;

        if(t==="embedding"){
          p.embedding_dim=settings.embedding_size;
          p.hidden_size=settings.embedding_size;
          p.dim=settings.embedding_size;
          p.vocab_size=settings.vocab_size;
          p.dtype=dtype;
        }else if(t==="esa"){
          p.embd=settings.embedding_size;
          p.dim=settings.embedding_size;
          p.head=settings.heads;
          p.heads=settings.heads;
          p.batch=settings.default_batch;
          p.block=settings.block;
          p.precision=settings.precision;
          p.dtype=dtype;
        }else if(t==="soup"){
          p.dim=settings.embedding_size;
          p.precision=settings.precision;
          p.mixer_config=soupMixerConfigWithHeads(p.mixer_config,settings.heads,p.depth);
        }else if(t==="stateaware_esa_stack"){
          p.dim=settings.embedding_size;
          p.heads=settings.heads;
          p.batch=settings.default_batch;
          p.block=settings.block;
        }else if(t==="rmsnorm"||t==="layernorm"){
          p.normalized_shape=settings.embedding_size;
          p.hidden_size=settings.embedding_size;
          p.dim=settings.embedding_size;
        }else if(t==="ffn"||t==="saffn"){
          const priorDim=numberOr(oldSettings?.embedding_size,384);
          const priorIntermediate=numberOr(p.intermediate_size ?? p.ffn_dim,priorDim*4);
          const ratio=Math.max(1,priorIntermediate/priorDim);
          p.hidden_size=settings.embedding_size;
          p.dim=settings.embedding_size;
          p.intermediate_size=Math.round(settings.embedding_size*ratio);
          p.ffn_dim=Math.round(settings.embedding_size*ratio);
        }else if(t==="lm_head"){
          p.hidden_size=settings.embedding_size;
          p.dim=settings.embedding_size;
          p.vocab_size=settings.vocab_size;
        }else if(t==="classifier"){
          p.hidden_size=settings.embedding_size;
          p.dim=settings.embedding_size;
        }else if(["vesa","bolt","visualbolt"].includes(t)){
          p.dim=settings.embedding_size;
          if("d_model" in p)p.d_model=settings.embedding_size;
          if("heads" in p)p.heads=settings.heads;
          if("num_heads" in p)p.num_heads=settings.heads;
        }
      });
    }

    function updateBuiltModelSetting(entry,key,value){
      if(!entry)return;
      const oldSettings=deriveModelSettings(entry);
      const next={...oldSettings};
      if(key==="precision"){
        next[key]=String(value||"fp16");
      }else{
        const n=Number(value);
        if(!Number.isFinite(n)||n<=0){
          setStatus(key.replaceAll("_"," ")+" must be greater than 0.");
          draw();
          return;
        }
        next[key]=Math.round(n);
      }

      if(key==="embedding_size" && next.embedding_size%next.heads!==0){
        setStatus("Embedding Size must be divisible by Heads.");
        draw();
        return;
      }
      if(key==="heads" && next.embedding_size%next.heads!==0){
        setStatus("Heads must divide Embedding Size exactly.");
        draw();
        return;
      }

      checkpoint("Update Model Settings");
      syncModelSettingsToGraph(next,oldSettings);

      // Architecture-affecting model settings invalidate the previous build.
      entry.context_length=next.block;
      entry.batch_size=next.default_batch;
      entry.status="needs_rebuild";
      entry.weights_ready=false;
      entry.training_status="untrained";
      entry.requirements=inferModelRequirements(modelRootComponent());
      entry.requirements.context_length=next.block;
      entry.model_settings={...next};
      entry.architecture=cp(modelRootComponent());
      entry.fingerprint=modelFingerprint(modelRootComponent());

      // Keep training defaults aligned with the model-wide default batch.
      ensureRuntimeConfigs(entry);
      entry.training_config.batch_size=next.default_batch;
      if(entry.training_config.precision==="auto" || !entry.training_config.precision){
        entry.training_config.precision=next.precision;
      }
      if(entry.generation_config && (entry.generation_config.precision==="auto" || !entry.generation_config.precision)){
        entry.generation_config.precision=next.precision;
      }

      setStatus("Model setting updated. Rebuild required before training.");
      draw();
    }

    function modelSettingField(label,key,value,entry,options=null,help=""){
      const field=document.createElement("div");field.className="mlb-model-setting-field";
      const top=document.createElement("div");top.className="mlb-model-setting-label";
      const lab=document.createElement("label");lab.textContent=label;top.appendChild(lab);
      if(help){const hint=document.createElement("span");hint.textContent=help;top.appendChild(hint);}
      field.appendChild(top);

      let input;
      if(options){
        input=document.createElement("select");
        options.forEach(opt=>{
          const value=typeof opt==="object"?opt.value:opt;
          const text=typeof opt==="object"?opt.label:opt;
          const o=document.createElement("option");o.value=value;o.textContent=text;
          input.appendChild(o);
        });
        input.value=String(value);
      }else{
        input=document.createElement("input");input.type="number";input.min="1";input.step="1";input.value=String(value);
      }
      input.addEventListener("change",()=>updateBuiltModelSetting(entry,key,input.value));
      field.appendChild(input);
      return field;
    }

    function renderModelSettings(body,entry){
      const settings=deriveModelSettings(entry);
      const title=document.createElement("div");title.className="mlb-section-title";title.textContent="MODEL SETTINGS";
      body.appendChild(title);

      const box=document.createElement("div");box.className="mlb-model-settings";
      box.append(
        modelSettingField("Embedding Size","embedding_size",settings.embedding_size,entry,null,"hidden width"),
        modelSettingField("Heads","heads",settings.heads,entry,null,"attention / ESA"),
        modelSettingField("Block / Context","block",settings.block,entry,null,"sequence length"),
        modelSettingField("Default Batch","default_batch",settings.default_batch,entry,null,"training default"),
        modelSettingField("Vocabulary","vocab_size",settings.vocab_size,entry,null,"token count"),
        modelSettingField("Precision","precision",settings.precision,entry,[
          {value:"fp16",label:"FP16"},
          {value:"bf16",label:"BF16"},
          {value:"fp32",label:"FP32"}
        ],"model default")
      );
      body.appendChild(box);

      const note=document.createElement("div");note.className="mlb-model-settings-note";
      note.textContent="Block / Context, width, heads, vocabulary and precision change the architecture/runtime contract. Changing them marks this build as Rebuild Required. Default Batch can still be overridden per training run.";
      body.appendChild(note);
    }

    function builtModelById(id){
      return (state.model_outputs||[]).find(item=>item.id===id)||null;
    }

    function selectedOutputModel(){
      if(state.active_workspace!=="model" || bottomView!=="outputs" || !outputDirectorySelection)return null;
      return builtModelById(outputDirectorySelection);
    }

    function modelFingerprint(model){
      return JSON.stringify({
        nodes:(model?.nodes||[]).map(n=>({
          id:n.id,type:n.type,name:n.name,definition_id:n.definition_id||null,
          repeat:n.repeat||1,params:n.params||{}
        })),
        edges:(model?.edges||[]).map(e=>({
          source:e.source,target:e.target,kind:e.kind||"main",
          source_port:e.source_port||null,target_port:e.target_port||null
        }))
      });
    }

    function inferModelRequirements(model){
      const nodes=model?.nodes||[];
      const types=new Set(nodes.map(n=>n.type));
      let modality="unknown";
      if(types.has("text_input"))modality="text";
      else if(types.has("image_input"))modality="image";
      else if(types.has("audio_input"))modality="audio";

      const terminal=[...nodes].reverse().find(n=>
        ["text_output","logits_output","classifier","lm_head"].includes(n.type)
      );

      return {
        modality,
        output_type:terminal?.type||"unknown",
        requires_tokenizer:modality==="text" && (types.has("embedding")||types.has("lm_head")),
        context_length:Number(state.project?.context_length||0)||null,
        batch_size:Number(state.project?.batch_size||0)||null,
      };
    }

    function validateModelBuild(){
      const model=modelRootComponent();
      const errors=[];
      if(!model || !(model.nodes||[]).length){
        return [{node_ids:[],message:"Model canvas is empty. Add model components before Build."}];
      }

      const nodes=model.nodes||[];
      const byId=new Map(nodes.map(n=>[n.id,n]));
      const mainEdges=(model.edges||[]).filter(e=>(e.kind||"main")==="main");
      const inputTypes=new Set(["text_input","image_input","audio_input"]);
      const outputTypes=new Set(["text_output","logits_output","classifier","lm_head"]);

      const inputs=nodes.filter(n=>inputTypes.has(n.type));
      const outputs=nodes.filter(n=>outputTypes.has(n.type));
      if(!inputs.length)errors.push({node_ids:[],message:"Add at least one model Input before Build."});
      if(!outputs.length)errors.push({node_ids:[],message:"Add a model output/head before Build."});

      const degree=new Map(nodes.map(n=>[n.id,0]));
      mainEdges.forEach(e=>{
        if(byId.has(e.source)&&byId.has(e.target)){
          degree.set(e.source,(degree.get(e.source)||0)+1);
          degree.set(e.target,(degree.get(e.target)||0)+1);
        }
      });
      const disconnected=nodes.filter(n=>nodes.length>1 && (degree.get(n.id)||0)===0);
      if(disconnected.length){
        errors.push({
          node_ids:disconnected.map(n=>n.id),
          message:"Disconnected model components: "+disconnected.map(n=>n.name).join(", ")
        });
      }

      // Detect cycles on the Main lane, while allowing branches/parallel graphs.
      const incoming=new Map(nodes.map(n=>[n.id,0]));
      const outgoing=new Map(nodes.map(n=>[n.id,[]]));
      mainEdges.forEach(e=>{
        if(byId.has(e.source)&&byId.has(e.target)){
          incoming.set(e.target,(incoming.get(e.target)||0)+1);
          outgoing.get(e.source).push(e.target);
        }
      });
      const queue=nodes.filter(n=>(incoming.get(n.id)||0)===0).map(n=>n.id);
      let visited=0;
      while(queue.length){
        const id=queue.shift();visited++;
        (outgoing.get(id)||[]).forEach(next=>{
          incoming.set(next,incoming.get(next)-1);
          if(incoming.get(next)===0)queue.push(next);
        });
      }
      if(visited!==nodes.length){
        errors.push({node_ids:[],message:"Main model flow contains a cycle. Remove the cycle before Build."});
      }

      return errors;
    }

    function registerBuiltModel(){
      const model=modelRootComponent();
      const requirements=inferModelRequirements(model);
      const name=state.project?.name||model?.name||"Built Model";
      const fingerprint=modelFingerprint(model);
      let entry=(state.model_outputs||[]).find(item=>item.kind==="built_model" && item.name===name);

      const latestData=selectedModelDataset()||latestPreparedDataset();
      const snapshot={
        name,
        kind:"built_model",
        status:"built",
        built_at:new Date().toISOString(),
        revision:(entry?.revision||0)+1,
        nodes:(model?.nodes||[]).length,
        connections:(model?.edges||[]).length,
        context_length:state.project?.context_length??null,
        batch_size:state.project?.batch_size??null,
        estimated_parameters:state.project?.estimated_parameters??null,
        model_settings:{...deriveModelSettings(entry)},
        architecture:cp(model),
        custom_components_snapshot:cp(state.custom_components||{}),
        requirements,
        fingerprint,
        selected_dataset_id:entry?.selected_dataset_id || latestData?.id || null,
        training_status:"untrained",
        weights_ready:false,
      };

      if(entry){
        Object.assign(entry,snapshot);
      }else{
        entry={id:uid("model"),...snapshot};
        state.model_outputs=state.model_outputs||[];
        state.model_outputs.push(entry);
      }
      return entry;
    }

    function showModelBuildErrors(errors){
      const states={};
      (modelRootComponent()?.nodes||[]).forEach(n=>states[n.id]={status:"queued",message:"Waiting"});
      errors.forEach(err=>(err.node_ids||[]).forEach(id=>{
        states[id]={status:"error",message:err.message};
      }));
      execution={status:"error",overall:0,message:errors[0]?.message||"Model Build failed.",nodes:states};
      setStatus(execution.message);
      draw();
    }

    function requestModelBuild(){
      if(state.active_workspace!=="model")return;
      const errors=validateModelBuild();
      if(errors.length){showModelBuildErrors(errors);return;}

      if(modelBuildTimer){clearInterval(modelBuildTimer);modelBuildTimer=null;}
      const model=modelRootComponent();
      const nodes=model.nodes||[];
      const states={};
      nodes.forEach(n=>states[n.id]={status:"queued",message:"Waiting to build"});
      execution={status:"running",overall:0,message:"Building model design…",nodes:states};
      setStatus(execution.message);
      draw();

      let index=0;
      const finish=()=>{
        const entry=registerBuiltModel();
        execution={
          status:"done",overall:100,message:"Model built: "+entry.name,
          nodes:Object.fromEntries(nodes.map(n=>[n.id,{status:"done",message:"Built"}]))
        };
        bottomView="outputs";
        // Keep the workspace collapsed by default, but open it after a successful
        // build so the newly built model is immediately available for selection.
        bottomExpanded=true;
        outputDirectorySelection=entry.id;
        selected=null;
        scrollBuiltModelActionsOnce=true;
        setStatus("Build complete. Select training data and check compatibility.");
        draw();
      };

      modelBuildTimer=setInterval(()=>{
        if(index>0){
          const prev=nodes[index-1];
          states[prev.id]={status:"done",message:"Built"};
        }
        if(index>=nodes.length){
          clearInterval(modelBuildTimer);modelBuildTimer=null;finish();return;
        }
        const node=nodes[index];
        states[node.id]={status:"running",message:"Building "+node.name+"…"};
        execution={
          status:"running",
          overall:Math.round(index/Math.max(nodes.length,1)*100),
          message:"Building "+node.name+"…",
          nodes:cp(states)
        };
        applyExecutionProgress(execution);
        index++;
      },110);
    }

    function datasetModality(meta){
      const p=meta?.pipeline||{};
      if(p.image_processing)return "image";
      if(p.audio_processing)return "audio";
      return "text";
    }

    function datasetTrainingCapabilities(datasetMeta){
      const pipeline=datasetMeta?.pipeline||{};
      const tokenizer=pipeline.tokenizer||null;
      const cols=datasetMeta?.splits?.train?.columns||[];
      const hasInputIds=cols.includes("input_ids");
      const declaredRepack=datasetMeta?.capabilities?.runtime_context_repack===true;
      return {
        tokenizer,
        columns:cols,
        hasInputIds,
        repackableTokenStream:declaredRepack||hasInputIds,
        preparedContext:Number(tokenizer?.context_length||0)||null,
      };
    }

    function modelDatasetCompatibility(modelEntry,datasetMeta){
      const checks=[];
      const add=(label,ok,detail)=>checks.push({label,ok,detail});
      if(!datasetMeta){
        add("Prepared dataset",false,"Select a prepared dataset.");
        return {ok:false,checks};
      }

      const req=modelEntry?.requirements||{};
      add(
        "Build",
        modelEntry?.status==="built" || modelEntry?.status==="trained",
        modelEntry?.status==="needs_rebuild"?"Model settings changed · Build again":"Current build"
      );
      const modality=datasetModality(datasetMeta);
      add(
        "Modality",
        req.modality==="unknown" || req.modality===modality,
        "Model: "+(req.modality||"unknown")+" · Data: "+modality
      );

      const trainRows=datasetMeta?.splits?.train?.rows;
      add("Train split",Number(trainRows)>0,"Train rows: "+(trainRows??0));

      const caps=datasetTrainingCapabilities(datasetMeta);
      if(req.modality==="text" && req.requires_tokenizer){
        add("Tokenizer",!!caps.tokenizer,caps.tokenizer?.tokenizer_name||"Tokenizer missing");

        // Context is a model/runtime packing choice, not a property of token IDs.
        // Any prepared text dataset that exposes input_ids is a repackable token
        // stream, so one compatibility rule works for every current/future LM.
        add(
          "Tokenized fields",
          caps.hasInputIds,
          caps.hasInputIds?"input_ids available":"input_ids not found"
        );

        const modelContext=Number(req.context_length||0)||null;
        if(caps.repackableTokenStream && modelContext){
          const prepared=caps.preparedContext;
          const detail=prepared
            ? (prepared===modelContext
                ? "Prepared max "+prepared+" · Training "+modelContext+" · Exact packing"
                : "Prepared max "+prepared+" · Training "+modelContext+" · Auto repack token stream")
            : "Training "+modelContext+" · Packed from input_ids at runtime";
          add("Context packing",true,detail);
        }else if(caps.repackableTokenStream){
          add("Context packing",true,"Packed from input_ids at runtime");
        }
      }

      return {ok:checks.every(c=>c.ok),checks};
    }

    function setBuiltModelDataset(entry,datasetId){
      if(!entry)return;
      entry.selected_dataset_id=datasetId||null;
      const meta=preparedDatasetById(datasetId);
      if(meta){
        // Keep the editable model Text Input aligned with the training selection.
        const model=modelRootComponent();
        (model?.nodes||[]).filter(n=>n.type==="text_input").forEach(n=>configureTextInputForDataset(n,meta));
        state.project=state.project||{};
        state.project.dataset=meta.name;
      }
      setStatus(meta?meta.name+" selected for compatibility check.":"Training dataset cleared.");
      draw();
    }

    function defaultTrainingConfig(entry,dataset){
      const validationSplit=dataset?.splits?.validation ? "validation" : (dataset?.splits?.test ? "test" : "train");
      return {
        budget_type:"steps",
        max_steps:1000,
        max_tokens:1000000,
        epochs:1,
        batch_size:Number(entry?.batch_size||state.project?.batch_size||16),
        gradient_accumulation:1,
        optimizer:"adamw",
        learning_rate:0.0005,
        weight_decay:0.1,
        beta1:0.9,
        beta2:0.95,
        warmup_steps:0,
        validation_split:validationSplit,
        validate_every:100,
        validation_steps:20,
        generate_on_validation:true,
        validation_prompt:"Once upon a time",
        validation_generate_tokens:64,
        checkpoint_every:500,
        seed:42,
        device:"auto",
        backend:"pytorch",
        execution_mode:"eager",
        compile_mode:"reduce-overhead",
        precision:"fp16",
        output_dir:localPaths.models||((localDefaultRoot.replace(/[\\/]+$/,"")||".")+"/mlbricks/models"),
      };
    }

    function defaultGenerationConfig(entry){
      return {
        prompt:"Once upon a time",
        max_new_tokens:128,
        temperature:0.8,
        top_k:50,
        top_p:0.95,
        seed:42,
        device:"auto",
        backend:"pytorch",
        execution_mode:"eager",
        compile_mode:"reduce-overhead",
        precision:"fp16",
      };
    }

    function mergeRuntimeDefaults(defaults,saved){
      const out={...defaults};
      Object.entries(saved||{}).forEach(([key,value])=>{
        if(value!==null && value!==undefined && !(typeof value==="string" && value.trim()==="")){
          out[key]=value;
        }
      });
      return out;
    }

    function defaultServeConfig(entry){
      const gen=defaultGenerationConfig(entry);
      return {
        host:"127.0.0.1",port:8000,cors_origin:"same-origin",require_api_key:true,public_tunnel:"off",
        max_request_bytes:1048576,max_prompt_chars:32768,max_server_new_tokens:2048,request_timeout_seconds:120,max_concurrent_requests:2,rate_limit_per_minute:60,
        device:entry?.generation_config?.device||gen.device,
        backend:entry?.generation_config?.backend||gen.backend,
        execution_mode:entry?.generation_config?.execution_mode||gen.execution_mode,
        compile_mode:entry?.generation_config?.compile_mode||gen.compile_mode,
        precision:entry?.generation_config?.precision||gen.precision
      };
    }

    function ensureRuntimeConfigs(entry){
      const dataset=preparedDatasetById(entry?.selected_dataset_id)||null;
      entry.training_config=mergeRuntimeDefaults(defaultTrainingConfig(entry,dataset),entry.training_config);
      entry.generation_config=mergeRuntimeDefaults(defaultGenerationConfig(entry),entry.generation_config);
      entry.serve_config=mergeRuntimeDefaults(defaultServeConfig(entry),entry.serve_config);
      serveSecrets[entry.id]=serveSecrets[entry.id]||{api_key:"",ngrok_token:""};
    }

    function openRuntimePanel(mode,entry){
      if(!entry)return;
      ensureRuntimeConfigs(entry);
      runtimePanel={mode,modelId:entry.id,tab:"setup"};
      // Keep MODEL WORKSPACE open normally and while serving. Only training
      // and generation collapse it automatically to maximize runtime space.
      if(mode==="train"||mode==="generate")bottomExpanded=false;
      selected=null;
      outputDirectorySelection=entry.id;
      setStatus(mode==="train"?"Training setup opened.":mode==="generate"?"Generation setup opened.":"Model API server setup opened.");
      draw();
    }

    function requestBuiltModelTraining(entry,compat){
      if(!entry||!compat?.ok)return;
      entry.training_status="configured";
      openRuntimePanel("train",entry);
    }

    function requestTokenGeneration(entry){
      if(!entry)return;
      openRuntimePanel("generate",entry);
    }

    function requestModelServing(entry){
      if(!entry||!entry.weights_ready)return;
      openRuntimePanel("serve",entry);
    }

    function runtimeDeviceOptions(){
      const devices=runtimeCaps.devices||[];
      return devices.length?devices:[{id:"auto",label:"Auto"},{id:"cpu",label:"CPU"}];
    }

    function selectedRuntimeDevice(config){
      return runtimeDeviceOptions().find(d=>d.id===config.device)||runtimeDeviceOptions()[0];
    }

    function runtimeField(label,type,value,onChange,options){
      const wrap=document.createElement("div");wrap.className="mlb-runtime-field";
      const l=document.createElement("label");l.textContent=label;wrap.appendChild(l);
      let input;
      if(type==="select"){
        input=document.createElement("select");
        (options||[]).forEach(opt=>{
          const item=typeof opt==="string"?{value:opt,label:opt}:opt;
          const o=document.createElement("option");o.value=item.value;o.textContent=item.label;
          if(String(item.value)===String(value))o.selected=true;input.appendChild(o);
        });
      }else if(type==="textarea"){
        input=document.createElement("textarea");input.rows=4;input.value=value??"";
      }else if(type==="checkbox"){
        input=document.createElement("input");input.type="checkbox";input.checked=!!value;
      }else{
        input=document.createElement("input");input.type=type||"text";input.value=value??"";
        if(type==="number")input.step="any";
      }
      const commit=()=>{
        const value=type==="checkbox"
          ?input.checked
          :(type==="number"?(input.value.trim()===""?null:Number(input.value)):input.value);
        onChange(value);
      };
      input.addEventListener("change",commit);
      wrap.appendChild(input);return wrap;
    }

    function runtimeSection(title){
      const s=document.createElement("section");s.className="mlb-runtime-section";
      const h=document.createElement("h3");h.textContent=title;s.appendChild(h);return s;
    }

    function deviceCards(config){
      const box=document.createElement("div");box.className="mlb-device-grid";
      runtimeDeviceOptions().forEach(device=>{
        const card=document.createElement("button");card.type="button";
        card.className="mlb-device-card"+(config.device===device.id?" selected":"");
        const icon=device.kind==="cpu"?"CPU":device.kind==="cuda"?"GPU":device.kind==="xpu"?"XPU":device.kind==="mps"?"GPU":"AUTO";
        card.innerHTML="<strong>"+icon+"</strong><span>"+device.label+"</span>"+
          (device.compute_capability?"<small>Compute "+device.compute_capability+"</small>":"");
        card.addEventListener("click",()=>{config.device=device.id;draw();});box.appendChild(card);
      });
      return box;
    }

    function runtimeCompatibilitySummary(entry){
      const dataset=preparedDatasetById(entry.selected_dataset_id)||null;
      return modelDatasetCompatibility(entry,dataset);
    }

    function trainingConfigValid(entry,config){
      const compat=runtimeCompatibilitySummary(entry);
      const errors=[];
      if(!compat.ok)errors.push("Training data is not compatible.");
      const positive=(value)=>value!==null&&value!==undefined&&value!==""&&Number.isFinite(Number(value))&&Number(value)>0;
      if(config.budget_type==="steps"&&!positive(config.max_steps))errors.push("Training steps must be a number greater than 0.");
      if(config.budget_type==="tokens"&&!positive(config.max_tokens))errors.push("Token budget must be a number greater than 0.");
      if(config.budget_type==="epochs"&&!positive(config.epochs))errors.push("Epochs must be a number greater than 0.");
      if(!positive(config.batch_size))errors.push("Batch size must be a number greater than 0.");
      if(!positive(config.learning_rate))errors.push("Learning rate must be a number greater than 0.");
      const betaValid=(value)=>value!==null&&value!==undefined&&Number.isFinite(Number(value))&&Number(value)>=0&&Number(value)<1;
      if(["adamw","adam"].includes(String(config.optimizer||"").toLowerCase())){
        if(!betaValid(config.beta1))errors.push("Adam Beta 1 must be between 0 and 1.");
        if(!betaValid(config.beta2))errors.push("Adam Beta 2 must be between 0 and 1.");
      }
      return {ok:errors.length===0,errors,compat};
    }

    function runtimeHistory(entry,mode){
      const key=mode==="train"?"training_history":"generation_history";
      if(!Array.isArray(entry[key]))entry[key]=[];
      return entry[key];
    }

    function recordRuntimeEvent(next){
      if(!next || !["train","generate"].includes(next.runtime_kind))return;
      const modelId=next.model_id||runtimePanel?.modelId;
      const entry=builtModelById(modelId);
      if(!entry)return;
      const history=runtimeHistory(entry,next.runtime_kind);
      const key=[next.ts||"",next.phase||"",next.step??"",next.generated_tokens??"",next.message||"",next.checkpoint_path||""].join("|");
      const event={
        key,ts:next.ts||Date.now()/1000,status:next.status||"running",phase:next.phase||"runtime",
        step:next.step??null,max_steps:next.max_steps??null,tokens_seen:next.tokens_seen??null,
        generated_tokens:next.generated_tokens??null,loss:next.loss??null,ppl:next.ppl??null,val_loss:next.val_loss??null,val_ppl:next.val_ppl??null,
        best_val_loss:next.best_val_loss??null,tokens_per_sec:next.tokens_per_sec??null,avg_tokens_per_sec:next.avg_tokens_per_sec??null,
        end_to_end_tokens_per_sec:next.end_to_end_tokens_per_sec??null,avg_end_to_end_tokens_per_sec:next.avg_end_to_end_tokens_per_sec??null,
        memory_allocated_gb:next.memory_allocated_gb??null,memory_reserved_gb:next.memory_reserved_gb??null,memory_peak_gb:next.memory_peak_gb??null,memory_total_gb:next.memory_total_gb??null,
        lr:next.lr??null,elapsed_seconds:next.elapsed_seconds??null,compile_seconds:next.compile_seconds??null,
        message:next.message||"",checkpoint_path:next.checkpoint_path||null
      };
      if(!history.length||history[history.length-1].key!==key)history.push(event);
      if(history.length>250)history.splice(0,history.length-250);

      if(next.runtime_kind==="train"){
        entry.training_live={
          status:event.status,phase:event.phase,overall:Number(next.overall||0),step:event.step,max_steps:event.max_steps,
          tokens_seen:event.tokens_seen??entry.training_live?.tokens_seen,
          loss:event.loss??entry.training_live?.loss,ppl:event.ppl??entry.training_live?.ppl,
          val_loss:event.val_loss??entry.training_live?.val_loss,val_ppl:event.val_ppl??entry.training_live?.val_ppl,
          best_val_loss:event.best_val_loss??entry.training_live?.best_val_loss,
          tokens_per_sec:event.tokens_per_sec??entry.training_live?.tokens_per_sec,avg_tokens_per_sec:event.avg_tokens_per_sec??entry.training_live?.avg_tokens_per_sec,
          end_to_end_tokens_per_sec:event.end_to_end_tokens_per_sec??entry.training_live?.end_to_end_tokens_per_sec,avg_end_to_end_tokens_per_sec:event.avg_end_to_end_tokens_per_sec??entry.training_live?.avg_end_to_end_tokens_per_sec,
          memory_allocated_gb:event.memory_allocated_gb??entry.training_live?.memory_allocated_gb,memory_reserved_gb:event.memory_reserved_gb??entry.training_live?.memory_reserved_gb,
          memory_peak_gb:event.memory_peak_gb??entry.training_live?.memory_peak_gb,memory_total_gb:event.memory_total_gb??entry.training_live?.memory_total_gb,
          lr:event.lr??entry.training_live?.lr,elapsed_seconds:event.elapsed_seconds??entry.training_live?.elapsed_seconds,
          compile_seconds:event.compile_seconds??entry.training_live?.compile_seconds,message:event.message,
          checkpoint_path:event.checkpoint_path||entry.training_live?.checkpoint_path||entry.checkpoint_path||null
        };
        if(next.sample_text){entry.latest_validation_sample=next.sample_text;entry.latest_validation_sample_step=event.step;}
        if(event.val_loss!==null){entry.latest_validation_loss=event.val_loss;entry.latest_validation_step=event.step;}
        if(event.checkpoint_path)entry.latest_checkpoint_path=event.checkpoint_path;
      }else{
        entry.generation_live={
          status:event.status,phase:event.phase,overall:Number(next.overall||0),generated_tokens:event.generated_tokens,
          message:event.message,generated_text:next.generated_text||entry.generation_live?.generated_text||entry.last_generation||""
        };
        if(next.generated_text)entry.last_generation=next.generated_text;
      }
    }

    function scheduleRuntimeStatusDraw(){
      if(!runtimePanel||runtimePanel.tab!=="status"||runtimeStatusRedrawTimer)return;
      runtimeStatusRedrawTimer=setTimeout(()=>{runtimeStatusRedrawTimer=null;draw();},120);
    }

    function runtimeTabButton(label,tab,entry,mode){
      const current=(runtimePanel?.tab||"setup")===tab;
      const button=btn(label,"mlb-runtime-tab"+(current?" active":""));
      button.addEventListener("click",()=>{runtimePanel={mode,modelId:entry.id,tab};draw();});
      return button;
    }

    function escapeRuntimeText(value){
      return String(value??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    }

    function formatDuration(seconds){
      const n=Number(seconds||0);if(!n)return "—";if(n<60)return n.toFixed(1)+"s";
      return Math.floor(n/60)+"m "+Math.floor(n%60)+"s";
    }

    function statusMetric(label,value,sub){
      const box=document.createElement("div");box.className="mlb-status-metric";
      const a=document.createElement("span");a.textContent=label;
      const b=document.createElement("strong");b.textContent=value??"—";box.append(a,b);
      if(sub){const c=document.createElement("small");c.textContent=sub;box.appendChild(c);}return box;
    }

    function trainingLive(entry){
      const live=entry.training_live||{};
      if(execution.runtime_kind==="train"&&runtimePanel?.modelId===entry.id){
        return {...live,status:execution.status||live.status,phase:execution.phase||live.phase,overall:Number(execution.overall??live.overall??0),
          step:execution.step??live.step,max_steps:execution.max_steps??live.max_steps,tokens_seen:execution.tokens_seen??live.tokens_seen,
          loss:execution.loss??live.loss,ppl:execution.ppl??live.ppl,val_loss:execution.val_loss??live.val_loss,val_ppl:execution.val_ppl??live.val_ppl,
          best_val_loss:execution.best_val_loss??live.best_val_loss,tokens_per_sec:execution.tokens_per_sec??live.tokens_per_sec,avg_tokens_per_sec:execution.avg_tokens_per_sec??live.avg_tokens_per_sec,
          end_to_end_tokens_per_sec:execution.end_to_end_tokens_per_sec??live.end_to_end_tokens_per_sec,avg_end_to_end_tokens_per_sec:execution.avg_end_to_end_tokens_per_sec??live.avg_end_to_end_tokens_per_sec,
          memory_allocated_gb:execution.memory_allocated_gb??live.memory_allocated_gb,memory_reserved_gb:execution.memory_reserved_gb??live.memory_reserved_gb,
          memory_peak_gb:execution.memory_peak_gb??live.memory_peak_gb,memory_total_gb:execution.memory_total_gb??live.memory_total_gb,lr:execution.lr??live.lr,
          elapsed_seconds:execution.elapsed_seconds??live.elapsed_seconds,compile_seconds:execution.compile_seconds??live.compile_seconds,message:execution.message||live.message,
          checkpoint_path:execution.checkpoint_path||live.checkpoint_path};
      }return live;
    }

    function generationLive(entry){
      const live=entry.generation_live||{};
      if(execution.runtime_kind==="generate"&&runtimePanel?.modelId===entry.id){
        return {...live,status:execution.status||live.status,phase:execution.phase||live.phase,overall:Number(execution.overall??live.overall??0),
          generated_tokens:execution.generated_tokens??live.generated_tokens,message:execution.message||live.message,
          generated_text:execution.generated_text||live.generated_text};
      }return live;
    }

    function renderEventLog(section,history,emptyText){
      const log=document.createElement("div");log.className="mlb-training-log";
      const events=(history||[]).slice(-100);
      if(!events.length){log.innerHTML="<div class='mlb-log-empty'>"+emptyText+"</div>";}
      else events.forEach(ev=>{
        const row=document.createElement("div");row.className="mlb-log-row "+(ev.status||"");
        const meta=[];if(ev.step!==null)meta.push("step "+ev.step);if(ev.generated_tokens!==null)meta.push(ev.generated_tokens+" tokens");if(ev.phase)meta.push(ev.phase);
        const extra=[];
        if(ev.tokens_per_sec!==null)extra.push("GPU "+Math.round(Number(ev.tokens_per_sec)).toLocaleString()+" tok/s");
        if(ev.end_to_end_tokens_per_sec!==null)extra.push("E2E "+Math.round(Number(ev.end_to_end_tokens_per_sec)).toLocaleString()+" tok/s");
        if(ev.memory_allocated_gb!==null)extra.push("mem "+Number(ev.memory_allocated_gb).toFixed(2)+" GB");
        if(ev.loss!==null)extra.push("loss "+Number(ev.loss).toFixed(4));
        if(ev.ppl!==null)extra.push("ppl "+Number(ev.ppl).toFixed(2));
        if(ev.val_loss!==null)extra.push("val "+Number(ev.val_loss).toFixed(4));
        if(ev.val_ppl!==null)extra.push("val ppl "+Number(ev.val_ppl).toFixed(2));
        row.innerHTML="<span>"+escapeRuntimeText(meta.join(" · "))+"</span><strong>"+escapeRuntimeText(ev.message||"Runtime event")+(extra.length?" · "+extra.join(" · "):"")+"</strong>";
        log.appendChild(row);
      });
      section.appendChild(log);
    }

    function renderTrainingStatus(main,side,entry){
      const config=entry.training_config||{},live=trainingLive(entry),history=runtimeHistory(entry,"train");
      const dataset=preparedDatasetById(entry.selected_dataset_id)||null;
      const hero=runtimeSection("Training Status");hero.classList.add("mlb-training-status-hero");
      const top=document.createElement("div");top.className="mlb-training-status-top";
      const stateBox=document.createElement("div");stateBox.className="mlb-training-state "+(live.status||entry.training_status||"idle");
      const stateLabel=live.status==="running"?"TRAINING":live.status==="done"?"COMPLETE":live.status==="error"?"ERROR":live.status==="stopped"?"STOPPED":entry.weights_ready?"TRAINED":"NOT STARTED";
      stateBox.innerHTML="<strong>"+stateLabel+"</strong><span>"+escapeRuntimeText(live.message||"Configure training, then press Start Training.")+"</span>";
      const pct=document.createElement("div");pct.className="mlb-training-percent";pct.innerHTML="<strong>"+Math.round(Number(live.overall||0))+"%</strong><span>"+(live.phase||"idle")+"</span>";
      top.append(stateBox,pct);hero.appendChild(top);
      const bar=document.createElement("div");bar.className="mlb-status-progress";bar.innerHTML="<i style='width:"+Math.max(0,Math.min(100,Number(live.overall||0)))+"%'></i>";hero.appendChild(bar);
      const metrics=document.createElement("div");metrics.className="mlb-status-metrics";
      // While a new/current attempt exists, use only that attempt's telemetry.
      // Stored metrics from the previous completed run are shown only when no
      // active/error attempt is being displayed. This prevents an immediate
      // setup error from appearing beside stale values such as Step 1000.
      const currentAttempt=["running","error","stopped"].includes(live.status);
      const pick=(current,stored)=>current!=null?current:(currentAttempt?null:stored);
      const memoryNow=pick(live.memory_allocated_gb,null),peakMemory=pick(live.memory_peak_gb,entry.memory_peak_gb);
      const stepNow=pick(live.step,entry.trained_steps);
      const tokNow=pick(live.tokens_per_sec,entry.avg_tokens_per_sec);
      const e2eTokNow=pick(live.end_to_end_tokens_per_sec,entry.avg_end_to_end_tokens_per_sec);
      const lossNow=pick(live.loss,entry.last_loss);
      const pplNow=pick(live.ppl,entry.last_ppl);
      const valLossStored=entry.latest_validation_loss??entry.last_val_loss;
      const valLossNow=pick(live.val_loss,valLossStored);
      const valPplNow=pick(live.val_ppl,entry.last_val_ppl);
      const tokensNow=pick(live.tokens_seen,entry.tokens_seen);
      metrics.append(statusMetric("Step",(stepNow??0)+(live.max_steps?" / "+live.max_steps:"")),
        statusMetric("GPU Tok/s",tokNow==null?"—":Math.round(Number(tokNow)).toLocaleString()),
        statusMetric("E2E Tok/s",e2eTokNow==null?"—":Math.round(Number(e2eTokNow)).toLocaleString()),
        statusMetric("Loss",lossNow==null?"—":Number(lossNow).toFixed(4)),
        statusMetric("PPL",pplNow==null?"—":Number(pplNow).toFixed(2)),
        statusMetric("Val Loss",valLossNow==null?"—":Number(valLossNow).toFixed(4)),
        statusMetric("Val PPL",valPplNow==null?"—":Number(valPplNow).toFixed(2)),
        statusMetric("GPU Memory",memoryNow==null?"—":Number(memoryNow).toFixed(2)+" GB",live.memory_total_gb==null?null:"of "+Number(live.memory_total_gb).toFixed(1)+" GB"),
        statusMetric("Peak Memory",peakMemory==null?"—":Number(peakMemory).toFixed(2)+" GB"),
        statusMetric("Compile",config.execution_mode==="compiled"?(live.compile_seconds==null?(currentAttempt?"Pending":(entry.compile_seconds==null?"Pending":Number(entry.compile_seconds).toFixed(1)+"s")):Number(live.compile_seconds).toFixed(1)+"s"):"Not used"),
        statusMetric("Tokens",Number(tokensNow??0).toLocaleString()),statusMetric("Elapsed",formatDuration(live.elapsed_seconds)));
      hero.appendChild(metrics);main.appendChild(hero);

      const validation=runtimeSection("Validation + Generated Sample");
      const vg=document.createElement("div");vg.className="mlb-validation-status-grid";
      vg.append(statusMetric("Dataset",dataset?.name||"—"),statusMetric("Validation Split",config.validation_split||"—"),
        statusMetric("Validate Every",(config.validate_every||0)+" steps"),statusMetric("Validation Steps",config.validation_steps??"—"),
        statusMetric("Latest Val",entry.latest_validation_loss==null?"—":Number(entry.latest_validation_loss).toFixed(4),entry.latest_validation_step?"step "+entry.latest_validation_step:null),
        statusMetric("Sample Tokens",config.generate_on_validation?config.validation_generate_tokens:"Off"));
      validation.appendChild(vg);
      const sample=document.createElement("div");sample.className="mlb-status-sample";
      sample.innerHTML="<div><strong>VALIDATION GENERATION</strong><span>"+(config.generate_on_validation?("Prompt: "+escapeRuntimeText(config.validation_prompt||"")):"Disabled in Training Setup")+"</span></div><pre>"+escapeRuntimeText(entry.latest_validation_sample||"No validation sample generated yet.")+"</pre>";
      validation.appendChild(sample);main.appendChild(validation);

      const logs=runtimeSection("Training Log");renderEventLog(logs,history,"Training has not started yet.");main.appendChild(logs);
      const cp=runtimeSection("Checkpoints + Output");const cg=document.createElement("div");cg.className="mlb-validation-status-grid";
      cg.append(statusMetric("Checkpoint Every",(config.checkpoint_every||0)+" steps"),
        statusMetric("Latest Checkpoint",entry.latest_checkpoint_path||entry.checkpoint_path||live.checkpoint_path||"—"),statusMetric("Weights",entry.weights_ready?"Available":"Not yet"),
        statusMetric("Training Status",entry.training_status||"untrained"),statusMetric("Trained At",entry.trained_at||"—"));cp.appendChild(cg);main.appendChild(cp);

      const summary=document.createElement("div");summary.className="mlb-runtime-summary";const dev=selectedRuntimeDevice(config);
      summary.innerHTML="<h3>Training Control</h3><div><span>Status</span><strong>"+stateLabel+"</strong></div><div><span>Device</span><strong>"+dev.label+"</strong></div><div><span>Backend</span><strong>"+config.backend+"</strong></div><div><span>Execution</span><strong>"+config.execution_mode+"</strong></div><div><span>Precision</span><strong>"+config.precision+"</strong></div>";side.appendChild(summary);
      const statusValid=trainingConfigValid(entry,config);
      side.appendChild(trainingActionButton(entry,statusValid));
      const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title=trainingIsRunning()?"Stop training and return to Model Builder":"Return to Model Builder";cancel.addEventListener("click",()=>cancelTrainingToModelEditor(entry));side.appendChild(cancel);
      if(entry.weights_ready){const gen=btn("Open Generation","mlb-generate-btn");gen.addEventListener("click",()=>openRuntimePanel("generate",entry));side.appendChild(gen);}
    }

    function renderGenerationStatus(main,side,entry){
      const config=entry.generation_config||{},live=generationLive(entry),history=runtimeHistory(entry,"generate");
      const hero=runtimeSection("Generation Status");hero.classList.add("mlb-training-status-hero");
      const top=document.createElement("div");top.className="mlb-training-status-top";
      const stateBox=document.createElement("div");stateBox.className="mlb-training-state "+(live.status||"idle");
      const stateLabel=live.status==="running"?"GENERATING":live.status==="done"?"COMPLETE":live.status==="error"?"ERROR":live.status==="stopped"?"STOPPED":"READY";
      stateBox.innerHTML="<strong>"+stateLabel+"</strong><span>"+escapeRuntimeText(live.message||"Configure generation, then press Generate Tokens.")+"</span>";
      const pct=document.createElement("div");pct.className="mlb-training-percent";pct.innerHTML="<strong>"+Math.round(Number(live.overall||0))+"%</strong><span>"+(live.phase||"idle")+"</span>";
      top.append(stateBox,pct);hero.appendChild(top);
      const bar=document.createElement("div");bar.className="mlb-status-progress";bar.innerHTML="<i style='width:"+Math.max(0,Math.min(100,Number(live.overall||0)))+"%'></i>";hero.appendChild(bar);
      const metrics=document.createElement("div");metrics.className="mlb-status-metrics";
      metrics.append(statusMetric("Generated",Number(live.generated_tokens||0).toLocaleString()),statusMetric("Target",Number(config.max_new_tokens||0).toLocaleString()),
        statusMetric("Temperature",config.temperature),statusMetric("Top K",config.top_k),statusMetric("Top P",config.top_p),statusMetric("Seed",config.seed));hero.appendChild(metrics);main.appendChild(hero);

      const output=runtimeSection("Generated Text");const prompt=document.createElement("div");prompt.className="mlb-status-prompt";prompt.innerHTML="<strong>PROMPT</strong><pre>"+escapeRuntimeText(config.prompt||"")+"</pre>";output.appendChild(prompt);
      const text=document.createElement("div");text.className="mlb-status-sample generation";text.innerHTML="<div><strong>OUTPUT</strong><span>"+Number(live.generated_tokens||0)+" / "+Number(config.max_new_tokens||0)+" tokens</span></div><pre>"+escapeRuntimeText(live.generated_text||entry.last_generation||"No generated text yet.")+"</pre>";output.appendChild(text);main.appendChild(output);

      const logs=runtimeSection("Generation Log");renderEventLog(logs,history,"Generation has not started yet.");main.appendChild(logs);
      const runtime=runtimeSection("Runtime Used");const rg=document.createElement("div");rg.className="mlb-validation-status-grid";const dev=selectedRuntimeDevice(config);
      rg.append(statusMetric("Device",dev.label),statusMetric("Backend",config.backend),statusMetric("Execution",config.execution_mode),statusMetric("Compile",config.execution_mode==="compiled"?config.compile_mode:"Not used"),statusMetric("Precision",config.precision),statusMetric("Generated At",entry.generated_at||"—"));runtime.appendChild(rg);main.appendChild(runtime);

      const summary=document.createElement("div");summary.className="mlb-runtime-summary";summary.innerHTML="<h3>Generation Control</h3><div><span>Status</span><strong>"+stateLabel+"</strong></div><div><span>Device</span><strong>"+dev.label+"</strong></div><div><span>Generated</span><strong>"+Number(live.generated_tokens||0)+" / "+Number(config.max_new_tokens||0)+"</strong></div><div><span>Weights</span><strong>"+(entry.weights_ready?"Available":"Missing")+"</strong></div>";side.appendChild(summary);
      side.appendChild(generationActionButton(entry));
      const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title=generationIsRunning()?"Stop generation and return to Model Builder":"Return to Model Builder";cancel.addEventListener("click",()=>cancelRuntimeToModelEditor(entry,"generate"));side.appendChild(cancel);
    }

    async function copyTextRobust(text,label="Text"){
      const value=String(text||"");
      if(!value){setStatus(label+" is empty.");return false;}

      try{
        if(navigator.clipboard&&navigator.clipboard.writeText){
          await navigator.clipboard.writeText(value);
          setStatus(label+" copied.");
          return true;
        }
      }catch(_){/* Kaggle iframe may block clipboard permission. */}

      try{
        const doc=root.ownerDocument||document;
        const area=doc.createElement("textarea");
        area.value=value;
        area.setAttribute("readonly","");
        area.style.position="fixed";
        area.style.left="-9999px";
        area.style.top="0";
        doc.body.appendChild(area);
        area.focus();area.select();area.setSelectionRange(0,value.length);
        const ok=doc.execCommand&&doc.execCommand("copy");
        area.remove();
        if(ok){setStatus(label+" copied.");return true;}
      }catch(_){/* fall through */}

      // Last-resort Kaggle-safe behavior: select/show the secret so Ctrl+C works.
      try{
        const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
        if(win&&typeof win.prompt==="function"){
          win.prompt("Copy "+label+" (Ctrl+C, Enter):",value);
          setStatus("Copy "+label+" from the opened box.");
          return false;
        }
      }catch(_){ }
      setStatus("Clipboard blocked. Select the "+label+" value and press Ctrl+C.");
      return false;
    }

    function serveUrlCard(label,url,kind,emptyLabel="Unavailable"){
      const card=document.createElement("div");card.className="mlb-serve-url "+kind;
      const top=document.createElement("div");top.innerHTML="<strong>"+label+"</strong><span>"+(url||emptyLabel)+"</span>";card.appendChild(top);
      if(url){const actions=document.createElement("div");
        const open=btn("Open","mlb-serve-mini");open.addEventListener("click",()=>window.open(url,"_blank","noopener"));
        const copyBtn=btn("Copy","mlb-serve-mini");copyBtn.addEventListener("click",()=>copyTextRobust(url,label));
        actions.append(open,copyBtn);card.appendChild(actions);}
      return card;
    }

    function serveCodeExample(entry,info){
      const base=info?.public_url||info?.lan_url||info?.local_url||"http://127.0.0.1:8000";
      const secret=serveSecrets[entry.id]?.api_key||"YOUR_API_KEY";
      const lines=[
        'fetch("'+base+'/v1/generate", {',
        '  method: "POST",',
        '  headers: {',
        '    "Content-Type": "application/json",'
      ];
      if(entry.serve_config?.require_api_key!==false){
        lines.push('    "Authorization": "Bearer '+secret+'",');
      }
      lines.push(
        '  },',
        '  body: JSON.stringify({',
        '    prompt: "Once upon a time",',
        '    max_new_tokens: 128',
        '  })',
        '})',
        '.then(r => r.json())',
        '.then(console.log);'
      );
      return lines.join("\n");
    }


    function renderServingWorkspace(canvas,entry){
      ensureRuntimeConfigs(entry);
      const config=entry.serve_config,secret=serveSecrets[entry.id]||(serveSecrets[entry.id]={api_key:"",ngrok_token:""});
      const tab=runtimePanel?.tab||"setup",info=entry.serve_live||{};
      const outer=document.createElement("div");outer.className="mlb-runtime-workspace";
      const top=document.createElement("div");top.className="mlb-runtime-head";
      const title=document.createElement("div");title.innerHTML="<strong>SERVE MODEL / API</strong><span>"+entry.name+"</span>";
      const tabs=document.createElement("div");tabs.className="mlb-runtime-tabs";
      tabs.append(runtimeTabButton("API Server Setup","setup",entry,"serve"),runtimeTabButton("API Server Status","status",entry,"serve"));
      top.append(title,tabs);outer.appendChild(top);
      const layout=document.createElement("div");layout.className="mlb-runtime-layout";
      const main=document.createElement("div");main.className="mlb-runtime-main",side=document.createElement("aside");side.className="mlb-runtime-side";
      layout.append(main,side);outer.appendChild(layout);
      const update=(key,value)=>{config[key]=value;setStatus("Server setting updated: "+key);draw();};

      if(tab==="status"){
        const running=entry.serve_status==="running"||!!info.local_url;
        const failed=entry.serve_status==="error"||!!info.error;
        const tunnelError=info.public_tunnel_error||entry.serve_tunnel_error||null;
        const hero=runtimeSection("API Server Status"),status=document.createElement("div");
        status.className="mlb-serve-status "+(running?"running":failed?"error":"stopped");
        status.innerHTML="<strong>"+(running?(tunnelError?"● RUNNING · LOCAL":"● RUNNING"):failed?"✕ ERROR":"○ STOPPED")+"</strong><span>"+
          (running
            ?(tunnelError?"HTTP server is running, but Public HTTPS failed. Check the ngrok error below.":"Model is accepting HTTP inference requests.")
            :failed?(info.error||"API server failed to start."):"Start the server from API Server Setup.")+
          "</span>";
        hero.appendChild(status);
        if(running&&info.used_port_fallback){
          const portNotice=document.createElement("div");portNotice.className="mlb-serve-port-notice";
          portNotice.innerHTML="<strong>Port "+(info.requested_port||config.port)+" was busy.</strong><span>Builder automatically selected port "+info.port+". All links and examples below use the actual port.</span>";
          hero.appendChild(portNotice);
        }
        if(running&&tunnelError){
          const tunnelNotice=document.createElement("div");tunnelNotice.className="mlb-serve-tunnel-error";
          tunnelNotice.innerHTML="<strong>Public HTTPS tunnel failed</strong><span>"+tunnelError+"</span><small>The local HTTP server and API key are still valid. Fix the ngrok token/setup, then Restart API Server.</small>";
          hero.appendChild(tunnelNotice);
        }
        main.appendChild(hero);
        const links=runtimeSection("Access Links"),linkGrid=document.createElement("div");linkGrid.className="mlb-serve-links";
        linkGrid.append(serveUrlCard("Localhost",info.local_url||entry.serve_urls?.local_url,"local"),
          serveUrlCard("LAN / Same Wi‑Fi",info.lan_url||entry.serve_urls?.lan_url,"lan"),
          serveUrlCard(
            "Public HTTPS",
            info.public_url||entry.serve_urls?.public_url,
            "public",
            tunnelError?"Tunnel failed":"Unavailable"
          ));links.appendChild(linkGrid);main.appendChild(links);
        if(info.remote_notebook&&!info.public_url){const warn=document.createElement("div");warn.className="mlb-serve-warning";
          warn.innerHTML="<strong>"+(info.environment||"Remote notebook")+" detected</strong><span>localhost and LAN belong to the remote kernel. Enable ngrok Public HTTPS in Setup for your phone or local web app.</span>";main.appendChild(warn);}
        const endpoints=runtimeSection("API Endpoints"),ep=document.createElement("div");ep.className="mlb-serve-endpoints";
        [["Playground","GET /"],["Health","GET /health"],["Generate","POST /v1/generate"],["OpenAI-style","POST /v1/completions"],["Models","GET /v1/models"]].forEach(([a,b])=>{const row=document.createElement("div");row.innerHTML="<span>"+a+"</span><strong>"+b+"</strong>";ep.appendChild(row);});endpoints.appendChild(ep);main.appendChild(endpoints);
        const code=runtimeSection("Web App Example"),pre=document.createElement("pre");pre.className="mlb-serve-code";pre.textContent=serveCodeExample(entry,info);code.appendChild(pre);main.appendChild(code);
        const summary=document.createElement("div");summary.className="mlb-runtime-summary";summary.innerHTML="<h3>Server</h3><div><span>Status</span><strong>"+(running?"Running":"Stopped")+"</strong></div><div><span>Port</span><strong>"+(info.port||config.port)+"</strong></div><div><span>API Key</span><strong>"+(config.require_api_key?"Required":"Off")+"</strong></div><div><span>Public Tunnel</span><strong>"+(config.public_tunnel||"off")+"</strong></div>";side.appendChild(summary);
        if(config.require_api_key){const keyBox=document.createElement("div");keyBox.className="mlb-serve-secret";keyBox.innerHTML="<strong>API KEY</strong><code title='Click to select'>"+(secret.api_key||"Restart server to generate key")+"</code>";
          const keyCode=keyBox.querySelector("code");
          keyCode?.addEventListener("click",()=>{
            try{const range=(root.ownerDocument||document).createRange();range.selectNodeContents(keyCode);const sel=(root.ownerDocument.defaultView||window).getSelection();sel.removeAllRanges();sel.addRange(range);setStatus("API key selected — press Ctrl+C.");}catch(_){}
          });const copyKey=btn("Copy API Key","mlb-dark-btn");copyKey.addEventListener("click",()=>copyTextRobust(secret.api_key||"","API key"));keyBox.appendChild(copyKey);side.appendChild(keyBox);}
        const check=btn("Refresh Status","mlb-dark-btn");check.addEventListener("click",()=>requestServeCommand("serve_status",entry));side.appendChild(check);
        const stop=btn("Stop API Server","mlb-runtime-stop");stop.disabled=!running;stop.addEventListener("click",()=>requestServeCommand("serve_stop",entry));side.appendChild(stop);
        const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title="Stop server and return to Model Builder";cancel.addEventListener("click",()=>cancelRuntimeToModelEditor(entry,"serve"));side.appendChild(cancel);
        canvas.appendChild(outer);return;
      }

      const access=runtimeSection("Server Access"),accessGrid=document.createElement("div");accessGrid.className="mlb-runtime-grid";
      accessGrid.append(runtimeField("Host","text",config.host,v=>update("host",v)),runtimeField("Port","number",config.port,v=>update("port",v)),
        runtimeField("CORS Origin","text",config.cors_origin,v=>update("cors_origin",v)),runtimeField("Require API Key","checkbox",config.require_api_key,v=>update("require_api_key",v)),
        runtimeField("Public Link","select",config.public_tunnel,v=>update("public_tunnel",v),[{value:"off",label:"Off — Local / LAN only"},{value:"ngrok",label:"ngrok — Public HTTPS"}]));
      access.appendChild(accessGrid);main.appendChild(access);

      const limits=runtimeSection("Safety & Request Limits"),limitsGrid=document.createElement("div");limitsGrid.className="mlb-runtime-grid";
      limitsGrid.append(
        runtimeField("Max Request Bytes","number",config.max_request_bytes,v=>update("max_request_bytes",v)),
        runtimeField("Max Prompt Characters","number",config.max_prompt_chars,v=>update("max_prompt_chars",v)),
        runtimeField("Max New Tokens","number",config.max_server_new_tokens,v=>update("max_server_new_tokens",v)),
        runtimeField("Request Timeout (sec)","number",config.request_timeout_seconds,v=>update("request_timeout_seconds",v)),
        runtimeField("Max Concurrent Requests","number",config.max_concurrent_requests,v=>update("max_concurrent_requests",v)),
        runtimeField("Rate Limit / Minute","number",config.rate_limit_per_minute,v=>update("rate_limit_per_minute",v))
      );
      limits.appendChild(limitsGrid);main.appendChild(limits);

      const secretsSection=runtimeSection("Session Credentials"),secGrid=document.createElement("div");secGrid.className="mlb-runtime-grid";
      secGrid.appendChild(runtimeField("API Key (blank = generate one)","password",secret.api_key,v=>secret.api_key=v));
      if(config.public_tunnel==="ngrok")secGrid.appendChild(runtimeField("ngrok Authtoken","password",secret.ngrok_token,v=>secret.ngrok_token=v));
      secretsSection.appendChild(secGrid);const sn=document.createElement("div");sn.className="mlb-serve-secret-note";sn.textContent="API keys and ngrok tokens are session-only and are not saved in Builder files.";secretsSection.appendChild(sn);main.appendChild(secretsSection);

      const dev=runtimeSection("Available Devices");dev.appendChild(deviceCards(config));main.appendChild(dev);
      const runtime=runtimeSection("Inference Runtime"),grid=document.createElement("div");grid.className="mlb-runtime-grid";
      const deviceOpts=runtimeDeviceOptions().map(d=>({value:d.id,label:d.label}));
      grid.append(runtimeField("Device","select",config.device,v=>update("device",v),deviceOpts),
        runtimeField("Backend","select",config.backend,v=>update("backend",v),runtimeCaps.backends||["auto","native","pytorch"]),
        runtimeField("Execution","select",config.execution_mode,v=>update("execution_mode",v),runtimeCaps.execution_modes||["eager","compiled"]),
        runtimeField("Compile Mode","select",config.compile_mode,v=>update("compile_mode",v),runtimeCaps.compile_modes||["default","reduce-overhead","max-autotune"]),
        runtimeField("Precision","select",config.precision,v=>update("precision",v),runtimeCaps.precisions||["auto","fp32","fp16","bf16"]));runtime.appendChild(grid);main.appendChild(runtime);

      const device=selectedRuntimeDevice(config),summary=document.createElement("div");summary.className="mlb-runtime-summary";
      summary.innerHTML="<h3>Serve Summary</h3><div><span>Device</span><strong>"+device.label+"</strong></div><div><span>Host</span><strong>"+config.host+":"+config.port+"</strong></div><div><span>Auth</span><strong>"+(config.require_api_key?"API key":"None")+"</strong></div><div><span>Public</span><strong>"+(config.public_tunnel==="ngrok"?"ngrok HTTPS":"Off")+"</strong></div><div><span>Execution</span><strong>"+config.execution_mode+"</strong></div>";side.appendChild(summary);
      const weights=document.createElement("div");weights.className="mlb-weight-status "+(entry.weights_ready?"ready":"missing");weights.textContent=entry.weights_ready?"✓ Trained / loaded weights available":"✕ No weights available";side.appendChild(weights);
      const start=btn(entry.serve_status==="running"?"Restart API Server":"Start API Server","mlb-runtime-start");start.disabled=!entry.weights_ready||execution.status==="running";
      start.addEventListener("click",()=>{
        runtimePanel={mode:"serve",modelId:entry.id,tab:"status"};
        entry.serve_status="starting";
        entry.serve_live={running:false,error:null,message:"Starting API server…"};
        draw();
        setTimeout(()=>requestServeCommand("serve_start",entry),80);
      });side.appendChild(start);
      const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title="Return to Model Builder";cancel.addEventListener("click",()=>cancelRuntimeToModelEditor(entry,"serve"));side.appendChild(cancel);
      if(config.public_tunnel==="ngrok"){const note=document.createElement("div");note.className="mlb-serve-warning compact";note.innerHTML="<strong>Remote access</strong><span>ngrok creates the HTTPS URL needed to reach a Kaggle/Colab model from your phone or local web app.</span>";side.appendChild(note);}
      canvas.appendChild(outer);
    }

    function renderRuntimeWorkspace(canvas,entry,mode){
      if(mode==="serve"){renderServingWorkspace(canvas,entry);return;}
      const outer=document.createElement("div");outer.className="mlb-runtime-workspace";
      const top=document.createElement("div");top.className="mlb-runtime-head";
      const title=document.createElement("div");title.innerHTML="<strong>"+(mode==="train"?"TRAIN MODEL":"GENERATE TOKENS")+"</strong><span>"+entry.name+"</span>";
      const tabs=document.createElement("div");tabs.className="mlb-runtime-tabs";
      tabs.append(
        runtimeTabButton(mode==="train"?"Training Setup":"Generation Setup","setup",entry,mode),
        runtimeTabButton(mode==="train"?"Training Status":"Generation Status","status",entry,mode)
      );
      top.append(title,tabs);outer.appendChild(top);

      const layout=document.createElement("div");layout.className="mlb-runtime-layout";
      const main=document.createElement("div");main.className="mlb-runtime-main";
      const side=document.createElement("aside");side.className="mlb-runtime-side";
      layout.append(main,side);outer.appendChild(layout);

      const config=mode==="train"?entry.training_config:entry.generation_config;
      const update=(key,value)=>{config[key]=value;setStatus((mode==="train"?"Training":"Generation")+" setting updated: "+key);draw();};
      const tab=runtimePanel?.tab||"setup";
      if(tab==="status"){
        if(mode==="train")renderTrainingStatus(main,side,entry);
        else renderGenerationStatus(main,side,entry);
        canvas.appendChild(outer);
        return;
      }

      const dev=runtimeSection("Available Devices");dev.appendChild(deviceCards(config));main.appendChild(dev);

      if(mode==="train"){
        const dataset=preparedDatasetById(entry.selected_dataset_id)||null;
        const budget=runtimeSection("Training Budget");
        const budgetGrid=document.createElement("div");budgetGrid.className="mlb-runtime-grid";
        budgetGrid.append(
          runtimeField("Budget By","select",config.budget_type,v=>update("budget_type",v),["steps","tokens","epochs"]),
          runtimeField("Training Steps","number",config.max_steps,v=>update("max_steps",v)),
          runtimeField("Token Budget","number",config.max_tokens,v=>update("max_tokens",v)),
          runtimeField("Epochs","number",config.epochs,v=>update("epochs",v)),
          runtimeField("Batch Size","number",config.batch_size,v=>update("batch_size",v)),
          runtimeField("Gradient Accumulation","number",config.gradient_accumulation,v=>update("gradient_accumulation",v))
        );budget.appendChild(budgetGrid);main.appendChild(budget);

        const opt=runtimeSection("Optimizer");const optGrid=document.createElement("div");optGrid.className="mlb-runtime-grid";
        optGrid.append(
          runtimeField("Optimizer","select",config.optimizer,v=>update("optimizer",v),["adamw","adam","sgd"]),
          runtimeField("Learning Rate","number",config.learning_rate,v=>update("learning_rate",v)),
          runtimeField("Weight Decay","number",config.weight_decay,v=>update("weight_decay",v)),
          runtimeField("Adam Beta 1","number",config.beta1,v=>update("beta1",v)),
          runtimeField("Adam Beta 2","number",config.beta2,v=>update("beta2",v)),
          runtimeField("Warmup Steps","number",config.warmup_steps,v=>update("warmup_steps",v)),
          runtimeField("Seed","number",config.seed,v=>update("seed",v))
        );opt.appendChild(optGrid);main.appendChild(opt);

        const val=runtimeSection("Validation + Sample Generation");const valGrid=document.createElement("div");valGrid.className="mlb-runtime-grid";
        const splitOpts=dataset?Object.keys(dataset.splits||{}).map(x=>({value:x,label:datasetSplitLabel(x,dataset)})):[{value:"validation",label:"Validation"}];
        valGrid.append(
          runtimeField("Validation Split","select",config.validation_split,v=>update("validation_split",v),splitOpts),
          runtimeField("Validate Every N Steps","number",config.validate_every,v=>update("validate_every",v)),
          runtimeField("Validation Steps","number",config.validation_steps,v=>update("validation_steps",v)),
          runtimeField("Generate Sample at Validation","checkbox",config.generate_on_validation,v=>update("generate_on_validation",v)),
          runtimeField("Validation Sample Tokens","number",config.validation_generate_tokens,v=>update("validation_generate_tokens",v)),
          runtimeField("Checkpoint Every N Steps","number",config.checkpoint_every,v=>update("checkpoint_every",v))
        );val.appendChild(valGrid);
        if(config.generate_on_validation)val.appendChild(runtimeField("Validation Prompt","textarea",config.validation_prompt,v=>update("validation_prompt",v)));
        main.appendChild(val);
      }else{
        const gen=runtimeSection("Prompt + Sampling");
        gen.appendChild(runtimeField("Prompt","textarea",config.prompt,v=>update("prompt",v)));
        const genGrid=document.createElement("div");genGrid.className="mlb-runtime-grid";
        genGrid.append(
          runtimeField("New Token Count","number",config.max_new_tokens,v=>update("max_new_tokens",v)),
          runtimeField("Temperature","number",config.temperature,v=>update("temperature",v)),
          runtimeField("Top K","number",config.top_k,v=>update("top_k",v)),
          runtimeField("Top P","number",config.top_p,v=>update("top_p",v)),
          runtimeField("Seed","number",config.seed,v=>update("seed",v))
        );gen.appendChild(genGrid);main.appendChild(gen);
      }

      const runtime=runtimeSection("Runtime");const runtimeGrid=document.createElement("div");runtimeGrid.className="mlb-runtime-grid";
      const deviceOpts=runtimeDeviceOptions().map(d=>({value:d.id,label:d.label}));
      runtimeGrid.append(
        runtimeField("Device","select",config.device,v=>update("device",v),deviceOpts),
        runtimeField("Backend","select",config.backend,v=>update("backend",v),runtimeCaps.backends||["auto","native","pytorch"]),
        runtimeField("Execution","select",config.execution_mode,v=>update("execution_mode",v),runtimeCaps.execution_modes||["eager","compiled"]),
        runtimeField("Compile Mode","select",config.compile_mode,v=>update("compile_mode",v),runtimeCaps.compile_modes||["default","reduce-overhead","max-autotune"]),
        runtimeField("Precision","select",config.precision,v=>update("precision",v),runtimeCaps.precisions||["auto","fp32","fp16","bf16"])
      );runtime.appendChild(runtimeGrid);main.appendChild(runtime);


      const device=selectedRuntimeDevice(config);
      const summary=document.createElement("div");summary.className="mlb-runtime-summary";
      summary.innerHTML="<h3>Runtime Summary</h3>"+
        "<div><span>Device</span><strong>"+device.label+"</strong></div>"+
        "<div><span>Backend</span><strong>"+config.backend+"</strong></div>"+
        "<div><span>Execution</span><strong>"+config.execution_mode+"</strong></div>"+
        "<div><span>Compile</span><strong>"+(config.execution_mode==="compiled"?config.compile_mode:"Not used")+"</strong></div>"+
        "<div><span>Precision</span><strong>"+config.precision+"</strong></div>";
      side.appendChild(summary);

      if(mode==="train"){
        const valid=trainingConfigValid(entry,config);
        side.appendChild(compatibilityCard(valid.compat));
        if(!valid.ok){const errors=document.createElement("div");errors.className="mlb-runtime-errors";errors.innerHTML=valid.errors.map(x=>"<div>✕ "+x+"</div>").join("");side.appendChild(errors);}
        side.appendChild(trainingActionButton(entry,valid));
        const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title=trainingIsRunning()?"Stop training and return to Model Builder":"Return to Model Builder";cancel.addEventListener("click",()=>cancelTrainingToModelEditor(entry));side.appendChild(cancel);
      }else{
        const weights=document.createElement("div");weights.className="mlb-weight-status "+(entry.weights_ready?"ready":"missing");
        weights.textContent=entry.weights_ready?"✓ Model weights available":"✕ No trained/loaded weights yet";side.appendChild(weights);
        side.appendChild(generationActionButton(entry));
        const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title=generationIsRunning()?"Stop generation and return to Model Builder":"Return to Model Builder";cancel.addEventListener("click",()=>cancelRuntimeToModelEditor(entry,"generate"));side.appendChild(cancel);
      }

      const reset=btn("Reset Runtime Defaults","mlb-dark-btn");
      reset.addEventListener("click",()=>{
        const dataset=preparedDatasetById(entry?.selected_dataset_id)||null;
        if(mode==="train")entry.training_config=defaultTrainingConfig(entry,dataset);
        else entry.generation_config=defaultGenerationConfig(entry);
        setStatus((mode==="train"?"Training":"Generation")+" runtime settings reset to safe defaults.");
        draw();
      });
      side.appendChild(reset);

      const live=document.createElement("div");live.className="mlb-runtime-live "+(execution.status||"idle");
      live.innerHTML="<div class='mlb-runtime-live-head'><strong>RUNTIME</strong><span>"+Math.round(Number(execution.overall||0))+"%</span></div><div class='mlb-runtime-live-message'>"+(execution.runtime_kind===mode?(execution.message||"Ready"):"Ready")+"</div><div class='mlb-runtime-progress'><i style='width:"+(execution.runtime_kind===mode?Number(execution.overall||0):0)+"%'></i></div>";
      side.appendChild(live);
      const note=document.createElement("div");note.className="mlb-runtime-executor-note";
      note.textContent="Training uses packed fixed-shape causal-LM batches for both eager and compiled execution. Compiled mode captures one full model + LM-head + loss graph with fullgraph=True and dynamic=False. Supported training components include Embedding, Learned/Sinusoidal Position, ESA, StateAware ESA Stack, SOUP, RMSNorm/LayerNorm, Linear, FFN, Residual, Dropout, LM Head and reusable Modules built from them.";side.appendChild(note);

      canvas.appendChild(outer);
    }


    function compatibilityCard(compat){
      const box=document.createElement("div");
      box.className="mlb-compat-card "+(compat.ok?"compatible":"incompatible");
      const head=document.createElement("div");head.className="mlb-compat-head";
      head.innerHTML="<strong>"+(compat.ok?"✓ Compatible":"✕ Not Compatible")+"</strong><span>"+(compat.ok?"Ready for training":"Fix the items below")+"</span>";
      box.appendChild(head);
      (compat.checks||[]).forEach(check=>{
        const row=document.createElement("div");row.className="mlb-compat-row "+(check.ok?"pass":"fail");
        row.innerHTML="<span>"+(check.ok?"✓":"✕")+" "+check.label+"</span><strong>"+check.detail+"</strong>";
        box.appendChild(row);
      });
      return box;
    }

    function renderRuntimeContextInspector(body,entry,mode){
      if(!entry)return;
      ensureRuntimeConfigs(entry);
      const dataset=preparedDatasetById(entry.selected_dataset_id)||null;
      const req=entry.requirements||{};
      const isTrain=mode==="train";
      const running=execution.status==="running"&&execution.runtime_kind===mode&&execution.model_id===entry.id;
      const config=isTrain?entry.training_config:(mode==="generate"?entry.generation_config:entry.serve_config);
      const device=selectedRuntimeDevice(config||{});

      const head=document.createElement("div");head.className="mlb-selected";
      head.innerHTML="<strong>"+entry.name+"</strong><span class='mlb-pill'>"+(isTrain?"Training":"Runtime")+"</span>";
      body.appendChild(head);

      const statusBox=document.createElement("div");
      statusBox.className="mlb-api-status "+(running?"ok":"utility");
      statusBox.textContent=running
        ?("● "+(isTrain?"Training":"Runtime")+" active · "+(execution.message||"Running"))
        :(isTrain?"Training configuration ready":"Runtime configuration ready");
      body.appendChild(statusBox);

      detailSection(body,"MODEL",[
        ["Status",entry.status||"built"],
        ["Layers",entry.nodes??"—"],
        ["Parameters",entry.estimated_parameters??"—"],
        ["Input",req.modality||"unknown"],
        ["Output",req.output_type||"unknown"],
      ]);

      detailSection(body,"RUNTIME",[
        ["Device",device?.label||"Auto"],
        ["Backend",config?.backend||"auto"],
        ["Execution",config?.execution_mode||"eager"],
        ["Precision",config?.precision||"auto"],
      ]);

      if(isTrain){
        const compat=modelDatasetCompatibility(entry,dataset);
        const compTitle=document.createElement("div");compTitle.className="mlb-section-title";compTitle.textContent="COMPATIBILITY";body.appendChild(compTitle);
        body.appendChild(compatibilityCard(compat));
        if(dataset)body.appendChild(datasetSummaryCard(dataset,"TRAINING DATA"));
        else{
          const empty=document.createElement("div");empty.className="mlb-api-path";empty.textContent="No prepared training dataset selected.";body.appendChild(empty);
        }
        const note=document.createElement("div");note.className="mlb-runtime-note";
        note.textContent=running
          ?"Model and dataset details are read-only context while training is running. Use the center runtime controls to stop or cancel training."
          :"Training controls and settings are in the center workspace. This panel shows the model, runtime and selected data context.";
        body.appendChild(note);
      }else if(mode==="generate"){
        const weights=document.createElement("div");weights.className="mlb-weight-status "+(entry.weights_ready?"ready":"missing");weights.textContent=entry.weights_ready?"✓ Model weights available":"✕ No trained/loaded weights yet";body.appendChild(weights);
      }
    }

    function renderBuiltModelInspector(body,entry){
      const dataset=preparedDatasetById(entry.selected_dataset_id)||null;
      const compat=modelDatasetCompatibility(entry,dataset);
      const req=entry.requirements||{};

      const head=document.createElement("div");head.className="mlb-selected";
      head.innerHTML="<strong>"+entry.name+"</strong><span class='mlb-pill'>Built Model</span>";
      body.appendChild(head);

      const built=document.createElement("div");built.className="mlb-api-status ok";
      built.textContent="✓ Build complete · revision "+(entry.revision||1);
      body.appendChild(built);

      detailSection(body,"MODEL DETAILS",[
        ["Status",entry.status||"built"],
        ["Layers",entry.nodes??"—"],
        ["Connections",entry.connections??"—"],
        ["Input",req.modality||"unknown"],
        ["Output",req.output_type||"unknown"],
        ["Parameters",entry.estimated_parameters??"—"],
        ["Built",entry.built_at||"—"],
      ]);

      renderModelSettings(body,entry);

      const dataTitle=document.createElement("div");dataTitle.className="mlb-section-title";dataTitle.textContent="TRAINING DATA";
      body.appendChild(dataTitle);

      const field=document.createElement("div");field.className="mlb-field";
      const label=document.createElement("label");label.textContent="Prepared Dataset";
      const select=document.createElement("select");
      select.className="mlb-training-data-select";
      const datasets=availablePreparedDatasets();
      if(!datasets.length){
        const o=document.createElement("option");o.value="";o.textContent="No prepared datasets available";
        select.appendChild(o);select.disabled=true;
      }else{
        const blank=document.createElement("option");blank.value="";blank.textContent="Select prepared dataset…";select.appendChild(blank);
        datasets.forEach(meta=>{
          const o=document.createElement("option");o.value=meta.id;
          o.textContent=meta.name;
          o.title=meta.name+" — "+compactDatasetSummary(meta);
          if(entry.selected_dataset_id===meta.id)o.selected=true;
          select.appendChild(o);
        });
        select.addEventListener("change",()=>setBuiltModelDataset(entry,select.value));
      }
      field.append(label,select);body.appendChild(field);

      const compTitle=document.createElement("div");compTitle.className="mlb-section-title";compTitle.textContent="COMPATIBILITY";
      body.appendChild(compTitle);
      body.appendChild(compatibilityCard(compat));

      if(dataset){
        body.appendChild(datasetSummaryCard(dataset,"SELECTED TRAINING DATA"));
      }

      const actionTitle=document.createElement("div");actionTitle.className="mlb-section-title mlb-model-actions-title";actionTitle.textContent="ACTIONS";
      body.appendChild(actionTitle);
      const actions=document.createElement("div");actions.className="mlb-model-actions";

      if(compat.ok && entry.status!=="needs_rebuild"){
        const train=btn("Train","mlb-train-btn");
        train.addEventListener("click",()=>requestBuiltModelTraining(entry,compat));
        actions.appendChild(train);
      }else{
        const blocked=document.createElement("div");blocked.className="mlb-train-blocked";
        blocked.textContent=entry.status==="needs_rebuild"
          ?"Model settings changed. Click Build before training."
          :"Train appears when the selected data is compatible.";
        actions.appendChild(blocked);
      }

      if(req.modality==="text"){
        const generate=btn(entry.weights_ready?"Generate Tokens":"Configure Generation","mlb-generate-btn");
        generate.title=entry.weights_ready
          ?"Open generation settings"
          :"Configure generation now; actual token generation needs trained/loaded weights";
        generate.addEventListener("click",()=>requestTokenGeneration(entry));
        actions.appendChild(generate);

        const serve=btn("Serve Model / API","mlb-serve-btn");
        serve.disabled=!entry.weights_ready;
        serve.title=entry.weights_ready
          ?"Create localhost, LAN, or public API links"
          :"Train or load model weights before serving";
        serve.addEventListener("click",()=>requestModelServing(entry));
        actions.appendChild(serve);
      }
      body.appendChild(actions);

      const note=document.createElement("div");note.className="mlb-runtime-note";
      note.textContent=entry.weights_ready
        ?"Weights are available. Train/Generation open the runtime workspace in the center."
        :"Train opens full runtime settings. Generation can be configured now, but token generation needs trained/loaded weights.";
      body.appendChild(note);
    }

    function dataStorageLabel(meta){
      if(meta.storage==="disk+memory")return "Memory + Disk";
      if(meta.storage==="disk")return "Disk";
      return "Memory";
    }

    function makeDirectoryEmpty(message,help){
      const empty=document.createElement("div");empty.className="mlb-output-empty";
      empty.innerHTML="<strong>"+message+"</strong><span>"+help+"</span>";
      return empty;
    }

    function useDatasetInModel(meta){
      if(!meta)return;
      checkpoint("Use "+meta.name+" in Model");
      autoBindDatasetToModel(meta);
      setStatus(meta.name+" selected for Model Builder Text Input.");
      draw();
    }

    function renderDataOutputDirectory(container){
      const entries=availablePreparedDatasets();
      const head=document.createElement("div");head.className="mlb-output-head";head.innerHTML="<div><strong>DATA REPOSITORY</strong><span>"+entries.length+" dataset"+(entries.length===1?"":"s")+" · processed, loaded and imported data</span></div>";container.appendChild(head);
      if(!entries.length){container.appendChild(makeDirectoryEmpty("No prepared datasets yet.","Run a Data Processing pipeline. Completed datasets will appear here automatically."));return;}
      const list=document.createElement("div");list.className="mlb-output-list compact";
      entries.forEach(meta=>{const card=document.createElement("div");card.className="mlb-output-entry compact"+(outputDirectorySelection===meta.id?" selected":"");const top=document.createElement("div");top.className="mlb-output-entry-top";const sourceLabel=meta.local_source?"Local Environment Data":dataStorageLabel(meta);top.innerHTML="<div class='mlb-output-name'><strong>"+meta.name+"</strong><span>"+sourceLabel+"</span></div><span class='mlb-output-type data'>DATA</span>";card.appendChild(top);const stats=document.createElement("div");stats.className="mlb-output-stats compact";[["train","Train"],["validation","Val"],["test","Test"]].forEach(([key,label])=>{if(meta.splits?.[key]){const item=document.createElement("div");item.innerHTML="<span>"+label+"</span><strong>"+splitRows(meta,key)+"</strong>";stats.appendChild(item);}});card.appendChild(stats);const foot=document.createElement("div");foot.className="mlb-output-compact-foot";foot.innerHTML="<span>"+(meta.total_rows??"?")+" rows</span><span>Details →</span>";card.appendChild(foot);card.addEventListener("click",()=>{outputDirectorySelection=meta.id;selected=null;inspectorTab="settings";setStatus(meta.name+" details opened.");draw();});list.appendChild(card);});container.appendChild(list);
    }

    function renderModelOutputDirectory(container){
      const entries=modelDirectoryEntries();
      const head=document.createElement("div");head.className="mlb-output-head";
      head.innerHTML="<div><strong>MODEL REPOSITORY</strong><span>"+entries.length+" model"+(entries.length===1?"":"s")+" · built, trained, loaded and imported models</span></div>";
      container.appendChild(head);

      if(!entries.length){
        container.appendChild(makeDirectoryEmpty(
          "No built model yet.",
          "Finish the architecture in Model Builder, then click Build."
        ));
        return;
      }

      const list=document.createElement("div");list.className="mlb-output-list compact";
      entries.forEach(entry=>{
        const card=document.createElement("div");
        card.className="mlb-output-entry compact"+(outputDirectorySelection===entry.id?" selected":"");

        const top=document.createElement("div");top.className="mlb-output-entry-top";
        const sourceLabel=entry.legacy_recovered
          ?"Recovered Legacy Checkpoint"
          :(entry.local_source?"Local Environment Model":"Built Model · r"+(entry.revision||1));
        top.innerHTML="<div class='mlb-output-name'><strong>"+entry.name+"</strong><span>"+sourceLabel+"</span></div>"+
          "<span class='mlb-output-type model'>"+(entry.legacy_recovered?"RECOVERED":"MODEL")+"</span>";
        card.appendChild(top);

        const stats=document.createElement("div");stats.className="mlb-output-stats compact model-three";
        [["Layers",entry.nodes??"—"],["Context",entry.context_length??"—"],["Batch",entry.batch_size??"—"]].forEach(([label,value])=>{
          const item=document.createElement("div");item.innerHTML="<span>"+label+"</span><strong>"+value+"</strong>";stats.appendChild(item);
        });
        card.appendChild(stats);

        const foot=document.createElement("div");foot.className="mlb-output-compact-foot";
        const ds=preparedDatasetById(entry.selected_dataset_id);
        foot.innerHTML="<span>"+(ds?ds.name:"No training data")+"</span><span>"+(entry.weights_ready?"Train / Generate / Serve →":"Train →")+"</span>";
        card.appendChild(foot);

        card.addEventListener("click",()=>{
          runtimePanel=null;
          outputDirectorySelection=entry.id;
          selected=null;inspectorTab="settings";
          setStatus(entry.name+" model details opened.");
          draw();
        });
        list.appendChild(card);
      });
      container.appendChild(list);
    }

    function renderOutputDirectory(container){
      container.className="mlb-output-directory";
      if(state.active_workspace==="data")renderDataOutputDirectory(container);
      else renderModelOutputDirectory(container);
    }

    function safeProjectStem(){
      return safeFilename(state.project?.name||"mlbricks-project");
    }

    function projectFileEntries(){
      const stem=safeProjectStem();
      const files=[
        {
          id:"design_json",
          name:stem+".mlbricks.json",
          category:"design",
          type:"Builder Design",
          location:"Browser download",
          description:"Complete MLBricks project: model graph, data graph, registries and settings."
        },
        {
          id:"design_bin",
          name:stem+".mlbricks.bin",
          category:"design",
          type:"Binary Design",
          location:"Browser download",
          description:"Binary Builder project file for Save BIN / Load."
        },
        {
          id:"model_config",
          name:stem+".model-config.json",
          category:"config",
          type:"Model Config",
          location:"Generated from Model Builder",
          description:"Current model architecture/configuration represented by the Model Builder graph."
        }
      ];

      availablePreparedDatasets().forEach(meta=>{
        files.push({
          id:"data_"+meta.id,
          name:meta.name,
          category:"data",
          type:"Prepared Dataset",
          location:meta.hub_repo_id ? ("HF: "+meta.hub_repo_id) : (meta.path || "Python memory"),
          path:meta.path || null,
          description:compactDatasetSummary(meta)+" · "+dataStorageLabel(meta),
          dataset_id:meta.id
        });
      });

      (state.model_outputs||[]).forEach((item,index)=>{
        files.push({
          id:"model_"+(item.id||index),
          name:item.name||("Model Artifact "+(index+1)),
          category:"model",
          type:item.format||item.kind||"Model Artifact",
          location:item.hub_repo_id ? ("HF: "+item.hub_repo_id) : (item.path||"Registered artifact"),
          path:item.path||null,
          description:item.dataset?("Dataset: "+item.dataset):"Trained/exported model artifact"
        });
      });

      (state.project_files||[]).forEach(item=>{
        if(!files.some(x=>x.id===item.id))files.push(cp(item));
      });
      return files;
    }

    function fileCategoryLabel(category){
      return category==="data"?"DATA":category==="model"?"MODEL":category==="config"?"CONFIG":"DESIGN";
    }

    function downloadModelConfig(){
      const model=modelRootComponent();
      if(!model)return;
      const config={
        format:"mlbricks-model-config",
        builder_version:"1.0.0",
        project:cp(state.project||{}),
        model:cp(model),
        selected_dataset:selectedModelDataset(),
      };
      const blob=new Blob([JSON.stringify(config,null,2)],{type:"application/json"});
      const url=URL.createObjectURL(blob);
      const a=document.createElement("a");
      a.href=url;a.download=safeProjectStem()+".model-config.json";
      document.body.appendChild(a);a.click();a.remove();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
      setStatus("Model config downloaded.");
    }

    function renderGalleryView(container){
      container.className="mlb-gallery-view";
      const head=document.createElement("div");head.className="mlb-gallery-head";
      const title=document.createElement("div");title.innerHTML="<strong>GALLERY</strong><span>Sample models and data, plus reusable designs saved by you.</span>";
      const saveLabel=state.active_workspace==="data"?"+ Save Current Data":"+ Save Current Model";
      head.appendChild(title);
      if(current(state)?.kind!=="custom_edit"){const save=btn(saveLabel,"mlb-gallery-save");save.addEventListener("click",saveCurrentToGallery);head.appendChild(save);}
      container.appendChild(head);

      const grid=document.createElement("div");grid.className="mlb-gallery-grid";
      const makeSection=(heading,countText,extraClass="")=>{
        const section=document.createElement("section");section.className="mlb-gallery-section"+(extraClass?" "+extraClass:"");
        const st=document.createElement("div");st.className="mlb-gallery-section-title";st.innerHTML="<strong>"+heading+"</strong><span>"+countText+"</span>";section.appendChild(st);return section;
      };
      const makeSampleCard=(name,meta,actionLabel,onLoad)=>{
        const card=document.createElement("div");card.className="mlb-gallery-card mlb-gallery-sample-card";
        const info=document.createElement("div");info.innerHTML="<strong>"+name+"</strong><span>"+meta+"</span>";
        const acts=document.createElement("div");const load=btn(actionLabel,"mlb-gallery-action sample");load.addEventListener("click",onLoad);acts.append(load);card.append(info,acts);return card;
      };

      // Built-in examples live only in Gallery so the canvas toolbar stays clean.
      // Add future examples to these registries rather than adding toolbar buttons.
      const builtInSampleModels=[
        {name:"TinyStories 30M",meta:"6 layers · Context 512 · Batch 16 · ~30M parameters",action:"Load Model",load:loadTinyStories},
        {name:"SOUP 30M 1L",meta:"1 SOUP layer · Context 512 · Batch 16 · 30,003,528 parameters",action:"Load Model",load:loadSOUP30M1L},
        {name:"StateAware ESA 200M",meta:"8 layers · Context 256 · Batch 16 · 199,982,344 parameters",action:"Load Model",load:loadStateAwareESA200M},
        {name:"SOUP 200M",meta:"3 SOUP layers · Context 256 · Batch 16 · 199,916,160 parameters",action:"Load Model",load:loadSOUP200M}
      ];
      const builtInSampleData=[
        {name:"TinyStories Text Pipeline",meta:"Hugging Face → Text Processing → Train 90% · Validation 5% · Test 5% → GPT-2 Tokenize → Prepared Dataset",action:"Load Pipeline",load:loadTextDataStarter}
      ];
      const sampleModels=makeSection("SAMPLE MODELS",builtInSampleModels.length+" built-in","sample");
      builtInSampleModels.forEach(item=>sampleModels.appendChild(makeSampleCard(item.name,item.meta,item.action,item.load)));
      const sampleData=makeSection("SAMPLE DATA",builtInSampleData.length+" built-in","sample");
      builtInSampleData.forEach(item=>sampleData.appendChild(makeSampleCard(item.name,item.meta,item.action,item.load)));

      const componentSection=makeSection("MODULES & API COMPONENTS",(state.gallery.components||[]).length+" saved");
      if(!(state.gallery.components||[]).length){const e=document.createElement("div");e.className="mlb-gallery-empty";e.textContent="Create a Module or API Component and save it here for reuse.";componentSection.appendChild(e);}
      (state.gallery.components||[]).forEach(entry=>{
        const card=document.createElement("div");card.className="mlb-gallery-card";
        const apiDef=String(entry.definition?.implementation||"graph")==="api";const steps=apiStepNodes(entry.definition);
        const allBlocks=(entry.definition?.nodes||[]).length;const detail=apiDef?((allBlocks||steps.length||1)+" blocks · "+steps.length+" API functions · "+((entry.definition?.edges||[]).length)+" connections"):((allBlocks)+" internal components · Module");
        const meta=document.createElement("div");meta.innerHTML="<strong>"+entry.name+"</strong><span>"+detail+" · "+(entry.saved_at?new Date(entry.saved_at).toLocaleDateString():"Saved")+"</span>";
        const acts=document.createElement("div");
        const installed=Object.values(state.custom_components||{}).some(def=>def.gallery_entry_id===entry.id&&def.palette_installed===true&&def.palette_hidden!==true);
        const add=btn(installed?"Added":"Add to Library","mlb-gallery-action");add.disabled=installed;if(!installed)add.addEventListener("click",()=>addGalleryComponent(entry));
        const edit=btn("Edit","mlb-gallery-action");edit.addEventListener("click",()=>editGalleryComponent(entry));
        const remove=btn("Remove","mlb-gallery-action danger");remove.addEventListener("click",()=>removeGalleryEntry("components",entry.id));acts.append(add,edit,remove);card.append(meta,acts);componentSection.appendChild(card);
      });

      const modelSection=makeSection("MY MODELS",(state.gallery.models||[]).length+" saved");
      if(!(state.gallery.models||[]).length){const e=document.createElement("div");e.className="mlb-gallery-empty";e.textContent="Save your current Model Builder layout to keep it for later.";modelSection.appendChild(e);}
      (state.gallery.models||[]).forEach(entry=>{
        const card=document.createElement("div");card.className="mlb-gallery-card";
        const meta=document.createElement("div");meta.innerHTML="<strong>"+entry.name+"</strong><span>"+((entry.architecture?.nodes||[]).length)+" components · "+(entry.saved_at?new Date(entry.saved_at).toLocaleDateString():"Saved")+"</span>";
        const acts=document.createElement("div");const load=btn("Load to Canvas","mlb-gallery-action");load.addEventListener("click",()=>loadGalleryModel(entry));const remove=btn("Remove","mlb-gallery-action danger");remove.addEventListener("click",()=>removeGalleryEntry("models",entry.id));acts.append(load,remove);card.append(meta,acts);modelSection.appendChild(card);
      });

      const dataSection=makeSection("MY DATA PIPELINES",(state.gallery.data||[]).length+" saved");
      if(!(state.gallery.data||[]).length){const e=document.createElement("div");e.className="mlb-gallery-empty";e.textContent="Save a Data Processing pipeline here for reuse.";dataSection.appendChild(e);}
      (state.gallery.data||[]).forEach(entry=>{
        const card=document.createElement("div");card.className="mlb-gallery-card";
        const meta=document.createElement("div");meta.innerHTML="<strong>"+entry.name+"</strong><span>"+((entry.architecture?.nodes||[]).length)+" steps · "+(entry.saved_at?new Date(entry.saved_at).toLocaleDateString():"Saved")+"</span>";
        const acts=document.createElement("div");const load=btn("Load Pipeline","mlb-gallery-action");load.addEventListener("click",()=>loadGalleryData(entry));const remove=btn("Remove","mlb-gallery-action danger");remove.addEventListener("click",()=>removeGalleryEntry("data",entry.id));acts.append(load,remove);card.append(meta,acts);dataSection.appendChild(card);
      });

      grid.append(sampleModels,sampleData,componentSection,modelSection,dataSection);container.appendChild(grid);
      const note=document.createElement("div");note.className="mlb-gallery-note";note.textContent="Built-in samples stay available in Gallery. Your saved items are stored in the Builder project and mirrored to browser storage when available.";container.appendChild(note);
    }


    function renderCentralGallery(canvas){
      canvas.classList.add("gallery-active");
      const outer=document.createElement("div");outer.className="mlb-central-gallery";

      const head=document.createElement("div");head.className="mlb-gallery-page-head";
      const copy=document.createElement("div");copy.className="mlb-gallery-page-copy";
      copy.innerHTML="<strong>GALLERY</strong><span>Prebuilt MLBricks models, reusable Modules/API Components, data pipelines, and your saved designs.</span>";
      const close=btn("×","mlb-gallery-page-close");close.title="Close Gallery";close.addEventListener("click",closeGallery);
      head.append(copy,close);outer.appendChild(head);

      const tabsRow=document.createElement("div");tabsRow.className="mlb-gallery-tabs-row";
      const tabs=document.createElement("div");tabs.className="mlb-central-gallery-tabs";
      [["models","Models"],["components","Components"],["data","Data"]].forEach(([key,label])=>{
        const b=btn(label,"mlb-central-gallery-tab"+(galleryWorkspace.tab===key?" active":""));
        b.addEventListener("click",()=>{galleryWorkspace.tab=key;draw();});tabs.appendChild(b);
      });
      tabsRow.appendChild(tabs);

      const galleryActions=document.createElement("div");galleryActions.className="mlb-gallery-page-actions";
      const galleryLoad=btn("⇧ Load","mlb-gallery-action mlb-gallery-file-action");
      galleryLoad.title="Load .mlbricks.json or .mlbricks.bin";galleryLoad.addEventListener("click",loadDesign);galleryActions.appendChild(galleryLoad);
      const galleryExport=btn("⇩ Export","mlb-gallery-action mlb-gallery-file-action");
      galleryExport.title="Export model config or workspace data";galleryExport.addEventListener("click",exportWorkspace);galleryActions.appendChild(galleryExport);

      let canSave=false,saveLabel="";
      if(galleryWorkspace.tab==="models"&&state.active_workspace==="model"&&current(state)?.kind!=="custom_edit"){canSave=true;saveLabel="+ Save Current Model";}
      if(galleryWorkspace.tab==="data"&&state.active_workspace==="data"){canSave=true;saveLabel="+ Save Current Data";}
      if(canSave){const save=btn(saveLabel,"mlb-gallery-save mlb-gallery-page-save");save.addEventListener("click",saveCurrentToGallery);galleryActions.appendChild(save);}
      tabsRow.appendChild(galleryActions);
      outer.appendChild(tabsRow);

      // Only this content region scrolls. The banner and tabs never shrink.
      const body=document.createElement("div");body.className="mlb-central-gallery-body";
      const makeSection=(heading,countText,kind="")=>{
        const s=document.createElement("section");s.className="mlb-central-gallery-section"+(kind?" "+kind:"");
        const sh=document.createElement("div");sh.className="mlb-central-gallery-section-head";
        sh.innerHTML="<strong>"+heading+"</strong><span>"+countText+"</span>";s.appendChild(sh);return s;
      };
      const empty=(text)=>{const e=document.createElement("div");e.className="mlb-central-gallery-empty";e.textContent=text;return e;};
      const card=(name,meta,tag,actions)=>{
        const c=document.createElement("div");c.className="mlb-central-gallery-card";
        const icon=document.createElement("div");icon.className="mlb-central-gallery-icon";icon.textContent=tag;
        const info=document.createElement("div");info.className="mlb-central-gallery-card-info";
        const title=document.createElement("strong");title.textContent=name;
        const detail=document.createElement("span");detail.textContent=meta||"";
        info.append(title,detail);
        const acts=document.createElement("div");acts.className="mlb-central-gallery-card-actions";(actions||[]).forEach(a=>acts.appendChild(a));
        c.append(icon,info,acts);return c;
      };
      const modelMeta=(entry)=>{
        const p=entry?.project||{};
        const parts=[];
        const params=p.estimated_parameters||entry?.estimated_parameters;
        const batch=p.batch_size||entry?.batch_size;
        const block=p.context_length||entry?.context_length;
        if(params)parts.push("Parameters "+params);
        if(batch)parts.push("Batch "+batch);
        if(block)parts.push("Block "+block);
        const count=(entry?.architecture?.nodes||[]).length;
        if(count)parts.push(count+" components");
        if(!parts.length)parts.push(entry?.saved_at?"Saved "+new Date(entry.saved_at).toLocaleDateString():"Saved model");
        return parts.join(" · ");
      };
      const openAndClose=(fn)=>()=>{galleryWorkspace.open=false;bottomExpanded=galleryPreviousBottomExpanded;fn();};

      if(galleryWorkspace.tab==="models"){
        body.classList.add("models-tab");
        const samples=makeSection("PREBUILT MODELS","4 available","featured full-width");
        const sampleGrid=document.createElement("div");sampleGrid.className="mlb-central-gallery-card-grid prebuilt-grid";
        const loadTiny=btn("Open Model","mlb-gallery-action sample");loadTiny.addEventListener("click",openAndClose(loadTinyStories));
        sampleGrid.appendChild(card("TinyStories 30M","Parameters ~30M · Batch 16 · Block 512 · 6 layers","MODEL",[loadTiny]));
        const loadSoup30=btn("Open Model","mlb-gallery-action sample");loadSoup30.addEventListener("click",openAndClose(loadSOUP30M1L));
        sampleGrid.appendChild(card("SOUP 30M 1L","Parameters 30,003,528 · Batch 16 · Block 512 · 1 SOUP layer","MODEL",[loadSoup30]));
        const loadEsa200=btn("Open Model","mlb-gallery-action sample");loadEsa200.addEventListener("click",openAndClose(loadStateAwareESA200M));
        sampleGrid.appendChild(card("StateAware ESA 200M","Parameters 199,982,344 · Batch 16 · Block 256 · 8 layers","MODEL",[loadEsa200]));
        const loadSoup200=btn("Open Model","mlb-gallery-action sample");loadSoup200.addEventListener("click",openAndClose(loadSOUP200M));
        sampleGrid.appendChild(card("SOUP 200M","Parameters 199,916,160 · Batch 16 · Block 256 · 3 SOUP layers","MODEL",[loadSoup200]));
        samples.appendChild(sampleGrid);
        body.appendChild(samples);

        const mine=makeSection("MY MODELS",(state.gallery.models||[]).length+" saved","full-width saved-models");
        if(!(state.gallery.models||[]).length){mine.appendChild(empty("Models you save to Gallery will appear here."));}
        else{
          const savedGrid=document.createElement("div");savedGrid.className="mlb-central-gallery-card-grid saved-model-grid";
          (state.gallery.models||[]).forEach(entry=>{
            const load=btn("Open","mlb-gallery-action");load.addEventListener("click",openAndClose(()=>loadGalleryModel(entry)));
            const remove=btn("Remove","mlb-gallery-action danger");remove.addEventListener("click",()=>removeGalleryEntry("models",entry.id));
            savedGrid.appendChild(card(entry.name,modelMeta(entry),"MODEL",[load,remove]));
          });
          mine.appendChild(savedGrid);
        }
        body.appendChild(mine);
      }else if(galleryWorkspace.tab==="components"){
        body.classList.add("components-tab");
        const samples=makeSection("CREATE","Choose API Component or Module","featured full-width compact-create");
        const createActions=document.createElement("div");createActions.className="mlb-gallery-component-create-actions";
        const createApi=btn("API Component","mlb-gallery-action sample mlb-gallery-create-small");createApi.title="Bind Mamba, Flash Attention, torch.nn modules, or another Python/PyTorch API";createApi.addEventListener("click",createAPICustom);
        const createVisual=btn("Module","mlb-gallery-action mlb-gallery-create-small");createVisual.title="Compose a reusable Module from Components, API Components, and nested Modules";createVisual.addEventListener("click",createCustom);
        createActions.append(createApi,createVisual);
        samples.appendChild(createActions);body.appendChild(samples);

        const mine=makeSection("MODULES & API COMPONENTS",(state.gallery.components||[]).length+" saved","full-width saved-components");
        if(!(state.gallery.components||[]).length)mine.appendChild(empty("Modules and API Components you save will appear here."));
        else{
          const savedGrid=document.createElement("div");savedGrid.className="mlb-central-gallery-card-grid saved-component-grid";
          (state.gallery.components||[]).forEach(entry=>{
            const installed=Object.values(state.custom_components||{}).some(def=>def.gallery_entry_id===entry.id&&def.palette_installed===true&&def.palette_hidden!==true);
            const add=btn(installed?"Added":"Add to Library","mlb-gallery-action");add.disabled=installed;if(!installed)add.addEventListener("click",()=>addGalleryComponent(entry));
            const edit=btn("Edit","mlb-gallery-action");edit.addEventListener("click",()=>editGalleryComponent(entry));
            const remove=btn("Remove","mlb-gallery-action danger");remove.addEventListener("click",()=>removeGalleryEntry("components",entry.id));
            const apiDef=String(entry.definition?.implementation||"graph")==="api";
            const steps=apiStepNodes(entry.definition);
            const allBlocks=(entry.definition?.nodes||[]).length;
            const detail=apiDef
              ?((allBlocks||steps.length||1)+" blocks · "+steps.length+" API functions · "+((entry.definition?.edges||[]).length)+" connections")
              :((allBlocks)+" components");
            savedGrid.appendChild(card(entry.name,detail+" · "+(entry.saved_at?new Date(entry.saved_at).toLocaleDateString():"Saved"),apiDef?"API":"MOD",[add,edit,remove]));
          });
          mine.appendChild(savedGrid);
        }
        body.appendChild(mine);
      }else{
        body.classList.add("data-tab");
        const samples=makeSection("PREBUILT DATA","1 available","featured full-width");
        const load=btn("Open Pipeline","mlb-gallery-action sample");load.addEventListener("click",openAndClose(loadTextDataStarter));
        const sampleGrid=document.createElement("div");sampleGrid.className="mlb-central-gallery-card-grid prebuilt-grid";
        sampleGrid.appendChild(card("TinyStories Text Pipeline","Hugging Face → Text Processing → Train 90% · Validation 5% · Test 5% → GPT-2 Tokenize → Prepared Dataset","DATA",[load]));
        samples.appendChild(sampleGrid);
        body.appendChild(samples);

        const mine=makeSection("MY DATA",(state.gallery.data||[]).length+" saved","full-width saved-data");
        if(!(state.gallery.data||[]).length)mine.appendChild(empty("Data pipelines you save to Gallery will appear here."));
        else{
          const savedGrid=document.createElement("div");savedGrid.className="mlb-central-gallery-card-grid saved-data-grid";
          (state.gallery.data||[]).forEach(entry=>{
            const load=btn("Open","mlb-gallery-action");load.addEventListener("click",openAndClose(()=>loadGalleryData(entry)));
            const remove=btn("Remove","mlb-gallery-action danger");remove.addEventListener("click",()=>removeGalleryEntry("data",entry.id));
            savedGrid.appendChild(card(entry.name,((entry.architecture?.nodes||[]).length)+" steps · "+(entry.saved_at?new Date(entry.saved_at).toLocaleDateString():"Saved"),"DATA",[load,remove]));
          });
          mine.appendChild(savedGrid);
        }
        body.appendChild(mine);
      }

      outer.appendChild(body);canvas.appendChild(outer);
    }

    function renderCentralCloud(canvas){
      canvas.classList.add("gallery-active","cloud-active");
      const outer=document.createElement("div");outer.className="mlb-central-cloud";
      const head=document.createElement("div");head.className="mlb-cloud-page-head";
      const copy=document.createElement("div");copy.className="mlb-cloud-page-copy";
      copy.innerHTML="<strong>CLOUD & REPOSITORIES</strong><span>Connect providers, push or load models, datasets, reusable Modules/API Components, and projects.</span>";
      const close=btn("×","mlb-gallery-page-close");close.title="Close Cloud & Repositories";close.addEventListener("click",closeCloudWorkspace);
      head.append(copy,close);outer.appendChild(head);
      const body=document.createElement("div");body.className="mlb-central-cloud-body";
      renderCloudView(body,false);
      outer.appendChild(body);canvas.appendChild(outer);
    }

    function renderLocalView(container){
      container.className="mlb-local-view";
      const importingData=state.active_workspace==="data";
      const kind=importingData?"data":"model";
      const report=localImportReports[kind];
      const pathKey=importingData?"data_path":"model_path";
      const action=importingData?"local_import_data":"local_import_models";

      const head=document.createElement("div");head.className="mlb-local-head";
      const copyHead=document.createElement("div");
      copyHead.innerHTML=importingData
        ?"<strong>LOCAL ENVIRONMENT DATA IMPORT</strong><span>Detected "+localEnvironment.name+". Base Path starts at the current workspace root. Builder can also scan other available environment roots.</span>"
        :"<strong>LOCAL ENVIRONMENT MODEL IMPORT</strong><span>Detected "+localEnvironment.name+". Base Path starts at the current workspace root. Builder can also scan other available environment roots.</span>";
      const badge=document.createElement("span");badge.className="mlb-local-badge";badge.textContent=String(localEnvironment.name||"AUTO").toUpperCase();
      head.append(copyHead,badge);container.appendChild(head);

      const box=document.createElement("div");box.className="mlb-local-auto-box";
      const field=document.createElement("div");field.className="mlb-local-field";
      const label=document.createElement("label");label.textContent="Base Path";field.appendChild(label);
      const input=document.createElement("input");
      input.value=localForm[pathKey]||localDefaultRoot;
      input.placeholder=localDefaultRoot;
      input.addEventListener("input",()=>localForm[pathKey]=input.value);
      field.appendChild(input);

      const button=btn(importingData?"Scan Current Data":"Scan Current Models","mlb-local-load");
      button.addEventListener("click",()=>{
        const path=String(localForm[pathKey]||localDefaultRoot).trim()||localDefaultRoot;
        requestLocalCommand(action,{path,max_depth:12,max_entries:1000});
      });
      const pathButton=btn("Scan This Path","mlb-local-load secondary");
      pathButton.addEventListener("click",()=>{
        const path=String(localForm[pathKey]||"").trim();
        if(!path){setStatus("Enter a local environment directory path.");return;}
        requestLocalCommand(action,{path,max_depth:12,max_entries:1000});
      });
      const buttons=document.createElement("div");buttons.className="mlb-local-scan-actions";buttons.append(button,pathButton);
      box.append(field,buttons);container.appendChild(box);

      const examples=document.createElement("div");examples.className="mlb-local-path-examples";
      const quickPaths=[localDefaultRoot,...(localEnvironment.roots||[])].filter((value,index,array)=>value&&array.indexOf(value)===index);
      quickPaths.forEach(value=>{
        const chip=document.createElement("button");chip.textContent=value;
        chip.addEventListener("click",()=>{localForm[pathKey]=value;draw();});
        examples.appendChild(chip);
      });
      container.appendChild(examples);

      const flow=document.createElement("div");flow.className="mlb-local-flow";
      flow.innerHTML=importingData
        ?"<span>1</span><strong>Path</strong><i>→</i><span>2</span><strong>Recursive Scan</strong><i>→</i><span>3</span><strong>Detect Data</strong><i>→</i><span>4</span><strong>Data Repository</strong>"
        :"<span>1</span><strong>Path</strong><i>→</i><span>2</span><strong>Recursive Scan</strong><i>→</i><span>3</span><strong>Detect Models</strong><i>→</i><span>4</span><strong>Model Repository</strong>";
      container.appendChild(flow);

      if(report){
        const panel=document.createElement("div");panel.className="mlb-local-report";
        const rh=document.createElement("div");rh.className="mlb-local-report-head";
        rh.innerHTML="<strong>LAST IMPORT</strong><span>"+(report.root||"")+"</span>";
        panel.appendChild(rh);

        const stats=document.createElement("div");stats.className="mlb-local-report-stats";
        [["Found",report.found||0],["Imported",report.imported_count||0],["Skipped",report.skipped_count||0],["Errors",report.error_count||0]].forEach(([name,value])=>{
          const item=document.createElement("div");item.innerHTML="<span>"+name+"</span><strong>"+value+"</strong>";stats.appendChild(item);
        });
        panel.appendChild(stats);

        const imported=report.imported||[];
        if(imported.length){
          const list=document.createElement("div");list.className="mlb-local-imported-list";
          imported.forEach(item=>{
            const row=document.createElement("div");
            const title=item.name||(importingData?"Imported Dataset":"Imported Model");
            const path=item.local_path||item.checkpoint_path||item.path||"";
            const badgeText=(!importingData&&item.legacy_recovered)?"RECOVERED":"IMPORTED";
            row.innerHTML="<div><strong>"+title+"</strong><span>"+path+"</span></div><b>"+badgeText+"</b>";
            list.appendChild(row);
          });
          panel.appendChild(list);
        }

        if((report.errors||[]).length){
          const details=document.createElement("details");details.className="mlb-local-errors";
          const summary=document.createElement("summary");
          summary.textContent=report.errors.length+" incompatible "+(importingData?"data item":"checkpoint")+(report.errors.length===1?"":"s");
          details.appendChild(summary);
          report.errors.forEach(item=>{
            const row=document.createElement("div");
            row.innerHTML="<strong>"+item.path+"</strong><span>"+item.error+"</span>";
            details.appendChild(row);
          });
          panel.appendChild(details);
        }
        container.appendChild(panel);
      }

      const note=document.createElement("div");note.className="mlb-local-note";
      note.innerHTML=importingData
        ?"<strong>Environment-aware.</strong> Builder recursively detects Hugging Face <code>save_to_disk()</code> folders, <code>.txt</code>, <code>.csv</code>, <code>.json</code>, <code>.jsonl</code>, <code>.parquet</code>, <code>.arrow</code> and MLBricks dataset bundles. Duplicate paths are ignored. Imported datasets are added automatically to <strong>Data Repository</strong>."
        :"<strong>Environment-aware.</strong> Builder recursively detects <code>last.pt</code>, <code>.pt</code>, <code>.pth</code>, <code>.ckpt</code> and MLBricks model bundles. Duplicate paths are ignored. Imported models are added automatically to <strong>Model Repository</strong>.";
      container.appendChild(note);
    }


    function cloudArtifactOptions(type){
      if(type==="dataset"){
        return availablePreparedDatasets().map(x=>({id:x.id,name:x.name,detail:compactDatasetSummary(x)}));
      }
      if(type==="model"){
        return modelDirectoryEntries().map(x=>({
          id:x.id,name:x.name,detail:x.weights_ready?"Trained weights ready":"Architecture / build"
        }));
      }
      return [{id:"project",name:state.project?.name||"Current Project",detail:"Complete Builder project"}];
    }

    function cloudField(label,type,value,placeholder,onChange,secret=false){
      const field=document.createElement("div");field.className="mlb-cloud-field";
      const l=document.createElement("label");l.textContent=label;field.appendChild(l);
      let input;
      if(type==="textarea"){
        input=document.createElement("textarea");
        input.rows=3;
      }else{
        input=document.createElement("input");
        input.type=secret?"password":(type||"text");
      }
      input.value=value||"";
      input.placeholder=placeholder||"";
      input.autocomplete="off";
      input.addEventListener("input",()=>onChange(input.value));
      field.appendChild(input);
      return field;
    }

    function cloudSelect(label,value,options,onChange){
      const field=document.createElement("div");field.className="mlb-cloud-field";
      const l=document.createElement("label");l.textContent=label;field.appendChild(l);
      const select=document.createElement("select");
      options.forEach(item=>{
        const v=typeof item==="object"?item.value:item;
        const text=typeof item==="object"?item.label:item;
        const o=document.createElement("option");o.value=v;o.textContent=text;
        if(String(v)===String(value))o.selected=true;
        select.appendChild(o);
      });
      select.addEventListener("change",()=>onChange(select.value));
      field.appendChild(select);return field;
    }

    function providerLabel(provider){
      return {
        huggingface:"Hugging Face",
        github:"GitHub",
        aws:"AWS S3",
        gcp:"Google Cloud Storage",
        azure:"Azure Blob Storage"
      }[provider]||provider;
    }

    function renderProviderCredentials(card){
      const p=cloudForm.provider;
      const title=document.createElement("div");title.className="mlb-cloud-subtitle";
      title.textContent="🔑  SESSION CREDENTIALS";card.appendChild(title);

      if(p==="huggingface"){
        card.appendChild(cloudField(
          "API Token / Access Token","text",cloudSecrets.huggingface.token,
          "hf_...  (optional if already logged in)",
          v=>cloudSecrets.huggingface.token=v,true
        ));
      }else if(p==="github"){
        card.appendChild(cloudField(
          "GitHub Personal Access Token","text",cloudSecrets.github.token,
          "github_pat_... / ghp_...",
          v=>cloudSecrets.github.token=v,true
        ));
      }else if(p==="aws"){
        const grid=document.createElement("div");grid.className="mlb-cloud-mini-grid";
        grid.append(
          cloudField("Access Key ID","text",cloudSecrets.aws.access_key,"AKIA...",v=>cloudSecrets.aws.access_key=v,true),
          cloudField("Secret Access Key","text",cloudSecrets.aws.secret_key,"••••••",v=>cloudSecrets.aws.secret_key=v,true)
        );
        card.appendChild(grid);
        card.appendChild(cloudField(
          "Session Token (optional)","text",cloudSecrets.aws.session_token,"Temporary session token",
          v=>cloudSecrets.aws.session_token=v,true
        ));
      }else if(p==="gcp"){
        card.appendChild(cloudField(
          "Service Account JSON","textarea",cloudSecrets.gcp.service_account_json,
          '{"type":"service_account", ...}  (blank = Application Default Credentials)',
          v=>cloudSecrets.gcp.service_account_json=v,true
        ));
      }else if(p==="azure"){
        card.appendChild(cloudField(
          "Connection String","textarea",cloudSecrets.azure.connection_string,
          "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...",
          v=>cloudSecrets.azure.connection_string=v,true
        ));
      }

      const note=document.createElement("div");note.className="mlb-cloud-secret-note";
      note.textContent="Session only · masked · never included in Save JSON, BIN, model, dataset, or project metadata.";
      card.appendChild(note);
    }

    function currentCloudCredentials(){
      const p=cloudForm.provider;
      if(p==="huggingface")return {token:cloudSecrets.huggingface.token};
      if(p==="github")return {token:cloudSecrets.github.token};
      if(p==="aws")return {
        access_key:cloudSecrets.aws.access_key,
        secret_key:cloudSecrets.aws.secret_key,
        session_token:cloudSecrets.aws.session_token,
        region:cloudForm.region
      };
      if(p==="gcp")return {service_account_json:cloudSecrets.gcp.service_account_json};
      if(p==="azure")return {connection_string:cloudSecrets.azure.connection_string};
      return {};
    }

    function providerTargetFields(card){
      const p=cloudForm.provider;
      if(p==="huggingface"){
        card.appendChild(cloudField("Repository ID","text",cloudForm.repo,"username-or-org/repo-name",v=>cloudForm.repo=v));
        card.appendChild(cloudField("Revision","text",cloudForm.revision,"main",v=>cloudForm.revision=v));
      }else if(p==="github"){
        card.appendChild(cloudField("Repository","text",cloudForm.repo,"owner/repository",v=>cloudForm.repo=v));
        const grid=document.createElement("div");grid.className="mlb-cloud-mini-grid";
        grid.append(
          cloudField("Branch","text",cloudForm.branch,"main",v=>cloudForm.branch=v),
          cloudField("File Path","text",cloudForm.object_path,"mlbricks/project.mlbricks.zip",v=>cloudForm.object_path=v)
        );
        card.appendChild(grid);
      }else if(p==="aws"||p==="gcp"){
        const grid=document.createElement("div");grid.className="mlb-cloud-mini-grid";
        grid.append(
          cloudField("Bucket","text",cloudForm.bucket,"my-mlbricks-bucket",v=>cloudForm.bucket=v),
          cloudField(p==="aws"?"Object Key":"Object Name","text",cloudForm.object_path,"models/model.mlbricks.zip",v=>cloudForm.object_path=v)
        );
        card.appendChild(grid);
        if(p==="aws"){
          card.appendChild(cloudField("Region","text",cloudForm.region,"us-east-1",v=>cloudForm.region=v));
        }
      }else if(p==="azure"){
        const grid=document.createElement("div");grid.className="mlb-cloud-mini-grid";
        grid.append(
          cloudField("Container","text",cloudForm.container,"mlbricks",v=>cloudForm.container=v),
          cloudField("Blob Name","text",cloudForm.object_path,"models/model.mlbricks.zip",v=>cloudForm.object_path=v)
        );
        card.appendChild(grid);
      }
    }

    function cloudCommandConfig(contentType,artifactId){
      return {
        provider:cloudForm.provider,
        content_type:contentType,
        artifact_id:artifactId,
        repo:cloudForm.repo,
        branch:cloudForm.branch||"main",
        revision:cloudForm.revision||"main",
        bucket:cloudForm.bucket,
        container:cloudForm.container,
        object_path:cloudForm.object_path,
        region:cloudForm.region,
        private:!!cloudForm.private,
        credentials:currentCloudCredentials()
      };
    }

    function renderCloudView(container,showHead=true){
      container.classList.add("mlb-cloud-view");
      if(showHead){
        const head=document.createElement("div");head.className="mlb-cloud-head";
        head.innerHTML="<div><strong>CLOUD & REPOSITORIES</strong><span>Push and load Builder data, models and projects</span></div><span class='mlb-cloud-badge'>CLOUD</span>";
        container.appendChild(head);
      }

      const providerCard=document.createElement("section");providerCard.className="mlb-cloud-card mlb-cloud-provider-card";
      const providerTitle=document.createElement("div");providerTitle.className="mlb-cloud-section-title";providerTitle.innerHTML="<span>☁</span><strong>PROVIDER & CONNECTION</strong>";providerCard.appendChild(providerTitle);
      const providerBar=document.createElement("div");providerBar.className="mlb-cloud-provider-bar";
      const providerField=cloudSelect("Provider",cloudForm.provider,[
        {value:"huggingface",label:"Hugging Face"},{value:"github",label:"GitHub"},{value:"aws",label:"AWS S3"},{value:"gcp",label:"Google Cloud Storage"},{value:"azure",label:"Azure Blob Storage"}
      ],v=>{cloudForm.provider=v;cloudStatus[v]=cloudStatus[v]||{};draw();});
      const status=cloudStatus[cloudForm.provider]||{};
      const connectionField=document.createElement("div");connectionField.className="mlb-cloud-field mlb-cloud-connection-field";
      const connectionLabel=document.createElement("label");connectionLabel.textContent="Connection";
      const indicator=document.createElement("div");indicator.className="mlb-cloud-status "+(status.ok||status.authenticated?"ok":status.message?"warn":"idle");
      const connectionText=document.createElement("span");connectionText.textContent=status.message||providerLabel(cloudForm.provider)+" · not checked";
      indicator.appendChild(connectionText);connectionField.append(connectionLabel,indicator);
      const check=btn("Check Connection","mlb-cloud-check");check.addEventListener("click",()=>requestCloudCommand("cloud_status",{provider:cloudForm.provider,credentials:currentCloudCredentials(),region:cloudForm.region}));
      providerBar.append(providerField,connectionField,check);providerCard.appendChild(providerBar);container.appendChild(providerCard);

      const credentials=document.createElement("section");credentials.className="mlb-cloud-card credentials";
      renderProviderCredentials(credentials);container.appendChild(credentials);

      const grid=document.createElement("div");grid.className="mlb-cloud-grid";
      const push=document.createElement("section");push.className="mlb-cloud-card";
      const pt=document.createElement("div");pt.className="mlb-cloud-card-title";pt.innerHTML="<strong>↑ PUSH</strong><span>Send local Builder content to "+providerLabel(cloudForm.provider)+"</span>";push.appendChild(pt);
      push.appendChild(cloudSelect("Content Type",cloudForm.push_type,[{value:"dataset",label:"Prepared Dataset"},{value:"model",label:"Built / Trained Model"},{value:"project",label:"Builder Project"}],v=>{cloudForm.push_type=v;cloudForm.push_artifact="";draw();}));
      const artifacts=cloudArtifactOptions(cloudForm.push_type);if(!cloudForm.push_artifact&&artifacts.length)cloudForm.push_artifact=artifacts[0].id;
      push.appendChild(cloudSelect("Local Content",cloudForm.push_artifact,artifacts.length?artifacts.map(x=>({value:x.id,label:x.name+" — "+x.detail})):[{value:"",label:"Nothing available yet"}],v=>cloudForm.push_artifact=v));
      providerTargetFields(push);
      if(cloudForm.provider==="huggingface"){
        const privacy=document.createElement("label");privacy.className="mlb-cloud-private";const box=document.createElement("input");box.type="checkbox";box.checked=!!cloudForm.private;box.addEventListener("change",()=>cloudForm.private=box.checked);
        const text=document.createElement("span");text.innerHTML="<strong>Private repository</strong><small>Uncheck to publish publicly</small>";privacy.append(box,text);push.appendChild(privacy);
      }
      const pushBtn=btn("↑ Push","mlb-cloud-primary");pushBtn.disabled=!artifacts.length;pushBtn.addEventListener("click",()=>requestCloudCommand("cloud_push",cloudCommandConfig(cloudForm.push_type,cloudForm.push_artifact)));push.appendChild(pushBtn);grid.appendChild(push);

      const load=document.createElement("section");load.className="mlb-cloud-card";
      const lt=document.createElement("div");lt.className="mlb-cloud-card-title";lt.innerHTML="<strong>↓ LOAD</strong><span>Restore content from "+providerLabel(cloudForm.provider)+"</span>";load.appendChild(lt);
      load.appendChild(cloudSelect("Content Type",cloudForm.load_type,[{value:"dataset",label:"Prepared Dataset"},{value:"model",label:"MLB Studio Model"},{value:"project",label:"Builder Project"}],v=>cloudForm.load_type=v));
      providerTargetFields(load);const loadBtn=btn("↓ Load","mlb-cloud-primary secondary");loadBtn.addEventListener("click",()=>requestCloudCommand("cloud_load",cloudCommandConfig(cloudForm.load_type,null)));load.appendChild(loadBtn);grid.appendChild(load);
      container.appendChild(grid);
    }


    function renderFilesView(container){
      container.className="mlb-files-view";

      const head=document.createElement("div");head.className="mlb-files-head";
      const title=document.createElement("div");
      title.innerHTML="<strong>PROJECT FILES</strong><span>Data, model, config and design files in one place</span>";
      const filters=document.createElement("div");filters.className="mlb-files-filters";
      [["all","All"],["data","Data"],["model","Models"],["config","Config"],["design","Design"]].forEach(([value,label])=>{
        const button=btn(label,"mlb-file-filter"+(filesFilter===value?" active":""));
        button.addEventListener("click",()=>{filesFilter=value;draw();});
        filters.appendChild(button);
      });
      head.append(title,filters);container.appendChild(head);

      const entries=projectFileEntries().filter(item=>filesFilter==="all"||item.category===filesFilter);
      if(!entries.length){
        container.appendChild(makeDirectoryEmpty(
          "No files in this category yet.",
          "Run data processing or create/export a model artifact and it will appear here."
        ));
        return;
      }

      const table=document.createElement("div");table.className="mlb-files-table";
      const header=document.createElement("div");header.className="mlb-files-row header";
      header.innerHTML="<span>Name</span><span>Type</span><span>Location</span><span>Actions</span>";
      table.appendChild(header);

      entries.forEach(item=>{
        const row=document.createElement("div");row.className="mlb-files-row";
        const name=document.createElement("div");name.className="mlb-file-name";
        name.innerHTML="<strong>"+item.name+"</strong><span>"+(item.description||"")+"</span>";
        const type=document.createElement("div");
        type.innerHTML="<span class='mlb-file-type "+item.category+"'>"+fileCategoryLabel(item.category)+"</span><small>"+(item.type||"File")+"</small>";
        const location=document.createElement("div");location.className="mlb-file-location";
        location.textContent=item.location||item.path||"—";
        location.title=item.location||item.path||"";

        const actions=document.createElement("div");actions.className="mlb-file-actions";
        if(item.id==="design_json"){
          const a=btn("Save JSON","mlb-file-action");a.addEventListener("click",saveDesign);actions.appendChild(a);
        }else if(item.id==="design_bin"){
          const a=btn("Save BIN","mlb-file-action");a.addEventListener("click",saveDesignBin);actions.appendChild(a);
        }else if(item.id==="model_config"){
          const a=btn("Download","mlb-file-action");a.addEventListener("click",downloadModelConfig);actions.appendChild(a);
        }else if(item.category==="data" && item.dataset_id){
          const meta=preparedDatasetById(item.dataset_id);
          const a=btn("Use in Model","mlb-file-action");
          a.addEventListener("click",()=>useDatasetInModel(meta));actions.appendChild(a);
        }

        row.append(name,type,location,actions);
        table.appendChild(row);
      });
      container.appendChild(table);
    }

    function currentDataPipelineSnapshot(){
      const ws=state.workspaces?.data;
      const comp=state.components?.[ws?.root_component_id];
      const snap={source:null,text_processing:null,split:null,tokenizer:null,image_processing:null,audio_processing:null,batch:null,output:null,steps:[]};
      if(!comp)return snap;
      const sourceTypes=new Set(["manual_dataset","hf_dataset","kaggle_dataset","url_dataset","local_dataset"]);
      (comp.nodes||[]).forEach(node=>{
        const value={type:node.type,name:node.name,...cp(node.params||{})};
        snap.steps.push({id:node.id,type:node.type,name:node.name,params:cp(node.params||{})});
        if(sourceTypes.has(node.type))snap.source=value;
        else if(node.type==="text_process")snap.text_processing=value;
        else if(node.type==="train_test_split")snap.split=value;
        else if(node.type==="tokenize_text")snap.tokenizer=value;
        else if(node.type==="image_process")snap.image_processing=value;
        else if(node.type==="audio_process")snap.audio_processing=value;
        else if(node.type==="batch_data")snap.batch=value;
        else if(node.type==="prepared_dataset")snap.output=value;
      });
      return snap;
    }
    function datasetPipeline(meta){return meta?.pipeline||currentDataPipelineSnapshot();}
    function prettyBool(value){if(value===undefined||value===null||value==="")return "—";const v=String(value).toLowerCase();return v==="true"?"Yes":v==="false"?"No":String(value);}
    function sourceDisplay(source){if(!source)return "—";if(source.type==="hf_dataset")return source.dataset_id||"Hugging Face";if(source.type==="kaggle_dataset")return source.dataset_handle||"Kaggle";if(source.type==="url_dataset")return source.url||"URL";if(source.type==="local_dataset")return source.path||"Local File";if(source.type==="manual_dataset")return "Manual Text Data";return source.name||source.type||"—";}
    function detailSection(body,title,rows){const st=document.createElement("div");st.className="mlb-section-title";st.textContent=title;body.appendChild(st);const box=document.createElement("div");box.className="mlb-dataset-detail-box";rows.filter(row=>row&&row[1]!==undefined&&row[1]!==null&&row[1]!=="").forEach(([label,value])=>{const r=document.createElement("div");r.className="mlb-dataset-detail-row";const a=document.createElement("span");a.textContent=label;const v=document.createElement("strong");v.textContent=String(value);v.title=String(value);r.append(a,v);box.appendChild(r);});body.appendChild(box);}
    function renderPreparedDatasetInspector(body,meta){
      const p=datasetPipeline(meta),source=p.source||{},process=p.text_processing||{},split=p.split||{},tok=p.tokenizer||{},output=p.output||{};
      const head=document.createElement("div");head.className="mlb-selected";head.innerHTML="<strong>"+meta.name+"</strong><span class='mlb-pill'>Prepared Data</span>";body.appendChild(head);
      const ready=document.createElement("div");ready.className="mlb-api-status ok";ready.textContent="✓ Dataset ready for Model Builder";body.appendChild(ready);
      const st=document.createElement("div");st.className="mlb-section-title";st.textContent="SPLITS";body.appendChild(st);body.appendChild(datasetSummaryCard(meta,"DATASET OUTPUT"));
      detailSection(body,"SOURCE",[["Source Type",source.name||source.type||"—"],["Source",sourceDisplay(source)],["Hub Source Split",source.split],["Text Column",source.text_column],["Max Rows",Number(source.max_rows)===0?"All":source.max_rows]]);
      detailSection(body,"TRAIN / VALIDATION / TEST",[["Train %",split.train_size!==undefined?split.train_size+"%":"—"],["Validation %",split.validation_size!==undefined?split.validation_size+"%":"—"],["Test %",split.test_size!==undefined?split.test_size+"%":"—"],["Seed",split.seed],["Shuffle",prettyBool(split.shuffle)]]);
      if(Object.keys(process).length)detailSection(body,"TEXT PROCESSING",[["Text Column",process.text_column],["Lowercase",prettyBool(process.lowercase)],["Trim Spaces",prettyBool(process.strip)],["Normalize Whitespace",prettyBool(process.normalize_whitespace)],["Normalize Unicode",prettyBool(process.unicode_nfkc)],["Remove Empty",prettyBool(process.remove_empty)],["Min Characters",process.min_chars],["Max Characters",!process.max_chars||Number(process.max_chars)===0?"All":process.max_chars]]);
      if(Object.keys(tok).length)detailSection(body,"TOKENIZER",[["Tokenizer",tok.tokenizer_name],["Text Column",tok.text_column],["Tokenizer Max Length",tok.context_length],["Truncation",prettyBool(tok.truncation)],["Padding",tok.padding],["Special Tokens",prettyBool(tok.add_special_tokens)]]);
      detailSection(body,"STORAGE",[["Storage",dataStorageLabel(meta)],["Total Rows",meta.total_rows??"—"],["Save To Disk",output.save_to_disk!==undefined?prettyBool(output.save_to_disk):(meta.path?"Yes":"No")],["Path",meta.path||"Python memory"],["Created",meta.created_at||"—"]]);
      const actions=document.createElement("div");actions.className="mlb-action-grid";const use=btn("Use in Model","mlb-dark-btn");use.addEventListener("click",()=>useDatasetInModel(meta));actions.appendChild(use);body.appendChild(actions);
    }
    function selectedOutputDataset(){if(state.active_workspace!=="data"||bottomView!=="outputs"||!outputDirectorySelection)return null;return preparedDatasetById(outputDirectorySelection);}

    function normalizedBrickName(name){
      return String(name||"").trim().replace(/\s+/g," ").toLowerCase();
    }

    function customDefinitionsRelated(aId,bId){
      if(!aId||!bId||aId===bId)return aId===bId;
      return customDefinitionDependsOn(aId,bId,new Set())||customDefinitionDependsOn(bId,aId,new Set());
    }

    function customNameExists(name, exceptId=null, allowRelatedTo=null){
      const wanted=normalizedBrickName(name);
      if(!wanted) return false;
      return Object.values(state.custom_components||{}).some(def=>{
        if(exceptId && def.id===exceptId) return false;
        if(allowRelatedTo&&customDefinitionsRelated(allowRelatedTo,def.id))return false;
        return normalizedBrickName(def.name)===wanted;
      });
    }

    function askUniqueCustomName(defaultName, titleText){
      let proposed=prompt(titleText||"Module / API Component name:",defaultName||"");
      if(proposed===null) return null;
      proposed=String(proposed).trim().replace(/\s+/g," ");
      if(!proposed){
        setStatus("Name cannot be empty.");
        return null;
      }
      if(customNameExists(proposed)){
        setStatus('An unrelated Module/API Component named "'+proposed+'" already exists.');
        alert('An unrelated Module/API Component named "'+proposed+'" already exists. Choose another name. Parent and nested child Modules may share a display name.');
        return null;
      }
      return proposed;
    }

    function pythonValue(v){
      if(v===null||v===undefined||v==="") return "None";
      if(typeof v==="boolean") return v?"True":"False";
      if(typeof v==="number") return String(v);
      if(v==="true") return "True";
      if(v==="false") return "False";
      if(v==="None"||v==="none") return "None";
      if(typeof v==="string" && v.startsWith("torch.")) return v;
      return JSON.stringify(v);
    }

    function builderDataPreview(node){
      const p=node.params||{};
      const arg=(k,def)=>{
        let v=p[k];
        if(v===undefined||v===null||v==="")v=def;
        return pythonValue(v);
      };
      const varname=(node.name||"data").toLowerCase().replace(/[^a-z0-9]+/g,"_").replace(/^_+|_+$/g,"")||"data";

      if(node.type==="manual_dataset"){
        return "from mlb_studio.data import load_manual_text_dataset\n\n"+
          varname+" = load_manual_text_dataset(\n"+
          "    "+arg("text","Once upon a time")+", text_column="+arg("text_column","text")+",\n"+
          "    one_line_per_sample="+arg("one_line_per_sample","true")+",\n)";
      }
      if(node.type==="hf_dataset"){
        return "from mlb_studio.data import load_huggingface_dataset\n\n"+
          varname+" = load_huggingface_dataset(\n"+
          "    "+arg("dataset_id","roneneldan/TinyStories")+",\n"+
          "    config="+arg("config","")+", split="+arg("split","train")+",\n"+
          "    text_column="+arg("text_column","text")+", streaming="+arg("streaming","false")+",\n"+
          "    max_rows="+(Number(p.max_rows||0)>0?pythonValue(Number(p.max_rows)):"None")+",\n)";
      }
      if(node.type==="kaggle_dataset"){
        return "from mlb_studio.data import load_kaggle_dataset\n\n"+
          varname+" = load_kaggle_dataset(\n"+
          "    "+arg("dataset_handle","owner/dataset-name")+",\n"+
          "    file_pattern="+arg("file_pattern","*.csv")+", format="+arg("format","auto")+",\n"+
          "    text_column="+arg("text_column","text")+",\n"+
          "    max_rows="+(Number(p.max_rows||0)>0?pythonValue(Number(p.max_rows)):"None")+",\n)";
      }
      if(node.type==="url_dataset"){
        return "from mlb_studio.data import load_url_dataset\n\n"+
          varname+" = load_url_dataset(\n"+
          "    "+arg("url","https://example.com/data.txt")+",\n"+
          "    format="+arg("format","auto")+", text_column="+arg("text_column","text")+",\n"+
          "    max_rows="+(Number(p.max_rows||0)>0?pythonValue(Number(p.max_rows)):"None")+",\n)";
      }
      if(node.type==="local_dataset"){
        return "from mlb_studio.data import load_local_dataset\n\n"+
          varname+" = load_local_dataset(\n"+
          "    "+arg("path",localDefaultRoot)+",\n"+
          "    format="+arg("format","auto")+", text_column="+arg("text_column","text")+",\n"+
          "    max_rows="+(Number(p.max_rows||0)>0?pythonValue(Number(p.max_rows)):"None")+",\n)";
      }
      if(node.type==="text_process"){
        return "from mlb_studio.data import process_text_dataset\n\n"+
          "processed = process_text_dataset(\n"+
          "    dataset,\n"+
          "    text_column="+arg("text_column","text")+", lowercase="+arg("lowercase","false")+",\n"+
          "    strip="+arg("strip","true")+", normalize_whitespace="+arg("normalize_whitespace","true")+",\n"+
          "    unicode_nfkc="+arg("unicode_nfkc","true")+", remove_empty="+arg("remove_empty","true")+",\n"+
          "    min_chars="+arg("min_chars",1)+", max_chars="+(Number(p.max_chars||0)>0?pythonValue(Number(p.max_chars)):"None")+",\n)";
      }
      if(node.type==="train_test_split"){
        const tr=Math.max(0,Number(p.train_size??90))/100;
        const va=Math.max(0,Number(p.validation_size??5))/100;
        const te=Math.max(0,Number(p.test_size??5))/100;
        return "from mlb_studio.data import train_validation_test_split\n\n"+
          "splits = train_validation_test_split(\n"+
          "    dataset, train_size="+pythonValue(tr)+", validation_size="+pythonValue(va)+", test_size="+pythonValue(te)+",\n"+
          "    seed="+arg("seed",42)+", shuffle="+arg("shuffle","true")+",\n)";
      }
      if(node.type==="tokenize_text"){
        return "from mlb_studio.data import tokenize_text_dataset\n\n"+
          "tokenized = tokenize_text_dataset(\n"+
          "    dataset, tokenizer_name="+arg("tokenizer_name","gpt2")+",\n"+
          "    text_column="+arg("text_column","text")+", context_length="+arg("context_length",512)+",\n"+
          "    truncation="+arg("truncation","true")+", padding="+arg("padding","false")+",\n"+
          "    add_special_tokens="+arg("add_special_tokens","true")+",\n)";
      }
      if(node.type==="image_process"){
        return "from mlb_studio.data import process_image_dataset\n\n"+
          "processed = process_image_dataset(\n"+
          "    dataset, image_column="+arg("image_column","image")+", width="+arg("width",224)+", height="+arg("height",224)+",\n"+
          "    mode="+arg("mode","RGB")+", center_crop="+arg("center_crop","false")+",\n)";
      }
      if(node.type==="audio_process"){
        return "from mlb_studio.data import process_audio_dataset\n\n"+
          "processed = process_audio_dataset(\n"+
          "    dataset, audio_column="+arg("audio_column","audio")+", sample_rate="+arg("sample_rate",16000)+",\n"+
          "    normalize="+arg("normalize","true")+", trim_silence="+arg("trim_silence","false")+",\n"+
          "    silence_threshold="+arg("silence_threshold",0.01)+",\n)";
      }
      if(node.type==="batch_data"){
        return "from mlb_studio.data import make_torch_dataloader\n\n"+
          "loader = make_torch_dataloader(\n"+
          "    dataset, batch_size="+arg("batch_size",16)+", shuffle="+arg("shuffle","true")+",\n"+
          "    num_workers="+arg("num_workers",2)+", drop_last="+arg("drop_last","false")+",\n)";
      }
      if(node.type==="prepared_dataset"){
        return "# Registered in Builder as: "+String(p.dataset_name||"Prepared Dataset")+"\n"+
          "from mlb_studio.data import prepared_dataset_output\n\n"+
          "prepared = prepared_dataset_output(\n"+
          "    dataset, save_to_disk="+arg("save_to_disk","false")+", path="+arg("path",(localPaths.data||"mlbricks/data")+"/prepared_dataset")+",\n)";
      }
      return "";
    }

    function pythonDeepValue(value){
      if(Array.isArray(value))return "["+value.map(pythonDeepValue).join(", ")+"]";
      if(value&&typeof value==="object"){
        return "{"+Object.entries(value).map(([k,v])=>pythonValue(k)+": "+pythonDeepValue(v)).join(", ")+"}";
      }
      return pythonValue(value);
    }

    function pythonJsonish(value,fallback){
      return pythonDeepValue(parseJsonish(value,fallback));
    }

    function pythonScalarOrList(value,{numeric=false}={}){
      if(Array.isArray(value))return pythonDeepValue(value);
      const text=String(value??"").trim();
      if(!text)return "None";
      if(text.startsWith("[")){
        try{return pythonDeepValue(JSON.parse(text));}catch(_){}
      }
      if(text.includes(",")){
        const parts=text.split(",").map(x=>x.trim()).filter(Boolean);
        return pythonDeepValue(numeric?parts.map(Number):parts);
      }
      if(numeric){const n=Number(text);return Number.isFinite(n)?String(n):pythonValue(text);}
      return pythonValue(text);
    }

    function constructorPreview(node){
      const api=apiInfo(node);
      if(node.type==="custom"){
        const def=state.custom_components?.[node.definition_id];
        if(String(def?.implementation||"graph")==="api"){
          const binding=def.api_binding||{};const path=String(binding.import_path||"module.Symbol");const parts=path.split(".");const symbol=parts.pop()||"Symbol";const mod=parts.join(".")||"module";
          const specs=binding.parameters||[];const renderSpec=spec=>{const source=String(spec.source||"user");if(source==="input"||source==="main")return "x";if(source==="skip")return "skip";if(source==="extra")return "extra";if(source!=="user")return "<"+source+">";return pythonValue(node.params?.[spec.name]??spec.default);};
          const args=stage=>specs.filter(x=>String(x.stage||"init")===stage).map(spec=>(spec.positional?"":String(spec.name||"arg")+"=")+renderSpec(spec)).join(", ");
          return "from "+mod+" import "+symbol+"\n\n"+(binding.target_kind==="function"?("y = "+symbol+"("+(args("call")||"x")+")"):("layer = "+symbol+"("+args("init")+")\ny = layer("+(args("call")||"x")+")"));
        }
        return "# Nested Module";
      }
      if(api?.builder_utility) return api.builder_python_api ? builderDataPreview(node) : "";
      if(node.type==="stateaware_esa_stack"){
        const p=node.params||{};
        return "from mlbricks.esa import ESA\nfrom mlbricks.components import RMSNorm\nfrom mlbricks.ffnbrick import StateAwareFFN\nfrom mlbricks.residualbrick import ResController\n\n"+
          "# Builder compound stack matching the StateAware ESA notebook\n"+
          "# dim="+String(p.dim??384)+", state_dim="+String(p.state_dim??2749)+", layers="+String(p.layers??8)+", heads="+String(p.heads??6)+"\n"+
          "# Each layer: RMSNorm → ESA + StateAwareFFN → gated ResController\n";
      }
      if(!api?.available) return "# MLBricks API unavailable";
      if(node.type==="soup"){
        const p=node.params||{};
        const varname=(node.name||"soup").toLowerCase().replace(/[^a-z0-9]+/g,"_").replace(/^_+|_+$/g,"")||"soup";
        return "from mlbricks.soup import SOUP\n\n"+varname+" = SOUP(\n"+
          "    dim="+pythonValue(Number(p.dim??512))+",\n"+
          "    width="+pythonScalarOrList(p.width??1116,{numeric:true})+",\n"+
          "    depth="+pythonValue(Number(p.depth??2))+",\n"+
          "    mixer="+pythonScalarOrList(p.mixer??"esa")+",\n"+
          "    ffn="+pythonScalarOrList(p.ffn??"saffn")+",\n"+
          "    mixer_config="+pythonJsonish(p.mixer_config,{head:8,num_heads:8})+",\n"+
          "    ffn_config="+pythonJsonish(p.ffn_config,{})+",\n"+
          "    backend="+pythonValue(p.backend??"auto")+", precision="+pythonValue(p.precision??"fp16")+",\n"+
          "    memory_dim="+pythonValue(Number(p.memory_dim??128))+", fusion_hidden="+pythonValue(Number(p.fusion_hidden??768))+",\n)";
      }
      if(node.type==="elasticbit_runtime"){
        const p=node.params||{};
        const threshold=Number(p.threshold??0.01),minBits=Number(p.min_bits??4),maxBits=Number(p.max_bits??32);
        const mode=String(p.runtime_mode||"compact");
        const varname=(node.name||"elasticbit_matrix").toLowerCase().replace(/[^a-z0-9]+/g,"_").replace(/^_+|_+$/g,"")||"elasticbit_matrix";
        return "from mlbricks.elasticbit import ElasticBit\n\n"+
          "analysis = ElasticBit.bitsAnaliser(\n"+
          "    weights, calibration, threshold="+pythonValue(threshold)+", min_bits="+pythonValue(minBits)+", max_bits="+pythonValue(maxBits)+",\n)\n\n"+
          varname+" = ElasticBit.RuntimeMatrix.from_auto(\n"+
          "    weights, calibration, threshold="+pythonValue(threshold)+", runtime_mode="+pythonValue(mode)+",\n"+
          "    min_bits="+pythonValue(minBits)+", max_bits="+pythonValue(maxBits)+",\n)";
      }
      const args=[];
      (api.parameters||[]).forEach(f=>{
        let v=node.params?.[f.key];
        if(v===undefined || v===null || v==="") v=f.value;
        if((v===undefined || v===null) && f.required) return;
        if(v===undefined || v===null) return;
        args.push(f.key+"="+pythonValue(v));
      });
      const varname=(node.name||"layer").toLowerCase().replace(/[^a-z0-9]+/g,"_").replace(/^_+|_+$/g,"")||"layer";
      const importModule=api.import_module||(api.import_path?api.import_path.split(".").slice(0,-1).join("."):"mlbricks");
      if(api.config_api){
        const cfgName=api.config_api.public_name;
        const cfgModule=api.config_api.import_module||(api.config_api.import_path?api.config_api.import_path.split(".").slice(0,-1).join("."):importModule);
        const imports=cfgModule===importModule
          ?"from "+importModule+" import "+api.public_name+", "+cfgName
          :"from "+importModule+" import "+api.public_name+"\nfrom "+cfgModule+" import "+cfgName;
        return imports+"\n\n"+
          "config = "+cfgName+"("+args.join(", ")+")\n"+
          varname+" = "+api.public_name+"(config)";
      }
      return "from "+importModule+" import "+api.public_name+"\n\n"+varname+" = "+api.public_name+"("+args.join(", ")+")";
    }

    function connect(a,b,kind="main",sourcePort="main_out",targetPort="main_in",record=true){
      if(record&&!requireEditableLayout("change connections"))return;
      if(a===b){setStatus("A layer cannot connect to itself.");return;}
      const c=current(state);
      if(c.edges.some(e=>e.source===a&&e.target===b&&e.kind===kind&&e.source_port===sourcePort&&e.target_port===targetPort)){
        setStatus("Connection already exists.");return;
      }
      if(record) checkpoint(kind==="residual"?"Create skip connection":(kind==="aux"?"Create extra connection":(kind==="named"?"Create named connection":"Create main connection")));
      const e=edge(a,b,kind);
      e.source_port=sourcePort;
      e.target_port=targetPort;
      c.edges.push(e);
      setStatus(kind==="residual"?"Skip connection created.":(kind==="aux"?"Extra connection created.":(kind==="named"?"Named port connection created.":"Main connection created.")));
    }

    function isMainLaneEdge(e){
      return e.kind==="main" && (e.source_port||"main_out")==="main_out" && (e.target_port||"main_in")==="main_in";
    }

    function rebuildMainFlow(){
      const c=current(state);
      if(isApiComposerView())return;
      // Visual Module editors keep their ordered Main lane connected even
      // though the model-level Auto Connect toggle is intentionally hidden.
      if(!state.auto_connect&&!isGraphCustomEditor())return;
      // In Auto Connect mode the middle lane represents the ordered model flow.
      // Rebuild only that lane; Skip and Extra connections remain untouched.
      c.edges=(c.edges||[]).filter(e=>!isMainLaneEdge(e));
      for(let i=0;i<c.nodes.length-1;i++){
        connect(c.nodes[i].id,c.nodes[i+1].id,"main","main_out","main_in",false);
      }
    }

    function insertAfterSelection(node){
      const c=current(state);
      let insertAt=c.nodes.length;
      if(selected){
        const idx=c.nodes.findIndex(x=>x.id===selected);
        if(idx>=0) insertAt=idx+1;
      }
      c.nodes.splice(insertAt,0,node);
      rebuildMainFlow();
      selected=node.id;
      return insertAt;
    }

    function addPrimitive(item){
      const apiMode=isApiComposerView();
      if(apiMode&&!apiComposerAllowsCatalogItem(item)){
        setStatus((item?.name||"This component")+" is not available inside API Components yet. Use a supported model component or Add Function in the top toolbar.");draw();return;
      }
      if(!requireEditableLayout("add components"))return;
      checkpoint("Add "+item.name);
      const c=current(state);
      const source=apiMode?(selectedNode()||c.nodes[c.nodes.length-1]||null):null;
      const n=makeNode(item);n.name=uniqueNodeName(item.name);n.display_name=item.name;
      if(n.type==="text_input")configureTextInputForLatest(n);
      if(n.type==="soup"){
        const settings=deriveModelSettings(null);
        n.params=n.params||{};
        n.params.dim=settings.embedding_size;
        n.params.precision=settings.precision;
        n.params.mixer_config=soupMixerConfigWithHeads(n.params.mixer_config,settings.heads,n.params.depth);
      }
      const pos=insertAfterSelection(n);
      if(apiMode&&source&&source.id!==n.id){
        const e=edge(source.id,n.id,"main");e.source_port="main_out";e.target_port="main_in";c.edges.push(e);
      }
      setStatus(apiMode
        ?(n.name+" added to the API Component execution graph. Rewire its ports for serial or parallel execution as needed.")
        :(n.name+" inserted at layer "+(pos+1)+"."));
      draw();
      queueComponentImport(n.type);
    }

    function customArgDefault(index=0){
      return {
        id:uid("arg"),name:"arg_"+(index+1),label:"Argument "+(index+1),stage:"init",
        type:"int",source:"user",default:0,required:false,positional:false,options:[]
      };
    }

    function defaultAPIBinding(){
      return {
        module_path:"",
        symbol:"",
        import_path:"",
        // target_kind is kept for backward compatibility with V1.0 designs.
        target_kind:"function",
        call_type:"function",
        object_mode:"new",
        object_id:"",
        object_name:"",
        object_ref:"",
        call_method:"",
        auto_main_input:true,
        register_result_object:false,
        result_object_id:"",
        result_object_name:"",
        result_output_mode:"result",
        output_selector:"auto",
        user_function_name:"custom_function",
        user_code:"def custom_function(x):\n    return x",
        user_class_name:"CustomClass",
        user_class_code:"class CustomClass:\n    def __init__(self):\n        pass",
        source_cache_id:"",
        source_hash:"",
        source_revision:1,
        dependencies:[],
        port_mode:"standard",
        input_ports:[],
        output_ports:[],
        multi_output:false,
        output_map:{main:"0",skip:"1",extra:"2"},
        parameters:[]
      };
    }

    const apiCallTypeLabels={
      function:"Function",
      user_function:"User Defined Function",
      user_class:"User Defined Class",
      static_method:"Static Method",
      class_method:"Class Method",
      instance_method:"Instance Method",
      constructor:"Create Object"
    };

    function apiCallTypeLabel(value){
      return apiCallTypeLabels[String(value||"")]||"Instance Method";
    }

    function normalizeAPIBinding(binding){
      binding=binding||defaultAPIBinding();
      if((!binding.module_path||!binding.symbol)&&binding.import_path){
        const parts=String(binding.import_path).split(".").filter(Boolean);
        if(parts.length>1){
          if(!binding.symbol)binding.symbol=parts.pop();
          if(!binding.module_path)binding.module_path=parts.join(".");
        }
      }
      binding.module_path=String(binding.module_path||"").trim();
      binding.symbol=String(binding.symbol||"").trim();
      binding.import_path=[binding.module_path,binding.symbol].filter(Boolean).join(".");
      if(!binding.call_type){
        binding.call_type=String(binding.target_kind||"module").toLowerCase()==="function"?"function":"instance_method";
      }
      binding.call_type=String(binding.call_type||"instance_method").toLowerCase();
      if(!Object.prototype.hasOwnProperty.call(apiCallTypeLabels,binding.call_type))binding.call_type="instance_method";
      binding.object_mode=String(binding.object_mode||"new").toLowerCase()==="existing"?"existing":"new";
      binding.object_id=String(binding.object_id||"").trim();
      binding.object_name=String(binding.object_name||"").trim();
      binding.object_ref=String(binding.object_ref||"").trim();
      binding.call_method=String(binding.call_method||"").trim();
      if(binding.auto_main_input===undefined)binding.auto_main_input=true;
      binding.auto_main_input=!!binding.auto_main_input;
      binding.register_result_object=!!binding.register_result_object;
      binding.result_object_id=String(binding.result_object_id||"").trim();
      binding.result_object_name=String(binding.result_object_name||"").trim();
      binding.result_output_mode=String(binding.result_output_mode||"result").toLowerCase()==="passthrough"?"passthrough":"result";
      binding.user_function_name=String(binding.user_function_name||"custom_function").trim()||"custom_function";
      binding.user_code=String(binding.user_code||"def custom_function(x):\n    return x");
      binding.user_class_name=String(binding.user_class_name||"CustomClass").trim()||"CustomClass";
      binding.user_class_code=String(binding.user_class_code||"class CustomClass:\n    def __init__(self):\n        pass");
      binding.source_cache_id=String(binding.source_cache_id||"").trim();
      binding.source_hash=String(binding.source_hash||"").trim();
      binding.source_revision=Math.max(1,Number(binding.source_revision||1));
      if(!Array.isArray(binding.dependencies))binding.dependencies=[];
      binding.dependencies=binding.dependencies.map(x=>String(x||"").trim()).filter(Boolean);
      binding.port_mode=String(binding.port_mode||"standard").toLowerCase()==="named"?"named":"standard";
      if(!Array.isArray(binding.input_ports))binding.input_ports=[];
      if(!Array.isArray(binding.output_ports))binding.output_ports=[];
      binding.input_ports=binding.input_ports.map((port,i)=>({id:String(port?.id||("in_"+(i+1))),name:String(port?.name||("input_"+(i+1))).trim()||("input_"+(i+1)),parameter:String(port?.parameter||port?.name||("input_"+(i+1))).trim()||("input_"+(i+1)),positional:!!port?.positional,required:port?.required!==false}));
      binding.output_ports=binding.output_ports.map((port,i)=>({id:String(port?.id||("out_"+(i+1))),name:String(port?.name||("output_"+(i+1))).trim()||("output_"+(i+1)),selector:String(port?.selector??(i===0?"auto":String(i)))}));
      binding.multi_output=!!binding.multi_output;
      if(!binding.output_map||typeof binding.output_map!=="object")binding.output_map={main:"0",skip:"1",extra:"2"};
      binding.output_map.main=String(binding.output_map.main??"0");
      binding.output_map.skip=String(binding.output_map.skip??"1");
      binding.output_map.extra=String(binding.output_map.extra??"2");
      // Keep the legacy field synchronized for old runtimes/design readers.
      binding.target_kind=["function","user_function"].includes(binding.call_type)?"function":"module";
      if(!Array.isArray(binding.parameters))binding.parameters=[];
      return binding;
    }

    function apiBindingImportPath(binding){
      return normalizeAPIBinding(binding).import_path||"";
    }

    function apiSafeObjectName(value,fallback="object"){
      const clean=String(value||"").trim().replace(/[^A-Za-z0-9_]+/g,"_").replace(/^_+|_+$/g,"");
      const base=clean||fallback;
      return /^[A-Za-z_]/.test(base)?base:("obj_"+base);
    }

    function apiSafePortName(value,fallback="port"){
      return apiSafeObjectName(value,fallback);
    }
    function defaultUserInputPort(index=0){
      const n=index+1;return {id:uid("inport"),name:"input_"+n,parameter:"input_"+n,positional:false,required:true};
    }
    function defaultUserOutputPort(index=0){
      const n=index+1;return {id:uid("outport"),name:"output_"+n,selector:index===0?"auto":String(index)};
    }

    function ensureAPIStepObjectIds(step){
      if(!step)return defaultAPIBinding();
      step.api_binding=normalizeAPIBinding(step.api_binding||defaultAPIBinding());
      const b=step.api_binding;
      if(!b.object_id)b.object_id="object::"+step.id;
      if(!b.result_object_id)b.result_object_id="result::"+step.id;
      return b;
    }

    function apiObjectCandidates(def,excludeStepId=""){
      const out=[];
      apiStepNodes(def).forEach(step=>{
        if(!step||step.id===excludeStepId)return;
        const b=ensureAPIStepObjectIds(step);
        const source=step.name||"API Function";
        const createsDirect=b.call_type==="constructor"||b.call_type==="user_class"||(b.call_type==="instance_method"&&b.object_mode==="new");
        if(createsDirect){
          out.push({id:b.object_id,name:b.object_name||apiSafeObjectName(b.symbol||source),source,kind:"created"});
        }
        if(b.register_result_object){
          out.push({id:b.result_object_id,name:b.result_object_name||apiSafeObjectName(source+"_result"),source,kind:"result"});
        }
      });
      return out;
    }

    function defaultAPIStep(index=0){
      const step={
        id:uid("api_step"),type:"api_step",name:"Function "+(index+1),display_name:"Function "+(index+1),
        repeat:1,params:{},input_count:3,output_count:3,position:{x:0,y:0},api_binding:defaultAPIBinding()
      };
      ensureAPIStepObjectIds(step);
      return step;
    }

    function apiStepNodes(def){
      // While an API Component is open, its editable graph lives in the
      // transient custom_edit component until Save/Done copies it back to the
      // definition. Use that live graph so newly created objects are
      // immediately available to later API nodes in the same editor session.
      const live=current(state);
      const nodes=(live?.kind==="custom_edit"&&live.definition_id===def?.id)
        ?(live.nodes||[])
        :(def?.nodes||[]);
      return nodes.filter(n=>n&&n.type==="api_step");
    }

    function ensureAPIDefinitionSteps(def){
      if(!def||String(def.implementation||"graph")!=="api")return def;
      if(apiStepNodes(def).length)return def;
      const first=defaultAPIStep(0);
      if(def.api_binding)first.api_binding=normalizeAPIBinding(cp(def.api_binding));
      def.nodes=[first];def.edges=[];
      return def;
    }

    function isApiComposerView(){
      const c=current(state);if(!c||c.kind!=="custom_edit")return false;
      const def=state.custom_components?.[c.definition_id];
      return String(def?.implementation||"graph")==="api";
    }

    function activeCustomDefinition(){
      const c=current(state);
      return c?.kind==="custom_edit"?state.custom_components?.[c.definition_id]||null:null;
    }

    function isGraphCustomEditor(){
      const def=activeCustomDefinition();
      return !!def && String(def.implementation||"graph")!=="api";
    }

    function customDefinitionDependsOn(startId,targetId,seen=new Set()){
      if(!startId||!targetId)return false;
      if(startId===targetId)return true;
      if(seen.has(startId))return false;
      seen.add(startId);
      const def=state.custom_components?.[startId];
      if(!def)return false;
      for(const n of (def.nodes||[])){
        const childId=n?.type==="custom"?n.definition_id:null;
        if(!childId)continue;
        if(childId===targetId)return true;
        if(customDefinitionDependsOn(childId,targetId,seen))return true;
      }
      return false;
    }

    function customCanNest(parentDef,childDef){
      if(!parentDef||!childDef)return false;
      if(parentDef.id===childDef.id)return false;
      // Adding child inside parent is invalid if child already reaches parent.
      return !customDefinitionDependsOn(childDef.id,parentDef.id,new Set());
    }

    function nestedComponentChoices(){
      const parent=activeCustomDefinition();
      if(!parent)return [];
      const parentIsApi=String(parent.implementation||"graph")==="api";
      const out=[];const seen=new Set();
      Object.values(state.custom_components||{}).forEach(def=>{
        if(!def||def.id===parent.id||!customCanNest(parent,def))return;
        // An API Component may reuse Modules, but does not nest another API
        // Component. This keeps its API graph readable while still allowing
        // reusable FFN/norm/etc. Modules between API functions.
        if(parentIsApi&&String(def.implementation||"graph")==="api")return;
        if(def.palette_installed!==true && !def.gallery_entry_id)return;
        const key=def.gallery_entry_id||("def:"+def.id);if(seen.has(key))return;seen.add(key);
        out.push({name:def.name||"Module",kind:String(def.implementation||"graph")==="api"?"API":"MOD",def,entry:null});
      });
      (state.gallery?.components||[]).forEach(entry=>{
        if(!entry?.definition)return;
        const rootId=entry.source_definition_id||entry.definition.id;
        if(rootId===parent.id||entry.id===parent.gallery_entry_id)return;
        if(parentIsApi&&String(entry.definition.implementation||"graph")==="api")return;
        if(seen.has(entry.id))return;
        const existing=Object.values(state.custom_components||{}).find(d=>d.gallery_entry_id===entry.id||d.id===rootId);
        if(existing&&!customCanNest(parent,existing))return;
        seen.add(entry.id);
        out.push({name:entry.name||entry.definition.name||"Module",kind:String(entry.definition.implementation||"graph")==="api"?"API":"MOD",def:existing||null,entry});
      });
      return out.sort((a,b)=>String(a.name).localeCompare(String(b.name)));
    }

    // API Components are explicit execution graphs. They may mix API function
    // blocks with supported built-in MLBricks/PyTorch components, while saved
    // custom components remain excluded to avoid recursive/circular nesting.
    const apiComposerBuiltInTypes=new Set([
      "embedding","esa","stateaware_esa_stack","soup","rmsnorm","layernorm",
      "linear","ffn","residual","dropout","learned_position","sinusoidal_position","lm_head"
    ]);
    function apiComposerAllowsCatalogItem(item){
      return !!item && apiComposerBuiltInTypes.has(String(item.type||""));
    }

    function apiParamKey(step,spec){
      const explicit=String(spec?.expose_key||"").trim();
      if(explicit)return explicit;
      return String(step?.id||"step")+"::"+String(spec?.name||spec?.key||"arg");
    }

    function addAPIFunction(){
      const c=current(state);if(!c||!isApiComposerView())return;
      if(!requireEditableLayout("add API functions"))return;
      checkpoint("Add API function");
      const n=defaultAPIStep(c.nodes.length);
      n.name=uniqueNodeName(n.name,c);n.display_name=n.name;
      const source=selectedNode()||c.nodes[c.nodes.length-1]||null;
      c.nodes.push(n);
      if(source&&source.id!==n.id){
        const e=edge(source.id,n.id,"main");e.source_port="main_out";e.target_port="main_in";c.edges.push(e);
      }
      selected=n.id;pendingPort=null;
      setStatus(n.name+" added. Configure its import, function/class and parameters.");draw();
    }

    function customFieldType(spec){
      const t=String(spec?.type||"str").toLowerCase();
      if(t==="int"||t==="integer"||t==="float"||t==="number")return "number";
      if(t==="bool"||t==="boolean")return "select";
      if(t==="select")return "select";
      if(t==="json"||t==="dict"||t==="list"||t==="tuple")return "textarea";
      return "text";
    }

    function customExposedFields(def){
      if(!def||String(def.implementation||"graph")!=="api")return [];
      const steps=apiStepNodes(def);
      if(steps.length){
        const fields=[];
        steps.forEach(step=>{
          const binding=normalizeAPIBinding(step.api_binding||defaultAPIBinding());
          (binding.parameters||[]).filter(spec=>String(spec.source||"user")==="user").forEach(spec=>{
            const key=apiParamKey(step,spec);
            const field={
              key,label:(step.name||"Function")+" · "+(spec.label||spec.name||"Argument"),type:customFieldType(spec),value:spec.default,
              required:!!spec.required,help:"Custom API · "+(spec.stage||"init")+" argument · "+(apiBindingImportPath(binding)||"unbound")
            };
            if(field.type==="select"){
              if(String(spec.type||"").toLowerCase()==="bool"||String(spec.type||"").toLowerCase()==="boolean")field.options=["true","false"];
              else field.options=Array.isArray(spec.options)?spec.options:[];
            }
            fields.push(field);
          });
        });
        return fields;
      }
      const binding=normalizeAPIBinding(def.api_binding||defaultAPIBinding());
      return (binding.parameters||[]).filter(spec=>String(spec.source||"user")==="user").map(spec=>{
        const field={
          key:spec.name,label:spec.label||spec.name,type:customFieldType(spec),value:spec.default,
          required:!!spec.required,help:"Custom API · "+(spec.stage||"init")+" argument"
        };
        if(field.type==="select"){
          if(String(spec.type||"").toLowerCase()==="bool"||String(spec.type||"").toLowerCase()==="boolean")field.options=["true","false"];
          else field.options=Array.isArray(spec.options)?spec.options:[];
        }
        return field;
      });
    }

    function boundContextValue(source,spec){
      const settings=deriveModelSettings(null);
      if(source==="model_dim")return settings.embedding_size;
      if(source==="heads")return settings.heads;
      if(source==="context")return numberOr(state.project?.context_length,512);
      if(source==="batch")return numberOr(state.project?.batch_size,16);
      if(source==="device")return "auto";
      if(source==="dtype")return precisionToDtype(settings.precision);
      return spec?.default;
    }

    function customNodeParams(def){
      const params={};
      if(!def||String(def.implementation||"graph")!=="api")return params;
      const steps=apiStepNodes(def);
      if(steps.length){
        steps.forEach(step=>{
          const binding=normalizeAPIBinding(step.api_binding||defaultAPIBinding());
          (binding.parameters||[]).forEach(spec=>{
            const name=String(spec.name||"").trim();if(!name)return;
            const source=String(spec.source||"user");
            if(["input","main","skip","extra"].includes(source))return;
            const key=apiParamKey(step,spec);
            params[key]=source==="user"?spec.default:boundContextValue(source,spec);
          });
        });
        return params;
      }
      const binding=normalizeAPIBinding(def.api_binding||defaultAPIBinding());
      (binding.parameters||[]).forEach(spec=>{
        const name=String(spec.name||"").trim();if(!name)return;
        const source=String(spec.source||"user");
        if(["input","main","skip","extra"].includes(source))return;
        params[name]=source==="user"?spec.default:boundContextValue(source,spec);
      });
      return params;
    }

    function createAPICustom(){
      const name=askUniqueCustomName("API Component","New API component name:");
      if(!name){draw();return;}
      beginCustomEditorTransaction();
      rememberWorkspaceView();state.active_workspace="model";
      const modelWs=state.workspaces?.model;if(modelWs){state.view_component_id=modelWs.view_component_id||modelWs.root_component_id;state.breadcrumbs=cp(modelWs.breadcrumbs||[{id:modelWs.root_component_id,name:modelWs.name||"Model Builder"}]);}
      const id=uid("custom");
      const first=defaultAPIStep(0);
      state.custom_components[id]={
        id,name,description:"Python/PyTorch API execution graph",revision:1,
        implementation:"api",api_binding:null,nodes:[cp(first)],edges:[],input_count:3,output_count:3,
        palette_hidden:true,palette_installed:false,gallery_entry_id:null
      };
      const vid="view_"+id+"_"+uid("n");
      state.components[vid]={
        id:vid,name,kind:"custom_edit",definition_id:id,revision:1,
        input_count:3,output_count:3,nodes:[cp(first)],edges:[]
      };
      galleryWorkspace.open=false;bottomExpanded=galleryPreviousBottomExpanded;
      state.view_component_id=vid;state.breadcrumbs=[{id:vid,name}];selected=first.id;pendingPort=null;
      setStatus(name+" created with Function 1. Configure the API on the right, then add more functions as needed.");draw();
    }

    function requestCustomAPIImport(def,step=null){
      const binding=step?ensureAPIStepObjectIds(step):normalizeAPIBinding(def?.api_binding||defaultAPIBinding());
      const statusKey=step?def.id+":"+step.id:def.id;
      if(binding.call_type==="user_function"){
        const source=String(binding.user_code||"").trim();
        const functionName=String(binding.user_function_name||"").trim();
        if(!source||!functionName){
          customImportStatus[statusKey]={status:"error",message:"Enter a Python function name and source code first."};
          setStatus("Enter a Python function name and source code first.");draw();return;
        }
        if(!bridgeReady()){setStatus("Kernel bridge is offline. Re-run the Builder cell, then validate the function again.");return;}
        const command={action:"validate_user_function",source,function_name:functionName,label:(step?.name||def?.name||functionName),definition_id:statusKey,ts:Date.now()};
        if(!setBridgeState()||!setBridgeCommand(command)){setStatus("Could not send User Function validation to Python.");return;}
        const button=bridgeControl(bridge.run,"button");if(!button){setStatus("Python Run control was not found.");return;}
        customImportStatus[statusKey]={status:"running",message:"Validating "+functionName+"…"};
        setStatus("Validating "+functionName+"…");clickBridgeButton(button);draw();return;
      }
      if(binding.call_type==="user_class"){
        const source=String(binding.user_class_code||"").trim();
        const className=String(binding.user_class_name||"").trim();
        if(!source||!className){
          customImportStatus[statusKey]={status:"error",message:"Enter a Python class name and source code first."};
          setStatus("Enter a Python class name and source code first.");draw();return;
        }
        if(!bridgeReady()){setStatus("Kernel bridge is offline. Re-run the Builder cell, then validate the class again.");return;}
        const command={action:"validate_user_class",source,class_name:className,label:(step?.name||def?.name||className),definition_id:statusKey,ts:Date.now()};
        if(!setBridgeState()||!setBridgeCommand(command)){setStatus("Could not send User Class validation to Python.");return;}
        const button=bridgeControl(bridge.run,"button");if(!button){setStatus("Python Run control was not found.");return;}
        customImportStatus[statusKey]={status:"running",message:"Validating "+className+"…"};
        setStatus("Validating "+className+"…");clickBridgeButton(button);draw();return;
      }
      if(binding.call_type==="instance_method"&&binding.object_mode==="existing"){
        const candidate=apiObjectCandidates(def,step?.id||"").find(x=>x.id===binding.object_ref);
        if(!candidate){
          customImportStatus[statusKey]={status:"error",message:"Choose an existing object first."};
          setStatus("Choose an existing object from this API Component first.");draw();return;
        }
        customImportStatus[statusKey]={status:"done",message:"Using object "+candidate.name+" from "+candidate.source+"."};
        setStatus("Bound to existing object "+candidate.name+" from "+candidate.source+".");draw();return;
      }
      const importPath=apiBindingImportPath(binding);
      if(!importPath){setStatus("Enter Import / Module and Function / Class first.");return;}
      if(!bridgeReady()){setStatus("Kernel bridge is offline. Re-run the Builder cell, then test the API again.");return;}
      const label=(step?.name||def?.name||importPath);
      const command={action:"ensure_external_import",import_path:importPath,label,definition_id:statusKey,ts:Date.now()};
      if(!setBridgeState()||!setBridgeCommand(command)){setStatus("Could not send custom API import check to Python.");return;}
      const button=bridgeControl(bridge.run,"button");if(!button){setStatus("Python Run control was not found.");return;}
      customImportStatus[statusKey]={status:"running",message:"Checking "+importPath+"…"};
      setStatus("Checking "+importPath+"…");clickBridgeButton(button);draw();
    }

    function editorRow(labelText,value,onInput,opts={}){
      const row=document.createElement("div");row.className="mlb-custom-binding-field";
      const label=document.createElement("label");label.textContent=labelText;row.appendChild(label);
      let input;
      if(opts.select){
        input=document.createElement("select");
        (opts.options||[]).forEach(opt=>{
          const isObj=opt&&typeof opt==="object";
          const optValue=isObj?opt.value:opt;
          const optLabel=isObj?(opt.label??opt.value):opt;
          const o=document.createElement("option");o.value=String(optValue??"");o.textContent=String(optLabel??"");
          if(String(value??"")===String(optValue??""))o.selected=true;input.appendChild(o);
        });
      }
      else if(opts.textarea){input=document.createElement("textarea");input.rows=opts.rows||2;input.value=value??"";}
      else {input=document.createElement("input");input.type=opts.type||"text";input.value=value??"";}
      input.addEventListener(opts.select?"change":"input",()=>onInput(input.value));row.appendChild(input);return row;
    }

    function renderAPIStepBindingEditor(body,def,step){
      const binding=ensureAPIStepObjectIds(step);
      const intro=document.createElement("div");intro.className="mlb-api-path";
      intro.textContent="Bind a Python API as a function, class/static method, object constructor, or instance method. Objects created by one node can be reused by later nodes without recreating the instance.";body.appendChild(intro);

      const title=document.createElement("div");title.className="mlb-section-title";title.textContent="FUNCTION / API";body.appendChild(title);
      body.appendChild(editorRow("Node Name",step.name||"Function",v=>{const clean=String(v||"").trim();if(clean){step.name=clean;step.display_name=clean;}},{type:"text"}));

      const callTypeOptions=Object.entries(apiCallTypeLabels).map(([value,label])=>({value,label}));
      body.appendChild(editorRow("Call Type",binding.call_type||"function",v=>{
        binding.call_type=v;
        binding.target_kind=["function","user_function"].includes(v)?"function":"module";
        const args=Array.isArray(binding.parameters)?binding.parameters:[];
        if(v==="constructor"||v==="user_class")args.forEach(spec=>spec.stage="init");
        else if(v!=="instance_method"||binding.object_mode==="existing")args.forEach(spec=>spec.stage="call");
        customImportStatus[def.id+":"+step.id]=null;draw();
      },{select:true,options:callTypeOptions}));

      const callType=binding.call_type;
      const isExisting=callType==="instance_method"&&binding.object_mode==="existing";
      const isUserFunction=callType==="user_function";
      const isUserClass=callType==="user_class";
      const isUserSource=isUserFunction||isUserClass;
      const needsImport=!isExisting&&!isUserSource;
      const classLike=["static_method","class_method","instance_method","constructor","user_class"].includes(callType);

      if(callType==="instance_method"){
        body.appendChild(editorRow("Object Source",binding.object_mode||"new",v=>{
          binding.object_mode=v;
          if(v==="existing"){
            (binding.parameters||[]).forEach(spec=>spec.stage="call");
          }
          customImportStatus[def.id+":"+step.id]=null;draw();
        },{select:true,options:[{value:"new",label:"Create New Object"},{value:"existing",label:"Use Existing Object"}]}));
      }

      if(isUserFunction){
        const userTitle=document.createElement("div");userTitle.className="mlb-section-title";userTitle.textContent="PYTHON FUNCTION";body.appendChild(userTitle);
        body.appendChild(editorRow("Function Name",binding.user_function_name||"custom_function",v=>{binding.user_function_name=String(v||"").trim()||"custom_function";customImportStatus[def.id+":"+step.id]=null;},{type:"text"}));
        body.appendChild(editorRow("Python Code",binding.user_code||"def custom_function(x):\n    return x",v=>{binding.user_code=v;binding.dependencies=extractPythonDependencies(v);binding.source_hash=userSourceHash(v);customImportStatus[def.id+":"+step.id]=null;},{textarea:true,rows:10}));
        const userHelp=document.createElement("div");userHelp.className="mlb-api-path";userHelp.textContent="Define one Python function. torch and torch.nn are available automatically; imports inside the editor use the notebook/kernel environment.";body.appendChild(userHelp);
        body.appendChild(editorRow("Visual Port Mode",binding.port_mode||"standard",v=>{
          binding.port_mode=v;
          if(v==="named"){
            binding.auto_main_input=false;binding.multi_output=false;
            if(!binding.input_ports.length)binding.input_ports=[defaultUserInputPort(0)];
            if(!binding.output_ports.length)binding.output_ports=[defaultUserOutputPort(0)];
          }
          pendingPort=null;draw();
        },{select:true,options:[{value:"standard",label:"Main / Skip / Extra"},{value:"named",label:"Custom Named Ports"}]}));
        if(binding.port_mode==="named"){
          const inTitle=document.createElement("div");inTitle.className="mlb-section-title";inTitle.textContent="INPUT PORTS";body.appendChild(inTitle);
          const inHelp=document.createElement("div");inHelp.className="mlb-api-path";inHelp.textContent="Each connected input port is passed to the Python function using its Function Parameter name. Named ports can connect directly between User Defined Function nodes.";body.appendChild(inHelp);
          (binding.input_ports||[]).forEach((port,index)=>{
            const box=document.createElement("div");box.className="mlb-custom-arg-card";
            const head=document.createElement("div");head.className="mlb-custom-arg-head";const nm=document.createElement("strong");nm.textContent=port.name||("Input "+(index+1));
            const rm=btn("×","mlb-custom-arg-remove");rm.title="Remove input port";rm.addEventListener("click",()=>{checkpoint("Remove user input port");current(state).edges=(current(state).edges||[]).filter(e=>!(e.target===step.id&&e.target_port==="named_in:"+port.id));binding.input_ports.splice(index,1);pendingPort=null;draw();});head.append(nm,rm);box.appendChild(head);
            box.appendChild(editorRow("Port Name",port.name||"",v=>{port.name=apiSafePortName(v,"input_"+(index+1));draw();}));
            box.appendChild(editorRow("Function Parameter",port.parameter||port.name||"",v=>port.parameter=apiSafePortName(v,port.name||("input_"+(index+1)))));
            box.appendChild(editorRow("Pass As",port.positional?"positional":"keyword",v=>port.positional=v==="positional",{select:true,options:["keyword","positional"]}));
            box.appendChild(editorRow("Required",port.required===false?"false":"true",v=>port.required=v==="true",{select:true,options:["true","false"]}));
            body.appendChild(box);
          });
          const addIn=btn("+ Add Input Port","mlb-create mlb-custom-add-arg");addIn.addEventListener("click",()=>{checkpoint("Add user input port");binding.input_ports.push(defaultUserInputPort(binding.input_ports.length));draw();});body.appendChild(addIn);
          const outTitle=document.createElement("div");outTitle.className="mlb-section-title";outTitle.textContent="OUTPUT PORTS";body.appendChild(outTitle);
          const outHelp=document.createElement("div");outHelp.className="mlb-api-path";outHelp.textContent="Map each visual output to the complete result (auto), a tuple/list index, or a dict/object key. Add as many outputs as your function returns.";body.appendChild(outHelp);
          (binding.output_ports||[]).forEach((port,index)=>{
            const box=document.createElement("div");box.className="mlb-custom-arg-card";
            const head=document.createElement("div");head.className="mlb-custom-arg-head";const nm=document.createElement("strong");nm.textContent=port.name||("Output "+(index+1));
            const rm=btn("×","mlb-custom-arg-remove");rm.title="Remove output port";rm.addEventListener("click",()=>{checkpoint("Remove user output port");current(state).edges=(current(state).edges||[]).filter(e=>!(e.source===step.id&&e.source_port==="named_out:"+port.id));binding.output_ports.splice(index,1);pendingPort=null;draw();});head.append(nm,rm);box.appendChild(head);
            box.appendChild(editorRow("Port Name",port.name||"",v=>{port.name=apiSafePortName(v,"output_"+(index+1));draw();}));
            box.appendChild(editorRow("Return Index / Key",port.selector??"auto",v=>port.selector=v));body.appendChild(box);
          });
          const addOut=btn("+ Add Output Port","mlb-create mlb-custom-add-arg");addOut.addEventListener("click",()=>{checkpoint("Add user output port");binding.output_ports.push(defaultUserOutputPort(binding.output_ports.length));draw();});body.appendChild(addOut);
        }
      }
      if(isUserClass){
        const userTitle=document.createElement("div");userTitle.className="mlb-section-title";userTitle.textContent="PYTHON CLASS";body.appendChild(userTitle);
        body.appendChild(editorRow("Class Name",binding.user_class_name||"CustomClass",v=>{binding.user_class_name=String(v||"").trim()||"CustomClass";if(!binding.object_name)binding.object_name=apiSafeObjectName(binding.user_class_name);customImportStatus[def.id+":"+step.id]=null;},{type:"text"}));
        body.appendChild(editorRow("Python Code",binding.user_class_code||"class CustomClass:\n    def __init__(self):\n        pass",v=>{binding.user_class_code=v;binding.dependencies=extractPythonDependencies(v);binding.source_hash=userSourceHash(v);customImportStatus[def.id+":"+step.id]=null;},{textarea:true,rows:12}));
        const userHelp=document.createElement("div");userHelp.className="mlb-api-path";userHelp.textContent="The class source is saved with this component. This node constructs one reusable object; later Instance Method nodes can select that same object. Third-party libraries are never installed automatically.";body.appendChild(userHelp);
      }

      if(isUserSource){
        const source=isUserClass?binding.user_class_code:binding.user_code;
        binding.dependencies=extractPythonDependencies(source);
        binding.source_hash=userSourceHash(source);
        const depTitle=document.createElement("div");depTitle.className="mlb-section-title";depTitle.textContent="SOURCE CACHE / DEPENDENCIES";body.appendChild(depTitle);
        const depBox=document.createElement("div");depBox.className="mlb-summary";
        [["Cached Source",binding.source_cache_id?"Yes":"Saved when component is saved"],["Source Revision","v"+(binding.source_revision||1)],["Source Hash",binding.source_hash||"—"],["Dependencies",binding.dependencies.length?binding.dependencies.join(", "):"None detected"]].forEach(([a,b])=>{const r=document.createElement("div");r.className="mlb-summary-row";r.innerHTML="<span>"+a+"</span><strong>"+b+"</strong>";depBox.appendChild(r);});body.appendChild(depBox);
        const depHelp=document.createElement("div");depHelp.className="mlb-api-path";depHelp.textContent="MLB Studio stores your source code with the saved component/project. External dependencies must already be installed in the active Python environment; Studio never installs them automatically.";body.appendChild(depHelp);
      }

      if(needsImport){
        body.appendChild(editorRow("Import / Module",binding.module_path||"",v=>{binding.module_path=v;normalizeAPIBinding(binding);customImportStatus[def.id+":"+step.id]=null;},{type:"text"}));
        body.appendChild(editorRow(classLike?"Class / Symbol":"Function / Symbol",binding.symbol||"",v=>{
          binding.symbol=v;normalizeAPIBinding(binding);
          if(!binding.object_name&&(binding.call_type==="constructor"||(binding.call_type==="instance_method"&&binding.object_mode==="new"))){
            binding.object_name=apiSafeObjectName(v||step.name);
          }
          customImportStatus[def.id+":"+step.id]=null;
        },{type:"text"}));
      }

      if(callType==="constructor"||callType==="user_class"||(callType==="instance_method"&&binding.object_mode==="new")){
        if(!binding.object_name)binding.object_name=apiSafeObjectName((isUserClass?binding.user_class_name:binding.symbol)||step.name);
        body.appendChild(editorRow("Object Name",binding.object_name||"",v=>binding.object_name=apiSafeObjectName(v||"object"),{type:"text"}));
        const objInfo=document.createElement("div");objInfo.className="mlb-api-path";
        objInfo.textContent="Reusable object ID is tied to this node, not its visible name. Renaming the node will not break later references.";body.appendChild(objInfo);
      }

      let selectedObject=null;
      if(isExisting){
        const candidates=apiObjectCandidates(def,step.id);
        if(!binding.object_ref&&candidates.length)binding.object_ref=candidates[0].id;
        selectedObject=candidates.find(x=>x.id===binding.object_ref)||null;
        const options=candidates.length
          ?candidates.map(x=>({value:x.id,label:x.name+"  —  "+x.source}))
          :[{value:"",label:"No reusable objects yet"}];
        body.appendChild(editorRow("Existing Object",binding.object_ref||"",v=>{binding.object_ref=v;customImportStatus[def.id+":"+step.id]=null;draw();},{select:true,options}));
        const objInfo=document.createElement("div");objInfo.className="mlb-api-path";
        objInfo.textContent=selectedObject
          ?("Reusing “"+selectedObject.name+"” created by node “"+selectedObject.source+"”.")
          :"Create an object in another API node first, then select it here.";
        body.appendChild(objInfo);
      }

      if(["static_method","class_method","instance_method"].includes(callType)){
        body.appendChild(editorRow("Method",binding.call_method||"",v=>binding.call_method=v,{type:"text"}));
      }

      if(callType!=="constructor"&&callType!=="user_class"&&!(isUserFunction&&binding.port_mode==="named")){
        body.appendChild(editorRow("Implicit Main Input",binding.auto_main_input?"true":"false",v=>binding.auto_main_input=v==="true",{select:true,options:[{value:"true",label:"Auto prepend Main"},{value:"false",label:"Do not inject Main"}]}));
        body.appendChild(editorRow("Multiple Outputs",binding.multi_output?"true":"false",v=>{binding.multi_output=v==="true";draw();},{select:true,options:[{value:"false",label:"Single Output"},{value:"true",label:"Map Multiple Outputs"}]}));
        if(binding.multi_output){
          const mapTitle=document.createElement("div");mapTitle.className="mlb-section-title";mapTitle.textContent="OUTPUT MAPPING";body.appendChild(mapTitle);
          const mapHelp=document.createElement("div");mapHelp.className="mlb-api-path";mapHelp.textContent="For tuple/list returns use indexes such as 0, 1, 2. Dict/object returns may use a key or attribute name. Leave Skip or Extra blank when unused.";body.appendChild(mapHelp);
          body.appendChild(editorRow("Main Output Index / Key",binding.output_map?.main??"0",v=>binding.output_map.main=v,{type:"text"}));
          body.appendChild(editorRow("Skip Output Index / Key",binding.output_map?.skip??"1",v=>binding.output_map.skip=v,{type:"text"}));
          body.appendChild(editorRow("Extra Output Index / Key",binding.output_map?.extra??"2",v=>binding.output_map.extra=v,{type:"text"}));
        }else{
          body.appendChild(editorRow("Output Selector",binding.output_selector||"auto",v=>binding.output_selector=v,{type:"text"}));
        }
        body.appendChild(editorRow("Register Result as Object",binding.register_result_object?"true":"false",v=>{
          binding.register_result_object=v==="true";
          if(binding.register_result_object&&!binding.result_object_name)binding.result_object_name=apiSafeObjectName((step.name||"result")+"_result");
          draw();
        },{select:true,options:[{value:"false",label:"No"},{value:"true",label:"Yes"}]}));
        if(binding.register_result_object){
          if(!binding.result_object_name)binding.result_object_name=apiSafeObjectName((step.name||"result")+"_result");
          body.appendChild(editorRow("Result Object Name",binding.result_object_name||"",v=>binding.result_object_name=apiSafeObjectName(v||"result_object"),{type:"text"}));
          body.appendChild(editorRow("After Registering",binding.result_output_mode||"result",v=>binding.result_output_mode=v,{select:true,options:[{value:"result",label:"Send Result to Main Output"},{value:"passthrough",label:"Pass Main Input Through"}]}));
        }
      }

      const apiActions=document.createElement("div");apiActions.className="mlb-action-grid";
      const testLabel=isUserFunction?"Validate Function":(isUserClass?"Validate Class":(isExisting?"Bind Existing Object":"Bind / Test API"));
      const test=btn(testLabel,"mlb-custom-api-test");test.addEventListener("click",()=>requestCustomAPIImport(def,step));apiActions.appendChild(test);body.appendChild(apiActions);
      const st=customImportStatus[def.id+":"+step.id];if(st){
        const msg=document.createElement("div");msg.className="mlb-api-status "+(st.status==="done"?"available":st.status==="error"?"unavailable":"utility");msg.textContent=st.message||st.status;body.appendChild(msg);
        const checkedDeps=st.details?.dependencies||[];
        if(checkedDeps.length){
          const depCheck=document.createElement("div");depCheck.className="mlb-summary";
          checkedDeps.forEach(item=>{const r=document.createElement("div");r.className="mlb-summary-row";r.innerHTML="<span>"+(item.available?"✓ ":"✕ ")+item.name+"</span><strong>"+(item.available?"Available":"Install explicitly")+"</strong>";depCheck.appendChild(r);});
          body.appendChild(depCheck);
        }
      }

      const argTitle=document.createElement("div");argTitle.className="mlb-section-title";argTitle.textContent="PARAMETERS";body.appendChild(argTitle);
      const note=document.createElement("div");note.className="mlb-api-path";
      note.textContent=callType==="user_function"
        ?(binding.port_mode==="named"?"Your function uses custom named visual ports. Connected inputs are passed by parameter name and each output port selects a returned value.":"Your Python function executes on every pass. Use Main / Skip / Extra parameter sources for incoming lanes. Enable Multiple Outputs to map returned values to the node's three output ports.")
        :callType==="user_class"
        ?"Constructor parameters create one reusable instance from your cached class source. The object is registered once and later Instance Method nodes can reuse it."
        :callType==="constructor"
        ?"Constructor parameters are evaluated once when the API Component is built. The created object is registered and this node passes Main data through unchanged."
        :(callType==="instance_method"&&binding.object_mode==="new"
          ?"Init parameters create the reusable object once; Call parameters are supplied whenever this method executes."
          :"Call parameters execute on every pass. Tensor sources Main / Skip / Extra come from the visual lanes; other sources can bind model settings or user values.");
      body.appendChild(note);

      const args=Array.isArray(binding.parameters)?binding.parameters:(binding.parameters=[]);
      const allowedStages=(callType==="constructor"||callType==="user_class")?["init"]:(callType==="instance_method"&&binding.object_mode==="new"?["init","call"]:["call"]);
      args.forEach((spec,index)=>{
        if(!allowedStages.includes(String(spec.stage||"")))spec.stage=allowedStages[0];
        const box=document.createElement("div");box.className="mlb-custom-arg-card";
        const head=document.createElement("div");head.className="mlb-custom-arg-head";
        const name=document.createElement("strong");name.textContent=(spec.label||spec.name||("Parameter "+(index+1)));
        const remove=btn("×","mlb-custom-arg-remove");remove.title="Remove parameter";remove.addEventListener("click",()=>{checkpoint("Remove API parameter");args.splice(index,1);draw();});
        head.append(name,remove);box.appendChild(head);
        box.appendChild(editorRow("Parameter Name",spec.name||"",v=>{spec.name=v;spec.label=spec.label||v;}));
        box.appendChild(editorRow("UI Label",spec.label||spec.name||"",v=>spec.label=v));
        box.appendChild(editorRow("Stage",spec.stage||allowedStages[0],v=>spec.stage=v,{select:true,options:allowedStages}));
        box.appendChild(editorRow("Type",spec.type||"str",v=>{spec.type=v;setTimeout(draw,0);},{select:true,options:["int","float","str","bool","select","json","dict","list","tuple"]}));
        box.appendChild(editorRow("Source",spec.source||"user",v=>{spec.source=v;setTimeout(draw,0);},{select:true,options:["user","main","skip","extra","model_dim","heads","context","batch","device","dtype"]}));
        if(String(spec.source||"user")==="user"){
          box.appendChild(editorRow("Default",spec.default??"",v=>spec.default=v,{textarea:["json","dict","list","tuple"].includes(String(spec.type)),rows:2}));
          if(String(spec.type)==="select")box.appendChild(editorRow("Options (comma separated)",(spec.options||[]).join(", "),v=>spec.options=v.split(",").map(x=>x.trim()).filter(Boolean)));
        }
        box.appendChild(editorRow("Pass As",spec.positional?"positional":"keyword",v=>spec.positional=(v==="positional"),{select:true,options:["keyword","positional"]}));
        box.appendChild(editorRow("Required",spec.required?"true":"false",v=>spec.required=(v==="true"),{select:true,options:["false","true"]}));
        body.appendChild(box);
      });
      const add=btn("+ Add Parameter","mlb-create mlb-custom-add-arg");add.addEventListener("click",()=>{
        checkpoint("Add API parameter");const spec=customArgDefault(args.length);spec.stage=allowedStages[0];args.push(spec);draw();
      });body.appendChild(add);

      const previewTitle=document.createElement("div");previewTitle.className="mlb-section-title";previewTitle.textContent="FUNCTION PREVIEW";body.appendChild(previewTitle);
      const pre=document.createElement("pre");pre.className="mlb-code-preview";
      const path=apiBindingImportPath(binding)||"module.Symbol";const parts=path.split(".");const symbol=parts.pop()||"Symbol";const mod=parts.join(".")||"module";
      const renderArg=a=>(a.positional?"":String(a.name||"arg")+"=")+String(a.source||"user");
      const initArgs=args.filter(a=>String(a.stage||"init")==="init").map(renderArg).join(", ");
      const explicitCallArgs=args.filter(a=>String(a.stage||"call")==="call").map(renderArg).join(", ");
      const callArgs=explicitCallArgs||(binding.auto_main_input?"main":"");
      const objectName=binding.object_name||apiSafeObjectName((isUserClass?binding.user_class_name:symbol)||step.name);
      let code="";
      if(!isExisting&&!isUserSource)code+="from "+mod+" import "+symbol+"\n\n";
      if(isUserFunction)code+=(binding.user_code||"def custom_function(x):\n    return x")+"\n\n";
      if(isUserClass)code+=(binding.user_class_code||"class CustomClass:\n    def __init__(self):\n        pass")+"\n\n";
      if(callType==="user_function"){
        if(binding.port_mode==="named"){
          const namedArgs=(binding.input_ports||[]).map(p=>(p.positional?"":(p.parameter||p.name)+"=")+"<"+(p.name||"input")+">").join(", ");
          code+="result = "+(binding.user_function_name||"custom_function")+"("+namedArgs+")";
          (binding.output_ports||[]).forEach(p=>{code+="\n"+(p.name||"output")+" = result"+(String(p.selector||"auto")==="auto"?"":("["+p.selector+"]"));});
        }else code+="result = "+(binding.user_function_name||"custom_function")+"("+callArgs+")";
      }
      else if(callType==="user_class")code+=objectName+" = "+(binding.user_class_name||"CustomClass")+"("+initArgs+")\n# registered for later nodes\ny = main";
      else if(callType==="function")code+="y = "+symbol+"("+callArgs+")";
      else if(callType==="static_method"||callType==="class_method")code+="y = "+symbol+"."+(binding.call_method||"method")+"("+callArgs+")";
      else if(callType==="constructor")code+=objectName+" = "+symbol+"("+initArgs+")\n# registered for later nodes\ny = main";
      else if(binding.object_mode==="existing"){
        const reuseName=selectedObject?.name||"existing_object";
        code+="# object from "+(selectedObject?.source||"another node")+"\ny = "+reuseName+(binding.call_method?("."+binding.call_method):"")+"("+callArgs+")";
      }else{
        code+=objectName+" = "+symbol+"("+initArgs+")\n";
        code+="y = "+objectName+(binding.call_method?("."+binding.call_method):"")+"("+callArgs+")";
      }
      if(isUserFunction&&binding.port_mode!=="named"){
        if(binding.multi_output){
          const om=binding.output_map||{};
          code+="\nmain = result["+(om.main||"0")+"]";
          if(String(om.skip??"").trim())code+="\nskip = result["+om.skip+"]";
          if(String(om.extra??"").trim())code+="\nextra = result["+om.extra+"]";
        }else code+="\ny = result"+(binding.output_selector&&binding.output_selector!=="auto"?("["+binding.output_selector+"]"):"");
      }
      if(binding.register_result_object)code+="\n"+(binding.result_object_name||apiSafeObjectName(step.name+"_result"))+" = "+(isUserFunction?"result":"y")+"  # registered for reuse"+(binding.result_output_mode==="passthrough"?"\ny = main":"");
      pre.textContent=code;body.appendChild(pre);
    }

    function renderAPICustomOverview(body,def){
      const steps=apiStepNodes(def);
      const h=document.createElement("div");h.className="mlb-section-title";h.textContent="API EXECUTION GRAPH";body.appendChild(h);
      const help=document.createElement("div");help.className="mlb-api-path";
      help.textContent="This API Component is one execution graph. Add API functions from the top toolbar, or insert supported MLBricks components from the left. Use ports for serial/parallel paths and remove links from the selected block's Connections section.";body.appendChild(help);
      const summary=document.createElement("div");summary.className="mlb-summary";
      [["API Nodes",steps.length],["Reusable Objects",apiObjectCandidates(def).length],["Connections",(current(state)?.edges||[]).length]].forEach(([a,b])=>{const r=document.createElement("div");r.className="mlb-summary-row";r.innerHTML="<span>"+a+"</span><strong>"+b+"</strong>";summary.appendChild(r);});body.appendChild(summary);
    }

    function removeAPIFunction(step){
      const c=current(state);if(!c||!step)return;
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      if(win&&typeof win.confirm==="function"&&!win.confirm('Remove function "'+(step.name||"Function")+'" from this API Component?'))return;
      checkpoint("Remove API function");
      const removedBinding=ensureAPIStepObjectIds(step);
      const removedObjectIds=new Set([removedBinding.object_id,removedBinding.result_object_id].filter(Boolean));
      c.nodes=(c.nodes||[]).filter(n=>n.id!==step.id);
      c.edges=(c.edges||[]).filter(e=>e.source!==step.id&&e.target!==step.id);
      let clearedRefs=0;
      apiStepNodes(c).forEach(other=>{
        const b=ensureAPIStepObjectIds(other);
        if(removedObjectIds.has(b.object_ref)){b.object_ref="";clearedRefs++;}
      });
      selected=c.nodes[0]?.id||null;pendingPort=null;
      setStatus((step.name||"Function")+" removed from API Component."+(clearedRefs?" Cleared "+clearedRefs+" object reference"+(clearedRefs===1?"":"s")+".":""));draw();
    }

    function renderAPIStepConnections(body,step){
      const title=document.createElement("div");title.className="mlb-section-title";title.textContent="CONNECTIONS";body.appendChild(title);
      const help=document.createElement("div");help.className="mlb-api-path";help.textContent="Connect Main / Skip / Extra lanes normally, or connect custom named User Function ports directly. One output may feed multiple downstream blocks.";body.appendChild(help);
      const rel=(current(state).edges||[]).filter(e=>e.source===step.id||e.target===step.id);
      if(!rel.length){const empty=document.createElement("div");empty.className="mlb-api-path";empty.textContent="No connections for this function.";body.appendChild(empty);}
      else rel.forEach(ed=>{
        const row=document.createElement("div");row.className="mlb-connection-row";
        const src=current(state).nodes.find(x=>x.id===ed.source),tgt=current(state).nodes.find(x=>x.id===ed.target);
        const namedSource=String(ed.source_port||"").startsWith("named_out:")?String(ed.source_port).replace("named_out:",""):"";
        const namedTarget=String(ed.target_port||"").startsWith("named_in:")?String(ed.target_port).replace("named_in:",""):"";
        const lane=ed.kind==="residual"?"Skip":(ed.kind==="aux"?"Extra":(ed.kind==="named"?"Named":"Main"));
        const portText=(namedSource||namedTarget)?(" · "+(namedSource||"standard")+" → "+(namedTarget||lane)):" · "+lane;
        const txt=document.createElement("div");txt.className="mlb-connection-text";txt.textContent=(src?.name||"Function")+" → "+(tgt?.name||"Function")+portText;
        const del=btn("Remove","mlb-conn-remove");del.addEventListener("click",()=>{if(!requireEditableLayout("remove API connections"))return;checkpoint("Remove API connection");current(state).edges=current(state).edges.filter(x=>x.id!==ed.id);setStatus("API connection removed.");draw();});
        row.append(txt,del);body.appendChild(row);
      });
    }

    function renderAPIStepInspector(body,def,step){
      const selectedWrap=document.createElement("div");selectedWrap.className="mlb-selected";
      const title=document.createElement("strong");title.textContent=step.name||"Function";
      const pill=document.createElement("span");pill.className="mlb-pill";pill.textContent=apiCallTypeLabel(ensureAPIStepObjectIds(step).call_type);selectedWrap.append(title,pill);body.appendChild(selectedWrap);
      renderAPIStepBindingEditor(body,def,step);
      renderAPIStepConnections(body,step);
      const actions=document.createElement("div");actions.className="mlb-action-grid mlb-api-step-actions";
      const remove=btn("Remove API Node","mlb-danger-btn");remove.addEventListener("click",()=>removeAPIFunction(step));actions.appendChild(remove);body.appendChild(actions);
      appendCustomSaveActions(body);
    }

    function createCustom(){
      const name=askUniqueCustomName("My Module","New module name:");
      if(!name){draw();return;}

      beginCustomEditorTransaction();
      rememberWorkspaceView();
      const id=uid("custom");

      // IMPORTANT: creating a custom component creates an EMPTY reusable shell.
      // It never captures siblings or the current model canvas.
      state.custom_components[id]={
        id,
        name,
        description:"Reusable Module",
        revision:1,
        implementation:"graph",
        nodes:[],
        edges:[],
        input_count:3,
        output_count:3,
        palette_hidden:true,
        palette_installed:false,
        gallery_entry_id:null
      };

      // Open the new empty Module immediately. The focused editor is separate
      // from the model workspace and the outer Module is the save boundary.
      const vid="view_"+id+"_"+uid("n");
      state.components[vid]={
        id:vid,
        name,
        kind:"custom_edit",
        definition_id:id,
        revision:1,
        input_count:3,
        output_count:3,
        nodes:[],
        edges:[]
      };
      state.view_component_id=vid;
      state.breadcrumbs=[{id:vid,name}];
      selected=null;
      pendingPort=null;
      galleryWorkspace.open=false;
      setStatus(name+" opened in the Module Editor.");
      draw();
    }

    function customDefinitionSnapshot(def,c=null){
      return {
        id:def.id,name:def.name,description:def.description||"Reusable Module",
        revision:def.revision||1,implementation:def.implementation||"graph",
        api_binding:cp(def.api_binding||null),input_count:3,output_count:3,
        nodes:cp(c?.nodes||def.nodes||[]),edges:cp(c?.edges||def.edges||[])
      };
    }

    function collectCustomDependencySnapshots(rootDefId,seen=new Set()){
      const out={};
      if(!rootDefId||seen.has(rootDefId))return out;
      seen.add(rootDefId);
      const rootDef=state.custom_components?.[rootDefId];
      if(!rootDef)return out;
      (rootDef.nodes||[]).forEach(n=>{
        if(n?.type!=="custom"||!n.definition_id||seen.has(n.definition_id))return;
        const child=state.custom_components?.[n.definition_id];if(!child)return;
        out[child.id]=customDefinitionSnapshot(child);
        Object.assign(out,collectCustomDependencySnapshots(child.id,seen));
      });
      return out;
    }

    function cacheDefinitionTreeSources(rootDefId,seen=new Set(),liveRoot=null){
      if(!rootDefId||seen.has(rootDefId))return;
      seen.add(rootDefId);
      const def=state.custom_components?.[rootDefId];if(!def)return;
      cacheUserSourcesForDefinition(def,liveRoot&&liveRoot.definition_id===rootDefId?liveRoot:null);
      (def.nodes||[]).forEach(n=>{if(n?.type==="custom"&&n.definition_id)cacheDefinitionTreeSources(n.definition_id,seen,null);});
    }

    function customGallerySnapshot(def,c){
      cacheDefinitionTreeSources(def.id,new Set(),c||null);
      const snap=customDefinitionSnapshot(def,c);
      snap.component_cache={};
      const collectNodes=(nodes,seenDefs=new Set())=>{
        (nodes||[]).forEach(step=>{
          if(step?.type==="api_step"){
            const b=ensureAPIStepObjectIds(step);
            if(b.source_cache_id&&state.component_cache?.[b.source_cache_id])snap.component_cache[b.source_cache_id]=cp(state.component_cache[b.source_cache_id]);
          }else if(step?.type==="custom"&&step.definition_id&&!seenDefs.has(step.definition_id)){
            seenDefs.add(step.definition_id);
            const child=state.custom_components?.[step.definition_id];if(child)collectNodes(child.nodes||[],seenDefs);
          }
        });
      };
      collectNodes(snap.nodes||[],new Set([def.id]));
      snap.dependency_definitions=collectCustomDependencySnapshots(def.id,new Set());
      return snap;
    }

    function uniqueGalleryComponentName(base,exceptId=null){
      const clean=String(base||"Module").trim().replace(/\s+/g," ")||"Module";
      if(!galleryNameExists("components",clean,exceptId))return clean;
      let i=2;while(galleryNameExists("components",clean+" "+i,exceptId))i++;
      return clean+" "+i;
    }

    function upsertCustomInGallery(def,c){
      let entry=(state.gallery.components||[]).find(x=>x.id===def.gallery_entry_id);
      if(!entry)entry=(state.gallery.components||[]).find(x=>x.source_definition_id===def.id);
      if(entry){
        entry.name=uniqueGalleryComponentName(def.name,entry.id);
        entry.saved_at=new Date().toISOString();
        entry.definition=customGallerySnapshot(def,c);
        entry.source_definition_id=def.id;
      }else{
        entry={
          id:uid("gallery_component"),
          name:uniqueGalleryComponentName(def.name),
          kind:"component",saved_at:new Date().toISOString(),
          source_definition_id:def.id,definition:customGallerySnapshot(def,c)
        };
        state.gallery.components.push(entry);
      }
      def.gallery_entry_id=entry.id;
      persistGallery();persistComponentCache();
      return entry;
    }

    function editCustomDefinition(def){
      if(!def)return;
      beginCustomEditorTransaction();
      ensureAPIDefinitionSteps(def);
      customActionMenuId=null;
      rememberWorkspaceView();
      const vid="view_"+def.id+"_"+uid("n");
      state.components[vid]={
        id:vid,name:def.name,kind:"custom_edit",definition_id:def.id,revision:def.revision||1,
        input_count:3,output_count:3,nodes:cp(def.nodes||[]),edges:cp(def.edges||[])
      };
      const modelWs=state.workspaces?.model;
      if(modelWs){
        state.active_workspace="model";
      }
      state.view_component_id=vid;state.breadcrumbs=[{id:vid,name:def.name}];
      selected=null;pendingPort=null;galleryWorkspace.open=false;
      setStatus("Editing "+def.name+".");draw();
    }

    function renameCustomDefinition(def){
      if(!def)return;
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const kind=String(def.implementation||"graph")==="api"?"API Component":"Module";
      const proposed=win&&typeof win.prompt==="function"?win.prompt("Rename "+kind+":",def.name||kind):null;
      if(proposed===null)return;
      const name=String(proposed||"").trim().replace(/\s+/g," ");
      if(!name){setStatus(kind+" name cannot be empty.");return;}
      if(customNameExists(name,def.id,def.id)){setStatus('Another unrelated custom item named "'+name+'" already exists.');return;}
      checkpoint("Rename "+kind.toLowerCase());
      const oldName=def.name;def.name=name;
      Object.values(state.components||{}).forEach(comp=>{
        if(comp.kind==="custom_edit"&&comp.definition_id===def.id)comp.name=name;
        (comp.nodes||[]).forEach(n=>{if(n.definition_id===def.id){n.display_name=name;if(normalizedUserName(n.name)===normalizedUserName(oldName))n.name=uniqueNodeName(name,comp,n.id);}});
      });
      customActionMenuId=null;setStatus(kind+' renamed to "'+name+'".');draw();
    }

    function removeCustomFromPalette(def){
      if(!def)return;
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      if(win&&typeof win.confirm==="function"&&!win.confirm('Remove "'+def.name+'" from the Component Library? Existing model instances will remain unchanged.'))return;
      checkpoint("Remove custom item from Component Library");
      def.palette_hidden=true;def.palette_installed=false;customActionMenuId=null;
      setStatus(def.name+" removed from the Component Library. It remains available in Gallery if it was saved there.");draw();
    }

    function addCustom(def,options={}){
      const parentDef=activeCustomDefinition();
      if(parentDef&&String(parentDef.implementation||"graph")==="api"&&String(def?.implementation||"graph")==="api"){
        setStatus("An API Component can add reusable Modules, but cannot nest another API Component. Add API functions directly instead.");draw();return;
      }
      if(parentDef&&!customCanNest(parentDef,def)){
        setStatus('Cannot add "'+(def?.name||"Module")+'" here because it would create a circular Module dependency.');
        componentInsertPicker={open:false,afterNodeId:null};draw();return;
      }
      if(!requireEditableLayout("add components"))return;
      if(!options.skipCheckpoint)checkpoint("Add "+def.name);
      const c=current(state);
      const apiMode=isApiComposerView();
      const source=apiMode?(selectedNode()||c.nodes[c.nodes.length-1]||null):null;
      const n={
        id:uid("node"),
        type:"custom",
        name:uniqueNodeName(def.name),
        display_name:def.name,
        definition_id:def.id,
        repeat:1,
        params:customNodeParams(def),
        input_count:3,
        output_count:3,
        position:{x:0,y:0}
      };
      const pos=insertAfterSelection(n);
      if(apiMode&&source&&source.id!==n.id){
        const e=edge(source.id,n.id,"main");e.source_port="main_out";e.target_port="main_in";c.edges.push(e);
      }
      componentInsertPicker={open:false,afterNodeId:null};
      setStatus(def.name+" inserted at layer "+(pos+1)+".");
      draw();
      if(String(def.implementation||"graph")==="api"){
        const steps=apiStepNodes(def).filter(step=>apiBindingImportPath(step.api_binding));
        if(steps.length)setTimeout(()=>requestCustomAPIImport(def,steps[0]),60);
        else if(String(def.api_binding?.import_path||"").trim())setTimeout(()=>requestCustomAPIImport(def),60);
      }
    }

    function insertGalleryComponentIntoCurrent(entry){
      const parentDef=activeCustomDefinition();
      const def=restoreGalleryComponentDefinition(entry,{installed:false,allowNameConflictWith:parentDef?.id||null});
      if(!def)return;
      const after=componentInsertPicker.afterNodeId;
      if(after&&current(state).nodes.some(n=>n.id===after))selected=after;
      addCustom(def);
    }

    // v1.0.0: direct nested Module creation for both Module and API editors.
    // Add Module never opens a picker. It enters a blank child Module draft;
    // Done inserts that child into the parent graph and Cancel discards it.
    function createNestedCustom(afterNodeId=null){
      const parent=current(state);
      if(!parent||parent.kind!=="custom_edit")return;
      const parentDef=activeCustomDefinition();
      if(!parentDef)return;

      // Keep the insertion anchor stable before entering the child editor.
      const anchor=(afterNodeId&&parent.nodes.some(n=>n.id===afterNodeId))
        ?afterNodeId
        :(selected&&parent.nodes.some(n=>n.id===selected)?selected:null);

      const siblingCount=(parent.nodes||[]).filter(n=>n?.type==="custom").length;
      const name="Module "+(siblingCount+1);

      beginCustomEditorTransaction();
      const id=uid("custom");
      state.custom_components[id]={
        id,
        name,
        description:"Nested Module",
        revision:1,
        implementation:"graph",
        nodes:[],
        edges:[],
        input_count:3,
        output_count:3,
        palette_hidden:true,
        palette_installed:false,
        gallery_entry_id:null
      };

      const vid="view_"+id+"_"+uid("n");
      state.components[vid]={
        id:vid,
        name,
        kind:"custom_edit",
        definition_id:id,
        revision:1,
        input_count:3,
        output_count:3,
        nodes:[],
        edges:[],
        parent_edit_return:{
          view_id:parent.id,
          definition_id:parentDef.id,
          after_node_id:anchor,
          existing_node_id:null
        }
      };

      state.view_component_id=vid;
      state.breadcrumbs.push({id:vid,name});
      selected=null;
      pendingPort=null;
      componentInsertPicker={open:false,afterNodeId:null};
      setStatus(name+" opened inside "+parentDef.name+". Add Components, then use Done to insert it or Cancel to discard it.");
      draw();
    }

    function openInside(node){
      if(!node.definition_id)return;
      const def=state.custom_components[node.definition_id];if(!def)return;
      beginCustomEditorTransaction();
      ensureAPIDefinitionSteps(def);
      const parentView=current(state);
      const parentDef=parentView?.kind==="custom_edit"?state.custom_components?.[parentView.definition_id]:null;
      const vid="view_"+def.id+"_"+uid("n");
      state.components[vid]={
        id:vid,
        name:def.name,
        kind:"custom_edit",
        definition_id:def.id,
        revision:def.revision,
        input_count:3,
        output_count:3,
        nodes:cp(def.nodes),
        edges:cp(def.edges||[]),
        parent_edit_return:parentDef
          ?{view_id:parentView.id,definition_id:parentDef.id,after_node_id:node.id,existing_node_id:node.id}
          :null
      };
      if(parentDef)state.breadcrumbs.push({id:vid,name:def.name});
      else{rememberWorkspaceView();state.breadcrumbs=[{id:vid,name:def.name}];}
      state.view_component_id=vid;selected=null;pendingPort=null;componentInsertPicker={open:false,afterNodeId:null};draw();
    }

    function appendCustomSaveActions(container){
      const c=current(state);
      const actions=document.createElement("div");actions.className="mlb-action-grid mlb-custom-save-actions "+(c?.parent_edit_return?"nested":"outer");
      if(c?.parent_edit_return){
        const done=btn("Done");
        done.title="Apply this nested Module/API Component and return to its parent. The outer Module remains the only Gallery save boundary.";
        done.addEventListener("click",()=>saveCustom(false));
        const cancel=btn("Cancel");
        cancel.title="Discard changes made in this nested editor and return to the parent";
        cancel.addEventListener("click",cancelCustomEditor);
        actions.append(done,cancel);
      }else{
        const save=btn("Save");
        save.title="Save the complete outer Module/API Component, including all nested dependencies, to Gallery";
        save.addEventListener("click",()=>saveCustom(false));
        const saveAsNew=btn("Save As New");
        saveAsNew.title="Save the complete outer graph as a new Gallery item";
        saveAsNew.addEventListener("click",()=>saveCustom(true));
        const cancel=btn("Cancel");
        cancel.title="Discard this editor transaction and return without saving";
        cancel.addEventListener("click",cancelCustomEditor);
        actions.append(save,saveAsNew,cancel);
      }
      container.appendChild(actions);
    }

    function saveCustom(asNew){
      const c=current(state),def=state.custom_components[c.definition_id];if(!def)return;
      const returnInfo=c.parent_edit_return||null;

      // Nested editors are draft scopes. "Done" commits the child definition in
      // memory and returns to its parent; it never creates a separate Gallery
      // entry. The outermost Module/API Component is the transaction boundary.
      if(returnInfo){
        const frame=customEditorTransactions.pop()||null;
        def.nodes=cp(c.nodes||[]);def.edges=cp(c.edges||[]);def.input_count=3;def.output_count=3;
        def.revision=(def.revision||1)+1;c.revision=def.revision;
        if(state.components?.[returnInfo.view_id]){
          state.view_component_id=returnInfo.view_id;
          const crumbIndex=(state.breadcrumbs||[]).findIndex(x=>x.id===returnInfo.view_id);
          if(crumbIndex>=0)state.breadcrumbs=state.breadcrumbs.slice(0,crumbIndex+1);
          selected=returnInfo.after_node_id&&current(state).nodes.some(n=>n.id===returnInfo.after_node_id)?returnInfo.after_node_id:null;
          pendingPort=null;componentInsertPicker={open:false,afterNodeId:null};
          if(frame){
            restoreHistory(frame);
            undoStack.push({state:cp(frame.before_state),label:returnInfo.existing_node_id?"Update nested module":"Add nested module"});
            if(undoStack.length>historyLimit)undoStack.shift();
            redoStack.length=0;
          }
          if(!returnInfo.existing_node_id){
            addCustom(def,{skipCheckpoint:true});
            setStatus(def.name+" applied inside "+(activeCustomDefinition()?.name||"the parent Module")+". Save the outer Module when the full structure is complete.");draw();
          }else{
            setStatus(def.name+" updated. Returned to "+(activeCustomDefinition()?.name||"the parent Module")+". Save the outer Module when finished.");
            draw();
          }
        }
        return;
      }

      let savedDef=def,savedView=c;
      if(asNew){
        const name=askUniqueCustomName(def.name+" Copy","Save as new Module/API Component:");
        if(!name){draw();return;}
        const id=uid("custom");
        savedDef={
          id,name,description:def.description||"",revision:1,implementation:def.implementation||"graph",
          api_binding:cp(def.api_binding||null),nodes:cp(c.nodes),edges:cp(c.edges||[]),input_count:3,output_count:3,
          palette_hidden:true,palette_installed:false,gallery_entry_id:null
        };
        state.custom_components[id]=savedDef;
        savedView={nodes:cp(c.nodes),edges:cp(c.edges||[])};
      }else{
        def.nodes=cp(c.nodes);def.edges=cp(c.edges||[]);def.input_count=3;def.output_count=3;
        def.revision=(def.revision||1)+1;c.revision=def.revision;
      }
      // Saving the outer editor commits the transaction only after every
      // prompt/validation has succeeded. Editor history is discarded so Undo
      // on the Model/Gallery screen can never reopen or cancel the saved item.
      customEditorTransactions.pop();
      upsertCustomInGallery(savedDef,savedView);
      if(savedDef.palette_installed!==true)savedDef.palette_hidden=true;

      const label=String(savedDef.implementation||"graph")==="api"?"API Component":"Module";
      const savedMessage=savedDef.name+" saved as one "+label+" with all nested dependencies.";
      if(!galleryWorkspace.open)galleryPreviousBottomExpanded=bottomExpanded;
      // Restore the Model Builder workspace before opening Gallery.
      const modelWs=state.workspaces?.model;
      if(modelWs){state.active_workspace="model";state.view_component_id=modelWs.view_component_id||modelWs.root_component_id;state.breadcrumbs=cp(modelWs.breadcrumbs||[{id:modelWs.root_component_id,name:modelWs.name||"Model Builder"}]);}
      runtimePanel=null;cloudWorkspace.open=false;bottomExpanded=false;galleryWorkspace={open:true,tab:"components"};outputDirectorySelection=null;selected=null;componentInsertPicker={open:false,afterNodeId:null};
      undoStack.length=0;redoStack.length=0;
      setStatus(savedMessage);draw();
    }
    function deleteNode(id){
      if(!requireEditableLayout("delete components"))return;
      checkpoint("Delete node");
      const c=current(state);
      c.nodes=c.nodes.filter(n=>n.id!==id);
      c.edges=c.edges.filter(e=>e.source!==id&&e.target!==id);
      rebuildMainFlow();
      if(selected===id)selected=null;
      setStatus("Layer deleted.");
      draw();
    }

    function duplicateSelected(){
      const n=selectedNode();if(!n)return;
      if(!requireEditableLayout("duplicate components"))return;
      checkpoint("Duplicate "+n.name);
      const c=current(state),d=cp(n);d.id=uid("node");d.name=uniqueNodeName(n.name+" Copy",c);d.display_name=nodeDisplayName(n);
      const idx=c.nodes.findIndex(x=>x.id===n.id);
      c.nodes.splice(idx+1,0,d);
      rebuildMainFlow();
      selected=d.id;
      setStatus("Layer duplicated after "+n.name+".");
      draw();
    }

    function moveSelected(delta){
      const n=selectedNode();if(!n)return;
      if(!requireEditableLayout("move components"))return;
      const c=current(state);
      const from=c.nodes.findIndex(x=>x.id===n.id);
      if(from<0)return;
      const to=Math.max(0,Math.min(c.nodes.length-1,from+delta));
      if(to===from){
        setStatus(delta<0?"Layer is already first.":"Layer is already last.");
        draw();
        return;
      }
      checkpoint("Move "+n.name+(delta<0?" left":" right"));
      c.nodes.splice(from,1);
      c.nodes.splice(to,0,n);
      rebuildMainFlow();
      selected=n.id;
      setStatus(n.name+" moved to layer "+(to+1)+".");
      draw();
    }

    function portClick(nodeId,side,portIndex,ev,portKey="",portName="",portMode="standard"){
      ev.stopPropagation();
      if(!requireEditableLayout("edit connections"))return;
      const named=portMode==="named"&&portKey;
      if(side==="out"){
        pendingPort={nodeId,side,portIndex,portKey,portName,portMode:named?"named":"standard"};
        if(named)setStatus((portName||"Output")+" selected. Click a named input port or a standard Main / Skip / Extra input.");
        else{
          const lane=["Skip","Main","Extra"][portIndex]||"Lane";
          setStatus(lane+" output selected. Click the matching "+lane.toLowerCase()+" input.");
        }
        draw();return;
      }

      if(side==="in"&&pendingPort?.side==="out"){
        const pendingNamed=pendingPort.portMode==="named"&&pendingPort.portKey;
        if(named||pendingNamed){
          if(named&&pendingNamed){
            connect(pendingPort.nodeId,nodeId,"named","named_out:"+pendingPort.portKey,"named_in:"+portKey);
          }else if(named&&!pendingNamed){
            const srcLane=pendingPort.portIndex;
            const sourcePort=srcLane===0?"skip_out":(srcLane===2?"extra_out":"main_out");
            connect(pendingPort.nodeId,nodeId,"named",sourcePort,"named_in:"+portKey);
          }else{
            const lane=portIndex;
            const kind=lane===0?"residual":(lane===2?"aux":"main");
            const targetPort=lane===0?"skip_in":(lane===2?"extra_in":"main_in");
            connect(pendingPort.nodeId,nodeId,kind,"named_out:"+pendingPort.portKey,targetPort);
          }
          pendingPort=null;draw();return;
        }
        if(pendingPort.portIndex!==portIndex){
          const lane=["Skip","Main","Extra"][pendingPort.portIndex]||"Lane";
          setStatus("For a clean graph connect matching lanes: "+lane+" Out → "+lane+" In.");
          pendingPort=null;draw();return;
        }
        const lane=portIndex;
        const kind=lane===0?"residual":(lane===2?"aux":"main");
        const sourcePort=lane===0?"skip_out":(lane===2?"extra_out":"main_out");
        const targetPort=lane===0?"skip_in":(lane===2?"extra_in":"main_in");
        connect(pendingPort.nodeId,nodeId,kind,sourcePort,targetPort);
        pendingPort=null;draw();return;
      }

      pendingPort={nodeId,side,portIndex,portKey,portName,portMode:named?"named":"standard"};
      setStatus(named?(portName||"Input")+"  input selected. Choose a named or standard output port.":"Input selected. Choose an output from the same lane.");
      draw();
    }


    function splitPercentages(node){
      return {
        train:Math.max(0,Math.min(100,Number(fieldCurrentValue(node,"train_size")??90))),
        validation:Math.max(0,Math.min(100,Number(fieldCurrentValue(node,"validation_size")??5))),
        test:Math.max(0,Math.min(100,Number(fieldCurrentValue(node,"test_size")??5)))
      };
    }

    function splitTotal(node){
      const s=splitPercentages(node);
      return s.train+s.validation+s.test;
    }

    function splitIsValid(node){
      const s=splitPercentages(node);
      return s.train>0 && Math.abs((s.train+s.validation+s.test)-100)<0.0001;
    }

    function setSplitPreset(node,train,validation,test,label){
      checkpoint("Split preset "+label);
      node.params=node.params||{};
      node.params.train_size=train;
      node.params.validation_size=validation;
      node.params.test_size=test;
      setStatus("Split set to "+train+"% train / "+validation+"% validation / "+test+"% test.");
      draw();
    }

    function renderField(body,node,f){
      const wrap=document.createElement("div");wrap.className="mlb-field"+(f.type==="percent"?" mlb-percent-field":"");
      const label=document.createElement("label");label.textContent=f.label+(f.required?" *":"");
      let input;

      const commit=(value)=>{
        checkpoint("Edit "+node.name+"."+f.key);
        node.params=node.params||{};
        node.params[f.key]=f.type==="number"||f.type==="percent"?Number(value):value;
        if(node.type==="text_input" && f.key==="dataset_id"){
          const meta=preparedDatasetById(value);
          if(meta){
            const available=Object.keys(meta.splits||{});
            if(!available.includes(node.params.dataset_split)){
              node.params.dataset_split=meta.default_split||available[0]||"train";
            }
          }
        }
        if(node.type==="train_test_split"){
          const total=splitTotal(node);
          setStatus(splitIsValid(node)
            ?"Split valid: total 100%."
            :"Split needs attention: Train + Validation + Test = "+total+"%. It must equal 100%.");
        }else{
          setStatus(node.name+" settings updated.");
        }
        draw();
      };

      if(f.type==="dataset_select"){
        input=document.createElement("select");
        const datasets=availablePreparedDatasets();
        if(!datasets.length){
          const o=document.createElement("option");o.value="";o.textContent="No prepared datasets yet";
          input.appendChild(o);input.disabled=true;
        }else{
          datasets.forEach(meta=>{
            const o=document.createElement("option");o.value=meta.id;
            o.textContent=meta.name+" — "+compactDatasetSummary(meta);
            if(String(node.params?.[f.key]||"")===String(meta.id))o.selected=true;
            input.appendChild(o);
          });
          if(!node.params?.[f.key]){
            const latest=latestPreparedDataset();
            if(latest){node.params=node.params||{};node.params[f.key]=latest.id;input.value=latest.id;}
          }
          input.addEventListener("change",()=>{
            commit(input.value);
          });
        }
      }else if(f.type==="dataset_split_select"){
        input=document.createElement("select");
        const meta=preparedDatasetById(node.params?.dataset_id)||latestPreparedDataset();
        const splits=meta?Object.keys(meta.splits||{}):[];
        if(!splits.length){
          const o=document.createElement("option");o.value="";o.textContent="No splits available";
          input.appendChild(o);input.disabled=true;
        }else{
          splits.forEach(name=>{
            const o=document.createElement("option");o.value=name;o.textContent=datasetSplitLabel(name,meta);
            if(String(node.params?.[f.key]||meta.default_split||"train")===name)o.selected=true;
            input.appendChild(o);
          });
          input.addEventListener("change",()=>commit(input.value));
        }
      }else if(f.type==="percent"){
        const row=document.createElement("div");row.className="mlb-percent-row";
        const range=document.createElement("input");range.type="range";range.min=f.min??0;range.max=f.max??100;range.step=f.step??1;
        const number=document.createElement("input");number.type="number";number.min=f.min??0;number.max=f.max??100;number.step=f.step??1;
        const value=Number(node.params?.[f.key]??f.value??0);
        range.value=value;number.value=value;
        const suffix=document.createElement("span");suffix.className="mlb-percent-sign";suffix.textContent="%";
        range.addEventListener("input",()=>{number.value=range.value;});
        number.addEventListener("input",()=>{range.value=Math.max(Number(range.min),Math.min(Number(range.max),Number(number.value||0)));});
        range.addEventListener("change",()=>commit(range.value));
        number.addEventListener("change",()=>commit(Math.max(Number(number.min),Math.min(Number(number.max),Number(number.value||0)))));
        row.append(range,number,suffix);
        input=row;
      }else if(f.type==="select"){
        input=document.createElement("select");
        (f.options||[]).forEach(v=>{
          const o=document.createElement("option");o.value=v;o.textContent=v;
          if(String(node.params?.[f.key]??f.value)===String(v))o.selected=true;
          input.appendChild(o);
        });
        input.addEventListener("change",()=>commit(input.value));
      }else if(f.type==="textarea"){
        input=document.createElement("textarea");input.rows=4;input.value=node.params?.[f.key]??f.value??"";
        input.addEventListener("change",()=>commit(input.value));
      }else if(f.type==="bool"){
        input=document.createElement("select");
        ["true","false"].forEach(v=>{const o=document.createElement("option");o.value=v;o.textContent=v;if(String(node.params?.[f.key]??f.value)===v)o.selected=true;input.appendChild(o);});
        input.addEventListener("change",()=>commit(input.value));
      }else{
        input=document.createElement("input");input.type=f.type==="number"?"number":"text";input.step=f.step??"any";
        if(f.min!==undefined)input.min=f.min;if(f.max!==undefined)input.max=f.max;
        input.value=node.params?.[f.key]??f.value??"";
        input.addEventListener("change",()=>commit(input.value));
      }

      wrap.append(label,input);
      if(f.help){
        const help=document.createElement("div");help.className="mlb-field-help";help.textContent=f.help;wrap.appendChild(help);
      }
      body.appendChild(wrap);
    }

    function fieldCurrentValue(node,key){
      if(node.params && node.params[key]!==undefined)return node.params[key];
      const field=(cat(catalog,node.type).api||[]).find(x=>x.key===key);
      return field ? field.value : "";
    }

    function fieldVisible(node,f){
      if(f.show_when){
        return Object.entries(f.show_when).every(([k,v])=>String(fieldCurrentValue(node,k))===String(v));
      }
      if(f.show_when_any){
        return Object.entries(f.show_when_any).every(([k,values])=>{
          const current=String(fieldCurrentValue(node,k));
          return (values||[]).map(String).includes(current);
        });
      }
      return true;
    }

    function renderGroupedFields(body,node,fields){
      const groupOrder=[];
      (fields||[]).forEach(f=>{
        const group=f.group||"Settings";
        if(!groupOrder.includes(group))groupOrder.push(group);
      });

      groupOrder.forEach(group=>{
        const visible=(fields||[]).filter(f=>(f.group||"Settings")===group && fieldVisible(node,f));
        if(!visible.length)return;

        const collapsed=collapsedInspectorGroups.has(group);
        const header=document.createElement("button");
        header.type="button";
        header.className="mlb-ins-group";
        header.innerHTML="<span>"+group+"</span><span>"+(collapsed?"▸":"▾")+"</span>";
        header.addEventListener("click",()=>{
          if(collapsedInspectorGroups.has(group))collapsedInspectorGroups.delete(group);
          else collapsedInspectorGroups.add(group);
          draw();
        });
        body.appendChild(header);

        if(!collapsed){
          const section=document.createElement("div");
          section.className="mlb-ins-group-body";
          visible.forEach(f=>renderField(section,node,f));
          body.appendChild(section);
        }
      });
    }


    function namedUserPorts(node,side){
      if(node?.type!=="api_step")return null;
      const b=normalizeAPIBinding(node.api_binding||defaultAPIBinding());
      if(b.call_type!=="user_function"||b.port_mode!=="named")return null;
      return side==="in"?(b.input_ports||[]):(b.output_ports||[]);
    }

    function portButtons(node, side){
      const namedPorts=namedUserPorts(node,side);
      if(namedPorts){
        let html="";const count=Math.max(1,namedPorts.length);
        namedPorts.forEach((port,i)=>{
          const pct=((i+1)/(count+1))*100;
          const key=String(port.id||((side==="in"?"in_":"out_")+(i+1)));
          const name=apiSafePortName(port.name||(side==="in"?"input":"output"),side==="in"?"input":"output");
          const sideCss=side==="in"?"left:-6px":"right:-6px";
          const labelCss=side==="in"?"left:10px":"right:10px";
          html+='<button class="mlb-port '+side+' named-port" data-side="'+side+'" data-port-index="'+i+'" data-port-mode="named" data-port-key="'+key+'" data-port-name="'+name+'" style="'+sideCss+';top:'+pct+'%;transform:translateY(-50%)" type="button" aria-label="'+name+'" title="'+name+'"></button>';
          html+='<span class="mlb-user-port-label '+side+'" style="'+labelCss+';top:'+pct+'%;transform:translateY(-50%)">'+name+'</span>';
        });
        return html;
      }
      let html="";
      for(let i=0;i<3;i++){
        let style="";let posClass="";
        if(i===0){
          const left = side==="in" ? 28 : 72;
          style='left:'+left+'%;top:-6px;transform:translateX(-50%)';posClass="top-edge";
        }else if(i===1){style='top:50%;transform:translateY(-50%)';posClass="middle-side";}
        else{
          const left = side==="in" ? 28 : 72;
          style='left:'+left+'%;bottom:-6px;top:auto;transform:translateX(-50%)';posClass="bottom-edge";
        }
        html += '<button class="mlb-port '+side+' lane-'+i+' '+posClass+'" data-side="'+side+'" data-port-index="'+i+'" data-port-mode="standard" style="'+style+'" type="button" aria-label="'+portLabel(side,i)+'" title="'+portLabel(side,i)+'"></button>';
      }
      return html;
    }
    function nodeMiniFields(node,info){
      if(node.type==="text_input"){
        const mode=String(fieldCurrentValue(node,"input_mode")||"prompt");
        if(mode==="prepared_dataset"){
          const meta=preparedDatasetById(node.params?.dataset_id)||latestPreparedDataset();
          const split=node.params?.dataset_split||meta?.default_split||"train";
          return '<div class="mlb-mini-field"><span>Dataset</span><strong>'+(meta?.name||"No data")+'</strong></div>'+
                 '<div class="mlb-mini-field"><span>Split</span><strong>'+split+'</strong></div>';
        }
        return '<div class="mlb-mini-field"><span>Prompt</span><strong>'+String(node.params?.prompt||"Once upon a time")+'</strong></div>';
      }
      if(node.type==="train_test_split"){
        const s=splitPercentages(node),total=s.train+s.validation+s.test;
        return '<div class="mlb-mini-field"><span>Train</span><strong>'+s.train+'%</strong></div>'+ 
               '<div class="mlb-mini-field"><span>Validation</span><strong>'+s.validation+'%</strong></div>'+ 
               '<div class="mlb-mini-field"><span>Test</span><strong>'+s.test+'%</strong></div>'+ 
               '<div class="mlb-mini-field"><span>Total</span><strong>'+(total===100?'✓ 100%':'! '+total+'%')+'</strong></div>';
      }
      const api=apiInfo(node);const source=(api.parameters||info.api||[]).slice(0,4);
      return source.map(f=>{
        let v=node.params?.[f.key];if(v===undefined||v===null||v==="")v=f.value;
        if(v===undefined||v===null||v==="")return "";
        return '<div class="mlb-mini-field"><span>'+f.label+'</span><strong>'+String(v)+'</strong></div>';
      }).join("");
    }

    function drawEdges(wrap,flow){
      if(!wrap||!flow||!wrap.isConnected||!flow.isConnected)return;
      // A draw() rebuild can be followed by font/layout/scrollbar changes in
      // notebook hosts. Always replace the previous edge layer so connections
      // reflect the final measured port positions instead of a stale first frame.
      wrap.querySelectorAll(":scope > .mlb-edge-layer").forEach(el=>el.remove());
      const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");
      svg.setAttribute("class","mlb-edge-layer");
      const scaledRect=wrap.getBoundingClientRect();
      svg.setAttribute("width",Math.max(wrap.clientWidth,scaledRect.width));
      svg.setAttribute("height",Math.max(wrap.clientHeight,scaledRect.height,650));
      wrap.appendChild(svg);
      const wr=wrap.getBoundingClientRect();
      let skipRoute=0, extraRoute=0;

      function portRect(nodeEl,side,index,key=""){
        const selector=key
          ?('.mlb-port[data-side="'+side+'"][data-port-key="'+key+'"]')
          :('.mlb-port[data-side="'+side+'"][data-port-index="'+index+'"]');
        const el=nodeEl.querySelector(selector);
        return el ? el.getBoundingClientRect() : nodeEl.getBoundingClientRect();
      }

      function laneOf(e){
        if(e.kind==="named"||String(e.target_port||"").startsWith("named_in:")) return "named";
        if(e.kind==="residual") return 0;
        if(e.kind==="aux") return 2;
        const sp=String(e.source_port||"");
        if(sp.includes("skip")||sp.includes("res_")) return 0;
        if(sp.includes("extra")) return 2;
        return 1;
      }

      (current(state).edges||[]).forEach(e=>{
        const a=flow.querySelector('[data-node-id="'+e.source+'"]');
        const b=flow.querySelector('[data-node-id="'+e.target+'"]');
        if(!a||!b)return;
        const lane=laneOf(e);
        const sourcePortText=String(e.source_port||"");const targetPortText=String(e.target_port||"");
        const sourceKey=sourcePortText.startsWith("named_out:")?sourcePortText.replace(/^named_out:/,""):"";
        const targetKey=targetPortText.startsWith("named_in:")?targetPortText.replace(/^named_in:/,""):"";
        const routeIndex=lane==="named"?1:lane;
        const sourceIndex=sourceKey?routeIndex:(sourcePortText.includes("skip")?0:(sourcePortText.includes("extra")?2:routeIndex));
        const targetIndex=targetKey?routeIndex:(targetPortText.includes("skip")?0:(targetPortText.includes("extra")?2:routeIndex));
        const ar=portRect(a,"out",sourceIndex,sourceKey), br=portRect(b,"in",targetIndex,targetKey);
        const x1=ar.left-wr.left+ar.width/2, y1=ar.top-wr.top+ar.height/2;
        const x2=br.left-wr.left+br.width/2, y2=br.top-wr.top+br.height/2;
        const p=document.createElementNS("http://www.w3.org/2000/svg","path");
        p.setAttribute("data-edge-id",e.id);

        if(lane==="named"){
          const gap=Math.max(1,Math.abs(x2-x1));const dir=x2>=x1?1:-1;const handle=Math.max(18,Math.min(58,gap*0.42));
          p.setAttribute("d",`M ${x1} ${y1} C ${x1+dir*handle} ${y1}, ${x2-dir*handle} ${y2}, ${x2} ${y2}`);
          p.setAttribute("class","mlb-edge-main mlb-edge-named");
        }else if(lane===0){
          const ab=a.getBoundingClientRect(), bb=b.getBoundingClientRect();
          const route=skipRoute++;
          const topY=Math.min(ab.top-wr.top,bb.top-wr.top)-34-(route%4)*16;
          p.setAttribute("d",`M ${x1} ${y1} C ${x1+16} ${y1}, ${x1+16} ${topY}, ${x1+34} ${topY} L ${x2-34} ${topY} C ${x2-16} ${topY}, ${x2-16} ${y2}, ${x2} ${y2}`);
          p.setAttribute("class","mlb-edge-skip");
        }else if(lane===2){
          const ab=a.getBoundingClientRect(), bb=b.getBoundingClientRect();
          const route=extraRoute++;
          const bottomY=Math.max(ab.bottom-wr.top,bb.bottom-wr.top)+34+(route%4)*16;
          p.setAttribute("d",`M ${x1} ${y1} C ${x1+16} ${y1}, ${x1+16} ${bottomY}, ${x1+34} ${bottomY} L ${x2-34} ${bottomY} C ${x2-16} ${bottomY}, ${x2-16} ${y2}, ${x2} ${y2}`);
          p.setAttribute("class","mlb-edge-extra");
        }else{
          // Main lane routing:
          // 1) Neighboring blocks connect directly through the empty gap between
          //    their side ports.
          // 2) Only links that jump across another block (or travel backwards)
          //    are moved onto an outside rail.
          const ab=a.getBoundingClientRect(), bb=b.getBoundingClientRect();
          const left=Math.min(x1,x2), right=Math.max(x1,x2);
          let lowest=Math.max(ab.bottom-wr.top,bb.bottom-wr.top);
          let blocked=false;
          flow.querySelectorAll(".mlb-node").forEach(nodeEl=>{
            if(nodeEl===a||nodeEl===b)return;
            const nr=nodeEl.getBoundingClientRect();
            const nl=nr.left-wr.left, nrgt=nr.right-wr.left;
            if(nrgt>left+4&&nl<right-4){
              blocked=true;
              lowest=Math.max(lowest,nr.bottom-wr.top);
            }
          });

          const forward=x2>x1;
          if(forward&&!blocked){
            // Clean side-to-side connector for adjacent blocks. The control
            // points stay inside the inter-card gap, so the wire never dives
            // under the cards just to connect immediate neighbors.
            const gap=Math.max(1,x2-x1);
            const handle=Math.max(12,Math.min(44,gap*0.42));
            p.setAttribute("d",`M ${x1} ${y1} C ${x1+handle} ${y1}, ${x2-handle} ${y2}, ${x2} ${y2}`);
          }else{
            // Long/return connections use a lower rail so they cannot cut
            // through any component between source and target.
            const routeY=lowest+24;
            const dir=x2>=x1?1:-1;
            const exitX=x1+dir*18;
            const entryX=x2-dir*18;
            const corner=10;
            const down1=Math.max(y1+corner,routeY-corner);
            const down2=Math.max(y2+corner,routeY-corner);
            p.setAttribute("d",
              `M ${x1} ${y1} `+
              `C ${x1+dir*8} ${y1}, ${exitX} ${y1}, ${exitX} ${y1+corner} `+
              `L ${exitX} ${down1} `+
              `Q ${exitX} ${routeY} ${exitX+dir*corner} ${routeY} `+
              `L ${entryX-dir*corner} ${routeY} `+
              `Q ${entryX} ${routeY} ${entryX} ${down2} `+
              `L ${entryX} ${y2+corner} `+
              `C ${entryX} ${y2}, ${x2-dir*8} ${y2}, ${x2} ${y2}`
            );
          }
          p.setAttribute("class","mlb-edge-main");
        }
        svg.appendChild(p);
      });
    }

    function loadTextDataStarter(){
      checkpoint("Load Default Data Pipeline");
      rememberWorkspaceView();
      state.active_workspace="data";
      const ws=state.workspaces.data;
      state.view_component_id=ws.root_component_id;
      state.breadcrumbs=[{id:ws.root_component_id,name:"Data Processing"}];
      ws.view_component_id=ws.root_component_id;
      ws.breadcrumbs=cp(state.breadcrumbs);

      const starter=defaultDataNodes();
      state.components[ws.root_component_id]={
        id:ws.root_component_id,
        name:"Data Processing",
        kind:"data",
        revision:1,
        nodes:starter.nodes,
        edges:starter.edges
      };
      selected=null;pendingPort=null;
      execution={status:"idle",overall:0,message:"Ready",nodes:{}};
      setStatus("Default pipeline restored: Hugging Face → Clean → Train/Val/Test → Tokenize → Prepared Dataset.");
      switchingWorkspace=true;
      draw();
    }

    function loadTinyStories(){
      checkpoint("Load TinyStories 30M");
      rememberWorkspaceView();
      state.active_workspace="model";
      const rootId=state.workspaces.model.root_component_id;
      state.root_component_id=rootId;
      state.view_component_id=rootId;
      state.project={
        ...(state.project||{}),
        name:"TinyStories 30M",
        context_length:512,
        batch_size:16,
        model_settings:{
          embedding_size:384,
          heads:6,
          block:512,
          default_batch:16,
          vocab_size:32000,
          precision:"fp16"
        },
        dataset:"TinyStories",
        estimated_parameters:"~30M"
      };
      state.breadcrumbs=[{id:rootId,name:"TinyStories 30M"}];
      state.workspaces.model.view_component_id=rootId;
      state.workspaces.model.breadcrumbs=cp(state.breadcrumbs);
      const defId=uid("custom");
      const esa=makeNode(cat(catalog,"esa")),norm=makeNode(cat(catalog,"rmsnorm")),ffn=makeNode(cat(catalog,"ffn")),res=makeNode(cat(catalog,"residual"));
      state.custom_components[defId]={
        id:defId,
        name:"TinyStories ESA Block",
        revision:1,
        description:"ESA → RMSNorm → FFN → Residual",
        input_count:3,
        output_count:3,
        nodes:[esa,norm,ffn,res],
        edges:[
          edge(esa.id,norm.id),
          edge(norm.id,ffn.id),
          Object.assign(edge(ffn.id,res.id),{source_port:"main_out",target_port:"main_in"}),
          Object.assign(edge(esa.id,res.id,"residual"),{source_port:"skip_out",target_port:"skip_in"})
        ]
      };
      const nodes=[];
      const input=makeNode(cat(catalog,"text_input"));configureTextInputForLatest(input);nodes.push(input);
      const emb=makeNode(cat(catalog,"embedding"));nodes.push(emb);
      for(let i=1;i<=6;i++)nodes.push({id:uid("node"),type:"custom",name:"Layer "+i,definition_id:defId,repeat:1,params:{},input_count:3,output_count:3,position:{x:0,y:0}});
      const head=makeNode(cat(catalog,"lm_head")),out=makeNode(cat(catalog,"text_output"));nodes.push(head,out);
      const edges=[];for(let i=0;i<nodes.length-1;i++)edges.push(edge(nodes[i].id,nodes[i+1].id));
      state.components[rootId]={id:rootId,name:"TinyStories 30M",kind:"model",revision:1,nodes,edges};
      syncModelSettingsToGraph(state.project.model_settings,state.project.model_settings);
      selected=null;pendingPort=null;setStatus("TinyStories starter loaded.");draw();
    }

    function loadSequentialPrebuiltModel(spec){
      checkpoint("Load "+spec.name);rememberWorkspaceView();state.active_workspace="model";
      const rootId=state.workspaces.model.root_component_id;state.root_component_id=rootId;state.view_component_id=rootId;
      state.project={...(state.project||{}),name:spec.name,context_length:spec.block,batch_size:spec.batch,
        model_settings:{embedding_size:spec.dim,heads:spec.heads,block:spec.block,default_batch:spec.batch,vocab_size:spec.vocab,precision:spec.precision||"fp16"},
        dataset:spec.dataset??null,estimated_parameters:spec.parameters,description:spec.description||""};
      state.breadcrumbs=[{id:rootId,name:spec.name}];state.workspaces.model.view_component_id=rootId;state.workspaces.model.breadcrumbs=cp(state.breadcrumbs);
      const input=makeNode(cat(catalog,"text_input"));configureTextInputForLatest(input);
      const emb=makeNode(cat(catalog,"embedding"));emb.name="Token Embedding";emb.params={...(emb.params||{}),vocab_size:spec.vocab,embedding_dim:spec.dim,hidden_size:spec.dim,dim:spec.dim,dtype:precisionToDtype(spec.precision||"fp16")};
      const core=makeNode(cat(catalog,spec.coreType));core.name=spec.coreName;core.params={...(core.params||{}),...cp(spec.coreParams||{})};
      const norm=makeNode(cat(catalog,"rmsnorm"));norm.name="Final RMSNorm";norm.params={...(norm.params||{}),normalized_shape:spec.dim,hidden_size:spec.dim,dim:spec.dim,eps:1e-6,elementwise_affine:true};
      const head=makeNode(cat(catalog,"lm_head"));head.name="LM Head";head.params={...(head.params||{}),hidden_size:spec.dim,dim:spec.dim,vocab_size:spec.vocab,bias:false,tie_embeddings:true};
      const out=makeNode(cat(catalog,"text_output"));const nodes=[input,emb,core,norm,head,out];const edges=[];for(let i=0;i<nodes.length-1;i++)edges.push(edge(nodes[i].id,nodes[i+1].id));
      state.components[rootId]={id:rootId,name:spec.name,kind:"model",revision:1,nodes,edges};syncModelSettingsToGraph(state.project.model_settings,state.project.model_settings);
      selected=null;pendingPort=null;setStatus(spec.name+" loaded.");draw();
    }

    function loadStateAwareESA200M(){loadSequentialPrebuiltModel({name:"StateAware ESA 200M",parameters:"199,982,344",description:"Notebook-matched 8-layer StateAware ESA model",dataset:null,
      dim:384,heads:6,block:256,batch:16,vocab:50257,precision:"fp16",coreType:"stateaware_esa_stack",coreName:"StateAware ESA ×8",
      coreParams:{dim:384,state_dim:2749,layers:8,heads:6,block:256,batch:16,depth_dim:64,compass:16,update_ratio_start:0.20,update_ratio_end:0.14,stream_ratio:1.08}});}

    function loadSOUP200M(){loadSequentialPrebuiltModel({name:"SOUP 200M",parameters:"199,916,160",description:"Notebook-matched SOUP 200M with three physical layers",dataset:null,
      dim:1152,heads:18,block:256,batch:16,vocab:50257,precision:"fp16",coreType:"soup",coreName:"SOUP ×3",
      coreParams:{dim:1152,width:2864,depth:3,mixer:"esa",ffn:"saffn",mixer_config:{head:18,batch:16,block:256,compass:16,auto_compile:false},ffn_config:{depth_dim:128},memory_dim:256,fusion_hidden:1728}});}

    function loadSOUP30M1L(){loadSequentialPrebuiltModel({name:"SOUP 30M 1L",parameters:"30,003,528",description:"One-layer SOUP causal LM at ~30M parameters",dataset:"TinyStories",
      dim:384,heads:6,block:512,batch:16,vocab:50257,precision:"fp16",coreType:"soup",coreName:"SOUP ×1",
      coreParams:{dim:384,width:1408,depth:1,mixer:"esa",ffn:"saffn",mixer_config:{head:6,batch:16,block:512,compass:16,auto_compile:false},ffn_config:{depth_dim:64},memory_dim:128,fusion_hidden:928}});}

    function safeFilename(name){
      const base=String(name||"mlbricks-design").trim().replace(/[^a-zA-Z0-9._-]+/g,"-").replace(/^-+|-+$/g,"");
      return base||"mlbricks-design";
    }

    function sanitizedProjectState(){
      const clean=cp(state);
      delete clean._runtime_command;
      delete clean._session_secrets;
      return clean;
    }

    function designPayload(){
      rememberWorkspaceView();
      return {
        format:"mlb-studio-design",
        format_version:"1.0.0",
        builder_version:"1.0.0",
        saved_at:new Date().toISOString(),
        state:sanitizedProjectState()
      };
    }

    function downloadDesignBlob(blob,filename){
      const url=URL.createObjectURL(blob);
      const a=document.createElement("a");a.href=url;a.download=filename;document.body.appendChild(a);a.click();a.remove();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
    }

    function saveDesignBin(){
      const json=JSON.stringify(designPayload());
      const magic=new TextEncoder().encode("MLBRICKS-BIN-1\n");
      const payloadBytes=new TextEncoder().encode(json);
      const bytes=new Uint8Array(magic.length+payloadBytes.length);
      bytes.set(magic,0);bytes.set(payloadBytes,magic.length);
      downloadDesignBlob(new Blob([bytes],{type:"application/octet-stream"}),safeFilename(state.project?.name)+".mlbricks.bin");
      setStatus("Binary design saved.");draw();
    }

    function saveDesign(){
      const blob=new Blob([JSON.stringify(designPayload(),null,2)],{type:"application/json"});
      downloadDesignBlob(blob,safeFilename(state.project?.name)+".mlbricks.json");
      setStatus("JSON design saved.");draw();
    }

    function saveDesignChoice(){
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const choice=(win&&typeof win.prompt==="function")
        ? String(win.prompt('Save project as "bin" or "json":','bin')||"").trim().toLowerCase()
        : "bin";
      if(!choice){setStatus("Save cancelled.");return;}
      if(choice==="bin"||choice==="binary"||choice==="b")return saveDesignBin();
      if(choice==="json"||choice==="j")return saveDesign();
      setStatus('Unknown save type: '+choice+' — use "bin" or "json".');
    }

    function exportWorkspace(){
      if(state.active_workspace==="model"){
        const model=modelRootComponent();
        if(model){downloadModelConfig();return;}
      }
      const payload={
        format:"mlbricks-export",
        builder_version:"1.0.0",
        workspace:state.active_workspace,
        project:cp(state.project||{}),
        prepared_datasets:cp(state.prepared_datasets||[]),
        model_outputs:cp(state.model_outputs||[]),
        project_files:cp(state.project_files||[]),
        current_component:cp(current(state))
      };
      const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
      downloadDesignBlob(blob,safeFilename(state.project?.name||workspaceName())+"."+state.active_workspace+".export.json");
      setStatus((state.active_workspace==="model"?"Model":"Data")+" export downloaded.");
      draw();
    }

    async function shareWorkspace(){
      const lines=[
        "MLB Studio — "+(state.project?.name||workspaceName()),
        "Version: 1.0.0",
        "Workspace: "+workspaceName(),
        "Nodes: "+(current(state).nodes||[]).length,
        "Connections: "+(current(state).edges||[]).length
      ];
      const activeModel=selectedOutputModel()||builtModelById(outputDirectorySelection)||((state.model_outputs||[]).slice(-1)[0]||null);
      const serve=activeModel?.serve_runtime||{};
      const url=serve.public_url||serve.local_url||activeModel?.serve_urls?.public_url||activeModel?.serve_urls?.local_url||"";
      if(url)lines.push("Access URL: "+url);
      lines.push("Tip: use Save to send the full .mlbricks project file.");
      await copyTextRobust(lines.join("\n"),"share summary");
    }

    function showQuickHelp(){
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const help=[
        'MLB Studio V1.0',
        '',
        '• Add components or data steps from the left library.',
        '• Export downloads a model config or workspace export file.',
        '• Load opens .mlbricks.json or .mlbricks.bin files.',
        '• Select a node to edit config and read what it does in Inspector.',
      ].join('\n');
      if(win&&typeof win.alert==="function")win.alert(help);
      setStatus("Help opened.");
    }

    function openBuilderSettings(){
      const win=(root.ownerDocument&&root.ownerDocument.defaultView)||window;
      const currentName=state.project?.name||"Untitled Model";
      const nextName=win&&typeof win.prompt==="function"
        ? win.prompt("Project name:",currentName)
        : currentName;
      if(nextName===null){setStatus("Settings unchanged.");return;}
      const cleaned=String(nextName||"").trim();
      if(!cleaned){setStatus("Project name cannot be empty.");return;}
      checkpoint("Update project settings");
      state.project=state.project||{};
      state.project.name=cleaned;
      const modelId=state.workspaces?.model?.root_component_id||state.root_component_id;
      if(modelId&&state.components?.[modelId]&&!state.components[modelId].name)state.components[modelId].name=cleaned;
      if(Array.isArray(state.breadcrumbs)&&state.breadcrumbs.length)state.breadcrumbs[0].name=cleaned;
      if(state.workspaces?.model?.breadcrumbs?.length)state.workspaces.model.breadcrumbs[0].name=cleaned;
      setStatus("Project settings updated.");
      draw();
    }

    function loadDesign(){
      const input=document.createElement("input");
      input.type="file";
      input.accept=".json,.mlbricks,.bin,.mlbricks.bin,application/json,application/octet-stream";
      input.style.display="none";
      input.addEventListener("change",()=>{
        const file=input.files?.[0];
        if(!file){input.remove();return;}
        const reader=new FileReader();
        reader.onload=()=>{
          try{
            const bytes=new Uint8Array(reader.result);
            const magic="MLBRICKS-BIN-1\n";
            const magicBytes=new TextEncoder().encode(magic);
            let isBin=bytes.length>=magicBytes.length;
            for(let i=0;i<magicBytes.length&&isBin;i++) if(bytes[i]!==magicBytes[i])isBin=false;
            const text=new TextDecoder().decode(isBin?bytes.slice(magicBytes.length):bytes);
            const parsed=JSON.parse(text);
            const incoming=parsed.state||parsed;
            if(!incoming||!incoming.components||!incoming.root_component_id) throw new Error("This file is not an MLB Studio design.");
            checkpoint("Load design");
            state=cp(incoming);
            Object.values(state.components||{}).forEach(c=>{if(!c.edges)c.edges=[];});
            ensureWorkspaces();
            if(!state.view_component_id||!state.components[state.view_component_id])state.view_component_id=state.root_component_id;
            if(!Array.isArray(state.breadcrumbs)||!state.breadcrumbs.length)state.breadcrumbs=[{id:state.root_component_id,name:state.project?.name||"Model"}];
            selected=null;pendingPort=null;switchingWorkspace=true;
            setStatus((isBin?"Binary":"JSON")+" design loaded: "+file.name);
            draw();
          }catch(err){
            alert("Could not load design: "+err.message);setStatus("Design load failed.");draw();
          }finally{input.remove();}
        };
        reader.readAsArrayBuffer(file);
      });
      document.body.appendChild(input);input.click();
    }

    function draw(){
      if(pointerInteractionActive){
        deferredInteractionDraw=true;
        return;
      }
      if(bottomView==="hub")bottomView="cloud";
      const wsKey=state.active_workspace||"model";
      const oldCanvas=root.querySelector(".mlb-canvas");
      if(oldCanvas && !switchingWorkspace){
        workspaceScroll[wsKey]={left:oldCanvas.scrollLeft,top:oldCanvas.scrollTop};
      }
      const oldSidebar=root.querySelector(".mlb-sidebar");
      if(oldSidebar){
        sidebarScroll[wsKey]={left:oldSidebar.scrollLeft,top:oldSidebar.scrollTop};
      }
      const oldInspectorBody=root.querySelector(".mlb-ins-body");
      if(oldInspectorBody && lastInspectorRenderKey){
        inspectorScrollPositions[lastInspectorRenderKey]={left:oldInspectorBody.scrollLeft,top:oldInspectorBody.scrollTop};
      }
      switchingWorkspace=false;
      rememberWorkspaceView();
      root.innerHTML="";
      root.classList.toggle("mlb-custom-editor-active",current(state)?.kind==="custom_edit");
      const runtimeWorkspaceActive=!!(runtimePanel && state.active_workspace==="model" && !galleryWorkspace.open && !cloudWorkspace.open);
      root.classList.toggle("mlb-runtime-page-active",runtimeWorkspaceActive);

      // Top bar
      const top=document.createElement("div");top.className="mlb-topbar";
      const frontendVersion=root.dataset.mlbricksBuilderVersion||"1.0.0";

      const topLeft=document.createElement("div");topLeft.className="mlb-top-left";
      const logo=document.createElement("div");logo.className="mlb-logo";
      const logoBrand=document.createElement("img");logoBrand.className="mlb-logo-brand";logoBrand.alt="MLBRICKS STUDIO";logoBrand.src="data:image/png;base64,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";logo.appendChild(logoBrand);
      const versionBadge=document.createElement("span");versionBadge.className="mlb-beta";versionBadge.textContent="V1.0";logo.appendChild(versionBadge);
      const title=document.createElement("div");
      const focusedCustom=current(state)?.kind==="custom_edit";
      title.className="mlb-project-title"+(focusedCustom?"":" mlb-project-title-editable");
      title.textContent=focusedCustom
        ?(state.breadcrumbs?.[0]?.name||activeCustomDefinition()?.name||"Module")
        :(state.project?.name||"Untitled Model");
      title.spellcheck=false;
      if(focusedCustom){
        title.title="Outermost item currently being edited";
      }else{
        title.contentEditable="true";
        title.title="Click to rename model";
        title.setAttribute("role","textbox");
        title.setAttribute("aria-label","Model name");
        title.addEventListener("focus",()=>{title.dataset.originalName=title.textContent||"";});
        title.addEventListener("keydown",ev=>{
          if(ev.key==="Enter"){ev.preventDefault();title.blur();}
          else if(ev.key==="Escape"){
            ev.preventDefault();title.textContent=title.dataset.originalName||state.project?.name||"Untitled Model";title.blur();
          }
        });
        title.addEventListener("blur",()=>renameProjectInline(title.textContent));
      }
      const saved=document.createElement("div");saved.className="mlb-save-state";saved.textContent="• Saved";
      topLeft.append(logo,title,saved);
      top.appendChild(topLeft);

      // Build/Gallery belong to model/data workspaces, not the focused Module/API editor.
      const primary=document.createElement("div");primary.className="mlb-top-primary";
      const modelRuntimeBusy=state.active_workspace==="model" && execution.status==="running" &&
        (execution.runtime_kind==="train"||execution.runtime_kind==="generate");
      const dataFetchBusy=state.active_workspace==="data"&&execution.status==="running"&&execution.runtime_kind==="data";
      if(current(state)?.kind!=="custom_edit"){
        if(runtimeWorkspaceActive){
          const mode=runtimePanel?.mode||"train";
          const runtimeActuallyBusy=mode==="train"?trainingIsRunning():(execution.status==="running" && execution.runtime_kind===mode);
          if(runtimeActuallyBusy){
            const runtimeLabel=mode==="train"?"Training":mode==="generate"?"Generating":"Serving";
            const runtimeIndicator=actionBtn(runtimeLabel,"mlb-run mlb-build mlb-top-build-tab runtime-busy "+mode,"activity");
            runtimeIndicator.disabled=true;
            runtimeIndicator.title=runtimeLabel+" in progress";
            runtimeIndicator.setAttribute("aria-disabled","true");
            primary.appendChild(runtimeIndicator);
          }
        }else{
          const run=state.active_workspace==="model"
            ?actionBtn(
                modelRuntimeBusy
                  ?(execution.runtime_kind==="train"?"Training":"Generating")
                  :"Build",
                "mlb-run mlb-build mlb-top-build-tab"+(modelRuntimeBusy?" runtime-busy "+execution.runtime_kind:""),
                modelRuntimeBusy?"activity":"build"
              )
            :actionBtn(
                dataFetchBusy?"Fetching":"Fetch Data",
                "mlb-run mlb-build mlb-top-build-tab"+(dataFetchBusy?" runtime-busy data":""),
                dataFetchBusy?"activity":"fetch"
              );
          run.disabled=modelRuntimeBusy||dataFetchBusy;
          run.addEventListener("click",()=>{
            if(galleryWorkspace.open){closeGallery();return;}
            if(cloudWorkspace.open){closeCloudWorkspace();return;}
            (state.active_workspace==="model"?requestModelBuild:requestRun)();
          });
          primary.appendChild(run);
          if(state.active_workspace==="data"||modelRuntimeBusy){
            const stopBtn=actionBtn("Stop","mlb-stop mlb-center-stop","stop");
            stopBtn.addEventListener("click",requestStop);
            stopBtn.style.display=(dataFetchBusy||modelRuntimeBusy)?"inline-flex":"none";
            primary.appendChild(stopBtn);
          }
          const galleryBtn=actionBtn("Gallery","mlb-dark-btn mlb-top-gallery-btn"+(galleryWorkspace.open?" active":""),"gallery");
          galleryBtn.title="Open prebuilt Models, Components and Data";
          galleryBtn.addEventListener("click",()=>{
            if(galleryWorkspace.open){closeGallery();return;}
            openGallery(state.active_workspace==="data"?"data":"models");
          });
          primary.appendChild(galleryBtn);
        }
      }
      top.appendChild(primary);

      const acts=document.createElement("div");acts.className="mlb-top-actions";
      const fullBtn=!isPopout?document.createElement("a"):null;
      if(!runtimeWorkspaceActive){
        const undoBtn=btn("↶ Undo","mlb-dark-btn mlb-history-btn");undoBtn.disabled=undoStack.length===0;undoBtn.title=current(state)?.kind==="custom_edit"?"Undo last edit in this editor":"Undo last model edit";undoBtn.addEventListener("click",undo);
        const redoBtn=btn("↷ Redo","mlb-dark-btn mlb-history-btn");redoBtn.disabled=redoStack.length===0;redoBtn.title=current(state)?.kind==="custom_edit"?"Redo last edit in this editor":"Redo last undone edit";redoBtn.addEventListener("click",redo);
        const clearBtn=btn("↻ Clear","mlb-dark-btn");clearBtn.disabled=layoutIsLocked();clearBtn.addEventListener("click",()=>{
          if(!requireEditableLayout("clear components"))return;
          const c=current(state);if(!c.nodes.length&&!c.edges.length)return;
          checkpoint("Clear graph");c.nodes=[];c.edges=[];selected=null;pendingPort=null;setStatus("Graph cleared.");draw();
        });
        acts.append(undoBtn,redoBtn,clearBtn);
      }

      if(current(state)?.kind!=="custom_edit"){
        const cloudBtn=actionBtn("Cloud & Repositories","mlb-dark-btn mlb-top-cloud-btn mlb-top-cloud-action"+(cloudWorkspace.open?" active":""),"cloud");
        const lockCloud=runtimeWorkspaceActive && trainingIsRunning();
        cloudBtn.disabled=lockCloud;
        cloudBtn.title=lockCloud?"Cloud & Repositories is unavailable while training":"Open Cloud & Repositories";
        cloudBtn.addEventListener("click",()=>{
          if(lockCloud)return;
          cloudWorkspace.open?closeCloudWorkspace():openCloudWorkspace();
        });
        acts.appendChild(cloudBtn);
      }

      if(fullBtn){
        fullBtn.className="mlb-dark-btn mlb-full-window-btn";
        fullBtn.textContent="↗ Full Window";
        fullBtn.href="#";
        fullBtn.title="Open MLB Studio in a separate full-window browser tab";
        fullBtn.addEventListener("click",activateFullWindowLink);
        // Keep Full Window visible in notebook/Kaggle runtime pages too.
        acts.appendChild(fullBtn);
      }
      top.appendChild(acts);
      root.appendChild(top);

      const shell=document.createElement("div");shell.className="mlb-shell";

      // Sidebar
      const side=document.createElement("aside");side.className="mlb-sidebar";
      if(runtimeWorkspaceActive)side.classList.add("mlb-sidebar-hidden");
      const head=document.createElement("div");head.className="mlb-sidehead";head.innerHTML="<span>"+(state.active_workspace==="data"?"DATA LIBRARY":"COMPONENT LIBRARY")+"</span><span>×</span>";side.appendChild(head);
      const sr=document.createElement("div");sr.className="mlb-search-row";
      const searchInput=document.createElement("input");searchInput.className="mlb-search";searchInput.placeholder="Search...";searchInput.setAttribute("aria-label",state.active_workspace==="data"?"Search data steps":"Search components");searchInput.value=search;searchInput.addEventListener("input",()=>{
        search=searchInput.value;
        searchFocusRestore={start:searchInput.selectionStart??search.length,end:searchInput.selectionEnd??search.length};
        draw();
      });
      sr.appendChild(searchInput);side.appendChild(sr);
      if(current(state)?.kind!=="custom_edit"){
        const workspaceBox=document.createElement("div");workspaceBox.className="mlb-workspace-box";
        const workspaceLabel=document.createElement("label");workspaceLabel.textContent="BUILD WORKSPACE";
        const workspaceSelect=document.createElement("select");workspaceSelect.className="mlb-workspace-select";
        [["model","Model Builder"],["data","Data Processing"]].forEach(([value,label])=>{
          const o=document.createElement("option");o.value=value;o.textContent=label;
          if(state.active_workspace===value)o.selected=true;
          workspaceSelect.appendChild(o);
        });
        workspaceSelect.addEventListener("change",()=>switchWorkspace(workspaceSelect.value));
        workspaceBox.append(workspaceLabel,workspaceSelect);
        side.insertBefore(workspaceBox,sr);
      }

      const apiComposerMode=isApiComposerView();
      if(apiComposerMode){
        const modeBox=document.createElement("div");modeBox.className="mlb-api-composer-sidebar-note";
        modeBox.innerHTML="<strong>API COMPONENT</strong><span>Add API functions from the Inspector, Components below, or saved Modules from My Modules.</span><span>Example: API half → <b>FFN</b> → API half.</span>";
        side.appendChild(modeBox);
      }
      const visible=catalog.filter(item=>{
        // Some MLBricks APIs are code-level composition containers rather than
        // visual Builder components. Keep them in the catalog/API registry for
        // backward compatibility, but do not show them in the component library.
        if(item.library_hidden===true)return false;
        if(itemWorkspace(item)!==state.active_workspace)return false;
        if(apiComposerMode&&!apiComposerAllowsCatalogItem(item))return false;
        const q=(item.name+" "+item.description+" "+item.category).toLowerCase();
        return !search || q.includes(search.toLowerCase());
      });

      [...new Set(visible.map(x=>x.category))].forEach(category=>{
        // Search results always expand so a matching component can never be hidden.
        const collapsed=collapsedCategories.has(category) && !search;
        const h=document.createElement("button");
        h.type="button";
        h.className="mlb-category";
        h.setAttribute("aria-expanded",String(!collapsed));
        h.innerHTML="<span>"+category+"</span><span class='mlb-category-caret'>"+(collapsed?"▸":"▾")+"</span>";
        h.addEventListener("click",()=>{
          if(collapsedCategories.has(category)) collapsedCategories.delete(category);
          else collapsedCategories.add(category);
          draw();
        });
        side.appendChild(h);

        const pal=document.createElement("div");
        pal.className="mlb-palette"+(collapsed?" collapsed":"");
        if(!collapsed){
          visible.filter(x=>x.category===category).forEach(item=>{
            const b=document.createElement("button");b.type="button";b.dataset.type=item.type||"";
            const ico=document.createElement("span");ico.className="mlb-pal-icon";ico.textContent=compactIconLabel(item.icon||"ML");
            const text=document.createElement("span");text.innerHTML="<strong>"+item.name+'</strong><span class="mlb-pal-sub">'+(item.description||"MLBricks component")+"</span>";
            b.append(ico,text);b.disabled=layoutIsLocked();b.title=layoutIsLocked()?"Layout locked — click Edit Layout first":"Add "+item.name;b.addEventListener("click",()=>addPrimitive(item));pal.appendChild(b);
          });
        }
        side.appendChild(pal);
      });

      if(isGraphCustomEditor()){
        const nestedEntries=(state.gallery?.components||[]).filter(entry=>{
          if(!entry?.definition)return false;
          const parent=activeCustomDefinition();
          if(entry.id===parent?.gallery_entry_id||entry.source_definition_id===parent?.id||entry.definition.id===parent?.id)return false;
          const q=(String(entry.name||entry.definition.name||"")+" saved component").toLowerCase();
          return !search||q.includes(search.toLowerCase());
        });
        if(nestedEntries.length){
          const sh=document.createElement("button");sh.type="button";sh.className="mlb-category";sh.setAttribute("aria-expanded","true");
          sh.innerHTML="<span>SAVED MODULES / API</span><span class='mlb-category-caret'>▾</span>";side.appendChild(sh);
          const sp=document.createElement("div");sp.className="mlb-palette mlb-saved-nested-palette";
          nestedEntries.forEach(entry=>{
            const b=document.createElement("button");b.type="button";
            const isApi=String(entry.definition?.implementation||"graph")==="api";
            b.innerHTML='<span class="mlb-pal-icon">'+(isApi?"API":"MOD")+'</span><span><strong>'+String(entry.name||entry.definition.name||"Module")+'</strong><span class="mlb-pal-sub">Saved '+(isApi?"API Component":"Module")+' · click to insert</span></span>';
            b.disabled=layoutIsLocked();
            b.title=layoutIsLocked()?"Layout locked — click Edit Layout first":"Insert "+String(entry.name||entry.definition.name||"Component");
            b.addEventListener("click",()=>{componentInsertPicker={open:false,afterNodeId:selected||null};insertGalleryComponentIntoCurrent(entry);});
            sp.appendChild(b);
          });
          side.appendChild(sp);
        }
      }

      if(state.active_workspace==="model"){
        const mh=document.createElement("button");
        mh.type="button";
        mh.className="mlb-category";
        mh.setAttribute("aria-expanded",String(!myBricksCollapsed));
        mh.innerHTML="<span>"+(apiComposerMode?"MY MODULES":"MY MODULES / API")+"</span><span class='mlb-category-caret'>"+(myBricksCollapsed?"▸":"▾")+"</span>";
        mh.addEventListener("click",()=>{myBricksCollapsed=!myBricksCollapsed;draw();});
        side.appendChild(mh);

        if(!myBricksCollapsed){
          Object.values(state.custom_components||{}).filter(def=>{
            if(def.palette_hidden===true)return false;
            const parent=activeCustomDefinition();
            // API Component editors may insert saved Modules from the left, but
            // never another API Component.
            if(apiComposerMode&&String(def.implementation||"graph")==="api")return false;
            if(parent&&String(parent.implementation||"graph")==="api"&&String(def.implementation||"graph")==="api")return false;
            return !parent||customCanNest(parent,def);
          }).forEach(def=>{
            const wrap=document.createElement("div");wrap.className="mlb-custom-card-wrap";
            const row=document.createElement("div");row.className="mlb-custom-card-row";
            const b=document.createElement("button");b.className="mlb-custom-card";b.type="button";
            const isApi=String(def.implementation||"graph")==="api";
            const apiSteps=apiStepNodes(def);
            const emptyLabel=isApi?(" · "+((def.nodes||[]).length||apiSteps.length||1)+" blocks"):((def.nodes||[]).length===0?" · Empty":" · "+(def.nodes||[]).length+" components");
            b.innerHTML='<span class="mlb-pal-icon">'+(isApi?"API":"MOD")+'</span><span><strong>'+def.name+'</strong><span class="mlb-pal-sub">'+(isApi?"API Component":"Module")+' · v'+def.revision+emptyLabel+"</span></span>";
            b.disabled=layoutIsLocked();b.title=layoutIsLocked()?"Layout locked — click Edit Layout first":"Add "+def.name;b.addEventListener("click",()=>addCustom(def));
            const edit=document.createElement("button");edit.type="button";edit.className="mlb-custom-edit-icon";edit.textContent="✎";edit.title="Edit Module / API Component";edit.setAttribute("aria-label","Edit "+(String(def.implementation||"graph")==="api"?"API Component ":"Module ")+def.name);
            edit.addEventListener("click",ev=>{ev.stopPropagation();customActionMenuId=customActionMenuId===def.id?null:def.id;draw();});
            row.append(b,edit);wrap.appendChild(row);
            if(customActionMenuId===def.id){
              const menu=document.createElement("div");menu.className="mlb-custom-card-menu";
              const editAction=btn("Edit","mlb-custom-menu-action");editAction.addEventListener("click",()=>editCustomDefinition(def));
              const renameAction=btn("Rename","mlb-custom-menu-action");renameAction.addEventListener("click",()=>renameCustomDefinition(def));
              const removeAction=btn("Remove","mlb-custom-menu-action danger");removeAction.addEventListener("click",()=>removeCustomFromPalette(def));
              menu.append(editAction,renameAction,removeAction);wrap.appendChild(menu);
            }
            side.appendChild(wrap);
          });
        }
      }

      const sidePos=sidebarScroll[state.active_workspace]||{left:0,top:0};
      requestAnimationFrame(()=>{
        side.scrollLeft=sidePos.left||0;
        side.scrollTop=sidePos.top||0;
      });

      // Main
      const main=document.createElement("main");main.className="mlb-main";
      const toolbar=document.createElement("div");toolbar.className="mlb-toolbar";
      const workspaceBadge=document.createElement("div");workspaceBadge.className="mlb-workspace-badge";
      workspaceBadge.textContent=galleryWorkspace.open
        ?"GALLERY"
        :cloudWorkspace.open
        ?"CLOUD & REPOSITORIES"
        :runtimePanel
        ?(runtimePanel.mode==="train"
          ?((runtimePanel.tab||"setup")==="status"?"TRAINING STATUS":"TRAINING SETUP")
          :runtimePanel.mode==="generate"
            ?((runtimePanel.tab||"setup")==="status"?"GENERATION STATUS":"GENERATION SETUP")
            :((runtimePanel.tab||"setup")==="status"?"API SERVER STATUS":"API SERVER SETUP"))
        :workspaceName();
      toolbar.appendChild(workspaceBadge);

      if(galleryWorkspace.open){
        const gname=document.createElement("div");gname.className="mlb-runtime-toolbar-name";gname.textContent=galleryWorkspace.tab==="models"?"Models":galleryWorkspace.tab==="components"?"Components":"Data";toolbar.appendChild(gname);
        const tsp=document.createElement("div");tsp.className="mlb-toolspacer";toolbar.appendChild(tsp);
        const close=btn("× Close","mlb-tool mlb-gallery-toolbar-close");close.addEventListener("click",closeGallery);toolbar.appendChild(close);
      }else if(cloudWorkspace.open){
        const cname=document.createElement("div");cname.className="mlb-runtime-toolbar-name";cname.textContent=providerLabel(cloudForm.provider);toolbar.appendChild(cname);
        const tsp=document.createElement("div");tsp.className="mlb-toolspacer";toolbar.appendChild(tsp);
        const close=btn("× Close","mlb-tool mlb-gallery-toolbar-close");close.addEventListener("click",closeCloudWorkspace);toolbar.appendChild(close);
      }else if(runtimePanel && state.active_workspace==="model"){
        const entry=builtModelById(runtimePanel.modelId);
        if(entry){const name=document.createElement("div");name.className="mlb-runtime-toolbar-name";name.textContent=entry.name;toolbar.appendChild(name);}
        const tsp=document.createElement("div");tsp.className="mlb-toolspacer";toolbar.appendChild(tsp);
        const device=entry?selectedRuntimeDevice(runtimePanel.mode==="train"?entry.training_config:runtimePanel.mode==="generate"?entry.generation_config:entry.serve_config):null;
        if(device){const d=document.createElement("div");d.className="mlb-toolbar-device";d.textContent=device.label;toolbar.appendChild(d);}
      }else{
        const lockToggle=btn(layoutIsLocked()?"✎ Edit Layout":"🔒 Lock Layout","mlb-tool mlb-layout-toggle"+(layoutIsLocked()?" locked":" editing"));
        lockToggle.title=layoutIsLocked()?"Unlock structural editing":"Protect component positions, order and connections";
        lockToggle.addEventListener("click",toggleLayoutLock);
        toolbar.append(lockToggle);

        // Data runtime health/progress belongs only to the Data Processing workspace.
        // Module/API Component editors may be opened while Data is the active parent
        // workspace, but they are reusable component editors and should stay clean.
        if(state.active_workspace==="data" && current(state)?.kind!=="custom_edit"){
          // Data Processing only needs the notebook/kernel connectivity indicator here.
          // Progress percentage and prepared-validation summary are shown in their
          // relevant execution/output views instead of occupying the canvas toolbar.
          const kernel=document.createElement("div");kernel.className="mlb-kernel-badge";
          toolbar.appendChild(kernel);
          requestAnimationFrame(updateKernelBadge);
        }
        const tsp=document.createElement("div");tsp.className="mlb-toolspacer";toolbar.appendChild(tsp);
        if(isGraphCustomEditor()){
          const newModule=btn("Add Module","mlb-tool mlb-new-module-toolbar");
          newModule.title="Create an empty nested Module after the selected layer";
          newModule.disabled=layoutIsLocked();
          newModule.addEventListener("click",()=>createNestedCustom(selected||null));
          toolbar.appendChild(newModule);
        }else if(isApiComposerView()){
          const addFunction=btn("Add Function","mlb-tool mlb-add-function-toolbar");
          addFunction.title="Add a Python / PyTorch API function after the selected block";
          addFunction.disabled=layoutIsLocked();
          addFunction.addEventListener("click",addAPIFunction);
          toolbar.appendChild(addFunction);

          const addModule=btn("Add Module","mlb-tool mlb-new-module-toolbar");
          addModule.title="Create an empty nested Module after the selected API block";
          addModule.disabled=layoutIsLocked();
          addModule.addEventListener("click",()=>createNestedCustom(selected||null));
          toolbar.appendChild(addModule);
        }else{
          const toggle=document.createElement("label");toggle.className="mlb-toggle";
          const cb=document.createElement("input");cb.type="checkbox";
          cb.checked=!!state.auto_connect;cb.disabled=layoutIsLocked();
          cb.addEventListener("change",()=>{if(!requireEditableLayout("change Auto Connect"))return;checkpoint("Change Auto Connect");state.auto_connect=cb.checked;draw();});
          toggle.append(document.createTextNode("Auto Connect"),cb);
          toolbar.appendChild(toggle);
        }

        const z=document.createElement("div");z.className="mlb-zoom";
        const zm=btn("−");zm.addEventListener("click",()=>{zoom=Math.max(.65,zoom-.1);draw();});
        const zs=document.createElement("span");zs.textContent=Math.round(zoom*100)+"%";
        const zp=btn("+");zp.addEventListener("click",()=>{zoom=Math.min(1.5,zoom+.1);draw();});
        z.append(zm,zs,zp);toolbar.appendChild(z);
      }
      if(!galleryWorkspace.open&&!cloudWorkspace.open)main.appendChild(toolbar);

      const canvas=document.createElement("div");canvas.className="mlb-canvas"+((runtimePanel||galleryWorkspace.open||cloudWorkspace.open)?" runtime-active":"");
      if(galleryWorkspace.open){
        renderCentralGallery(canvas);
      }else if(cloudWorkspace.open){
        renderCentralCloud(canvas);
      }
      if(runtimeWorkspaceActive){
        const entry=builtModelById(runtimePanel.modelId);
        if(entry){renderRuntimeWorkspace(canvas,entry,runtimePanel.mode);}
        else{runtimePanel=null;}
      }
      let wrap=null,flow=null;
      if(!galleryWorkspace.open && !cloudWorkspace.open && !runtimeWorkspaceActive){
      const ctop=document.createElement("div");ctop.className="mlb-canvas-top";
      const crumbs=document.createElement("div");crumbs.className="mlb-breadcrumbs";
      state.breadcrumbs.forEach((c,i)=>{const b=btn(c.name,"mlb-crumb");b.addEventListener("click",()=>{
        const cur=current(state);
        if(cur?.parent_edit_return&&c.id!==cur.id){setStatus("Use Done to return to the parent so nested changes are applied before leaving this Module.");draw();return;}
        state.view_component_id=c.id;state.breadcrumbs=state.breadcrumbs.slice(0,i+1);selected=null;draw();
      });crumbs.appendChild(b);if(i<state.breadcrumbs.length-1){const sep=document.createElement("span");sep.textContent="/";crumbs.appendChild(sep);}});
      ctop.appendChild(crumbs);canvas.appendChild(ctop);

      const mini=document.createElement("div");mini.className="mlb-minimap";
      const miniTitle=document.createElement("div");miniTitle.className="mlb-minimap-title";miniTitle.textContent=state.active_workspace==="data"?"DATA BLUEPRINT":(isGraphCustomEditor()?"MODULE BLUEPRINT":(isApiComposerView()?"API BLUEPRINT":"MODEL BLUEPRINT"));
      const mg=document.createElement("div");mg.className="mlb-minimap-grid";
      current(state).nodes.forEach(()=>{const m=document.createElement("div");m.className="mlb-mini-node";mg.appendChild(m);});
      mini.append(miniTitle,mg);
      canvas.appendChild(mini);

      wrap=document.createElement("div");wrap.className="mlb-flow-wrap"+(isApiComposerView()?" mlb-api-composer-wrap":"");
      flow=document.createElement("div");flow.className="mlb-flow"+(isApiComposerView()?" mlb-api-composer-flow":"");
      flow.style.transformOrigin="left top";
      flow.style.transform="scale("+zoom+")";
      const comp=current(state);

      if(!comp.nodes.length){
        const e=document.createElement("div");e.className="mlb-empty";
        if(comp.kind==="custom_edit"){
          const def=state.custom_components?.[comp.definition_id];
          e.innerHTML=String(def?.implementation||"graph")==="api"
            ?"<strong>API Component execution graph.</strong><br><br>Use Add Function or Add Module in the top toolbar, or insert supported Components from the left."
            :"<strong>Empty Module.</strong><br><br>Add Components from the left, or use Add Module above for a nested Module.";
        }else if(state.active_workspace==="data"){
          e.innerHTML="<strong>Build your data pipeline step by step.</strong><br><br>Start with Hugging Face, Kaggle, URL, Local or Manual Data.";
        }else{
          e.innerHTML="<strong>Build your model layer by layer.</strong><br><br>Add a component from the left or open Gallery to load a sample model.";
        }
        flow.appendChild(e);
      }else{
        comp.nodes.forEach((n,i)=>{
          if(i&&!isApiComposerView()){
            const a=document.createElement("div");a.className="mlb-arrow";a.textContent="→";flow.appendChild(a);
          }
          const info=n.type==="api_step"
            ?{accent:"purple",description:"Python / PyTorch API function block",icon:"FX",api:[]}
            :(n.type==="custom"?{accent:"purple",description:"Nested Module",icon:"LAY",api:[]}:cat(catalog,n.type));
          const runState=execution.nodes?.[n.id];
          const card=document.createElement("div");
          card.className="mlb-node"+(n.type==="api_step"?" mlb-api-step-node":"")+(selected===n.id?" selected":"")+(runState?" run-"+runState.status:"");card.dataset.type=n.type||"";
          card.dataset.nodeId=n.id;card.dataset.accent=info.accent||"purple";
          card.innerHTML='<span class="index">'+(i+1)+'</span>'+portButtons(n,"in")+'<div class="node-head"><div class="node-name"></div><div class="node-icon"></div></div><div class="node-desc"></div><div class="mlb-node-fields"></div><div class="node-meta"></div>'+portButtons(n,"out");
          if(runState){
            const rb=document.createElement("div");rb.className="mlb-run-badge";rb.textContent=runLabel(runState.status);rb.title=runState.message||"";card.appendChild(rb);
            if(runState.status==="running"){const rt=document.createElement("div");rt.className="mlb-run-track";rt.innerHTML="<i></i>";card.appendChild(rt);}
          }
          card.querySelector(".node-name").textContent=nodeDisplayName(n);card.querySelector(".node-icon").textContent=compactIconLabel(info.icon||"ML");card.querySelector(".node-desc").textContent=info.description||"MLBricks layer";
          if(n.type==="api_step"){
            const binding=normalizeAPIBinding(n.api_binding||defaultAPIBinding());
            card.querySelector(".mlb-node-fields").innerHTML=
              '<div class="mlb-mini-field"><span>API</span><strong>'+(binding.call_type==="user_function"?("User: "+(binding.user_function_name||"custom_function")):binding.call_type==="user_class"?("Class: "+(binding.user_class_name||"CustomClass")):(apiBindingImportPath(binding)||"Not bound"))+'</strong></div>'+ 
              '<div class="mlb-mini-field"><span>Type</span><strong>'+apiCallTypeLabel(binding.call_type)+'</strong></div>'+ 
              (binding.call_type==="user_function"&&binding.port_mode==="named"
                ?'<div class="mlb-mini-field"><span>Ports</span><strong>'+((binding.input_ports||[]).length)+' → '+((binding.output_ports||[]).length)+'</strong></div>'
                :'<div class="mlb-mini-field"><span>Parameters</span><strong>'+((binding.parameters||[]).length)+'</strong></div>');
          }else if(n.type==="custom"){
            const def=state.custom_components?.[n.definition_id];const isApi=String(def?.implementation||"graph")==="api";
            const apiSteps=apiStepNodes(def);
            card.querySelector(".mlb-node-fields").innerHTML=isApi
              ?('<div class="mlb-mini-field"><span>Blocks</span><strong>'+((def?.nodes||[]).length||apiSteps.length||1)+'</strong></div>'+ '<div class="mlb-mini-field"><span>API</span><strong>'+apiSteps.length+' functions</strong></div>')
              :('<div class="mlb-mini-field"><span>Architecture</span><strong>Open</strong></div>'+ '<div class="mlb-mini-field"><span>Ports</span><strong>Skip / Main / Extra</strong></div>');
          }else card.querySelector(".mlb-node-fields").innerHTML=nodeMiniFields(n,info);
          card.querySelectorAll(".mlb-mini-field").forEach(row=>{
            const label=row.querySelector("span");
            const value=row.querySelector("strong");
            if(label)label.title=label.textContent||"";
            if(value)value.title=value.textContent||"";
          });
          const meta=card.querySelector(".node-meta");
          if(n.type==="api_step"){
            const metaBinding=ensureAPIStepObjectIds(n);
            meta.textContent=apiCallTypeLabel(metaBinding.call_type)+" · explicit graph connections";
          }else if(n.type==="custom"){
            const def=state.custom_components?.[n.definition_id];meta.textContent=String(def?.implementation||"graph")==="api"?"API execution graph · lazy imports":"Nested Module · 3-lane interface";
          }else meta.textContent=(apiInfo(n).public_name||n.type)+" · Skip / Main / Extra";
          card.querySelectorAll('.mlb-port').forEach(portEl=>{
            const side=portEl.dataset.side, idx=Number(portEl.dataset.portIndex||0),key=portEl.dataset.portKey||"",name=portEl.dataset.portName||"",mode=portEl.dataset.portMode||"standard";
            if(pendingPort?.nodeId===n.id&&pendingPort.side===side&&((mode==="named"&&pendingPort.portKey===key)||(mode!=="named"&&pendingPort.portIndex===idx))) portEl.classList.add("armed");
            portEl.addEventListener("click",ev=>portClick(n.id,side,idx,ev,key,name,mode));
          });
          const namedIn=namedUserPorts(n,"in"),namedOut=namedUserPorts(n,"out");
          if(namedIn||namedOut){card.classList.add("mlb-user-function-named");card.style.height=Math.max(315,(Math.max(namedIn?.length||0,namedOut?.length||0)+1)*38)+"px";}
          card.addEventListener("click",()=>{outputDirectorySelection=null;selected=n.id;draw();});card.addEventListener("dblclick",()=>{if(n.definition_id)openInside(n);});
          flow.appendChild(card);
        });
      }
      wrap.appendChild(flow);canvas.appendChild(wrap);
      // Keep only action-specific port guidance. Persistent canvas instruction
      // bars are intentionally disabled across every Studio workspace.
      if(pendingPort){
        const hint=document.createElement("div");hint.className="mlb-hint";
        hint.textContent="Choose the matching lane: Top ↔ Top, Main ↔ Main, Bottom ↔ Bottom.";
        canvas.appendChild(hint);
      }

      }
      main.appendChild(canvas);
      if(!galleryWorkspace.open && !cloudWorkspace.open && !runtimeWorkspaceActive){
        let edgeDrawGeneration=0;
        const renderConnections=()=>{
          if(!wrap||!flow||!wrap.isConnected||!flow.isConnected)return;
          const generation=++edgeDrawGeneration;
          // transform:scale changes pixels but not layout. Give the wrapper the
          // scaled dimensions so zoom creates the correct scroll area.
          const baseW=Math.max(flow.scrollWidth,flow.offsetWidth);
          const baseH=Math.max(flow.scrollHeight,flow.offsetHeight);
          wrap.style.width=Math.ceil(baseW*zoom)+"px";
          wrap.style.height=Math.ceil(baseH*zoom)+"px";
          drawEdges(wrap,flow);
          const pos=workspaceScroll[state.active_workspace]||{left:0,top:0};
          canvas.scrollLeft=pos.left||0;
          canvas.scrollTop=pos.top||0;
          return generation;
        };
        // Draw once after DOM insertion, once on the following frame, and once
        // after notebook/browser layout has settled. This fixes newly inserted
        // components whose ports were measured before their final card geometry.
        requestAnimationFrame(()=>{
          renderConnections();
          requestAnimationFrame(renderConnections);
        });
        setTimeout(renderConnections,90);
        setTimeout(renderConnections,240);
        // Keep edges synchronized when the flow width changes because a card is
        // inserted, renamed, expanded, or the notebook output is resized.
        if(typeof ResizeObserver!=="undefined" && flow){
          const edgeObserver=new ResizeObserver(()=>{
            if(!flow||!flow.isConnected){edgeObserver.disconnect();return;}
            requestAnimationFrame(renderConnections);
          });
          edgeObserver.observe(flow);
        }
      }

      // Bottom Model/Data Workspace drawer belongs only to design workspaces.
      // Runtime pages (Training / Generation / Serve) intentionally omit it.
      const details=document.createElement("div");details.className="mlb-details";

      const detailsBar=document.createElement("div");detailsBar.className="mlb-details-bar";
      const detailsLeft=document.createElement("div");detailsLeft.className="mlb-details-left";
      const detailsTitle=document.createElement("span");detailsTitle.className="mlb-details-title";
      detailsTitle.textContent=state.active_workspace==="data"?"DATA WORKSPACE":"MODEL WORKSPACE";

      const detailsSelect=document.createElement("select");detailsSelect.className="mlb-details-select";
      const options=state.active_workspace==="data"
        ?[["details","Pipeline Details"],["outputs","Output Directory"],["files","Files"],["local","Local Environment"]]
        :[["details","Model Details"],["outputs","Output Directory"],["files","Files"],["local","Local Environment"]];
      options.forEach(([value,label])=>{
        const o=document.createElement("option");o.value=value;o.textContent=label;
        if(bottomView===value)o.selected=true;
        detailsSelect.appendChild(o);
      });
      detailsSelect.addEventListener("change",ev=>{
        ev.stopPropagation();
        bottomView=detailsSelect.value;
        bottomExpanded=true;
        outputDirectorySelection=null;
        draw();
      });
      detailsLeft.append(detailsTitle,detailsSelect);

      const detailsToggle=btn(bottomExpanded?"▾ Hide":"▴ Show","mlb-details-toggle");
      detailsToggle.addEventListener("click",()=>{bottomExpanded=!bottomExpanded;draw();});
      detailsBar.append(detailsLeft,detailsToggle);
      details.appendChild(detailsBar);

      if(bottomView==="outputs"){
        const outputPanel=document.createElement("div");
        outputPanel.className="mlb-output-directory"+(bottomExpanded?" expanded":" collapsed");
        if(bottomExpanded)renderOutputDirectory(outputPanel);
        details.appendChild(outputPanel);
      }else if(bottomView==="files"){
        const filesPanel=document.createElement("div");
        filesPanel.className="mlb-files-view"+(bottomExpanded?" expanded":" collapsed");
        if(bottomExpanded)renderFilesView(filesPanel);
        details.appendChild(filesPanel);
      }else if(bottomView==="local"){
        const localPanelEl=document.createElement("div");
        localPanelEl.className="mlb-local-view"+(bottomExpanded?" expanded":" collapsed");
        if(bottomExpanded)renderLocalView(localPanelEl);
        details.appendChild(localPanelEl);
      }else if(bottomView==="cloud"){
        const cloudPanelEl=document.createElement("div");
        cloudPanelEl.className="mlb-cloud-view"+(bottomExpanded?" expanded":" collapsed");
        if(bottomExpanded)renderCloudView(cloudPanelEl);
        details.appendChild(cloudPanelEl);
      }else{
        const panels=document.createElement("div");panels.className="mlb-bottom-panels"+(bottomExpanded?" expanded":" collapsed");
        const p1=document.createElement("div");p1.className="mlb-bottom-card";
        const p2=document.createElement("div");p2.className="mlb-bottom-card";
        const p3=document.createElement("div");p3.className="mlb-bottom-card";
        const p4=document.createElement("div");p4.className="mlb-bottom-card";

        if(state.active_workspace==="data"){
          p1.innerHTML='<div class="mlb-bottom-title">GALLERY</div><div class="mlb-preset-card"><strong>▦ Sample & Saved Data</strong>Open sample data pipelines or reuse pipelines saved by you.</div>';
          p1.querySelector(".mlb-preset-card").addEventListener("click",openGallery);
          p2.innerHTML='<div class="mlb-bottom-title">PIPELINE INFO</div><div class="mlb-stat-row"><span>Steps</span><strong>'+current(state).nodes.length+'</strong></div><div class="mlb-stat-row"><span>Connections</span><strong>'+(current(state).edges||[]).length+'</strong></div><div class="mlb-stat-row"><span>Workspace</span><strong>Data</strong></div><div class="mlb-stat-row"><span>Status</span><strong class="mlb-good">✓ Designed</strong></div>';
          const latestData=latestPreparedDataset();
          p3.innerHTML=latestData
            ?('<div class="mlb-bottom-title">LATEST DATA</div>'+
              '<div class="mlb-stat-row"><span>Name</span><strong>'+latestData.name+'</strong></div>'+
              '<div class="mlb-stat-row"><span>Train</span><strong>'+splitRows(latestData,"train")+'</strong></div>'+
              '<div class="mlb-stat-row"><span>Validation</span><strong>'+splitRows(latestData,"validation")+'</strong></div>'+
              '<div class="mlb-stat-row"><span>Test</span><strong>'+splitRows(latestData,"test")+'</strong></div>')
            :'<div class="mlb-bottom-title">PROCESSING</div><div class="mlb-stat-row"><span>Text</span><strong>Clean / Tokenize</strong></div><div class="mlb-stat-row"><span>Image</span><strong>Resize / Crop</strong></div><div class="mlb-stat-row"><span>Audio</span><strong>Resample / Normalize</strong></div><div class="mlb-stat-row"><span>Split</span><strong>Train / Val / Test</strong></div>';
          p4.innerHTML='<div class="mlb-bottom-title">FLOW</div><div class="mlb-stat-row"><span>Main</span><strong>Processing order</strong></div><div class="mlb-stat-row"><span>Skip</span><strong>Optional branch</strong></div><div class="mlb-stat-row"><span>Extra</span><strong>Aux data</strong></div>';
        }else{
          p1.innerHTML='<div class="mlb-bottom-title">GALLERY</div><div class="mlb-preset-card"><strong>▦ Sample & Saved Models</strong>Open sample architectures or reuse models saved by you.</div>';
          p1.querySelector(".mlb-preset-card").addEventListener("click",openGallery);
          p2.innerHTML='<div class="mlb-bottom-title">GRAPH INFO</div><div class="mlb-stat-row"><span>Layers</span><strong>'+current(state).nodes.length+'</strong></div><div class="mlb-stat-row"><span>Connections</span><strong>'+(current(state).edges||[]).length+'</strong></div><div class="mlb-stat-row"><span>Context</span><strong>'+(state.project?.context_length||"—")+'</strong></div><div class="mlb-stat-row"><span>Batch Size</span><strong>'+(state.project?.batch_size||"—")+'</strong></div><div class="mlb-stat-row"><span>Status</span><strong class="mlb-good">Design Ready</strong></div>';
          p3.innerHTML='<div class="mlb-bottom-title">COMPUTE ESTIMATE</div><div class="mlb-stat-row"><span>Target Params</span><strong>'+(state.project?.estimated_parameters||"—")+'</strong></div><div class="mlb-stat-row"><span>Dataset</span><strong>'+(state.project?.dataset||"—")+'</strong></div><div class="mlb-stat-row"><span>Precision</span><strong>float16</strong></div><div class="mlb-stat-row"><span>Backend</span><strong>MLBricks</strong></div>';
          p4.innerHTML='<div class="mlb-bottom-title">CONNECTION LANES</div><div class="mlb-stat-row"><span>Skip</span><strong>Top Out → Top In</strong></div><div class="mlb-stat-row"><span>Main</span><strong>Middle Out → Middle In</strong></div><div class="mlb-stat-row"><span>Extra</span><strong>Bottom Out → Bottom In</strong></div><div class="mlb-stat-row"><span>Remove</span><strong>Inspector → Remove</strong></div>';
        }
        panels.append(p1,p2,p3,p4);
        details.appendChild(panels);
      }

      if(!galleryWorkspace.open&&!cloudWorkspace.open&&!runtimeWorkspaceActive)main.appendChild(details);

      // Inspector
      const ins=document.createElement("aside");ins.className="mlb-inspector";
      const runtimeInspectorEntry=runtimeWorkspaceActive?builtModelById(runtimePanel?.modelId):null;
      const tabs=document.createElement("div");tabs.className="mlb-ins-tabs";
      const primaryInspectorLabel=runtimeInspectorEntry?(runtimePanel?.mode==="train"?"Training":"Runtime"):"Inspector";
      [["settings",primaryInspectorLabel],["info","Info"]].forEach(([k,t])=>{const b=btn(t);if(inspectorTab===k)b.className="active";b.addEventListener("click",()=>{inspectorTab=k;draw();});tabs.appendChild(b);});ins.appendChild(tabs);
      const body=document.createElement("div");body.className="mlb-ins-body";
      const outputDataset=selectedOutputDataset();
      const outputModel=selectedOutputModel();
      const n=selectedNode();

      if(runtimeInspectorEntry){
        renderRuntimeContextInspector(body,runtimeInspectorEntry,runtimePanel?.mode||"train");
      }else if(outputDataset){
        renderPreparedDatasetInspector(body,outputDataset);
      }else if(outputModel){
        renderBuiltModelInspector(body,outputModel);
      }else if(!n){
        const compNow=current(state);const defNow=compNow?.kind==="custom_edit"?state.custom_components?.[compNow.definition_id]:null;
        if(defNow){
          const isApiCustom=String(defNow.implementation||"graph")==="api";
          const h=document.createElement("div");h.className="mlb-section-title";h.textContent=isApiCustom?"API COMPONENT":"MODULE";body.appendChild(h);
          const nameRow=editorRow(isApiCustom?"API Component Name":"Module Name",defNow.name||"",value=>{
            const name=String(value||"").trim().replace(/\s+/g," ");
            if(!name||name===defNow.name)return;
            checkpoint("Rename "+(isApiCustom?"API Component":"Module"));
            const oldName=defNow.name;defNow.name=name;compNow.name=name;
            const crumb=(state.breadcrumbs||[]).find(x=>x.id===compNow.id);if(crumb)crumb.name=name;
            Object.values(state.components||{}).forEach(comp=>{
              (comp.nodes||[]).forEach(node=>{if(node.definition_id===defNow.id){node.display_name=name;if(normalizedUserName(node.name)===normalizedUserName(oldName))node.name=name;}});
            });
          });
          body.appendChild(nameRow);
          if(isApiCustom){
            renderAPICustomOverview(body,defNow);
          }else{
            const help=document.createElement("div");help.className="mlb-api-path";help.textContent="Compose this reusable Module from built-in and saved components. You can nest Modules directly here without returning to Gallery.";body.appendChild(help);
            // Nested Modules are created from the top toolbar; built-in Components remain in the left library.
          }
          appendCustomSaveActions(body);
        }else{
          body.innerHTML='<div class="mlb-section-title">SELECT A NODE</div><div class="mlb-api-path">'+(state.active_workspace==="data"?"Choose a data step, or click a prepared dataset in Output Directory to inspect it.":"Choose a model component to edit its API.")+'</div>';
        }
      }else if(n.type==="api_step"&&isApiComposerView()&&inspectorTab==="settings"){
        const def=state.custom_components?.[current(state).definition_id];
        renderAPIStepInspector(body,def,n);
      }else if(n.type==="api_step"&&isApiComposerView()&&inspectorTab==="info"){
        const binding=normalizeAPIBinding(n.api_binding||defaultAPIBinding());
        body.innerHTML='<div class="mlb-selected"><strong>'+String(n.name||"Function")+'</strong><span class="mlb-pill">'+apiCallTypeLabel(binding.call_type)+'</span></div>';
        const s=document.createElement("div");s.className="mlb-summary";
        [["Import",(binding.call_type==="instance_method"&&binding.object_mode==="existing")?(apiObjectCandidates(state.custom_components?.[current(state).definition_id],n.id).find(x=>x.id===binding.object_ref)?.name||"Existing object"):(binding.call_type==="user_function"?("User: "+(binding.user_function_name||"custom_function")):binding.call_type==="user_class"?("Class: "+(binding.user_class_name||"CustomClass")):(apiBindingImportPath(binding)||"Not bound"))],["Type",apiCallTypeLabel(binding.call_type)],["Parameters",(binding.parameters||[]).length],["Connections",(current(state).edges||[]).filter(e=>e.source===n.id||e.target===n.id).length]].forEach(([a,b])=>{const r=document.createElement("div");r.className="mlb-summary-row";r.innerHTML="<span>"+a+"</span><strong>"+b+"</strong>";s.appendChild(r);});body.appendChild(s);
      }else if(inspectorTab==="info"){
        const api=apiInfo(n);const item=n.type==="custom"?{category:"Modules / API",description:"Reusable Module or API Component."}:cat(catalog,n.type);
        body.innerHTML='<div class="mlb-selected"><strong>'+nodeDisplayName(n)+'</strong><span class="mlb-pill">'+(api.public_name||"Custom")+'</span></div>';
        const s=document.createElement("div");s.className="mlb-summary";[["Type",n.type],["Definition",n.definition_id?"Custom":"Built-in"],["Category",item.category||"General"],["Repeat",n.repeat||1],["API",api.import_path||"custom"],["Status","Valid"]].forEach(([a,b])=>{const r=document.createElement("div");r.className="mlb-summary-row";r.innerHTML="<span>"+a+"</span><strong>"+b+"</strong>";s.appendChild(r);});body.appendChild(s);
      }else{
        const api=apiInfo(n);const info=n.type==="custom"?{api:[]}:cat(catalog,n.type);
        const sw=document.createElement("div");sw.className="mlb-selected";
        const displayName=nodeDisplayName(n);const apiName=api.public_name||"Custom Layer";
        const pill=document.createElement("span");pill.className="mlb-pill";pill.textContent=apiName;
        if(normalizedUserName(displayName)===normalizedUserName(apiName)){
          sw.classList.add("single-pill");sw.appendChild(pill);
        }else{
          const titleText=document.createElement("strong");titleText.textContent=displayName;sw.append(titleText,pill);
        }
        body.appendChild(sw);
        const renameComponentBtn=btn("✎ Rename Component","mlb-ins-rename");renameComponentBtn.addEventListener("click",renameSelectedComponent);body.appendChild(renameComponentBtn);
        const runLive=document.createElement("div");runLive.className="mlb-ins-run-live";
        const rs=execution.nodes?.[n.id];
        if(rs){
          runLive.className+=" "+rs.status;
          runLive.innerHTML="<strong>"+runLabel(rs.status)+"</strong><span>"+(rs.message||"")+"</span>";
        }else{
          runLive.style.display="none";
        }
        body.appendChild(runLive);

        const path=document.createElement("div");path.className="mlb-api-path";
        path.textContent=n.type==="custom"
          ?"custom://"+n.definition_id
          :(api.builder_utility
              ?(api.builder_python_api?"Builder data/text operation":"Builder workflow settings")
              :(api.signature||api.import_path||"MLBricks API"));
        body.appendChild(path);
        if(n.type!=="custom"){
          const apiStatus=document.createElement("div");
          apiStatus.className="mlb-api-status "+(api.available?"ok":"bad");
          if(api.builder_utility && api.builder_python_api){
            apiStatus.className="mlb-api-status utility";
            apiStatus.textContent="Builder data operation — executable with mlb_studio.data";
          }else if(api.builder_utility){
            apiStatus.className="mlb-api-status utility";
            apiStatus.textContent="Builder workflow node — no mlbricks Python API";
          }else if(api.available && api.runtime_available===true){
            apiStatus.textContent="✓ Real MLBricks API: "+(api.import_path||api.public_name);
          }else if(api.available){
            apiStatus.textContent="✓ API loaded from MLBricks source: "+(api.public_name||n.type);
            apiStatus.title=api.runtime_error||"Runtime import was not required for the inspector.";
          }else{
            apiStatus.textContent="✕ API unavailable";
          }
          body.appendChild(apiStatus);
        }
        if(n.type==="custom"){
          const def=state.custom_components[n.definition_id];const isApi=String(def?.implementation||"graph")==="api";
          const s=document.createElement("div");s.className="mlb-summary";
          (isApi?[["Implementation","API Execution Graph"],["Functions",apiStepNodes(def).length||1],["Connections",def?.edges?.length||0],["Revision","v"+(def?.revision||1)]]:[["Internal Components",def?.nodes?.length||0],["Connections",def?.edges?.length||0],["Revision","v"+(def?.revision||1)]]).forEach(([a,b])=>{const r=document.createElement("div");r.className="mlb-summary-row";r.innerHTML="<span>"+a+"</span><strong>"+b+"</strong>";s.appendChild(r);});
          body.appendChild(s);
          if(isApi){
            const fields=customExposedFields(def);if(fields.length){const st=document.createElement("div");st.className="mlb-section-title";st.textContent="CUSTOM ARGUMENTS";body.appendChild(st);fields.forEach(f=>renderField(body,n,f));}
            const bound=[];apiStepNodes(def).forEach(step=>{(normalizeAPIBinding(step.api_binding||defaultAPIBinding()).parameters||[]).filter(x=>String(x.source||"user")!=="user").forEach(x=>bound.push({label:(step.name||"Function")+" · "+(x.label||x.name),source:x.source}));});
            if(!bound.length&&def?.api_binding) (normalizeAPIBinding(def.api_binding).parameters||[]).filter(x=>String(x.source||"user")!=="user").forEach(x=>bound.push({label:x.label||x.name,source:x.source}));
            if(bound.length){const st=document.createElement("div");st.className="mlb-section-title";st.textContent="BOUND ARGUMENTS";body.appendChild(st);const fixed=document.createElement("div");fixed.className="mlb-api-path";fixed.textContent=bound.map(x=>x.label+" ← "+x.source).join(" · ");body.appendChild(fixed);}
          }else{
            const st=document.createElement("div");st.className="mlb-section-title";st.textContent="MODULE PORTS";body.appendChild(st);
            const fixed=document.createElement("div");fixed.className="mlb-api-path";fixed.textContent="Fixed clean interface: Top Skip, Middle Main, Bottom Extra — on both left and right sides.";body.appendChild(fixed);
          }
        }else{
          if(n.type==="train_test_split"){
            const s=splitPercentages(n),total=s.train+s.validation+s.test,valid=splitIsValid(n);
            const title=document.createElement("div");title.className="mlb-section-title";title.textContent="SPLIT PREVIEW";body.appendChild(title);
            const preview=document.createElement("div");preview.className="mlb-split-preview"+(valid?" valid":" invalid");
            preview.innerHTML='<div class="mlb-split-values"><span><b>'+s.train+'%</b> Train</span><span><b>'+s.validation+'%</b> Validation</span><span><b>'+s.test+'%</b> Test</span></div>'+
              '<div class="mlb-split-bar"><i style="width:'+s.train+'%"></i><i style="width:'+s.validation+'%"></i><i style="width:'+s.test+'%"></i></div>'+
              '<div class="mlb-split-total">'+(valid?'✓':'!')+' Total: <b>'+total+'%</b> '+(valid?'Ready':'— must equal 100%')+'</div>';
            body.appendChild(preview);
            const presets=document.createElement("div");presets.className="mlb-split-presets";
            [
              [90,5,5,"Train 90%","Validation 5% · Test 5%"],
              [80,10,10,"Train 80%","Validation 10% · Test 10%"],
              [90,10,0,"Train 90%","Validation 10% · No test split"]
            ].forEach(([tr,va,te,mainLabel,subLabel])=>{
              const b=btn("","mlb-split-preset");
              b.innerHTML="<strong>"+mainLabel+"</strong><span>"+subLabel+"</span>";
              if(s.train===tr&&s.validation===va&&s.test===te)b.classList.add("active");
              b.addEventListener("click",()=>setSplitPreset(n,tr,va,te,mainLabel+" / "+subLabel));
              presets.appendChild(b);
            });
            body.appendChild(presets);
          }

          const st=document.createElement("div");st.className="mlb-section-title";st.textContent="CONFIG";body.appendChild(st);
          const fields=(api.parameters||info.api||[]);
          if(fields.some(f=>f.group)) renderGroupedFields(body,n,fields);
          else fields.forEach(f=>renderField(body,n,f));

          if(n.type==="prepared_dataset"){
            const meta=availablePreparedDatasets().find(d=>d.output_node_id===n.id) || latestPreparedDataset();
            if(meta){
              const dt=document.createElement("div");dt.className="mlb-section-title";dt.textContent="DATA READY";body.appendChild(dt);
              body.appendChild(datasetSummaryCard(meta,"COMPLETED DATA"));
            }
          }else if(n.type==="text_input" && String(n.params?.input_mode||"prompt")==="prepared_dataset"){
            const meta=preparedDatasetById(n.params?.dataset_id)||latestPreparedDataset();
            if(meta){
              const dt=document.createElement("div");dt.className="mlb-section-title";dt.textContent="SELECTED DATA";body.appendChild(dt);
              body.appendChild(datasetSummaryCard(meta,"MODEL INPUT"));
            }
          }

          const preview=constructorPreview(n);
          if(preview){
            const ct=document.createElement("div");ct.className="mlb-section-title";
            ct.textContent=api.builder_python_api?"DATA PYTHON":"MLBRICKS PYTHON";
            body.appendChild(ct);
            const code=document.createElement("pre");code.className="mlb-code-preview";code.textContent=preview;body.appendChild(code);
          }
        }
        const edgeSectionTitle=document.createElement("div");edgeSectionTitle.className="mlb-section-title";edgeSectionTitle.textContent="CONNECTIONS";body.appendChild(edgeSectionTitle);
        const relEdges=(current(state).edges||[]).filter(e=>e.source===n.id||e.target===n.id);
        if(relEdges.length===0){
          const emptyEdge=document.createElement("div");emptyEdge.className="mlb-api-path";emptyEdge.textContent="No connections for this node.";body.appendChild(emptyEdge);
        } else {
          relEdges.forEach(ed=>{
            const row=document.createElement("div");row.className="mlb-connection-row";
            const src=current(state).nodes.find(x=>x.id===ed.source), tgt=current(state).nodes.find(x=>x.id===ed.target);
            const laneName=ed.kind==="residual"?"Skip":(ed.kind==="aux"?"Extra":"Main");
            const left=(src?nodeDisplayName(src):"Node")+" → "+(tgt?nodeDisplayName(tgt):"Node")+" · "+laneName;
            const txt=document.createElement("div");txt.className="mlb-connection-text";txt.textContent=left;
            const delBtn=btn("Remove","mlb-conn-remove");
            delBtn.disabled=layoutIsLocked();delBtn.addEventListener("click",()=>{
              if(!requireEditableLayout("remove connections"))return;
              checkpoint("Remove connection");
              current(state).edges=current(state).edges.filter(x=>x.id!==ed.id);
              setStatus("Connection removed.");
              draw();
            });
            row.append(txt,delBtn);body.appendChild(row);
          });
        }
        const moveTitle=document.createElement("div");moveTitle.className="mlb-section-title";moveTitle.textContent=state.active_workspace==="data"?"STEP POSITION":"LAYER POSITION";body.appendChild(moveTitle);
        const moveGrid=document.createElement("div");moveGrid.className="mlb-action-grid mlb-move-grid";
        const moveLeft=btn(state.active_workspace==="data"?"← Move Earlier":"← Move Left");
        const moveRight=btn(state.active_workspace==="data"?"Move Later →":"Move Right →");
        const nodeIndex=current(state).nodes.findIndex(x=>x.id===n.id);
        moveLeft.disabled=layoutIsLocked()||nodeIndex<=0;
        moveRight.disabled=layoutIsLocked()||nodeIndex<0||nodeIndex>=current(state).nodes.length-1;
        moveLeft.addEventListener("click",()=>moveSelected(-1));
        moveRight.addEventListener("click",()=>moveSelected(1));
        moveGrid.append(moveLeft,moveRight);body.appendChild(moveGrid);

        const actions=document.createElement("div");actions.className="mlb-action-grid";
        if(n.definition_id){const def=state.custom_components?.[n.definition_id];const open=btn(String(def?.implementation||"graph")==="api"?"Open API Component":"Open Module");open.addEventListener("click",()=>openInside(n));actions.appendChild(open);}
        const dup=btn("Duplicate");dup.disabled=layoutIsLocked();dup.addEventListener("click",duplicateSelected);actions.appendChild(dup);
        const disc=btn("Remove All Links");disc.disabled=layoutIsLocked();disc.addEventListener("click",()=>{
          if(!requireEditableLayout("remove connections"))return;
          checkpoint("Remove all links from "+n.name);
          current(state).edges=current(state).edges.filter(e=>e.source!==n.id&&e.target!==n.id);
          setStatus("All connections removed.");draw();
        });actions.appendChild(disc);
        const del=btn("Delete");del.disabled=layoutIsLocked();del.addEventListener("click",()=>deleteNode(n.id));actions.appendChild(del);body.appendChild(actions);
        if(current(state).kind==="custom_edit"){
          const parentCfg=document.createElement("div");
          parentCfg.className="mlb-summary";
          parentCfg.innerHTML='<div class="mlb-summary-row"><span>Module Interface</span><strong>Skip / Main / Extra</strong></div>';
          body.appendChild(parentCfg);
          appendCustomSaveActions(body);
        }
      }
      ins.appendChild(body);

      shell.append(side,main,ins);root.appendChild(shell);

      const stat=document.createElement("div");stat.className="mlb-statusbar";
      let statusDevice="Auto";
      if(runtimePanel){
        const e=builtModelById(runtimePanel.modelId);
        if(e){const cfg=runtimePanel.mode==="train"?e.training_config:e.generation_config;statusDevice=selectedRuntimeDevice(cfg).label;}
      }
      stat.innerHTML='<span>Workspace: '+workspaceName()+'</span><span>Backend: '+(state.active_workspace==="data"?"Builder Data API":"MLBricks Runtime")+'</span><span>Device: '+statusDevice+'</span><span class="right mlb-ready">● '+status+"</span>";
      root.appendChild(stat);

      const nextInspectorKey=inspectorRenderKey();
      const inspectorPos=inspectorScrollPositions[nextInspectorKey]||{left:0,top:0};
      lastInspectorRenderKey=nextInspectorKey;
      requestAnimationFrame(()=>{
        const liveBody=root.querySelector(".mlb-ins-body");
        if(liveBody){
          liveBody.scrollLeft=inspectorPos.left||0;
          liveBody.scrollTop=inspectorPos.top||0;
          if(scrollBuiltModelActionsOnce && outputModel){
            const actionsTitle=liveBody.querySelector(".mlb-model-actions-title");
            const actions=liveBody.querySelector(".mlb-model-actions");
            const target=actionsTitle||actions;
            if(target){
              liveBody.scrollTop=Math.max(0,target.offsetTop-14);
              inspectorScrollPositions[nextInspectorKey]={left:0,top:liveBody.scrollTop};
            }
            scrollBuiltModelActionsOnce=false;
          }
        }
        if(searchFocusRestore){
          const restore=searchFocusRestore;searchFocusRestore=null;
          const liveSearch=root.querySelector(".mlb-search");
          if(liveSearch){
            try{liveSearch.focus({preventScroll:true});}catch(_){liveSearch.focus();}
            try{liveSearch.setSelectionRange(restore.start,restore.end);}catch(_){}
          }
        }
      });
      if(isPopout)schedulePopoutStateSync();
    }

    setupPopoutBridge();
    draw();
    startBridgePolling();
  }

  window.MLBricksBuilder={mount};
}
window.__MLB_STUDIO_FACTORY__=__MLB_STUDIO_FACTORY__;
window.__MLB_STUDIO_JS_SOURCE__="("+__MLB_STUDIO_FACTORY__.toString()+")();";
__MLB_STUDIO_FACTORY__();
