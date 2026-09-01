(function(){
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
    const brandLogo=String(payload.brand_logo||"");
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
    let bottomExpanded=true;
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
            bottomExpanded=true;
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
      const assets=payload.popout_assets||{};
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
        bottomExpanded=true;
        outputDirectorySelection=entry.id;
        selected=null;
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
        host:"0.0.0.0",port:8000,cors_origin:"*",require_api_key:true,public_tunnel:"off",
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
      cg.append(statusMetric("Checkpoint Every",(config.checkpoint_every||0)+" steps"),statusMetric("Output Dir",config.output_dir||"—"),
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
      const stop=btn("Stop Generation","mlb-runtime-stop");stop.disabled=!(execution.status==="running"&&execution.runtime_kind==="generate");stop.addEventListener("click",requestStop);side.appendChild(stop);
      const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title="Stop generation and return to Model Builder";cancel.addEventListener("click",()=>cancelRuntimeToModelEditor(entry,"generate"));side.appendChild(cancel);
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

      if(mode==="train"){
        const out=runtimeSection("Output");const grid=document.createElement("div");grid.className="mlb-runtime-grid";
        grid.append(runtimeField("Output Directory","text",config.output_dir,v=>update("output_dir",v)));out.appendChild(grid);main.appendChild(out);
      }

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
        const start=btn("Generate Tokens","mlb-runtime-start");start.disabled=!entry.weights_ready||execution.status==="running";
        start.addEventListener("click",()=>{
          if(!entry.weights_ready)return;
          entry.generation_history=[];entry.generation_live={status:"running",phase:"starting",overall:0,generated_tokens:0,message:"Starting generation in Python…",generated_text:""};
          runtimePanel={mode:"generate",modelId:entry.id,tab:"status"};
          draw();
          setTimeout(()=>requestRuntimeCommand("generate",entry),80);
        });side.appendChild(start);
        const stop=btn("Stop Generation","mlb-runtime-stop");stop.disabled=execution.status!=="running";
        stop.addEventListener("click",requestStop);side.appendChild(stop);
        const cancel=btn("Cancel","mlb-runtime-cancel");cancel.title="Stop generation and return to Model Builder";cancel.addEventListener("click",()=>cancelRuntimeToModelEditor(entry,"generate"));side.appendChild(cancel);
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

      const actionTitle=document.createElement("div");actionTitle.className="mlb-section-title";actionTitle.textContent="ACTIONS";
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
        format_version:"0.8.0",
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
      if(brandLogo){
        const logoImg=document.createElement("img");logoImg.className="mlb-logo-brand";logoImg.src=brandLogo;logoImg.alt="MLBRICKS STUDIO";
        logo.appendChild(logoImg);
      }else{
        const fallback=document.createElement("span");fallback.className="mlb-studio-brand";fallback.setAttribute("aria-label","MLB Studio");fallback.innerHTML='<span class="mlb-studio-mark">MLB</span><span class="mlb-studio-word">Studio</span>';
        logo.appendChild(fallback);
      }
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

      const wrap=document.createElement("div");wrap.className="mlb-flow-wrap"+(isApiComposerView()?" mlb-api-composer-wrap":"");
      const flow=document.createElement("div");flow.className="mlb-flow"+(isApiComposerView()?" mlb-api-composer-flow":"");
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
          if(!wrap.isConnected||!flow.isConnected)return;
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
        if(typeof ResizeObserver!=="undefined"){
          const edgeObserver=new ResizeObserver(()=>{
            if(!flow.isConnected){edgeObserver.disconnect();return;}
            requestAnimationFrame(renderConnections);
          });
          edgeObserver.observe(flow);
        }
      }

      // Bottom project drawer is open by default. Train/Generate collapse it on
      // entry; normal design, local import and Serve Model/API keep it open.
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

      if(!galleryWorkspace.open&&!cloudWorkspace.open)main.appendChild(details);

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
        if(liveBody){liveBody.scrollLeft=inspectorPos.left||0;liveBody.scrollTop=inspectorPos.top||0;}
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
})();
