(function(global){
  'use strict';
  var React = global.React;
  var ReactDOM = global.ReactDOM;
  if(!React || !ReactDOM){
    global.__MLBReactRuntime={available:false,reason:'React runtime is unavailable'};
    return;
  }

  var h=React.createElement;
  var stores=new Map();
  var mounts=new Map();

  function shallowEqual(a,b){
    if(Object.is(a,b))return true;
    if(!a||!b||typeof a!=='object'||typeof b!=='object')return false;
    var ak=Object.keys(a),bk=Object.keys(b);
    if(ak.length!==bk.length)return false;
    for(var i=0;i<ak.length;i++)if(!Object.prototype.hasOwnProperty.call(b,ak[i])||!Object.is(a[ak[i]],b[ak[i]]))return false;
    return true;
  }

  function createStore(initial,actions){
    var state=initial||{};
    var listeners=new Set();
    return {
      actions:actions||{},
      getState:function(){return state;},
      setActions:function(next){this.actions=next||this.actions;},
      setState:function(next){
        state=Object.assign({},state,next||{});
        listeners.forEach(function(fn){try{fn();}catch(_){}});
      },
      replace:function(next){
        state=next||{};
        listeners.forEach(function(fn){try{fn();}catch(_){}});
      },
      subscribe:function(fn){listeners.add(fn);return function(){listeners.delete(fn);};}
    };
  }

  function connected(selector,render,equal){
    equal=equal||Object.is;
    return class ConnectedSlice extends React.PureComponent{
      constructor(props){
        super(props);
        this.state={slice:selector(props.store.getState())};
        this.unsubscribe=null;
      }
      componentDidMount(){
        var self=this;
        this.unsubscribe=this.props.store.subscribe(function(){
          var next=selector(self.props.store.getState());
          if(!equal(self.state.slice,next))self.setState({slice:next});
        });
      }
      componentWillUnmount(){if(this.unsubscribe)this.unsubscribe();}
      render(){return render(this.state.slice,this.props);}
    };
  }

  function escText(value){return String(value==null?'':value);}
  function fmtInt(value){var n=Number(value);return Number.isFinite(n)?Math.round(n).toLocaleString():'—';}
  function fmtFloat(value,digits){var n=Number(value);return Number.isFinite(n)?n.toFixed(digits):'—';}
  function fmtDuration(seconds){var n=Number(seconds||0);if(!n)return '—';if(n<60)return n.toFixed(1)+'s';return Math.floor(n/60)+'m '+Math.floor(n%60)+'s';}
  function pct(value){return Math.max(0,Math.min(100,Number(value||0)));}
  function asArray(value){return Array.isArray(value)?value:[];}

  function displayList(value,fallback){
    if(Array.isArray(value))return value.filter(function(x){return x!=null&&String(x).trim()!=='';}).map(String).join(' · ')||fallback;
    var text=value==null?'':String(value).trim();
    return text||fallback;
  }

  function maybeObject(value){return value&&typeof value==='object'&&!Array.isArray(value)?value:null;}
  function pickText(value){return value==null?'':String(value);}
  function guessOutputKind(kind,mime,data){
    kind=pickText(kind).trim().toLowerCase();
    mime=pickText(mime).trim().toLowerCase();
    if(kind)return kind;
    if(mime.indexOf('image/')===0)return 'image';
    if(mime.indexOf('audio/')===0)return 'audio';
    if(mime.indexOf('video/')===0)return 'video';
    if(mime.indexOf('application/json')===0)return 'json';
    if(typeof data==='string'&&data.indexOf('data:image/')===0)return 'image';
    if(typeof data==='string'&&data.indexOf('data:audio/')===0)return 'audio';
    if(Array.isArray(data))return 'signal';
    if(maybeObject(data))return 'json';
    return 'text';
  }
  function extractOutputData(raw){
    if(raw==null)return null;
    if(typeof raw==='string'||typeof raw==='number'||typeof raw==='boolean'||Array.isArray(raw))return raw;
    var obj=maybeObject(raw);
    if(!obj)return raw;
    if(obj.data!==undefined)return obj.data;
    if(obj.value!==undefined)return obj.value;
    if(obj.src!==undefined)return obj.src;
    if(obj.url!==undefined)return obj.url;
    return raw;
  }
  function normalizeOutputEnvelope(live,entry){
    live=live||{};entry=entry||{};
    var raw=live.generated_output;
    if(raw==null)raw=entry.last_generated_output;
    if(raw==null&&live.generated_text!=null)raw=live.generated_text;
    if(raw==null&&entry.last_generation!=null)raw=entry.last_generation;
    var obj=maybeObject(raw);
    var meta=(maybeObject(live.generated_output_meta)||maybeObject(entry.last_generated_output_meta)||maybeObject(obj&&obj.metadata)||{});
    var mime=pickText(live.generated_output_mime||entry.last_generated_output_mime||(obj&&(obj.mime||obj.content_type))||'');
    var kind=guessOutputKind(live.generated_output_kind||entry.last_generated_output_kind||(obj&&(obj.kind||obj.type)),mime,extractOutputData(raw));
    var data=extractOutputData(raw);
    var src=(obj&&(obj.src||obj.url))||(typeof data==='string'&&(kind==='image'||kind==='audio'||kind==='video'||kind==='file')?data:'');
    if((kind==='json'||kind==='tensor')&&typeof data==='string'){
      try{data=JSON.parse(data);}catch(_){/* leave string */}
    }
    return {kind:kind,mime:mime,data:data,src:src,meta:meta};
  }
  function numericSeries(value){
    if(Array.isArray(value))return value.map(Number).filter(function(v){return Number.isFinite(v);});
    if(typeof value==='string'){
      return value.split(/[\s,]+/).map(Number).filter(function(v){return Number.isFinite(v);});
    }
    return [];
  }
  function waveformPoints(series,width,height){
    if(!series.length)return '';
    var min=Math.min.apply(null,series),max=Math.max.apply(null,series),span=max-min||1;
    var pts=[];var last=series.length-1||1;
    for(var i=0;i<series.length;i++){
      var x=(i/last)*width;
      var y=height-((series[i]-min)/span)*height;
      pts.push(x.toFixed(1)+','+y.toFixed(1));
    }
    return pts.join(' ');
  }
  function renderUniversalOutput(env,stats){
    env=env||{kind:'text',data:''};stats=stats||{};
    var info=[];
    if(env.kind)info.push(env.kind.toUpperCase());
    if(env.mime)info.push(env.mime);
    if(stats.generated!=null&&stats.target!=null&&env.kind==='text')info.push(stats.generated+' / '+stats.target+' tokens');
    if(env.meta&&env.meta.size)info.push(String(env.meta.size));
    if(env.meta&&env.meta.dimensions)info.push(String(env.meta.dimensions));
    var header=h('div',null,h('strong',null,'OUTPUT'),h('span',null,info.join(' · ')||'Output preview'));
    if(env.kind==='image'){
      return h('div',{className:'mlb-status-sample generation mlb-output-viewer'},header,
        h('div',{className:'mlb-output-visual-card'}, env.src||typeof env.data==='string'
          ? h('img',{className:'mlb-output-image',src:env.src||env.data,alt:'Generated output'})
          : h('pre',null,pickText(env.data)||'No image available.')),
        env.meta&&Object.keys(env.meta).length?h('pre',{className:'mlb-output-meta'},JSON.stringify(env.meta,null,2)):null
      );
    }
    if(env.kind==='audio'){
      return h('div',{className:'mlb-status-sample generation mlb-output-viewer'},header,
        h('div',{className:'mlb-output-visual-card'}, env.src||typeof env.data==='string'
          ? h('audio',{className:'mlb-output-audio',controls:true,src:env.src||env.data})
          : h('pre',null,pickText(env.data)||'No audio available.')),
        env.meta&&Object.keys(env.meta).length?h('pre',{className:'mlb-output-meta'},JSON.stringify(env.meta,null,2)):null
      );
    }
    if(env.kind==='signal'){
      var series=numericSeries(env.data);
      var preview=series.length?series.slice(0,256):[];
      return h('div',{className:'mlb-status-sample generation mlb-output-viewer'},header,
        h('div',{className:'mlb-output-visual-card'},
          h('div',{className:'mlb-output-wave-head'},h('strong',null,'Waveform'),h('span',null,series.length?('samples: '+series.length):'No numeric samples')),
          series.length?h('svg',{className:'mlb-output-wave',viewBox:'0 0 320 96',preserveAspectRatio:'none'},
            h('polyline',{fill:'none',stroke:'currentColor',strokeWidth:'2',points:waveformPoints(preview,320,96)}))
          : h('pre',null,'No signal samples yet.'),
          h('pre',null,series.length?JSON.stringify(preview,null,2):pickText(env.data)||'No signal samples yet.')
        ),
        env.meta&&Object.keys(env.meta).length?h('pre',{className:'mlb-output-meta'},JSON.stringify(env.meta,null,2)):null
      );
    }
    if(env.kind==='json'||env.kind==='tensor'||env.kind==='embedding'||env.kind==='classification'){
      var text=typeof env.data==='string'?env.data:JSON.stringify(env.data,null,2);
      return h('div',{className:'mlb-status-sample generation mlb-output-viewer'},header,h('pre',null,text||'No structured output yet.'));
    }
    if(env.kind==='file'||env.kind==='video'){
      var source=env.src||pickText(env.data);
      return h('div',{className:'mlb-status-sample generation mlb-output-viewer'},header,
        h('div',{className:'mlb-output-visual-card'},
          env.kind==='video'&&source?h('video',{className:'mlb-output-video',controls:true,src:source}):null,
          source?h('a',{href:source,target:'_blank',rel:'noreferrer',className:'mlb-output-file-link'},source):h('pre',null,'No file available.')),
        env.meta&&Object.keys(env.meta).length?h('pre',{className:'mlb-output-meta'},JSON.stringify(env.meta,null,2)):null
      );
    }
    return h('div',{className:'mlb-status-sample generation mlb-output-viewer'},header,h('pre',null,pickText(env.data)||'No generated output yet.'));
  }

  class RuntimeErrorBoundary extends React.Component{
    constructor(props){
      super(props);
      this.state={error:null,epoch:0};
      this.unsubscribe=null;
    }
    componentDidMount(){
      var self=this;
      this.unsubscribe=this.props.store.subscribe(function(){
        if(self.state.error)self.setState(function(prev){return {error:null,epoch:prev.epoch+1};});
      });
    }
    componentWillUnmount(){if(this.unsubscribe)this.unsubscribe();}
    componentDidCatch(error){
      this.setState({error:error||new Error('Runtime view render failed')});
      try{console.error('MLBricks Studio runtime view render failed',error);}catch(_){}
    }
    render(){
      if(this.state.error){
        return h('section',{className:'mlb-runtime-section mlb-runtime-recovery'},
          h('h3',null,'Live view recovering'),
          h('div',{className:'mlb-runtime-recovery-copy'},'The model runtime is still active. Studio will rebuild this panel on the next telemetry update.'),
          h('small',null,escText(this.state.error&&this.state.error.message)||'Display update failed.'));
      }
      return h('div',{key:this.state.epoch,className:'mlb-runtime-boundary'},this.props.children);
    }
  }

  function Section(props){return h('section',{className:'mlb-runtime-section'+(props.className?' '+props.className:'')},h('h3',null,props.title),props.children);}
  function StaticMetric(props){
    return h('div',{className:'mlb-status-metric'},h('span',null,props.label),h('strong',null,props.value==null?'—':props.value),props.sub?h('small',null,props.sub):null);
  }
  function Grid(props){return h('div',{className:props.className||'mlb-status-metrics'},props.children);}

  function metric(selector,label,format,subSelector){
    var C=connected(function(s){return {value:selector(s),sub:subSelector?subSelector(s):null};},function(slice){
      var value=format?format(slice.value):slice.value;
      return h(StaticMetric,{label:label,value:value,sub:slice.sub});
    },shallowEqual);
    return function(store,key){return h(C,{store:store,key:key||label});};
  }

  function trainingPick(s,currentKey,storedKey){
    var live=s.live||{},entry=s.entry||{};
    var current=live[currentKey];
    var currentAttempt=['running','error','stopped'].indexOf(String(live.status||''))>=0;
    if(current!=null)return current;
    if(currentAttempt)return null;
    return entry[storedKey];
  }

  var TrainingHeroState=connected(function(s){
    var live=s.live||{},entry=s.entry||{};
    var label=live.status==='running'?'TRAINING':live.status==='done'?'COMPLETE':live.status==='error'?'ERROR':live.status==='stopped'?'STOPPED':entry.weights_ready?'TRAINED':'NOT STARTED';
    return {status:live.status||entry.training_status||'idle',label:label,message:live.message||'Configure training, then press Start Training.',overall:pct(live.overall),phase:live.phase||'idle'};
  },function(x){return h('div',{className:'mlb-training-status-top'},
    h('div',{className:'mlb-training-state '+x.status},h('strong',null,x.label),h('span',null,x.message)),
    h('div',{className:'mlb-training-percent'},h('strong',null,Math.round(x.overall)+'%'),h('span',null,x.phase))
  );},shallowEqual);

  var ProgressBar=connected(function(s){return pct((s.live||{}).overall);},function(value){return h('div',{className:'mlb-status-progress'},h('i',{style:{width:value+'%'}}));});

  var TrainingValidation=connected(function(s){
    var e=s.entry||{},c=s.config||{},d=s.dataset||{};
    return {dataset:d.name||'—',split:c.validation_split||'—',every:(c.validate_every||0)+' steps',steps:c.validation_steps==null?'—':c.validation_steps,
      latest:e.latest_validation_loss==null?'—':fmtFloat(e.latest_validation_loss,4),latestStep:e.latest_validation_step?'step '+e.latest_validation_step:null,
      sampleTokens:c.generate_on_validation?c.validation_generate_tokens:'Off',enabled:!!c.generate_on_validation,prompt:c.validation_prompt||'',sample:e.latest_validation_sample||'No validation sample generated yet.'};
  },function(x){return h(Section,{title:'Validation + Generated Sample'},
    h(Grid,{className:'mlb-validation-status-grid'},
      h(StaticMetric,{label:'Dataset',value:x.dataset}),h(StaticMetric,{label:'Validation Split',value:x.split}),
      h(StaticMetric,{label:'Validate Every',value:x.every}),h(StaticMetric,{label:'Validation Steps',value:x.steps}),
      h(StaticMetric,{label:'Latest Val',value:x.latest,sub:x.latestStep}),h(StaticMetric,{label:'Sample Tokens',value:x.sampleTokens})),
    h('div',{className:'mlb-status-sample'},
      h('div',null,h('strong',null,'VALIDATION GENERATION'),h('span',null,x.enabled?'Prompt: '+x.prompt:'Disabled in Training Setup')),
      h('pre',null,x.sample))
  );},shallowEqual);

  var EventLog=connected(function(s){return asArray(s.history);},function(history,props){
    var events=history.slice(-100);
    return h(Section,{title:props.title},h('div',{className:'mlb-training-log'},events.length?events.map(function(ev,index){
      var meta=[];if(ev.step!=null)meta.push('step '+ev.step);if(ev.generated_tokens!=null)meta.push(ev.generated_tokens+' tokens');if(ev.phase)meta.push(ev.phase);
      var extra=[];if(ev.tokens_per_sec!=null)extra.push(fmtInt(ev.tokens_per_sec)+' tok/s');if(ev.end_to_end_tokens_per_sec!=null)extra.push('E2E '+fmtInt(ev.end_to_end_tokens_per_sec)+' tok/s');
      if(ev.memory_allocated_gb!=null)extra.push('mem '+fmtFloat(ev.memory_allocated_gb,2)+' GB');if(ev.loss!=null)extra.push('loss '+fmtFloat(ev.loss,4));if(ev.ppl!=null)extra.push('ppl '+fmtFloat(ev.ppl,2));
      if(ev.val_loss!=null)extra.push('val '+fmtFloat(ev.val_loss,4));if(ev.val_ppl!=null)extra.push('val ppl '+fmtFloat(ev.val_ppl,2));
      return h('div',{className:'mlb-log-row '+(ev.status||''),key:ev.key||String(ev.event_seq||index)},h('span',null,meta.join(' · ')),h('strong',null,(ev.message||'Runtime event')+(extra.length?' · '+extra.join(' · '):'')));
    }):h('div',{className:'mlb-log-empty'},props.emptyText)));
  },function(a,b){return a===b;});

  function trainingEventStatus(ev){
    if(ev.step!=null)return 'Step '+ev.step;
    if(ev.phase){var phase=String(ev.phase).replace(/[_-]+/g,' ');return phase.charAt(0).toUpperCase()+phase.slice(1);}
    if(ev.status)return String(ev.status).toUpperCase();
    return 'Status';
  }

  var TrainingEventLog=connected(function(s){return asArray(s.history);},function(history,props){
    var events=history.slice(-100);
    return h(Section,{title:'Training Log'},h('div',{className:'mlb-training-log mlb-training-log-structured'},events.length?[
      h('div',{className:'mlb-training-log-head',key:'head'},
        h('span',null,'Status'),h('span',null,'Tok/s'),h('span',null,'E2E Tok/s'),h('span',null,'Loss'),h('span',null,'PPL'))
    ].concat(events.map(function(ev,index){
      var hasMetrics=ev.step!=null||ev.tokens_per_sec!=null||ev.end_to_end_tokens_per_sec!=null||ev.loss!=null||ev.ppl!=null;
      var key=ev.key||String(ev.event_seq||index);
      var status=trainingEventStatus(ev);
      if(!hasMetrics){
        return h('div',{className:'mlb-training-log-note '+(ev.status||''),key:key},h('strong',null,status),h('span',null,ev.message||'Runtime event'));
      }
      return h('div',{className:'mlb-training-log-metric-row '+(ev.status||''),key:key},
        h('strong',null,status),
        h('span',null,ev.tokens_per_sec==null?'—':fmtInt(ev.tokens_per_sec)),
        h('span',null,ev.end_to_end_tokens_per_sec==null?'—':fmtInt(ev.end_to_end_tokens_per_sec)),
        h('span',null,ev.loss==null?'—':fmtFloat(ev.loss,4)),
        h('span',null,ev.ppl==null?'—':fmtFloat(ev.ppl,2)));
    })):h('div',{className:'mlb-log-empty'},props.emptyText)));
  },function(a,b){return a===b;});

  var TrainingCheckpoint=connected(function(s){var e=s.entry||{},l=s.live||{},c=s.config||{};return {
    every:(c.checkpoint_every||0)+' steps',path:e.latest_checkpoint_path||e.checkpoint_path||l.checkpoint_path||'—',weights:e.weights_ready?'Available':'Not yet',status:e.training_status||'untrained',trainedAt:e.trained_at||'—'
  };},function(x){return h(Section,{title:'Checkpoints + Output'},h(Grid,{className:'mlb-validation-status-grid'},
    h(StaticMetric,{label:'Checkpoint Every',value:x.every}),h(StaticMetric,{label:'Latest Checkpoint',value:x.path}),h(StaticMetric,{label:'Weights',value:x.weights}),h(StaticMetric,{label:'Training Status',value:x.status}),h(StaticMetric,{label:'Trained At',value:x.trainedAt})));
  },shallowEqual);

  function TrainingMain(props){var store=props.store;return h('div',{className:'mlb-react-runtime-stack'},
    h(Section,{title:'Training Status',className:'mlb-training-status-hero'},h('div',{className:'mlb-react-runtime-hero'},h(TrainingHeroState,{store:store}),h(ProgressBar,{store:store}),
      h(Grid,null,
        metric(function(s){var l=s.live||{};var v=trainingPick(s,'step','trained_steps');return (v==null?0:v)+(l.max_steps?' / '+l.max_steps:'');},'Step')(store,'step'),
        metric(function(s){return trainingPick(s,'tokens_per_sec','avg_tokens_per_sec');},'Tok/s',fmtInt)(store,'tok'),
        metric(function(s){return trainingPick(s,'end_to_end_tokens_per_sec','avg_end_to_end_tokens_per_sec');},'E2E Tok/s',fmtInt)(store,'e2e'),
        metric(function(s){return trainingPick(s,'loss','last_loss');},'Loss',function(v){return fmtFloat(v,4);})(store,'loss'),
        metric(function(s){return trainingPick(s,'ppl','last_ppl');},'PPL',function(v){return fmtFloat(v,2);})(store,'ppl'),
        metric(function(s){var l=s.live||{},e=s.entry||{};var current=l.val_loss;var attempt=['running','error','stopped'].indexOf(String(l.status||''))>=0;return current!=null?current:(attempt?null:(e.latest_validation_loss!=null?e.latest_validation_loss:e.last_val_loss));},'Val Loss',function(v){return fmtFloat(v,4);})(store,'vloss'),
        metric(function(s){return trainingPick(s,'val_ppl','last_val_ppl');},'Val PPL',function(v){return fmtFloat(v,2);})(store,'vppl'),
        metric(function(s){return trainingPick(s,'memory_allocated_gb','__none');},'GPU Memory',function(v){return v==null?'—':fmtFloat(v,2)+' GB';},function(s){var n=(s.live||{}).memory_total_gb;return n==null?null:'of '+fmtFloat(n,1)+' GB';})(store,'mem'),
        metric(function(s){return trainingPick(s,'memory_peak_gb','memory_peak_gb');},'Peak Memory',function(v){return v==null?'—':fmtFloat(v,2)+' GB';})(store,'peak'),
        metric(function(s){var c=s.config||{},l=s.live||{},e=s.entry||{};if(c.execution_mode!=='compiled')return 'Not used';if(l.compile_seconds!=null)return fmtFloat(l.compile_seconds,1)+'s';var attempt=['running','error','stopped'].indexOf(String(l.status||''))>=0;return attempt?'Pending':(e.compile_seconds==null?'Pending':fmtFloat(e.compile_seconds,1)+'s');},'Compile')(store,'compile'),
        metric(function(s){return trainingPick(s,'tokens_seen','tokens_seen');},'Tokens',function(v){return Number(v||0).toLocaleString();})(store,'tokens'),
        metric(function(s){return (s.live||{}).elapsed_seconds;},'Elapsed',fmtDuration)(store,'elapsed')
      ))),
    h(TrainingValidation,{store:store}),
    h(TrainingEventLog,{store:store,emptyText:'Training has not started yet.'}),
    h(TrainingCheckpoint,{store:store})
  );}

  var GenerationHero=connected(function(s){var l=s.live||{},c=s.config||{};var label=l.status==='running'?'GENERATING':l.status==='done'?'COMPLETE':l.status==='error'?'ERROR':l.status==='stopped'?'STOPPED':'READY';return {
    status:l.status||'idle',label:label,message:l.message||'Configure generation, then press Generate Tokens.',overall:pct(l.overall),phase:l.phase||'idle',generated:Number(l.generated_tokens||0),target:Number(c.max_new_tokens||0),tok:l.tokens_per_sec,path:l.generation_mode||'Pending',temp:c.temperature,topk:c.top_k,topp:c.top_p,seed:c.seed
  };},function(x){return h(Section,{title:'Generation Status',className:'mlb-training-status-hero'},
    h('div',{className:'mlb-training-status-top'},h('div',{className:'mlb-training-state '+x.status},h('strong',null,x.label),h('span',null,x.message)),h('div',{className:'mlb-training-percent'},h('strong',null,Math.round(x.overall)+'%'),h('span',null,x.phase))),
    h('div',{className:'mlb-status-progress'},h('i',{style:{width:x.overall+'%'}})),
    h(Grid,null,h(StaticMetric,{label:'Generated',value:x.generated.toLocaleString()}),h(StaticMetric,{label:'Target',value:x.target.toLocaleString()}),h(StaticMetric,{label:'Tok/s',value:x.tok==null?'—':fmtFloat(x.tok,1)}),h(StaticMetric,{label:'Path',value:x.path}),h(StaticMetric,{label:'Temperature',value:x.temp}),h(StaticMetric,{label:'Top K',value:x.topk}),h(StaticMetric,{label:'Top P',value:x.topp}),h(StaticMetric,{label:'Seed',value:x.seed}))
  );},shallowEqual);

  var GenerationOutput=connected(function(s){var l=s.live||{},e=s.entry||{},c=s.config||{};return {prompt:c.prompt||'',output:normalizeOutputEnvelope(l,e),generated:Number(l.generated_tokens||0),target:Number(c.max_new_tokens||0)};},function(x){return h(Section,{title:'Generated Output'},
    h('div',{className:'mlb-status-prompt'},h('strong',null,'PROMPT'),h('pre',null,x.prompt)),
    renderUniversalOutput(x.output,{generated:x.generated,target:x.target})
  );},shallowEqual);

  var GenerationRuntime=connected(function(s){var l=s.live||{},e=s.entry||{},c=s.config||{},d=s.device||{};return {
    device:d.label||'Auto',backend:c.backend||'auto',execution:c.execution_mode||'eager',compile:c.execution_mode==='compiled'?(c.compile_mode||'default'):'Not used',precision:c.precision||'auto',
    source:l.runtime_source==='resident'?'Resident RAM/VRAM':l.runtime_source==='loaded'?'Loaded once':'Pending',path:l.generation_mode||'Pending',algos:displayList(l.generation_algorithms,'Compatibility path'),at:e.generated_at||'—'
  };},function(x){return h(Section,{title:'Runtime Used'},h(Grid,{className:'mlb-validation-status-grid'},
    h(StaticMetric,{label:'Device',value:x.device}),h(StaticMetric,{label:'Backend',value:x.backend}),h(StaticMetric,{label:'Execution',value:x.execution}),h(StaticMetric,{label:'Compile',value:x.compile}),h(StaticMetric,{label:'Precision',value:x.precision}),h(StaticMetric,{label:'Model Source',value:x.source}),h(StaticMetric,{label:'Generation Path',value:x.path}),h(StaticMetric,{label:'Algorithms',value:x.algos}),h(StaticMetric,{label:'Generated At',value:x.at})));
  },shallowEqual);

  function GenerationMain(props){var store=props.store;return h('div',{className:'mlb-react-runtime-stack'},h(GenerationHero,{store:store}),h(GenerationOutput,{store:store}),h(EventLog,{store:store,title:'Generation Log',emptyText:'Generation has not started yet.'}),h(GenerationRuntime,{store:store}));}

  var TrainingSide=connected(function(s){var l=s.live||{},e=s.entry||{},c=s.config||{},d=s.device||{},v=s.valid||{};var label=l.status==='running'?'TRAINING':l.status==='done'?'COMPLETE':l.status==='error'?'ERROR':l.status==='stopped'?'STOPPED':e.weights_ready?'TRAINED':'NOT STARTED';return {label:label,device:d.label||'Auto',backend:c.backend||'auto',execution:c.execution_mode||'eager',precision:c.precision||'auto',running:l.status==='running',valid:v.ok!==false,weights:!!e.weights_ready,locked:!!s.generationLocked,compat:v.compat||null};},function(x,props){var a=props.store.actions||{};return h('div',{className:'mlb-react-runtime-side'},
    h('div',{className:'mlb-runtime-summary'},h('h3',null,'Training Control'),h('div',null,h('span',null,'Status'),h('strong',null,x.label)),h('div',null,h('span',null,'Device'),h('strong',null,x.device)),h('div',null,h('span',null,'Backend'),h('strong',null,x.backend)),h('div',null,h('span',null,'Execution'),h('strong',null,x.execution)),h('div',null,h('span',null,'Precision'),h('strong',null,x.precision))),
    x.compat?h('div',{className:'mlb-compat-card '+(x.compat.ok?'compatible':'incompatible')},h('div',{className:'mlb-compat-head'},h('strong',null,x.compat.ok?'✓ Compatible':'✕ Not Compatible'),h('span',null,x.compat.ok?'Ready for training':'Fix the items below')),asArray(x.compat.checks).map(function(c,i){c=c||{};return h('div',{className:'mlb-compat-row '+(c.ok?'pass':'fail'),key:i},h('span',null,(c.ok?'✓ ':'✕ ')+escText(c.label)),h('strong',null,escText(c.detail)));})):null,
    x.running?h('button',{type:'button',className:'mlb-runtime-stop',onClick:a.stop},'Stop Training'):h('button',{type:'button',className:'mlb-runtime-start',disabled:!x.valid,onClick:a.start,title:!x.valid?'Fix training compatibility/settings before starting':'Start training'},'Start Training'),
    h('button',{type:'button',className:'mlb-vram-clean-btn',disabled:x.running,onClick:a.cleanVram,title:x.running?'Stop the active runtime before cleaning GPU memory':'Release cached model runtimes and empty the CUDA allocator cache'},'Clean GPU VRAM'),
    h('button',{type:'button',className:'mlb-runtime-cancel',onClick:a.cancel},'Cancel'),
    x.weights?h('button',{type:'button',className:'mlb-generate-btn',disabled:x.locked,onClick:a.openGeneration,title:x.locked?'Generation is disabled while training is running':'Open generation'},'Open Generation'):null
  );},shallowEqual);

  var GenerationSide=connected(function(s){var l=s.live||{},e=s.entry||{},c=s.config||{},d=s.device||{};var label=l.status==='running'?'GENERATING':l.status==='done'?'COMPLETE':l.status==='error'?'ERROR':l.status==='stopped'?'STOPPED':'READY';return {label:label,device:d.label||'Auto',generated:Number(l.generated_tokens||0),target:Number(c.max_new_tokens||0),weights:!!e.weights_ready,running:l.status==='running',locked:!!s.generationLocked,busy:!!s.otherRuntimeBusy};},function(x,props){var a=props.store.actions||{};var disabled=!x.weights||x.locked||x.busy;return h('div',{className:'mlb-react-runtime-side'},
    h('div',{className:'mlb-runtime-summary'},h('h3',null,'Generation Control'),h('div',null,h('span',null,'Status'),h('strong',null,x.label)),h('div',null,h('span',null,'Device'),h('strong',null,x.device)),h('div',null,h('span',null,'Generated'),h('strong',null,x.generated+' / '+x.target)),h('div',null,h('span',null,'Weights'),h('strong',null,x.weights?'Available':'Missing'))),
    x.running?h('button',{type:'button',className:'mlb-runtime-stop',onClick:a.stop},'Stop Generation'):h('button',{type:'button',className:'mlb-runtime-start',disabled:disabled,onClick:a.start,title:!x.weights?'Train or load model weights before generation':x.locked?'Generation is disabled while training is running':'Generate tokens'},'Generate Tokens'),
    h('button',{type:'button',className:'mlb-runtime-cancel',onClick:a.cancel},'Cancel')
  );},shallowEqual);

  function mountStatus(opts){
    var key=String(opts.instanceId)+'::'+String(opts.modelId)+'::'+String(opts.mode);
    var store=stores.get(key);
    if(!store){store=createStore(opts.snapshot||{},opts.actions||{});stores.set(key,store);}else{store.setActions(opts.actions||{});store.replace(opts.snapshot||{});}
    var old=mounts.get(key);
    if(old && (old.main!==opts.main||old.side!==opts.side)){
      try{ReactDOM.unmountComponentAtNode(old.main);}catch(_){}
      try{ReactDOM.unmountComponentAtNode(old.side);}catch(_){}
    }
    ReactDOM.render(h(RuntimeErrorBoundary,{store:store},h(opts.mode==='train'?TrainingMain:GenerationMain,{store:store})),opts.main);
    ReactDOM.render(h(RuntimeErrorBoundary,{store:store},h(opts.mode==='train'?TrainingSide:GenerationSide,{store:store})),opts.side);
    mounts.set(key,{main:opts.main,side:opts.side,instanceId:String(opts.instanceId)});
    return key;
  }

  function updateStatus(opts){
    var key=String(opts.instanceId)+'::'+String(opts.modelId)+'::'+String(opts.mode);
    var store=stores.get(key);
    if(store)store.replace(opts.snapshot||{});
    return !!store;
  }

  function unmountInstance(instanceId){
    instanceId=String(instanceId);
    Array.from(mounts.entries()).forEach(function(pair){var key=pair[0],m=pair[1];if(m.instanceId!==instanceId)return;try{ReactDOM.unmountComponentAtNode(m.main);}catch(_){}try{ReactDOM.unmountComponentAtNode(m.side);}catch(_){}mounts.delete(key);stores.delete(key);});
  }

  global.__MLBReactRuntime={available:true,version:'react-runtime-islands-v1',mountStatus:mountStatus,updateStatus:updateStatus,unmountInstance:unmountInstance};
})(typeof window!=='undefined'?window:this);
