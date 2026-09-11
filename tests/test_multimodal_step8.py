from __future__ import annotations

import math
import threading
from pathlib import Path

import torch

import mlb_studio.model_runtime as runtime
from mlb_studio import data as data_api


class Split:
    def __init__(self,data): self.data=data; self.column_names=list(data)
    def __getitem__(self,key):
        if isinstance(key,int): return {k:v[key] for k,v in self.data.items()}
        return self.data[key]
    def __len__(self): return len(next(iter(self.data.values()))) if self.data else 0


def edge(a,b,kind="main",target_port="main_in",source_port="main_out"):
    return {"id":f"{a}-{b}-{target_port}","source":a,"target":b,"kind":kind,"source_port":source_port,"target_port":target_port}


def named(a,b,key): return edge(a,b,"named",f"named_in:{key}","main_out")


def mm_jepa_entry():
    nodes=[
        {"id":"im","type":"image_input","name":"Image","params":{"channels":1}},
        {"id":"tx","type":"text_input","name":"Text","params":{"input_key":"text"}},
        {"id":"ie","type":"jepa_encoder","name":"Image Encoder","params":{"modality":"image","role":"context","latent_dim":16,"hidden_dim":16,"in_channels":1}},
        {"id":"te","type":"jepa_encoder","name":"Text Encoder","params":{"modality":"text","role":"context","latent_dim":16,"hidden_dim":16,"vocab_size":257}},
        {"id":"ip","type":"jepa_predictor","name":"I2T","params":{"latent_dim":16,"hidden_dim":24}},
        {"id":"tp","type":"jepa_predictor","name":"T2I","params":{"latent_dim":16,"hidden_dim":24}},
        {"id":"il","type":"jepa_latent_loss","name":"I Loss","params":{"loss":"mse","normalize":"true"}},
        {"id":"tl","type":"jepa_latent_loss","name":"T Loss","params":{"loss":"mse","normalize":"true"}},
        {"id":"sum","type":"tensor_add","name":"Loss Sum","params":{}},
        {"id":"out","type":"tensor_output","name":"Out","params":{}},
    ]
    edges=[edge("im","ie"),edge("tx","te"),edge("ie","ip"),edge("te","tp"),named("ip","il","prediction"),named("te","il","target"),named("tp","tl","prediction"),named("ie","tl","target"),named("il","sum","a"),named("tl","sum","b"),edge("sum","out")]
    return {"name":"Multimodal JEPA","task":"Cross-modal latent prediction","architecture":{"nodes":nodes,"edges":edges},"requirements":{"modality":"multimodal","training_mode":"multimodal","training_task":"multimodal_jepa","requires_tokenizer":False}}


def fusion_entry():
    nodes=[
        {"id":"im","type":"image_input","name":"Image","params":{"channels":1}},
        {"id":"ic","type":"conv2d","name":"IC","params":{"in_channels":1,"out_channels":4,"kernel_size":3,"padding":1}},
        {"id":"ip","type":"adaptive_avgpool2d","name":"IP","params":{"output_size":1}},
        {"id":"if","type":"flatten","name":"IF","params":{"start_dim":1}},
        {"id":"se","type":"signal_input","name":"Sensor","params":{"input_key":"sensor"}},
        {"id":"su","type":"unsqueeze","name":"SU","params":{"dim":1}},
        {"id":"sc","type":"conv1d","name":"SC","params":{"in_channels":1,"out_channels":4,"kernel_size":3,"padding":1}},
        {"id":"sp","type":"adaptive_avgpool1d","name":"SP","params":{"output_size":1}},
        {"id":"sf","type":"flatten","name":"SF","params":{"start_dim":1}},
        {"id":"cat","type":"concat","name":"Fuse","params":{"dim":-1}},
        {"id":"head","type":"classifier","name":"Head","params":{"dim":8,"hidden_size":8,"classes":3}},
    ]
    edges=[edge("im","ic"),edge("ic","ip"),edge("ip","if"),edge("se","su"),edge("su","sc"),edge("sc","sp"),edge("sp","sf"),named("if","cat","a"),named("sf","cat","b"),edge("cat","head")]
    return {"name":"Sensor + Vision Fusion","task":"Multimodal classification","architecture":{"nodes":nodes,"edges":edges},"requirements":{"modality":"multimodal","training_mode":"multimodal","training_task":"sensor_vision_fusion","requires_tokenizer":False}}


def config(tmp_path):
    return {"seed":7,"batch_size":2,"gradient_accumulation":1,"budget_type":"steps","max_steps":1,"optimizer":"sgd","learning_rate":0.01,"weight_decay":0.0,"warmup_steps":0,"validation_split":"validation","validate_every":1,"validation_steps":1,"checkpoint_every":0,"device":"cpu","backend":"pytorch","execution_mode":"eager","precision":"fp32","output_dir":str(tmp_path)}


def patch_save(monkeypatch):
    original=runtime.IMPORT_POOL.resolve_api
    def save(model,path,metadata=None):
        path=Path(path); path.mkdir(parents=True,exist_ok=True); (path/"model.pt").write_bytes(b"test")
    monkeypatch.setattr(runtime.IMPORT_POOL,"resolve_api",lambda key: save if key=="lifecycle.save" else original(key))


def test_step8_gallery_has_two_multimodal_models_and_glass_box_nodes():
    js=(Path(__file__).resolve().parents[1]/"frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    assert 'name:"Multimodal JEPA"' in js
    assert 'name:"Sensor + Vision Fusion"' in js
    assert 'input_key:"text"' in js
    assert 'input_key:"sensor"' in js
    assert 'Image → Text Predictor' in js
    assert 'Vision + Sensor Fusion' in js


def test_multimodal_demo_data_are_aligned(monkeypatch):
    class Factory:
        @staticmethod
        def from_dict(d): return Split(d)
    class DS: Dataset=Factory
    monkeypatch.setattr(data_api,"_datasets",lambda:DS)
    a=data_api.generate_demo_dataset("multimodal_image_text",samples=6)
    b=data_api.generate_demo_dataset("sensor_vision",samples=6)
    assert set(a.column_names)=={"image","text"}
    assert set(b.column_names)=={"image","sensor","label"}
    assert len(a)==len(b)==8  # generator enforces minimum 8


def test_multimodal_jepa_compiles_and_has_scalar_loss():
    e=mm_jepa_entry(); state={"project":{"task":e["task"],"context_length":16,"model_settings":{"vocab_size":257}},"custom_components":{}}
    compiled,_=runtime.compile_builder_model(state,e,{}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=True)
    image=torch.rand(2,1,16,16); text=torch.randint(0,257,(2,16))
    loss=compiled.raw_model(image,graph_named={"text":text})
    assert loss.numel()==1 and torch.isfinite(loss).all()
    loss.backward()
    assert any(p.grad is not None for p in compiled.raw_model.parameters() if p.requires_grad)


def test_sensor_vision_fusion_compiles():
    e=fusion_entry(); state={"project":{"task":e["task"]},"custom_components":{}}
    compiled,_=runtime.compile_builder_model(state,e,{}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=True)
    y=compiled.raw_model(torch.rand(2,1,16,16),graph_named={"sensor":torch.rand(2,32)})
    assert y.shape==(2,3)


def test_step8_multimodal_jepa_training_runs(monkeypatch,tmp_path):
    patch_save(monkeypatch); e=mm_jepa_entry(); n=8
    images=[[[float((r+c+i)%3==0) for c in range(16)] for r in range(16)] for i in range(n)]
    texts=["vertical pattern" if i%2==0 else "diagonal pattern" for i in range(n)]
    ds={"train":Split({"image":images,"text":texts}),"validation":Split({"image":images[:4],"text":texts[:4]})}
    result=runtime.train_builder_model(state={"project":{"task":e["task"],"context_length":16,"model_settings":{"vocab_size":257}},"custom_components":{}},model_entry=e,dataset=ds,dataset_meta={"name":"Aligned"},config=config(tmp_path),progress=lambda e:None,stop_event=threading.Event())
    assert result["model_update"]["training_mode"]=="multimodal"
    assert result["model_update"]["training_task"]=="multimodal_jepa"
    assert math.isfinite(result["model_update"]["last_loss"])


def test_step8_sensor_vision_training_runs(monkeypatch,tmp_path):
    patch_save(monkeypatch); e=fusion_entry(); n=8
    images=[[[float((r+c+i)%3==0) for c in range(16)] for r in range(16)] for i in range(n)]
    sensors=[[math.sin(2*math.pi*(1+i%3)*j/32) for j in range(32)] for i in range(n)]
    labels=[i%3 for i in range(n)]
    ds={"train":Split({"image":images,"sensor":sensors,"label":labels}),"validation":Split({"image":images[:4],"sensor":sensors[:4],"label":labels[:4]})}
    result=runtime.train_builder_model(state={"project":{"task":e["task"]},"custom_components":{}},model_entry=e,dataset=ds,dataset_meta={"name":"Fusion"},config=config(tmp_path),progress=lambda e:None,stop_event=threading.Event())
    assert result["model_update"]["training_task"]=="sensor_vision_fusion"
    assert "accuracy" in result["model_update"]["multimodal_metrics"]


def test_multimodal_universal_inference_accepts_dict():
    e=fusion_entry(); state={"project":{"task":e["task"]},"custom_components":{}}
    compiled,_=runtime.compile_builder_model(state,e,{}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=False)
    value={"image":[[0.0]*16 for _ in range(16)],"sensor":[0.0]*32}
    out=runtime.run_universal_inference(compiled,value,input_kind="multimodal",output_type="classifier",task="classification")
    assert out["kind"]=="classification"
