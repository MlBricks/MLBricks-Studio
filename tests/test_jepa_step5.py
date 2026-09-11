from __future__ import annotations

import math
import threading
from pathlib import Path

import pytest
import torch

from mlb_studio.data import prepare_jepa_dataset
from mlb_studio.graph import primitive_catalog
import mlb_studio.model_runtime as runtime
from mlb_studio.runner import EXECUTABLE_TYPES


class MiniDataset:
    def __init__(self, rows):
        self.rows=[dict(r) for r in rows]
        self.column_names=list(self.rows[0]) if self.rows else []
    def map(self, fn):
        return MiniDataset([fn(dict(r)) for r in self.rows])
    def __getitem__(self, key):
        if isinstance(key,str): return [r[key] for r in self.rows]
        return self.rows[key]
    def __len__(self): return len(self.rows)


class Split:
    def __init__(self,data): self.data=data; self.column_names=list(data)
    def __getitem__(self,key): return self.data[key]
    def __len__(self): return len(next(iter(self.data.values()))) if self.data else 0


def edge(a,b,*,source_port="main_out",target_port="main_in",kind="main"):
    return {"id":f"{a}-{b}-{source_port}-{target_port}","source":a,"target":b,"kind":kind,"source_port":source_port,"target_port":target_port}


def jepa_entry(modality="signal"):
    input_type={"image":"image_input","video":"video_input","text":"text_input","audio":"audio_input","signal":"signal_input"}[modality]
    nodes=[
        {"id":"x","type":input_type,"name":"Input","params":{}},
        {"id":"mask","type":"jepa_mask","name":"JEPA Mask","params":{"mask_ratio":0.35,"mask_value":0.0,"mode":"auto"}},
        {"id":"ctx","type":"jepa_encoder","name":"Context Encoder","params":{"modality":modality,"role":"context","latent_dim":16,"hidden_dim":12,"vocab_size":257,"in_channels":1}},
        {"id":"tgt","type":"jepa_encoder","name":"Target Encoder","params":{"modality":modality,"role":"target","latent_dim":16,"hidden_dim":12,"vocab_size":257,"in_channels":1}},
        {"id":"pred","type":"jepa_predictor","name":"JEPA Predictor","params":{"latent_dim":16,"hidden_dim":24,"dropout":0.0}},
        {"id":"loss","type":"jepa_latent_loss","name":"JEPA Latent Loss","params":{"loss":"mse","normalize":True}},
        {"id":"out","type":"tensor_output","name":"Loss Output","params":{}},
    ]
    edges=[
        edge("x","mask"),
        edge("mask","ctx",source_port="named_out:context"),
        edge("mask","tgt",source_port="named_out:target"),
        edge("ctx","pred"),
        edge("pred","loss",source_port="main_out",target_port="named_in:prediction",kind="named"),
        edge("tgt","loss",source_port="main_out",target_port="named_in:target",kind="named"),
        edge("loss","out"),
    ]
    return {
        "name":f"{modality.title()} JEPA",
        "context_length":32,
        "architecture":{"nodes":nodes,"edges":edges},
        "requirements":{"modality":modality,"training_mode":"jepa","training_task":"jepa","requires_tokenizer":False},
    }


def compile_entry(entry):
    return runtime.compile_builder_model(
        {"project":{"task":"Masked latent prediction"},"custom_components":{}},entry,{},
        {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=True,
    )[0]


def test_step5_catalog_exposes_universal_jepa_primitives_and_data_preparation():
    by_type={x["type"]:x for x in primitive_catalog()}
    assert {"jepa_prepare","jepa_mask","jepa_encoder","jepa_predictor","jepa_latent_loss"} <= set(by_type)
    assert "jepa_prepare" in EXECUTABLE_TYPES
    assert {p["id"] for p in by_type["jepa_mask"]["runtime_ports"]["outputs"]} == {"context","target"}
    assert {p["id"] for p in by_type["jepa_latent_loss"]["runtime_ports"]["inputs"]} == {"prediction","target"}


def test_jepa_preparation_makes_visible_common_tensor_field():
    text=prepare_jepa_dataset(MiniDataset([{"text":"hello"}]),modality="text",sequence_length=8)
    assert len(text["jepa_input"][0]) == 8
    assert text["jepa_input"][0][0] == ord("h") + 1

    signal=prepare_jepa_dataset(MiniDataset([{"signal":[1.0,2.0,3.0]}]),modality="signal",sequence_length=5)
    assert len(signal["jepa_input"][0]) == 5
    assert max(abs(v) for v in signal["jepa_input"][0]) <= 1.0

    image=prepare_jepa_dataset(MiniDataset([{"image":[[0.0,1.0],[1.0,0.0]]}]),modality="image",image_size=8)
    assert len(image["jepa_input"][0]) == 1
    assert len(image["jepa_input"][0][0]) == 8


@pytest.mark.parametrize("modality,shape,dtype",[
    ("image",(3,1,16,16),torch.float32),
    ("video",(2,4,1,12,12),torch.float32),
    ("text",(3,32),torch.long),
    ("audio",(3,64),torch.float32),
    ("signal",(3,64),torch.float32),
])
def test_all_five_jepa_modalities_compile_and_backprop(modality,shape,dtype):
    entry=jepa_entry(modality)
    compiled=compile_entry(entry)
    if dtype==torch.long:
        x=torch.randint(1,200,shape,dtype=dtype)
    else:
        x=torch.randn(*shape,dtype=dtype)
    loss=compiled.raw_model(x)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    ctx=compiled.raw_model.mods["ctx"]
    tgt=compiled.raw_model.mods["tgt"]
    assert any(p.grad is not None for p in ctx.parameters())
    assert all(p.grad is None and not p.requires_grad for p in tgt.parameters())


def test_text_jepa_compile_does_not_load_hf_tokenizer(monkeypatch):
    monkeypatch.setattr(runtime,"_tokenizer_for",lambda *a,**k: (_ for _ in ()).throw(AssertionError("HF tokenizer should not load for Text JEPA")))
    compiled,tokenizer=runtime.compile_builder_model(
        {"project":{},"custom_components":{}},jepa_entry("text"),{},
        {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=True,
    )
    assert tokenizer is None
    assert compiled.training_model is None


def test_signal_jepa_training_uses_ema_target_and_saves(monkeypatch,tmp_path):
    original=runtime.IMPORT_POOL.resolve_api
    saved={}
    def fake_save(model,path,metadata=None):
        path=Path(path); path.mkdir(parents=True,exist_ok=True); (path/"model.pt").write_bytes(b"test")
        saved["metadata"]=metadata or {}
    def resolve(key):
        if key=="lifecycle.save": return fake_save
        return original(key)
    monkeypatch.setattr(runtime.IMPORT_POOL,"resolve_api",resolve)

    signals=[]
    for i in range(24):
        signals.append([math.sin(2*math.pi*(1+(i%3))*j/63.0) for j in range(64)])
    ds={"train":Split({"signal":signals}),"validation":Split({"signal":signals[:8]})}
    events=[]
    result=runtime.train_builder_model(
        state={"project":{"task":"Masked latent prediction","context_length":64},"custom_components":{}},
        model_entry=jepa_entry("signal"),dataset=ds,dataset_meta={"name":"Signal JEPA Demo"},
        config={"seed":3,"batch_size":4,"gradient_accumulation":1,"budget_type":"steps","max_steps":2,"max_samples":100,
                "epochs":1,"optimizer":"adamw","learning_rate":0.002,"weight_decay":0.0,"beta1":0.9,"beta2":0.95,
                "validation_split":"validation","validate_every":1,"validation_steps":1,"checkpoint_every":0,"jepa_target_momentum":0.99,
                "device":"cpu","backend":"pytorch","execution_mode":"eager","precision":"fp32","output_dir":str(tmp_path)},
        progress=events.append,stop_event=threading.Event(),
    )
    update=result["model_update"]
    assert update["training_mode"] == "jepa"
    assert update["training_task"] == "jepa"
    assert update["jepa_modality"] == "signal"
    assert math.isfinite(update["last_loss"])
    assert saved["metadata"]["training_mode"] == "jepa"
    assert any(e.get("samples_per_sec") for e in events if e.get("phase")=="train")


def test_step5_gallery_contains_all_jepa_models_and_visible_data_preparation():
    root=Path(__file__).resolve().parents[1]
    js=(root/"frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    for name in ("Image JEPA","Video JEPA","Text JEPA","Audio JEPA","Signal JEPA"):
        assert f'name:"{name}"' in js
    assert 'const prep=makeNode(cat(catalog,"jepa_prepare"))' in js
    assert 'const context=add("jepa_encoder","Context Encoder"' in js
    assert 'const target=add("jepa_encoder","Target Encoder · EMA"' in js
    assert 'const predictor=add("jepa_predictor","JEPA Predictor"' in js
    assert 'const loss=add("jepa_latent_loss","JEPA Latent Loss"' in js
    assert 'training_mode:isJEPA?"jepa"' in js
