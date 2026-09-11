from __future__ import annotations

import math
import threading
from pathlib import Path

import torch

import mlb_studio.model_runtime as runtime
from mlb_studio.graph import primitive_catalog
from mlb_studio import data as data_api


class Split:
    def __init__(self, data):
        self.data=data; self.column_names=list(data)
    def __getitem__(self,key):
        if isinstance(key,int): return {k:v[key] for k,v in self.data.items()}
        return self.data[key]
    def __len__(self): return len(next(iter(self.data.values()))) if self.data else 0


def edge(a,b,kind="main",target_port="main_in",source_port="main_out"):
    return {"id":f"{a}-{b}-{target_port}","source":a,"target":b,"kind":kind,"source_port":source_port,"target_port":target_port}


def config(tmp_path):
    return {"seed":7,"batch_size":2,"gradient_accumulation":1,"budget_type":"steps","max_steps":1,
            "optimizer":"sgd","learning_rate":0.01,"weight_decay":0.0,"warmup_steps":0,
            "validation_split":"validation","validate_every":1,"validation_steps":1,"checkpoint_every":0,
            "device":"cpu","backend":"pytorch","execution_mode":"eager","precision":"fp32","output_dir":str(tmp_path)}


def patch_save(monkeypatch):
    original=runtime.IMPORT_POOL.resolve_api
    def save(model,path,metadata=None):
        path=Path(path); path.mkdir(parents=True,exist_ok=True); (path/"model.pt").write_bytes(b"test")
    monkeypatch.setattr(runtime.IMPORT_POOL,"resolve_api",lambda key: save if key=="lifecycle.save" else original(key))


def tts_entry():
    nodes=[
        {"id":"x","type":"text_input","name":"Text","params":{}},
        {"id":"e","type":"embedding","name":"Embedding","params":{"vocab_size":257,"embedding_dim":16}},
        {"id":"p","type":"reduce_mean","name":"Pool","params":{"dim":1,"keepdim":False}},
        {"id":"a","type":"audio_token_predictor","name":"Predictor","params":{"in_features":16,"latent_dim":24,"hidden_dim":24}},
        {"id":"d","type":"audio_codec_decoder","name":"Decoder","params":{"latent_dim":24,"output_samples":64,"hidden_dim":32}},
        {"id":"o","type":"audio_output","name":"Audio","params":{"sample_rate":16000}},
    ]
    edges=[edge(nodes[i]["id"],nodes[i+1]["id"]) for i in range(len(nodes)-1)]
    return {"name":"Neural TTS","task":"Text to speech","architecture":{"nodes":nodes,"edges":edges},
            "requirements":{"modality":"text","training_mode":"audio_generation","training_task":"tts","requires_tokenizer":False}}


def test_step7_audio_components_are_public():
    by={x["type"]:x for x in primitive_catalog()}
    for t in ("speaker_embedding","audio_codec_encoder","audio_token_predictor","audio_codec_decoder","audio_output"):
        assert t in by
    assert by["audio_output"]["category"]=="Outputs"


def test_audio_demo_multispeaker_has_reference_audio(monkeypatch):
    class Factory:
        @staticmethod
        def from_dict(d): return Split(d)
    class DS: Dataset=Factory
    monkeypatch.setattr(data_api,"_datasets",lambda:DS)
    ds=data_api.generate_demo_dataset("multispeaker_speech",samples=4,sequence_length=8)
    assert "reference_audio" in ds.column_names
    assert len(ds[0]["audio"])==len(ds[0]["reference_audio"])


def test_neural_tts_compiles_and_outputs_waveform():
    e=tts_entry()
    state={"project":{"model_settings":{"vocab_size":257}},"custom_components":{}}
    compiled,_=runtime.compile_builder_model(state,e,{}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=True)
    y=compiled.raw_model(torch.randint(0,257,(2,16)))
    assert y.shape==(2,64)
    assert torch.isfinite(y).all()


def test_voice_condition_named_root_input_routes_independently():
    nodes=[
        {"id":"x","type":"text_input","name":"Text","params":{}},
        {"id":"e","type":"embedding","name":"E","params":{"vocab_size":257,"embedding_dim":8}},
        {"id":"p","type":"reduce_mean","name":"P","params":{"dim":1}},
        {"id":"s","type":"feature_input","name":"Speaker","params":{"input_key":"speaker_id"}},
        {"id":"se","type":"speaker_embedding","name":"SE","params":{"num_speakers":4,"embedding_dim":4}},
        {"id":"c","type":"concat","name":"Fuse","params":{"dim":-1}},
        {"id":"a","type":"audio_token_predictor","name":"AP","params":{"in_features":12,"latent_dim":16,"hidden_dim":16}},
        {"id":"d","type":"audio_codec_decoder","name":"D","params":{"latent_dim":16,"output_samples":32,"hidden_dim":24}},
        {"id":"o","type":"audio_output","name":"O","params":{}},
    ]
    edges=[edge("x","e"),edge("e","p"),edge("s","se"),edge("p","c","named","named_in:a"),edge("se","c","named","named_in:b"),edge("c","a"),edge("a","d"),edge("d","o")]
    ent={"name":"Voice-conditioned TTS","task":"Speaker-conditioned TTS","architecture":{"nodes":nodes,"edges":edges},"requirements":{"modality":"text","training_mode":"audio_generation","training_task":"voice_tts"}}
    compiled,_=runtime.compile_builder_model({"project":{"model_settings":{"vocab_size":257}},"custom_components":{}},ent,{}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=True)
    y=compiled.raw_model(torch.randint(0,257,(2,12)),graph_named={"speaker_id":torch.tensor([0,1])})
    assert y.shape==(2,32)


def test_audio_training_runs_one_step(monkeypatch,tmp_path):
    patch_save(monkeypatch)
    e=tts_entry(); n=8; length=64
    texts=[f"hello sample {i}" for i in range(n)]
    audio=[[math.sin(2*math.pi*(1+i%3)*j/length) for j in range(length)] for i in range(n)]
    ds={"train":Split({"text":texts,"audio":audio}),"validation":Split({"text":texts[:4],"audio":audio[:4]})}
    result=runtime.train_builder_model(state={"project":{"task":"Text to speech","model_settings":{"vocab_size":257}},"custom_components":{}},model_entry=e,dataset=ds,dataset_meta={"name":"Speech"},config=config(tmp_path),progress=lambda e:None,stop_event=threading.Event())
    assert result["model_update"]["training_mode"]=="audio_generation"
    assert result["model_update"]["training_task"]=="tts"
    assert math.isfinite(result["model_update"]["last_loss"])


def test_step7_gallery_has_audio_models_and_glass_box_nodes():
    js=(Path(__file__).resolve().parents[1]/"frontend/src/legacy-builder.js").read_text(encoding="utf-8")
    for name in ("Neural TTS","Voice-conditioned TTS","Voice Clone educational template","Sound Generator","Music Generator"):
        assert f'name:"{name}"' in js
    assert 'add("speaker_embedding","Speaker Embedding"' in js
    assert 'add("audio_codec_encoder","Reference Audio Encoder"' in js
    assert 'add("audio_token_predictor"' in js
    assert 'add("audio_codec_decoder","Neural Audio Decoder · 256 Samples"' in js
    assert 'add("audio_output","Audio Waveform Output"' in js


def test_text_prompt_audio_models_use_universal_runtime_path():
    source=(Path(__file__).resolve().parents[1]/"src/mlb_studio/builder.py").read_text(encoding="utf-8")
    assert 'output_type == "audio_output"' in source
    assert 'training_mode == "audio_generation"' in source


def test_audio_universal_inference_returns_wav_data_uri():
    e=tts_entry()
    state={"project":{"model_settings":{"vocab_size":257}},"custom_components":{}}
    compiled,_=runtime.compile_builder_model(state,e,{}, {"device":"cpu","backend":"pytorch","precision":"fp32","execution_mode":"eager"},for_training=False)
    output=runtime.run_universal_inference(compiled,"hello audio",input_kind="text",output_type="audio_output",task="tts")
    assert output["kind"]=="audio"
    assert output["mime"]=="audio/wav"
    assert str(output["data"]).startswith("data:audio/wav;base64,")
