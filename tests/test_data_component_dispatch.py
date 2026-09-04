from __future__ import annotations

import pytest

from mlb_studio.graph import new_project, primitive_catalog
from mlb_studio import runner


def _defaults(component_type):
    item = next(x for x in primitive_catalog() if x["type"] == component_type)
    return {field["key"]: field.get("value") for field in item.get("api") or []}


def _state_for(types):
    state = new_project()
    ws = state["workspaces"]["data"]
    component = state["components"][ws["root_component_id"]]
    nodes = []
    for index, typ in enumerate(types):
        item = next(x for x in primitive_catalog() if x["type"] == typ)
        nodes.append({
            "id": f"n{index}",
            "type": typ,
            "name": item["name"],
            "params": _defaults(typ),
        })
    component["nodes"] = nodes
    component["edges"] = [
        {"id": f"e{i}", "source": f"n{i}", "target": f"n{i+1}", "kind": "main"}
        for i in range(len(nodes) - 1)
    ]
    return state


SOURCE_DISPATCH = {
    "manual_dataset": "load_manual_text_dataset",
    "hf_dataset": "load_huggingface_dataset",
    "kaggle_dataset": "load_kaggle_dataset",
    "url_dataset": "load_url_dataset",
    "local_dataset": "load_local_dataset",
}

PROCESSOR_DISPATCH = {
    "text_process": "process_text_dataset",
    "train_test_split": "train_validation_test_split",
    "tokenize_text": "tokenize_text_dataset",
    "image_process": "process_image_dataset",
    "audio_process": "process_audio_dataset",
    "batch_data": "make_torch_dataloader",
}


@pytest.mark.parametrize("component_type,function_name", sorted(SOURCE_DISPATCH.items()))
def test_every_data_source_dispatches_to_the_correct_data_api(monkeypatch, component_type, function_name):
    source_result = object()
    finished_result = object()
    calls = []

    def source(*args, **kwargs):
        calls.append((function_name, args, kwargs))
        return source_result

    def prepared(value, **kwargs):
        assert value is source_result
        calls.append(("prepared_dataset_output", (value,), kwargs))
        return finished_result

    monkeypatch.setattr(runner.data_api, function_name, source)
    monkeypatch.setattr(runner.data_api, "prepared_dataset_output", prepared)

    result = runner.execute_data_pipeline(_state_for([component_type, "prepared_dataset"]))
    assert result is finished_result
    assert calls[0][0] == function_name
    assert calls[-1][0] == "prepared_dataset_output"


@pytest.mark.parametrize("component_type,function_name", sorted(PROCESSOR_DISPATCH.items()))
def test_every_data_processor_dispatches_to_the_correct_data_api(monkeypatch, component_type, function_name):
    source_result = object()
    processed_result = object()
    finished_result = object()
    calls = []

    monkeypatch.setattr(
        runner.data_api,
        "load_manual_text_dataset",
        lambda *args, **kwargs: source_result,
    )

    def processor(value, *args, **kwargs):
        assert value is source_result
        calls.append((function_name, args, kwargs))
        return processed_result

    def prepared(value, **kwargs):
        assert value is processed_result
        return finished_result

    monkeypatch.setattr(runner.data_api, function_name, processor)
    monkeypatch.setattr(runner.data_api, "prepared_dataset_output", prepared)

    result = runner.execute_data_pipeline(
        _state_for(["manual_dataset", component_type, "prepared_dataset"])
    )
    assert result is finished_result
    assert calls and calls[0][0] == function_name


def test_prepared_dataset_output_dispatches_as_final_data_component(monkeypatch):
    source_result = object()
    finished_result = object()
    monkeypatch.setattr(
        runner.data_api,
        "load_manual_text_dataset",
        lambda *args, **kwargs: source_result,
    )
    monkeypatch.setattr(
        runner.data_api,
        "prepared_dataset_output",
        lambda value, **kwargs: finished_result if value is source_result else None,
    )
    assert runner.execute_data_pipeline(
        _state_for(["manual_dataset", "prepared_dataset"])
    ) is finished_result
