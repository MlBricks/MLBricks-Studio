from __future__ import annotations

import copy

from mlb_studio import data as data_api
from mlb_studio import runner
from mlb_studio.graph import new_project


class FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(self.rows[0].keys()) if self.rows else []

    def rename_column(self, old, new):
        rows = []
        for row in self.rows:
            item = dict(row)
            item[new] = item.pop(old)
            rows.append(item)
        return FakeDataset(rows)


class FakeIterable:
    def __init__(self, rows):
        self.rows = list(rows)

    def take(self, count):
        yield from self.rows[:count]


class FakeDatasetFactory:
    @staticmethod
    def from_list(rows):
        return FakeDataset(rows)

    @staticmethod
    def from_dict(mapping):
        keys = list(mapping)
        size = len(mapping[keys[0]]) if keys else 0
        return FakeDataset([{key: mapping[key][i] for key in keys} for i in range(size)])


class FakeDatasetsModule:
    Dataset = FakeDatasetFactory

    def __init__(self):
        self.calls = []

    def load_dataset(self, dataset_id, config, **kwargs):
        self.calls.append((dataset_id, config, kwargs))
        if dataset_id == "MlBricks/ultrachat-200k":
            raise RuntimeError("EmptyDatasetError: directory doesn't contain any data files")
        assert dataset_id == "HuggingFaceH4/ultrachat_200k"
        assert kwargs["split"] == "train_sft"
        assert kwargs["streaming"] is True
        return FakeIterable([
            {"prompt": "hello", "messages": []},
            {"prompt": "world", "messages": []},
            {"prompt": "again", "messages": []},
        ])


def test_hf_gallery_fallback_streams_only_requested_rows_and_normalizes_text_column(monkeypatch):
    fake = FakeDatasetsModule()
    monkeypatch.setattr(data_api, "_datasets", lambda: fake)
    events = []

    result = data_api.load_huggingface_dataset(
        "MlBricks/ultrachat-200k",
        split="train",
        text_column="text",
        streaming=True,
        max_rows=2,
        fallback_dataset_id="HuggingFaceH4/ultrachat_200k",
        fallback_split="train_sft",
        fallback_text_column="prompt",
        progress_callback=events.append,
    )

    assert [row["text"] for row in result.rows] == ["hello", "world"]
    assert fake.calls[0][0] == "MlBricks/ultrachat-200k"
    assert fake.calls[1][0] == "HuggingFaceH4/ultrachat_200k"
    assert any(event.get("fallback") for event in events)
    assert any(event.get("rows_loaded") == 2 for event in events)
    assert events[-1]["percent"] == 100


def test_runner_maps_hf_row_progress_into_pipeline_overall(monkeypatch):
    state = new_project()
    ws = state["workspaces"]["data"]
    comp = state["components"][ws["root_component_id"]]
    source = comp["nodes"][0]
    source["params"].update({
        "dataset_id": "MlBricks/wikipedia-en-1b",
        "fallback_dataset_id": "wikimedia/wikipedia",
        "fallback_config": "20231101.en",
        "streaming": "true",
        "max_rows": 10000,
    })

    source_result = object()
    final_result = object()

    def fake_hf(*args, progress_callback=None, **kwargs):
        progress_callback({"percent": 50, "message": "Loading rows 5,000 / 10,000…", "rows_loaded": 5000, "rows_total": 10000})
        return source_result

    monkeypatch.setattr(runner.data_api, "load_huggingface_dataset", fake_hf)
    monkeypatch.setattr(runner.data_api, "process_text_dataset", lambda value, **kwargs: value)
    monkeypatch.setattr(runner.data_api, "train_validation_test_split", lambda value, **kwargs: value)
    monkeypatch.setattr(runner.data_api, "tokenize_text_dataset", lambda value, **kwargs: value)
    monkeypatch.setattr(runner.data_api, "prepared_dataset_output", lambda value, **kwargs: final_result)

    progress = []
    assert runner.execute_data_pipeline(state, progress_callback=lambda event: progress.append(copy.deepcopy(event))) is final_result
    row_event = next(event for event in progress if event.get("rows_loaded") == 5000)
    assert row_event["node_percent"] == 50
    # First source node is 1 of 5 pipeline steps, so half of it is 10% overall.
    assert row_event["overall"] == 10
    assert row_event["nodes"][source["id"]]["percent"] == 50

class _FakeHttpResponse:
    def __init__(self, payload):
        import json
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def test_parquet_api_fast_path_selects_config_split_and_materializes_prefix(monkeypatch):
    payload = {
        "parquet_files": [
            {"config": "20231101.en", "split": "train", "url": "https://hf.invalid/wiki-0.parquet"},
            {"config": "20231101.en", "split": "train", "url": "https://hf.invalid/wiki-1.parquet"},
            {"config": "20231101.fr", "split": "train", "url": "https://hf.invalid/fr-0.parquet"},
        ]
    }
    monkeypatch.setattr(data_api, "urlopen", lambda request, timeout=20: _FakeHttpResponse(payload))

    class ParquetFake(FakeDatasetsModule):
        def load_dataset(self, dataset_id, config=None, **kwargs):
            self.calls.append((dataset_id, config, kwargs))
            assert dataset_id == "parquet"
            assert kwargs["data_files"]["train"] == [
                "https://hf.invalid/wiki-0.parquet",
                "https://hf.invalid/wiki-1.parquet",
            ]
            assert kwargs["streaming"] is True
            return FakeIterable([
                {"text": "a"},
                {"text": "b"},
                {"text": "c"},
            ])

    fake = ParquetFake()
    events = []
    result = data_api._load_hf_parquet_api_prefix(
        fake,
        "wikimedia/wikipedia",
        config="20231101.en",
        split="train",
        max_rows=2,
        progress_callback=events.append,
    )
    assert [row["text"] for row in result.rows] == ["a", "b"]
    assert fake.calls[0][0] == "parquet"
    assert any(event.get("fast_path") == "parquet_api" for event in events)
    assert any(event.get("rows_loaded") == 2 for event in events)
