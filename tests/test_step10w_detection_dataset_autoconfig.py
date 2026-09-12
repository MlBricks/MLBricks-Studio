from __future__ import annotations

from pathlib import Path

from mlb_studio.builder import Builder


ROOT = Path(__file__).resolve().parents[1]


def test_data_pipeline_snapshot_preserves_detection_processing_contract():
    builder = Builder.__new__(Builder)
    builder.state = {
        "workspaces": {"data": {"root_component_id": "data_root"}},
        "components": {
            "data_root": {
                "nodes": [
                    {"id": "src", "type": "coco128_cloud", "name": "COCO128 Cloud Source", "params": {}},
                    {
                        "id": "prep",
                        "type": "detection_process",
                        "name": "Detection Processing",
                        "params": {
                            "image_column": "image",
                            "boxes_column": "boxes",
                            "classes_column": "class_ids",
                            "width": 128,
                            "height": 128,
                            "mode": "RGB",
                        },
                    },
                    {"id": "out", "type": "prepared_dataset", "name": "Prepared COCO128 Cloud", "params": {}},
                ]
            }
        },
    }

    snap = builder._data_pipeline_snapshot()
    assert snap["source"]["type"] == "coco128_cloud"
    assert snap["detection_processing"]["type"] == "detection_process"
    assert snap["detection_processing"]["width"] == 128
    assert snap["detection_processing"]["height"] == 128
    assert snap["detection_processing"]["mode"] == "RGB"


def test_detection_dataset_schema_has_image_modality_precedence_and_autoconfig_ui():
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        js = (ROOT / rel).read_text(encoding="utf-8")
        assert 'cols.includes("image")&&cols.includes("boxes")&&cols.includes("class_ids")' in js
        assert 'function detectionDatasetContract(datasetMeta)' in js
        assert 'function configureDetectorForDataset(entry,datasetMeta)' in js
        assert 'n.params.channels=contract.channels;n.params.image_size=contract.width' in js
        assert 'n.params.classes=contract.classes' in js
        assert 'entry.status="needs_rebuild"' in js
        assert 'entry.weights_ready=false' in js
        assert 'Configure Model for ' in js
        assert 'Click Build to compile the updated graph.' in js


def test_builder_registration_marks_detection_contract_as_image_object_detection():
    source = (ROOT / "src/mlb_studio/builder.py").read_text(encoding="utf-8")
    assert 'pipeline.get("detection_processing")' in source
    assert '{"image", "boxes", "class_ids"}.issubset(default_columns)' in source
    assert 'metadata["modality"] = "image"' in source
    assert 'metadata["task"] = "object_detection"' in source
