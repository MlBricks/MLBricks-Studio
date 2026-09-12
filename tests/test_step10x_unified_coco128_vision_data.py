from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from mlb_studio import data as data_api


ROOT = Path(__file__).resolve().parents[1]


def _frontend_sources():
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        yield (ROOT / rel).read_text(encoding="utf-8")


def test_coco128_is_the_single_reusable_image_training_preset():
    for js in _frontend_sources():
        assert 'id:"vision_coco128",name:"COCO128 Cloud"' in js
        assert 'compatible_models:["CNN","Autoencoder","Image JEPA","Image Classifier","YOLO-style Detector","VESA-YOLO experimental","Custom vision models"]' in js
        # Redundant image-specific demo cards are intentionally removed from Data Gallery.
        for removed in (
            'id:"dl_image"',
            'id:"dl_autoencoder"',
            'id:"jepa_image",name:"Image JEPA Demo"',
            'id:"vision_classification"',
            'id:"vision_detection"',
            'id:"audio_unlabeled"',
            'id:"signal_unlabeled"',
        ):
            assert removed not in js


def test_all_directly_compatible_image_gallery_models_point_to_coco128():
    expected_ids = (
        "model_cnn",
        "model_autoencoder",
        "model_jepa_image",
        "model_image_classifier",
        "model_yolo_detector",
        "model_vesa_yolo",
    )
    for js in _frontend_sources():
        for model_id in expected_ids:
            start = js.index(f'id:"{model_id}"')
            snippet = js[start:start + 700]
            assert 'data_id:"vision_coco128"' in snippet


def test_coco128_gallery_models_use_rgb_128_and_80_classes_where_supervised():
    for js in _frontend_sources():
        assert 'Image Input · COCO128",{channels:3,image_size:128' in js
        assert 'COCO 80-Class Head",{dim:16,hidden_size:16,classes:80}' in js
        assert 'COCO 80-Class Head",{dim:32,hidden_size:32,classes:80}' in js
        assert 'P3/P4/P5 Detection Head · COCO80"' in js
        assert 'classes:80,slots:3' in js
        assert 'VESA P5 Projection · COCO128",{image_size:128,patch_size:4,in_channels:3' in js


def test_detection_processing_exposes_primary_object_label_for_classifiers():
    ds = pytest.importorskip("datasets")
    dataset = ds.Dataset.from_dict({
        "image": [Image.new("RGB", (20, 20), (10, 20, 30))],
        "boxes": [[[1.0, 1.0, 3.0, 3.0], [2.0, 2.0, 10.0, 8.0]]],
        "class_ids": [[0, 16]],
    })
    processed = data_api.process_detection_dataset(
        dataset,
        width=128,
        height=128,
        mode="RGB",
        normalize_images=True,
    )
    assert "label" in processed.column_names
    assert processed[0]["label"] == 16  # dog: largest annotated object
    assert processed[0]["class_ids"] == [0, 16]
    assert len(processed[0]["boxes"]) == 2


def test_shared_vision_autoconfig_supports_vesa_and_non_detector_image_models():
    for js in _frontend_sources():
        assert 'function visionDatasetAutoConfig(entry,datasetMeta)' in js
        assert 'function configureVisionModelForDataset(entry,datasetMeta)' in js
        assert 'n.params.in_channels=contract.channels;' in js
        assert 'n.params.image_size=contract.width;' in js
        assert 'plan.task==="classification"' in js
        assert 'plan.task==="reconstruction"' in js
        assert 'VESA detector geometry is editable but is not auto-resized yet' not in js


def test_training_backend_guards_classifier_class_count_too():
    source = (ROOT / "src/mlb_studio/model_runtime.py").read_text(encoding="utf-8")
    assert "Classification-class mismatch: model head has" in source
    assert "Set the Classifier Head classes to" in source
