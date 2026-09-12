import torch
import pytest

from mlb_studio import model_runtime as runtime


def _info(classes=80):
    return {
        "input_type": "image_input",
        "task": "classification",
        "output_classes": classes,
    }


def test_supervised_xy_drops_explicit_unlabeled_classification_rows():
    split = {
        "image": torch.randn(3, 3, 8, 8).tolist(),
        "label": [16, -1, 0],
    }
    x, y, _ = runtime._supervised_xy(split, _info(80))
    assert x.shape[0] == 2
    assert y.tolist() == [16, 0]


def test_supervised_xy_rejects_positive_classification_id_outside_head():
    split = {
        "image": torch.randn(2, 3, 8, 8).tolist(),
        "label": [0, 80],
    }
    with pytest.raises(ValueError, match="outside the model head range"):
        runtime._supervised_xy(split, _info(80))


def test_classification_loss_checks_target_range_before_cross_entropy():
    prediction = torch.randn(2, 80)
    target = torch.tensor([0, -1])
    with pytest.raises(ValueError, match="target range"):
        runtime._supervised_loss_and_metrics(prediction, target, "classification")
