from __future__ import annotations

import pytest
import torch

from mlb_studio.model_runtime import TensorGraph


def _audio_graph(sample_rate=16000):
    return TensorGraph(
        nodes=[
            {
                "id": "audio",
                "type": "audio_input",
                "name": "Audio Input",
                "params": {"sample_rate": sample_rate},
            }
        ],
        edges=[],
        custom_components={},
        runtime={
            "device": "cpu",
            "backend": "pytorch",
            "precision": "fp32",
        },
    )


@pytest.mark.parametrize(
    "shape",
    [
        (16000,),
        (2, 16000),
        (2, 2, 8000),
    ],
)
def test_audio_input_accepts_common_waveform_tensor_shapes(shape):
    graph = _audio_graph()
    waveform = torch.randn(*shape)

    output = graph(waveform)

    assert output is waveform
    assert output.shape == shape
    assert output.dtype == waveform.dtype


def test_audio_input_preserves_autograd_path():
    graph = _audio_graph(sample_rate=48000)
    waveform = torch.randn(2, 4096, requires_grad=True)

    output = graph(waveform)
    loss = output.square().mean()
    loss.backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
