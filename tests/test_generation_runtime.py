from __future__ import annotations

import torch
import torch.nn as nn

from mlb_studio.model_runtime import TensorGraph, generate_text


class _Tokenizer:
    eos_token_id = None
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        del text, add_special_tokens
        return [1, 2]

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(value) for value in ids)


def _next_logits(value, vocab=8):
    logits = torch.full((1, 1, vocab), -100.0)
    logits[..., value] = 100.0
    return logits


class _CachedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.prefill_calls = 0
        self.decode_calls = 0
        self.forward_calls = 0

    def recurrent_generation_support(self):
        return True, None

    def recurrent_generation_algorithms(self):
        return ["ESA Thunder prefill + Lightning decode"]

    def prefill(self, ids, *, capacity):
        self.prefill_calls += 1
        return _next_logits(3), {"position": ids.size(1), "capacity": capacity}

    def decode_step(self, token, cache, *, position):
        del token
        self.decode_calls += 1
        cache = dict(cache, position=position + 1)
        return _next_logits(3), cache

    def forward(self, ids):
        del ids
        self.forward_calls += 1
        raise AssertionError("full forward must not run on the recurrent path")


def test_generate_text_uses_prefill_once_and_streams_live_without_terminal_backlog():
    model = _CachedModel()
    events = []
    text, count = generate_text(
        model, _Tokenizer(), "prompt", max_new_tokens=3, context=16,
        device=torch.device("cpu"), precision="fp32",
        temperature=1.0, top_k=1, top_p=1.0, progress=events.append,
    )

    assert count == 3
    assert text == "1 2 3 3 3"
    assert model.prefill_calls == 1
    assert model.decode_calls == 2
    assert model.forward_calls == 0
    assert events[0]["phase"] == "prefill"
    token_events = [event for event in events if event["phase"] == "generate"]
    assert token_events
    assert token_events[0]["generated_tokens"] == 1
    # The terminal text is sent once by Builder's immediate done event instead
    # of being queued behind another growing per-token widget payload.
    assert token_events[-1]["generated_tokens"] < count
    assert all(event["generation_mode"] == "recurrent-cache" for event in events)
    assert token_events[-1]["generated_text"] in text


class _FakeBolt(nn.Module):
    def prefill_with_cache(self, x, *, start_pos=0):
        del start_pos
        batch, tokens, width = x.shape
        c = x.new_ones(batch, 2, tokens, width // 2)
        rho = x.new_ones(batch, 2, tokens)
        return x + 1, (c, rho)

    def project_decode_state(self, x, *, start_pos):
        del start_pos
        batch, _, width = x.shape
        q = x.new_ones(batch, 2, width // 2)
        return q, q[:, :, None, :], q.new_ones(batch, 2, 1)

    def decode_append_projected(self, q, c_now, rho_now, c, rho, *, position):
        c[:, :, position:position + 1].copy_(c_now)
        rho[:, :, position:position + 1].copy_(rho_now)
        return q.flatten(1)


def test_bolt_generation_cache_is_fixed_capacity_and_reused():
    bolt = _FakeBolt()
    x = torch.zeros(1, 3, 8)
    _, state = TensorGraph._bolt_prefill(bolt, x, capacity=10)
    c_ptr = state["c"].data_ptr()
    rho_ptr = state["rho"].data_ptr()

    y, state = TensorGraph._bolt_decode(bolt, torch.zeros(1, 1, 8), state, 3)

    assert y.shape == (1, 1, 8)
    assert state["length"] == 4
    assert state["c"].shape[2] == 10
    assert state["c"].data_ptr() == c_ptr
    assert state["rho"].data_ptr() == rho_ptr
