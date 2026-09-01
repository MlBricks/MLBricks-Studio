from __future__ import annotations

import http.client
import json
from types import SimpleNamespace

import pytest

import mlb_studio.serve as serve
from mlb_studio.serve import ModelHTTPRuntime, _playground_html


class DummyTokenizer:
    eos_token_id = 0
    pad_token_id = 0


class DummyCompiled:
    device = "cpu"
    precision = "fp32"
    model = object()


def runtime(**kwargs):
    return ModelHTTPRuntime(
        model_id="m1",
        model_name=kwargs.pop("model_name", "Demo"),
        compiled=DummyCompiled(),
        tokenizer=DummyTokenizer(),
        context=32,
        generation_defaults={},
        port=0,
        **kwargs,
    )


def test_playground_escapes_model_name():
    page = _playground_html('<script>alert("x")</script>', True, 128)
    assert '<script>alert("x")</script>' not in page
    assert "&lt;script&gt;" in page


def test_server_safe_defaults():
    r = runtime()
    assert r.host == "127.0.0.1"
    assert r.cors_origin == "same-origin"
    assert r.api_key_required is True
    assert r.max_request_bytes == 1_048_576
    assert r.max_new_tokens == 2048


def test_health_and_security_headers():
    r = runtime(model_name='<img src=x onerror=alert(1)>')
    info = r.start()
    try:
        conn = http.client.HTTPConnection("127.0.0.1", info.port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse(); body = resp.read().decode()
        assert resp.status == 200
        assert resp.getheader("X-Content-Type-Options") == "nosniff"
        assert "default-src 'none'" in resp.getheader("Content-Security-Policy")
        assert resp.getheader("Access-Control-Allow-Origin") is None
        assert '<img src=x onerror=alert(1)>' not in body
    finally:
        r.stop()


def test_request_size_and_token_limit(monkeypatch):
    monkeypatch.setattr(serve, "generate_text", lambda *a, **k: ("ok", 1))
    r = runtime(max_request_bytes=1024, max_new_tokens=16)
    info = r.start()
    try:
        headers = {"Authorization": f"Bearer {r.api_key}", "Content-Type": "application/json"}
        conn = http.client.HTTPConnection("127.0.0.1", info.port, timeout=5)
        oversized = json.dumps({"prompt": "x" * 1500})
        conn.request("POST", "/v1/generate", body=oversized, headers=headers)
        resp = conn.getresponse(); resp.read()
        assert resp.status == 413

        conn = http.client.HTTPConnection("127.0.0.1", info.port, timeout=5)
        body = json.dumps({"prompt": "hi", "max_new_tokens": 17})
        conn.request("POST", "/v1/generate", body=body, headers=headers)
        resp = conn.getresponse(); resp.read()
        assert resp.status == 400
    finally:
        r.stop()


def test_public_tunnel_requires_auth_before_ngrok_import():
    r = runtime(api_key_required=False)
    with pytest.raises(RuntimeError, match="requires API-key"):
        r.start_ngrok()


def test_generation_timeout_returns_408(monkeypatch):
    import time
    from mlb_studio.model_runtime import TrainingStopped

    def slow_generate(*args, **kwargs):
        stop_event = kwargs["stop_event"]
        while not stop_event.is_set():
            time.sleep(0.01)
        raise TrainingStopped("stopped")

    monkeypatch.setattr(serve, "generate_text", slow_generate)
    r = runtime(request_timeout_seconds=1, max_new_tokens=16)
    info = r.start()
    try:
        headers = {"Authorization": f"Bearer {r.api_key}", "Content-Type": "application/json"}
        conn = http.client.HTTPConnection("127.0.0.1", info.port, timeout=4)
        conn.request("POST", "/v1/generate", body=json.dumps({"prompt": "hi"}), headers=headers)
        resp = conn.getresponse(); payload = json.loads(resp.read())
        assert resp.status == 408
        assert "timeout" in payload["error"].lower()
    finally:
        r.stop()


def test_internal_errors_are_not_exposed_by_default(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("sensitive-internal-detail")

    monkeypatch.setattr(serve, "generate_text", fail)
    r = runtime(max_new_tokens=16)
    info = r.start()
    try:
        headers = {"Authorization": f"Bearer {r.api_key}", "Content-Type": "application/json"}
        conn = http.client.HTTPConnection("127.0.0.1", info.port, timeout=5)
        conn.request("POST", "/v1/generate", body=json.dumps({"prompt": "hi"}), headers=headers)
        resp = conn.getresponse(); payload = json.loads(resp.read())
        assert resp.status == 500
        assert "sensitive-internal-detail" not in payload["error"]
    finally:
        r.stop()


def test_playground_uses_studio_theme_and_custom_number_controls():
    page = _playground_html("Demo", True, 256)
    assert "MLBricks Studio Serve" in page
    assert "PUBLIC HTTPS API" in page
    assert "MAX NEW TOKENS" in page
    assert "TEMPERATURE" in page
    assert 'class="num-step"' in page
    assert "::-webkit-inner-spin-button" in page
    assert "GENERATED TEXT" in page
