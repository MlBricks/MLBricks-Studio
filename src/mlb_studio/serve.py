from __future__ import annotations

import errno
import html
import json
import os
import secrets
import socket
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .model_runtime import TrainingStopped, generate_text, runtime_float, runtime_int
from .version import SERVER_VERSION


class _PayloadTooLarge(ValueError):
    pass


class _RateLimited(RuntimeError):
    pass


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _clean_header(value: str, *, default: str = "") -> str:
    value = str(value or default).strip()
    if "\r" in value or "\n" in value:
        raise ValueError("HTTP header values may not contain newlines.")
    return value


def _lan_ip() -> str | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        value = sock.getsockname()[0]
        if value and not value.startswith("127."):
            return value
    except Exception:
        pass
    finally:
        sock.close()
    try:
        value = socket.gethostbyname(socket.gethostname())
        return value if value and not value.startswith("127.") else None
    except Exception:
        return None


def remote_notebook_environment() -> tuple[bool, str | None]:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.exists("/kaggle"):
        return True, "Kaggle"
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True, "Google Colab"
    return False, None


def _playground_html(model_name: str, api_key_required: bool, max_new_tokens: int) -> str:
    safe_name = html.escape(str(model_name), quote=True)
    auth_html = '<input id="key" type="password" autocomplete="off" placeholder="API key" />' if api_key_required else ""
    auth_js = 'if(key.value) headers["Authorization"]="Bearer "+key.value;' if api_key_required else ""
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_name} · MLBricks</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#0c1117;color:#e6edf3;margin:0}}
main{{max-width:760px;margin:0 auto;padding:28px 18px}} .card{{background:#111923;border:1px solid #2a3948;border-radius:14px;padding:18px}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:#8fa0b2}}
textarea,input{{width:100%;box-sizing:border-box;background:#0b1219;color:#e6edf3;border:1px solid #35485a;border-radius:9px;padding:11px;margin:6px 0}}
textarea{{min-height:120px;resize:vertical}} .row{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
button{{width:100%;padding:11px;border:0;border-radius:9px;background:#6d55e7;color:white;font-weight:700;cursor:pointer}}
pre{{white-space:pre-wrap;background:#0b1219;border:1px solid #263747;border-radius:9px;padding:12px;min-height:90px}} small{{color:#718396}}
</style></head><body><main><div class="card">
<h1>{safe_name}</h1><p>Served by MLB Studio V{SERVER_VERSION}</p>
{auth_html}
<textarea id="prompt">Once upon a time</textarea>
<div class="row"><input id="tokens" type="number" min="1" max="{int(max_new_tokens)}" value="{min(128, int(max_new_tokens))}"><input id="temp" type="number" step="0.1" value="0.8"></div>
<button id="go">Generate</button><p><small>POST /v1/generate · POST /v1/completions · GET /health</small></p><pre id="out">Ready.</pre>
</div></main><script>
const prompt=document.getElementById("prompt"),tokens=document.getElementById("tokens"),temp=document.getElementById("temp"),
out=document.getElementById("out"),key=document.getElementById("key")||{{value:""}};
document.getElementById("go").onclick=async()=>{{out.textContent="Generating…";const headers={{"Content-Type":"application/json"}};{auth_js}
try{{const r=await fetch("/v1/generate",{{method:"POST",headers,body:JSON.stringify({{prompt:prompt.value,max_new_tokens:Number(tokens.value),temperature:Number(temp.value)}})}});const data=await r.json();out.textContent=r.ok?data.text:JSON.stringify(data,null,2);}}
catch(e){{out.textContent=String(e)}}}};
</script></body></html>"""


@dataclass
class ServerInfo:
    model_id: str
    model_name: str
    host: str
    port: int
    local_url: str
    lan_url: str | None
    public_url: str | None
    api_key_required: bool
    api_key: str | None
    remote_notebook: bool
    environment: str | None
    started_at: float
    requested_port: int | None = None
    used_port_fallback: bool = False

    def to_dict(self, *, include_secret: bool = False) -> dict[str, Any]:
        result = {
            "model_id": self.model_id, "model_name": self.model_name,
            "host": self.host, "port": self.port,
            "local_url": self.local_url, "lan_url": self.lan_url, "public_url": self.public_url,
            "api_key_required": self.api_key_required,
            "remote_notebook": self.remote_notebook, "environment": self.environment,
            "started_at": self.started_at,
            "requested_port": self.requested_port,
            "used_port_fallback": self.used_port_fallback,
            "health_endpoint": "/health", "generate_endpoint": "/v1/generate",
            "completions_endpoint": "/v1/completions", "models_endpoint": "/v1/models",
        }
        if include_secret and self.api_key_required:
            result["api_key"] = self.api_key
        return result


class _HardenedThreadingHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class ModelHTTPRuntime:
    """Hardened notebook/local inference HTTP runtime.

    It is safe-by-default: loopback bind, API-key auth, same-origin CORS, bounded
    request/token sizes, request concurrency, rate limiting, and generation timeout.
    A reverse proxy/TLS terminator is still recommended for permanent internet-facing
    deployments; ngrok provides TLS for temporary notebook sharing.
    """

    def __init__(self, *, model_id, model_name, compiled, tokenizer, context,
                 generation_defaults, host="127.0.0.1", port=8000, cors_origin="same-origin",
                 api_key_required=True, api_key=None, max_request_bytes=1_048_576,
                 max_prompt_chars=32_768, max_new_tokens=2048, request_timeout_seconds=120,
                 max_concurrent_requests=2, rate_limit_per_minute=60, debug_errors=False):
        self.model_id = str(model_id); self.model_name = str(model_name)
        self.compiled = compiled; self.tokenizer = tokenizer
        self.context = runtime_int(context, 512, "Model Context", minimum=2)
        self.defaults = dict(generation_defaults or {})
        self.host = str(host or "127.0.0.1")
        self.port = runtime_int(port, 8000, "Server Port", minimum=0, maximum=65535)
        self.cors_origin = _clean_header(cors_origin, default="same-origin") or "same-origin"
        self.api_key_required = bool(api_key_required)
        self.api_key = api_key or (secrets.token_urlsafe(32) if self.api_key_required else None)
        self.max_request_bytes = runtime_int(max_request_bytes, 1_048_576, "Max Request Bytes", minimum=1024)
        self.max_prompt_chars = runtime_int(max_prompt_chars, 32_768, "Max Prompt Characters", minimum=1)
        self.max_new_tokens = runtime_int(max_new_tokens, 2048, "Max New Tokens", minimum=1)
        self.request_timeout_seconds = runtime_float(request_timeout_seconds, 120.0, "Request Timeout", minimum=1.0)
        self.max_concurrent_requests = runtime_int(max_concurrent_requests, 2, "Max Concurrent Requests", minimum=1, maximum=128)
        self.rate_limit_per_minute = runtime_int(rate_limit_per_minute, 60, "Rate Limit", minimum=1)
        self.debug_errors = bool(debug_errors)
        self.httpd = None; self.thread = None
        self.lock = threading.Lock()
        self._request_slots = threading.BoundedSemaphore(self.max_concurrent_requests)
        self._rate_lock = threading.Lock(); self._rate = {}
        self.public_url = None; self.tunnel = None; self.started_at = time.time()

    def _authorized(self, handler):
        if not self.api_key_required:
            return True
        auth = handler.headers.get("Authorization", "")
        supplied = auth[7:].strip() if auth.lower().startswith("bearer ") else handler.headers.get("X-API-Key", "")
        return bool(supplied) and secrets.compare_digest(str(supplied), str(self.api_key))

    def _rate_allowed(self, client_ip: str) -> bool:
        now = time.monotonic(); cutoff = now - 60.0
        with self._rate_lock:
            bucket = self._rate.setdefault(client_ip, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.rate_limit_per_minute:
                return False
            bucket.append(now)
            # Keep the map bounded over long notebook sessions.
            if len(self._rate) > 4096:
                stale = [key for key, values in self._rate.items() if not values or values[-1] < cutoff]
                for key in stale[:1024]:
                    self._rate.pop(key, None)
            return True

    def _handler_class(self):
        runtime = self

        class Handler(BaseHTTPRequestHandler):
            server_version = f"MLBricksModelServer/{SERVER_VERSION}"
            sys_version = ""

            def setup(self):
                super().setup()
                try:
                    self.connection.settimeout(max(5.0, runtime.request_timeout_seconds + 5.0))
                except Exception:
                    pass

            def log_message(self, fmt, *args):
                return

            def _headers(self):
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'",
                )
                if runtime.cors_origin not in {"", "same-origin"}:
                    self.send_header("Access-Control-Allow-Origin", runtime.cors_origin)
                    self.send_header("Vary", "Origin")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def _json(self, status, payload):
                body = _json_bytes(payload)
                self.send_response(status); self._headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def _html(self, status, page):
                body = page.encode("utf-8")
                self.send_response(status); self._headers()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def _auth(self):
                if runtime._authorized(self):
                    return True
                self._json(401, {"error": "Unauthorized. Supply a valid Bearer API key."})
                return False

            def _body(self):
                raw_length = self.headers.get("Content-Length", "0") or "0"
                try:
                    length = int(raw_length)
                except ValueError as exc:
                    raise ValueError("Invalid Content-Length header.") from exc
                if length < 0 or length > runtime.max_request_bytes:
                    raise _PayloadTooLarge(f"Request body exceeds {runtime.max_request_bytes} bytes.")
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    value = json.loads(raw.decode("utf-8"))
                except Exception as exc:
                    raise ValueError("Request body must be valid UTF-8 JSON.") from exc
                if not isinstance(value, dict):
                    raise ValueError("Request JSON must be an object.")
                return value

            def _check_rate(self):
                ip = str((self.client_address or ("unknown",))[0])
                if not runtime._rate_allowed(ip):
                    raise _RateLimited("Too many requests. Try again later.")

            def do_OPTIONS(self):
                self.send_response(204); self._headers(); self.end_headers()

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self._html(200, _playground_html(runtime.model_name, runtime.api_key_required, runtime.max_new_tokens)); return
                if path == "/health":
                    self._json(200, {"status": "ok", "model": runtime.model_name, "model_id": runtime.model_id,
                                     "device": str(runtime.compiled.device), "precision": runtime.compiled.precision}); return
                if path in {"/v1/model", "/v1/models"}:
                    if not self._auth(): return
                    item = {"id": runtime.model_id, "object": "model", "owned_by": "mlbricks",
                            "name": runtime.model_name, "context_length": runtime.context}
                    self._json(200, {"object": "list", "data": [item]} if path == "/v1/models" else item); return
                self._json(404, {"error": "Not found"})

            def do_POST(self):
                path = urlparse(self.path).path
                if path not in {"/v1/generate", "/v1/completions"}:
                    self._json(404, {"error": "Not found"}); return
                try:
                    self._check_rate()
                    if not self._auth(): return
                    if not runtime._request_slots.acquire(blocking=False):
                        self._json(429, {"error": "Server is at its generation concurrency limit."}); return
                    try:
                        body = self._body(); prompt = body.get("prompt")
                        if isinstance(prompt, list): prompt = prompt[0] if prompt else ""
                        prompt = str(prompt or "")
                        if not prompt: raise ValueError("prompt is required.")
                        if len(prompt) > runtime.max_prompt_chars:
                            raise _PayloadTooLarge(f"Prompt exceeds {runtime.max_prompt_chars} characters.")
                        requested_tokens = runtime_int(
                            body.get("max_new_tokens", body.get("max_tokens", runtime.defaults.get("max_new_tokens"))),
                            min(128, runtime.max_new_tokens), "New Token Count", minimum=1,
                        )
                        if requested_tokens > runtime.max_new_tokens:
                            raise ValueError(f"max_new_tokens may not exceed {runtime.max_new_tokens}.")

                        stop_event = threading.Event()
                        timer = threading.Timer(runtime.request_timeout_seconds, stop_event.set)
                        timer.daemon = True; timer.start()
                        try:
                            with runtime.lock:
                                text, count = generate_text(
                                    runtime.compiled.model, runtime.tokenizer, prompt,
                                    max_new_tokens=requested_tokens,
                                    context=runtime.context, device=runtime.compiled.device, precision=runtime.compiled.precision,
                                    temperature=runtime_float(body.get("temperature", runtime.defaults.get("temperature")), 0.8, "Temperature", minimum=.00001),
                                    top_k=runtime_int(body.get("top_k", runtime.defaults.get("top_k")), 50, "Top K", minimum=0),
                                    top_p=runtime_float(body.get("top_p", runtime.defaults.get("top_p")), .95, "Top P", minimum=0, maximum=1),
                                    seed=runtime_int(body.get("seed", runtime.defaults.get("seed")), 42, "Seed"),
                                    progress=None, stop_event=stop_event,
                                )
                        finally:
                            timer.cancel()
                        if path == "/v1/completions":
                            payload = {"id": "cmpl-" + uuid.uuid4().hex[:16], "object": "text_completion", "created": int(time.time()),
                                       "model": runtime.model_id, "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                                       "usage": {"completion_tokens": count, "total_tokens": count}}
                        else:
                            payload = {"model": runtime.model_id, "text": text, "generated_tokens": count}
                        self._json(200, payload)
                    finally:
                        runtime._request_slots.release()
                except _RateLimited as exc:
                    self._json(429, {"error": str(exc)})
                except _PayloadTooLarge as exc:
                    self._json(413, {"error": str(exc)})
                except TrainingStopped:
                    self._json(408, {"error": "Generation exceeded the server request timeout."})
                except ValueError as exc:
                    self._json(400, {"error": str(exc)})
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}" if runtime.debug_errors else "Generation request failed. Check the server logs/runtime configuration."
                    self._json(500, {"error": message})

        return Handler

    def start(self):
        if self.httpd is not None:
            return self.info()
        requested_port = int(self.port)
        candidate_ports = [requested_port] if requested_port == 0 else list(range(requested_port, min(requested_port + 21, 65536)))
        if requested_port != 0:
            candidate_ports.append(0)
        last_error = None; used_fallback = False
        for candidate in candidate_ports:
            try:
                self.httpd = _HardenedThreadingHTTPServer((self.host, candidate), self._handler_class())
                self.port = int(self.httpd.server_address[1]); used_fallback = self.port != requested_port; break
            except OSError as exc:
                last_error = exc
                if exc.errno not in {errno.EADDRINUSE, 98, 48, 10048}:
                    raise
                self.httpd = None
        if self.httpd is None:
            raise OSError(getattr(last_error, "errno", errno.EADDRINUSE),
                          f"Could not bind the model API server. Requested port {requested_port} and fallback ports were unavailable.")
        self.thread = threading.Thread(target=self.httpd.serve_forever, name=f"mlbricks-server-{self.model_id}", daemon=True)
        self.thread.start(); self.started_at = time.time(); self.requested_port = requested_port; self.used_port_fallback = used_fallback
        return self.info()

    def start_ngrok(self, auth_token=None):
        if not self.api_key_required:
            raise RuntimeError("Public ngrok serving requires API-key authentication. Enable Require API Key first.")
        try:
            from pyngrok import ngrok
        except ImportError as exc:
            raise RuntimeError("Public ngrok links need pyngrok. Install: pip install 'mlb-studio[serve]'") from exc
        if auth_token: ngrok.set_auth_token(auth_token)
        self.tunnel = ngrok.connect(addr=self.port, proto="http", bind_tls=True)
        self.public_url = str(self.tunnel.public_url)
        return self.public_url

    def info(self):
        ip = _lan_ip(); remote, env = remote_notebook_environment()
        lan = f"http://{ip}:{self.port}" if ip and self.host not in {"127.0.0.1", "localhost", "::1"} else None
        return ServerInfo(
            self.model_id, self.model_name, self.host, self.port,
            f"http://127.0.0.1:{self.port}", lan, self.public_url,
            self.api_key_required, self.api_key, remote, env, self.started_at,
            getattr(self, "requested_port", self.port), bool(getattr(self, "used_port_fallback", False)),
        )

    def stop(self):
        public = self.public_url
        if self.httpd is not None:
            try: self.httpd.shutdown()
            finally: self.httpd.server_close()
            self.httpd = None
        if public:
            try:
                from pyngrok import ngrok
                ngrok.disconnect(public)
            except Exception:
                pass
        self.public_url = None; self.tunnel = None
