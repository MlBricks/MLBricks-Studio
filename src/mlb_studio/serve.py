from __future__ import annotations

import json
import errno
import os
import secrets
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .model_runtime import generate_text, runtime_float, runtime_int


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


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


def _playground_html(model_name: str, api_key_required: bool) -> str:
    auth_html = '<input id="key" type="password" placeholder="API key" />' if api_key_required else ""
    auth_js = 'if(key.value) headers["Authorization"]="Bearer "+key.value;' if api_key_required else ""
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{model_name} · MLBricks</title>
<style>
body{{font-family:system-ui,-apple-system,sans-serif;background:#0c1117;color:#e6edf3;margin:0}}
main{{max-width:760px;margin:0 auto;padding:28px 18px}} .card{{background:#111923;border:1px solid #2a3948;border-radius:14px;padding:18px}}
h1{{font-size:22px;margin:0 0 4px}} p{{color:#8fa0b2}}
textarea,input{{width:100%;box-sizing:border-box;background:#0b1219;color:#e6edf3;border:1px solid #35485a;border-radius:9px;padding:11px;margin:6px 0}}
textarea{{min-height:120px;resize:vertical}} .row{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
button{{width:100%;padding:11px;border:0;border-radius:9px;background:#6d55e7;color:white;font-weight:700;cursor:pointer}}
pre{{white-space:pre-wrap;background:#0b1219;border:1px solid #263747;border-radius:9px;padding:12px;min-height:90px}} small{{color:#718396}}
</style></head><body><main><div class="card">
<h1>{model_name}</h1><p>Served by MLB Studio V1.0</p>
{auth_html}
<textarea id="prompt">Once upon a time</textarea>
<div class="row"><input id="tokens" type="number" min="1" value="128"><input id="temp" type="number" step="0.1" value="0.8"></div>
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


class ModelHTTPRuntime:
    def __init__(self, *, model_id, model_name, compiled, tokenizer, context,
                 generation_defaults, host="0.0.0.0", port=8000, cors_origin="*",
                 api_key_required=True, api_key=None):
        self.model_id=model_id; self.model_name=model_name
        self.compiled=compiled; self.tokenizer=tokenizer
        self.context=runtime_int(context,512,"Model Context",minimum=2)
        self.defaults=dict(generation_defaults or {})
        self.host=str(host or "0.0.0.0")
        self.port=runtime_int(port,8000,"Server Port",minimum=0,maximum=65535)
        self.cors_origin=str(cors_origin or "*")
        self.api_key_required=bool(api_key_required)
        self.api_key=api_key or (secrets.token_urlsafe(24) if self.api_key_required else None)
        self.httpd=None; self.thread=None; self.lock=threading.Lock()
        self.public_url=None; self.tunnel=None; self.started_at=time.time()

    def _authorized(self, handler):
        if not self.api_key_required: return True
        auth=handler.headers.get("Authorization","")
        supplied=auth[7:].strip() if auth.lower().startswith("bearer ") else handler.headers.get("X-API-Key","")
        return bool(supplied) and secrets.compare_digest(str(supplied),str(self.api_key))

    def _handler_class(self):
        runtime=self
        class Handler(BaseHTTPRequestHandler):
            server_version="MLBricksModelServer/0.7.0"
            def log_message(self,fmt,*args): return
            def _cors(self):
                self.send_header("Access-Control-Allow-Origin",runtime.cors_origin)
                self.send_header("Access-Control-Allow-Headers","Content-Type, Authorization, X-API-Key")
                self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS")
            def _json(self,status,payload):
                body=_json_bytes(payload); self.send_response(status); self._cors()
                self.send_header("Content-Type","application/json; charset=utf-8")
                self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
            def _html(self,status,html):
                body=html.encode(); self.send_response(status); self._cors()
                self.send_header("Content-Type","text/html; charset=utf-8")
                self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
            def _auth(self):
                if runtime._authorized(self): return True
                self._json(401,{"error":"Unauthorized. Supply Bearer API key."}); return False
            def _body(self):
                length=int(self.headers.get("Content-Length","0") or 0)
                try:return json.loads((self.rfile.read(length) if length else b"{}").decode())
                except Exception as exc: raise ValueError("Request body must be valid JSON.") from exc
            def do_OPTIONS(self):
                self.send_response(204); self._cors(); self.end_headers()
            def do_GET(self):
                path=urlparse(self.path).path
                if path=="/": self._html(200,_playground_html(runtime.model_name,runtime.api_key_required)); return
                if path=="/health":
                    self._json(200,{"status":"ok","model":runtime.model_name,"model_id":runtime.model_id,
                                    "device":str(runtime.compiled.device),"precision":runtime.compiled.precision}); return
                if path in {"/v1/model","/v1/models"}:
                    if not self._auth(): return
                    item={"id":runtime.model_id,"object":"model","owned_by":"mlbricks",
                          "name":runtime.model_name,"context_length":runtime.context}
                    self._json(200,{"object":"list","data":[item]} if path=="/v1/models" else item); return
                self._json(404,{"error":"Not found"})
            def do_POST(self):
                path=urlparse(self.path).path
                if path not in {"/v1/generate","/v1/completions"}:
                    self._json(404,{"error":"Not found"}); return
                if not self._auth(): return
                try:
                    body=self._body(); prompt=body.get("prompt")
                    if isinstance(prompt,list): prompt=prompt[0] if prompt else ""
                    prompt=str(prompt or "")
                    if not prompt: raise ValueError("prompt is required.")
                    with runtime.lock:
                        text,count=generate_text(
                            runtime.compiled.model,runtime.tokenizer,prompt,
                            max_new_tokens=runtime_int(body.get("max_new_tokens",body.get("max_tokens",runtime.defaults.get("max_new_tokens"))),128,"New Token Count",minimum=1),
                            context=runtime.context,device=runtime.compiled.device,precision=runtime.compiled.precision,
                            temperature=runtime_float(body.get("temperature",runtime.defaults.get("temperature")),0.8,"Temperature",minimum=.00001),
                            top_k=runtime_int(body.get("top_k",runtime.defaults.get("top_k")),50,"Top K",minimum=0),
                            top_p=runtime_float(body.get("top_p",runtime.defaults.get("top_p")),0.95,"Top P",minimum=0,maximum=1),
                            seed=runtime_int(body.get("seed",runtime.defaults.get("seed")),42,"Seed"),
                            progress=None,stop_event=None)
                    if path=="/v1/completions":
                        payload={"id":"cmpl-"+uuid.uuid4().hex[:16],"object":"text_completion","created":int(time.time()),
                                 "model":runtime.model_id,"choices":[{"index":0,"text":text,"finish_reason":"stop"}],
                                 "usage":{"completion_tokens":count,"total_tokens":count}}
                    else:
                        payload={"model":runtime.model_id,"text":text,"generated_tokens":count}
                    self._json(200,payload)
                except Exception as exc:
                    self._json(400,{"error":f"{type(exc).__name__}: {exc}"})
        return Handler

    def start(self):
        if self.httpd is not None:
            return self.info()

        requested_port = int(self.port)
        candidate_ports = [requested_port] if requested_port == 0 else list(range(requested_port, min(requested_port + 21, 65536)))
        if requested_port != 0:
            candidate_ports.append(0)  # final fallback: let the OS choose any free port

        last_error = None
        used_fallback = False
        for candidate in candidate_ports:
            try:
                self.httpd = ThreadingHTTPServer((self.host, candidate), self._handler_class())
                self.port = int(self.httpd.server_address[1])
                used_fallback = self.port != requested_port
                break
            except OSError as exc:
                last_error = exc
                address_in_use = exc.errno in {
                    errno.EADDRINUSE,
                    98,      # Linux
                    48,      # macOS
                    10048,   # Windows
                }
                if not address_in_use:
                    raise
                self.httpd = None
                continue

        if self.httpd is None:
            raise OSError(
                getattr(last_error, "errno", errno.EADDRINUSE),
                f"Could not bind the model API server. Requested port {requested_port} and fallback ports were unavailable."
            )

        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            name=f"mlbricks-server-{self.model_id}",
            daemon=True,
        )
        self.thread.start()
        self.started_at = time.time()
        self.requested_port = requested_port
        self.used_port_fallback = used_fallback
        return self.info()

    def start_ngrok(self,auth_token=None):
        try: from pyngrok import ngrok
        except ImportError as exc: raise RuntimeError("Public ngrok links need pyngrok. Install: pip install pyngrok") from exc
        if auth_token: ngrok.set_auth_token(auth_token)
        self.tunnel=ngrok.connect(addr=self.port,proto="http",bind_tls=True)
        self.public_url=str(self.tunnel.public_url)
        return self.public_url

    def info(self):
        ip=_lan_ip(); remote,env=remote_notebook_environment()
        return ServerInfo(
            self.model_id,
            self.model_name,
            self.host,
            self.port,
            f"http://127.0.0.1:{self.port}",
            f"http://{ip}:{self.port}" if ip else None,
            self.public_url,
            self.api_key_required,
            self.api_key,
            remote,
            env,
            self.started_at,
            getattr(self, "requested_port", self.port),
            bool(getattr(self, "used_port_fallback", False)),
        )

    def stop(self):
        public=self.public_url
        if self.httpd is not None:
            try:self.httpd.shutdown()
            finally:self.httpd.server_close()
            self.httpd=None
        if public:
            try:
                from pyngrok import ngrok
                ngrok.disconnect(public)
            except Exception: pass
        self.public_url=None; self.tunnel=None
