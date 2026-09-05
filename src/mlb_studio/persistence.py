from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

_SECRET_FIELD_NAMES = {
    "token",
    "api_token",
    "access_token",
    "refresh_token",
    "api_key",
    "secret",
    "secret_key",
    "session_token",
    "password",
    "service_account_json",
    "connection_string",
    "credentials",
}
_HEAVY_FIELD_NAMES = {
    "state_dict",
    "model_state_dict",
    "optimizer_state",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "parameter_tensors",
    "weight_tensors",
    "tensor_data",
}


def default_studio_home() -> Path:
    override = str(os.environ.get("MLBRICKS_STUDIO_HOME") or "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = str(os.environ.get("LOCALAPPDATA") or "").strip()
        if base:
            return Path(base) / "MLBricks" / "Studio"
    return Path.home() / ".mlbricks" / "studio"


def mask_secret(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    if len(text) <= 6:
        return "•" * len(text)
    if text.startswith("hf_") and len(text) > 8:
        return text[:3] + "•" * min(10, max(4, len(text) - 7)) + text[-4:]
    if len(text) <= 12:
        return text[:2] + "•" * (len(text) - 4) + text[-2:]
    return text[:4] + "•" * 10 + text[-4:]


def _json_safe_design(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    """Return a compact design-only representation.

    Studio persistence is intentionally for graphs, configs, source definitions,
    metadata and artifact *references*. It never serializes live tensors, model
    weights, optimizer state, bytes, file bodies, or credentials.
    """
    if depth > 80:
        return "<max-depth>"
    lowered = str(key or "").lower()
    if lowered in _SECRET_FIELD_NAMES:
        return None
    if lowered in _HEAVY_FIELD_NAMES:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            name = str(raw_key)
            cleaned = _json_safe_design(raw_value, key=name, depth=depth + 1)
            if cleaned is not None or raw_value is None:
                result[name] = cleaned
        return result
    if isinstance(value, (list, tuple)):
        # Large numeric arrays are model/data payloads rather than a design.
        if len(value) > 10000:
            return {"omitted": True, "reason": "large-array", "length": len(value)}
        return [_json_safe_design(item, depth=depth + 1) for item in value]
    # Torch / NumPy-like objects should never be materialized into the design DB.
    if hasattr(value, "detach") or hasattr(value, "numpy") or hasattr(value, "tobytes"):
        return None
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def sanitize_design(value: Any) -> Any:
    return _json_safe_design(copy.deepcopy(value))


class StudioPersistence:
    """Persistent local design/draft repository plus secure credential references."""

    _session_secrets: dict[str, dict[str, str]] = {}
    _session_lock = threading.RLock()

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser() if root is not None else default_studio_home()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "studio.db"
        self._db_lock = threading.RLock()
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self):
        with self._db_lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS drafts (
                    id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    workspace TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_drafts_updated ON drafts(updated_at DESC);

                CREATE TABLE IF NOT EXISTS repository_items (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    payload TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_repository_kind_updated
                    ON repository_items(kind, updated_at DESC);

                CREATE TABLE IF NOT EXISTS credential_profiles (
                    provider TEXT NOT NULL,
                    name TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    storage TEXT NOT NULL,
                    masks TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY(provider, name)
                );
                """
            )

    @staticmethod
    def _compact_json(value: Any) -> str:
        return json.dumps(sanitize_design(value), ensure_ascii=False, separators=(",", ":"))

    def save_draft(self, draft_id: str, project_name: str, state: dict[str, Any], *, workspace: str = "model") -> dict[str, Any]:
        draft_id = str(draft_id or "").strip() or f"draft_{uuid.uuid4().hex}"
        project_name = str(project_name or "Untitled Model").strip() or "Untitled Model"
        payload = self._compact_json(state)
        now = time.time()
        with self._db_lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO drafts(id, project_name, workspace, updated_at, payload)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     project_name=excluded.project_name,
                     workspace=excluded.workspace,
                     updated_at=excluded.updated_at,
                     payload=excluded.payload""",
                (draft_id, project_name, str(workspace or "model"), now, payload),
            )
        return {"id": draft_id, "project_name": project_name, "workspace": workspace, "updated_at": now}

    def list_drafts(self, *, limit: int = 30) -> list[dict[str, Any]]:
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, project_name, workspace, updated_at, payload FROM drafts ORDER BY updated_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = {
                "id": row["id"],
                "project_name": row["project_name"],
                "workspace": row["workspace"],
                "updated_at": row["updated_at"],
                "node_count": 0,
                "edge_count": 0,
            }
            try:
                state = json.loads(row["payload"] or "{}")
                workspace = str(item["workspace"] or state.get("active_workspace") or "model")
                components = state.get("components") or {}
                if workspace == "component":
                    # Component drafts capture the focused outer Module/API editor.
                    # The current view may be a nested editor, so walk back to the
                    # outer transaction boundary before reporting progress counts.
                    component = components.get(state.get("view_component_id")) or {}
                    seen = set()
                    while component.get("parent_edit_return") and component.get("id") not in seen:
                        seen.add(component.get("id"))
                        parent_id = (component.get("parent_edit_return") or {}).get("view_id")
                        parent = components.get(parent_id)
                        if not parent or parent.get("kind") != "custom_edit":
                            break
                        component = parent
                else:
                    workspaces = state.get("workspaces") or {}
                    ws = workspaces.get(workspace) or {}
                    root_id = ws.get("root_component_id") or state.get("root_component_id")
                    component = components.get(root_id) or {}
                item["node_count"] = len(component.get("nodes") or [])
                item["edge_count"] = len(component.get("edges") or [])
            except Exception:
                pass
            out.append(item)
        return out

    def load_draft(self, draft_id: str) -> dict[str, Any] | None:
        with self._db_lock, self._connect() as conn:
            row = conn.execute("SELECT payload FROM drafts WHERE id=?", (str(draft_id),)).fetchone()
        return json.loads(row["payload"]) if row else None

    def delete_draft(self, draft_id: str) -> bool:
        with self._db_lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM drafts WHERE id=?", (str(draft_id),))
            return bool(cur.rowcount)

    def save_repository_item(
        self,
        *,
        kind: str,
        name: str,
        payload: dict[str, Any],
        item_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        kind = str(kind or "project").strip().lower()
        if kind not in {"project", "model", "component", "data", "pipeline", "template"}:
            raise ValueError(f"Unsupported local repository item kind: {kind!r}")
        name = str(name or "Untitled").strip() or "Untitled"
        item_id = str(item_id or "").strip() or f"local_{kind}_{uuid.uuid4().hex}"
        payload_text = self._compact_json(payload)
        metadata_text = self._compact_json(metadata or {})
        now = time.time()
        with self._db_lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO repository_items(id, kind, name, updated_at, payload, metadata)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET
                     kind=excluded.kind,
                     name=excluded.name,
                     updated_at=excluded.updated_at,
                     payload=excluded.payload,
                     metadata=excluded.metadata""",
                (item_id, kind, name, now, payload_text, metadata_text),
            )
        return {"id": item_id, "kind": kind, "name": name, "updated_at": now}

    def list_repository_items(self, *, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        with self._db_lock, self._connect() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT id, kind, name, updated_at, metadata FROM repository_items WHERE kind=? ORDER BY updated_at DESC LIMIT ?",
                    (str(kind).lower(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, kind, name, updated_at, metadata FROM repository_items ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["metadata"] = json.loads(item.get("metadata") or "{}")
            except Exception:
                item["metadata"] = {}
            out.append(item)
        return out

    def load_repository_item(self, item_id: str) -> dict[str, Any] | None:
        with self._db_lock, self._connect() as conn:
            row = conn.execute(
                "SELECT id, kind, name, updated_at, payload, metadata FROM repository_items WHERE id=?",
                (str(item_id),),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "kind": row["kind"],
            "name": row["name"],
            "updated_at": row["updated_at"],
            "payload": json.loads(row["payload"]),
            "metadata": json.loads(row["metadata"] or "{}"),
        }

    def delete_repository_item(self, item_id: str) -> bool:
        with self._db_lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM repository_items WHERE id=?", (str(item_id),))
            return bool(cur.rowcount)

    @staticmethod
    def _credential_key(provider: str, name: str) -> str:
        return f"{str(provider).lower()}::{str(name).strip() or 'Default'}"

    @staticmethod
    def _keyring_backend():
        try:
            import keyring
            backend = keyring.get_keyring()
            # keyring.backends.fail.Keyring has priority <= 0 and throws on use.
            priority = getattr(backend, "priority", 0)
            if priority is None or float(priority) <= 0:
                return None
            return keyring
        except Exception:
            return None

    def save_credentials(self, provider: str, name: str, credentials: dict[str, Any]) -> dict[str, Any]:
        provider = str(provider or "").strip().lower()
        name = str(name or "Default").strip() or "Default"
        clean = {str(k): str(v) for k, v in (credentials or {}).items() if v not in {None, ""}}
        if not provider:
            raise ValueError("Credential provider is required.")
        if not clean:
            raise ValueError("Enter at least one credential value before saving.")

        storage = "session"
        key = self._credential_key(provider, name)
        backend = self._keyring_backend()
        if backend is not None:
            try:
                backend.set_password("MLBricks-Studio", key, json.dumps(clean, separators=(",", ":")))
                storage = "os-keyring"
            except Exception:
                storage = "session"

        if storage == "session":
            with self._session_lock:
                self._session_secrets[key] = dict(clean)

        masks = {field: mask_secret(value) for field, value in clean.items()}
        now = time.time()
        with self._db_lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO credential_profiles(provider, name, updated_at, storage, masks)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(provider, name) DO UPDATE SET
                     updated_at=excluded.updated_at,
                     storage=excluded.storage,
                     masks=excluded.masks""",
                (provider, name, now, storage, json.dumps(masks, separators=(",", ":"))),
            )
        return {
            "provider": provider,
            "name": name,
            "storage": storage,
            "persistent": storage == "os-keyring",
            "masks": masks,
            "updated_at": now,
        }

    def get_credentials(self, provider: str, name: str = "Default") -> dict[str, str]:
        provider = str(provider or "").strip().lower()
        name = str(name or "Default").strip() or "Default"
        key = self._credential_key(provider, name)
        with self._session_lock:
            if key in self._session_secrets:
                return dict(self._session_secrets[key])

        backend = self._keyring_backend()
        if backend is not None:
            try:
                raw = backend.get_password("MLBricks-Studio", key)
                parsed = json.loads(raw) if raw else {}
                if isinstance(parsed, dict):
                    return {str(k): str(v) for k, v in parsed.items()}
            except Exception:
                pass
        return {}

    def list_credentials(self) -> list[dict[str, Any]]:
        with self._db_lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT provider, name, updated_at, storage, masks FROM credential_profiles ORDER BY provider, name"
            ).fetchall()
        out = []
        for row in rows:
            try:
                masks = json.loads(row["masks"] or "{}")
            except Exception:
                masks = {}
            key = self._credential_key(row["provider"], row["name"])
            with self._session_lock:
                session_available = key in self._session_secrets
            out.append({
                "provider": row["provider"],
                "name": row["name"],
                "updated_at": row["updated_at"],
                "storage": row["storage"],
                "persistent": row["storage"] == "os-keyring",
                "available": row["storage"] == "os-keyring" or session_available,
                "masks": masks,
            })
        return out

    def delete_credentials(self, provider: str, name: str = "Default") -> bool:
        provider = str(provider or "").strip().lower()
        name = str(name or "Default").strip() or "Default"
        key = self._credential_key(provider, name)
        with self._session_lock:
            self._session_secrets.pop(key, None)
        backend = self._keyring_backend()
        if backend is not None:
            try:
                backend.delete_password("MLBricks-Studio", key)
            except Exception:
                pass
        with self._db_lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM credential_profiles WHERE provider=? AND name=?",
                (provider, name),
            )
        return bool(cur.rowcount)

    def summary(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "database": str(self.db_path),
            "drafts": self.list_drafts(limit=30),
            "repository": self.list_repository_items(limit=100),
            "credentials": self.list_credentials(),
        }


__all__ = [
    "StudioPersistence",
    "default_studio_home",
    "mask_secret",
    "sanitize_design",
]
