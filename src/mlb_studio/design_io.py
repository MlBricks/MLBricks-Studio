from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_MAGIC = b"MLBRICKS-BIN-1\n"


def save_design_file(design: dict[str, Any], path: str | Path) -> Path:
    """Save a Builder design as JSON or MLBricks binary depending on suffix."""
    path = Path(path)
    payload = design
    if path.name.endswith(".mlbricks.bin") or path.suffix.lower() == ".bin":
        raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        path.write_bytes(_MAGIC + raw)
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_design_file(path: str | Path) -> dict[str, Any]:
    """Load .mlbricks.json or .mlbricks.bin, auto-detecting the binary header."""
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(_MAGIC):
        raw = raw[len(_MAGIC):]
    return json.loads(raw.decode("utf-8"))


__all__ = ["save_design_file", "load_design_file"]
