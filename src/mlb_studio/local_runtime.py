from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DATA_FILE_EXTENSIONS = {'.txt', '.csv', '.json', '.jsonl', '.parquet', '.arrow'}
MODEL_EXTENSIONS = {'.pt', '.pth', '.ckpt'}


def human_size(size: int | None) -> str:
    if size is None:
        return '—'
    value = float(size)
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f'{value:.1f} {unit}' if unit != 'B' else f'{int(value)} B'
        value /= 1024
    return f'{size} B'


def _directory_size(path: Path, *, max_files: int = 2500) -> int | None:
    total = 0
    count = 0
    try:
        for child in path.rglob('*'):
            if child.is_file():
                count += 1
                if count > max_files:
                    return None
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
    except OSError:
        return None
    return total


def detect_local_kind(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {'kind': 'missing', 'label': 'Missing'}
    if p.is_dir():
        if (p / 'model.pt').exists() and (p / 'metadata.json').exists():
            try:
                payload = json.loads((p / 'metadata.json').read_text(encoding='utf-8'))
                if payload.get('format') == 'mlbricks.model':
                    return {'kind': 'model_artifact', 'label': 'MLBricks Model'}
            except Exception:
                # Still treat the canonical model.pt + metadata.json layout as
                # a model candidate so the loader can surface the real error.
                return {'kind': 'model_artifact', 'label': 'MLBricks Model'}
        if (p / 'dataset_dict.json').exists():
            return {'kind': 'dataset_dir', 'label': 'Prepared Dataset'}
        if (p / 'dataset_info.json').exists() and (p / 'state.json').exists():
            return {'kind': 'dataset_dir', 'label': 'Prepared Dataset'}
        if (p / 'manifest.json').exists():
            try:
                payload = json.loads((p / 'manifest.json').read_text(encoding='utf-8'))
                if payload.get('format') == 'mlbricks-cloud-bundle-v1':
                    return {'kind': 'bundle_dir', 'label': f"MLBricks {str(payload.get('content_type') or 'Bundle').title()}"}
            except Exception:
                pass
        return {'kind': 'folder', 'label': 'Folder'}
    name = p.name.lower()
    suffix = p.suffix.lower()
    if name.endswith('.mlbricks.zip'):
        return {'kind': 'bundle', 'label': 'MLBricks Bundle'}
    if name.endswith('.mlbricks.json'):
        return {'kind': 'project_json', 'label': 'Builder Project'}
    if name.endswith('.mlbricks.bin'):
        return {'kind': 'project_bin', 'label': 'Builder Project BIN'}
    if suffix in MODEL_EXTENSIONS:
        return {'kind': 'model_checkpoint', 'label': 'Model Checkpoint'}
    if suffix in DATA_FILE_EXTENSIONS:
        return {'kind': 'data_file', 'label': 'Dataset File'}
    return {'kind': 'file', 'label': 'File'}


def _existing_unique_paths(candidates: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        try:
            p = p.expanduser()
        except Exception:
            pass
        if not p.exists():
            continue
        try:
            resolved = str(p.resolve())
        except Exception:
            resolved = str(p)
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(p)
    return result


def detect_local_environment() -> dict[str, Any]:
    """Describe the notebook/Python filesystem Builder can scan and write to.

    ``workspace_root`` is the current writable notebook workspace. It is used as
    the default Base Path in the UI. Environment-specific read-only/input roots
    can still appear in ``roots`` for explicit environment scans.
    """
    cwd = Path.cwd()
    home = Path.home()

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        kind, name = "kaggle", "Kaggle"
        workspace_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else cwd
        candidates = [workspace_root, Path("/kaggle/input")]
    elif os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU") or "google.colab" in os.sys.modules:
        kind, name = "colab", "Google Colab"
        workspace_root = Path("/content") if Path("/content").exists() else cwd
        candidates = [workspace_root, Path("/content/drive/MyDrive")]
    elif os.environ.get("LIGHTNING_CLOUD_URL") or Path("/teamspace").exists():
        kind, name = "lightning", "Lightning AI"
        workspace_root = cwd if str(cwd) != "/" else Path("/teamspace/studios/this_studio")
        candidates = [workspace_root, Path("/teamspace/studios/this_studio"), Path("/teamspace")]
    elif os.environ.get("CODESPACES"):
        kind, name = "codespaces", "GitHub Codespaces"
        workspace_root = cwd if str(cwd) != "/" else Path("/workspaces")
        candidates = [workspace_root, Path("/workspaces")]
    elif os.environ.get("SAGEMAKER_REGION") or Path("/home/ec2-user/SageMaker").exists():
        kind, name = "sagemaker", "Amazon SageMaker"
        workspace_root = cwd if str(cwd) != "/" else Path("/home/ec2-user/SageMaker")
        candidates = [workspace_root, Path("/home/ec2-user/SageMaker")]
    elif Path("/workspace").exists() and cwd != Path("/"):
        kind, name = "cloud_workspace", "Cloud Workspace"
        workspace_root = cwd
        candidates = [workspace_root, Path("/workspace")]
    else:
        kind, name = "python", "Python / Jupyter Environment"
        # Do not use filesystem root as the default writable workspace.
        workspace_root = home if str(cwd) == "/" else cwd
        candidates = [workspace_root]
        try:
            if workspace_root.resolve() != home.resolve():
                candidates.append(home)
        except Exception:
            candidates.append(home)

    roots = _existing_unique_paths(candidates)
    if not roots:
        roots = [workspace_root]

    try:
        workspace_root = workspace_root.expanduser().resolve()
    except Exception:
        workspace_root = workspace_root.expanduser()

    resolved_roots = []
    for path in roots:
        try:
            resolved_roots.append(str(path.resolve()))
        except Exception:
            resolved_roots.append(str(path))

    mlbricks_root = workspace_root / "mlbricks"
    paths = {
        "root": str(mlbricks_root),
        "models": str(mlbricks_root / "models"),
        "data": str(mlbricks_root / "data"),
        "training": str(mlbricks_root / "training"),
        "projects": str(mlbricks_root / "projects"),
        "exports": str(mlbricks_root / "exports"),
    }

    return {
        "kind": kind,
        "name": name,
        "roots": resolved_roots,
        "workspace_root": str(workspace_root),
        "default_root": str(workspace_root),
        "cwd": str(cwd),
        "paths": paths,
    }


def ensure_mlbricks_workspace(environment: dict[str, Any] | None = None) -> dict[str, str]:
    """Create Builder's standard artifact directories in the current workspace."""
    env = environment or detect_local_environment()
    paths = dict(env.get("paths") or {})
    if not paths:
        root = Path(env.get("workspace_root") or env.get("default_root") or Path.cwd()) / "mlbricks"
        paths = {
            "root": str(root),
            "models": str(root / "models"),
            "data": str(root / "data"),
            "training": str(root / "training"),
            "projects": str(root / "projects"),
            "exports": str(root / "exports"),
        }
    try:
        for value in paths.values():
            Path(value).expanduser().mkdir(parents=True, exist_ok=True)
        return paths
    except OSError:
        # A few hosted runtimes expose a read-only cwd. Keep Builder usable by
        # falling back to the user's writable home directory only in that case.
        root = Path.home() / "mlbricks"
        fallback = {
            "root": str(root),
            "models": str(root / "models"),
            "data": str(root / "data"),
            "training": str(root / "training"),
            "projects": str(root / "projects"),
            "exports": str(root / "exports"),
        }
        for value in fallback.values():
            Path(value).mkdir(parents=True, exist_ok=True)
        return fallback


def _root_candidates() -> list[Path]:
    return [Path(x) for x in detect_local_environment().get("roots") or [str(Path.cwd())]]


def scan_local_files(roots: list[str] | None = None, *, max_entries: int = 300, max_depth: int = 5) -> dict[str, Any]:
    root_paths = [Path(x) for x in roots] if roots else _root_candidates()
    entries: list[dict[str, Any]] = []

    def add(path: Path, root: Path):
        if len(entries) >= max_entries:
            return
        info = detect_local_kind(path)
        if info['kind'] in {'folder', 'file', 'missing'}:
            return
        try:
            rel = str(path.relative_to(root))
        except Exception:
            rel = path.name
        size = _directory_size(path) if path.is_dir() else (path.stat().st_size if path.exists() else None)
        entries.append({
            'path': str(path), 'name': path.name or str(path), 'relative': rel,
            'root': str(root), 'kind': info['kind'], 'label': info['label'],
            'size': size, 'size_label': human_size(size), 'is_dir': path.is_dir(),
        })

    for root in root_paths:
        if len(entries) >= max_entries:
            break
        root = root.resolve()
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            try:
                depth = len(current_path.relative_to(root).parts)
            except Exception:
                depth = 0
            if depth > max_depth:
                dirs[:] = []
                continue
            current_kind = detect_local_kind(current_path)['kind']
            if current_kind in {'dataset_dir', 'bundle_dir', 'model_artifact'}:
                add(current_path, root)
                dirs[:] = []
                continue
            for filename in files:
                if len(entries) >= max_entries:
                    break
                add(current_path / filename, root)

    priority = {'model_artifact': 0, 'model_checkpoint': 1, 'dataset_dir': 2, 'bundle': 3, 'project_json': 4, 'project_bin': 5, 'data_file': 6}
    entries.sort(key=lambda x: (priority.get(x['kind'], 99), x['root'].lower(), x['path'].lower()))
    return {'roots': [str(x.resolve()) for x in root_paths], 'entries': entries, 'truncated': len(entries) >= max_entries}


def scan_model_candidates(
    base_path: str | Path,
    *,
    max_entries: int = 1000,
    max_depth: int = 12,
) -> dict[str, Any]:
    """Recursively find model checkpoints/bundles beneath one base path."""
    base = Path(base_path).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Local environment path was not found: {base}")
    base = base.resolve()

    if base.is_file():
        info = detect_local_kind(base)
        entries = []
        if info["kind"] in {"model_artifact", "model_checkpoint", "bundle"}:
            size = base.stat().st_size
            entries.append({
                "path": str(base),
                "name": base.name,
                "relative": base.name,
                "root": str(base.parent),
                "kind": info["kind"],
                "label": info["label"],
                "size": size,
                "size_label": human_size(size),
                "is_dir": False,
            })
        return {"root": str(base.parent), "entries": entries, "truncated": False}

    scan = scan_local_files([str(base)], max_entries=max_entries, max_depth=max_depth)
    return {
        "root": str(base),
        "entries": [
            item for item in (scan.get("entries") or [])
            if item.get("kind") in {"model_artifact", "model_checkpoint", "bundle"}
        ],
        "truncated": bool(scan.get("truncated")),
    }


def scan_data_candidates(
    base_path: str | Path,
    *,
    max_entries: int = 1000,
    max_depth: int = 12,
) -> dict[str, Any]:
    """Recursively find prepared/raw datasets and MLBricks bundles."""
    base = Path(base_path).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Local environment path was not found: {base}")
    base = base.resolve()

    if base.is_file():
        info = detect_local_kind(base)
        entries = []
        if info["kind"] in {"data_file", "bundle"}:
            size = base.stat().st_size
            entries.append({
                "path": str(base),
                "name": base.name,
                "relative": base.name,
                "root": str(base.parent),
                "kind": info["kind"],
                "label": info["label"],
                "size": size,
                "size_label": human_size(size),
                "is_dir": False,
            })
        return {"root": str(base.parent), "entries": entries, "truncated": False}

    scan = scan_local_files(
        [str(base)],
        max_entries=max_entries,
        max_depth=max_depth,
    )
    entries = [
        item for item in (scan.get("entries") or [])
        if item.get("kind") in {"dataset_dir", "data_file", "bundle"}
    ]
    return {
        "root": str(base),
        "entries": entries,
        "truncated": bool(scan.get("truncated")),
    }
