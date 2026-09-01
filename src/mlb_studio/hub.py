from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


DATASET_META_FILE = "mlbricks_dataset.json"
MODEL_META_FILE = "mlbricks_model.json"
PROJECT_META_FILE = "mlbricks_project.json"


class HubUnavailableError(RuntimeError):
    pass


def _hub():
    try:
        from huggingface_hub import HfApi, get_token, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise HubUnavailableError(
            "Hugging Face Hub support needs huggingface_hub. "
            "Install it with: pip install huggingface_hub"
        ) from exc
    return HfApi, get_token, hf_hub_download, snapshot_download


def hub_token(*, required: bool, token: str | None = None) -> str | None:
    _, get_token, _, _ = _hub()
    token = token or os.environ.get("HF_TOKEN") or get_token()
    if required and not token:
        raise RuntimeError(
            "Hugging Face login is required for Push. Run `hf auth login` "
            "in the notebook terminal/cell or set the HF_TOKEN environment variable. "
            "MLB Studio never stores your token in the project."
        )
    return token


def auth_status(token: str | None = None) -> dict[str, Any]:
    HfApi, _, _, _ = _hub()
    token = hub_token(required=False, token=token)
    result = {
        "package_available": True,
        "token_found": bool(token),
        "authenticated": False,
        "username": None,
        "organizations": [],
        "message": "No Hugging Face token found.",
    }
    if not token:
        return result
    try:
        info = HfApi(token=token).whoami(token=token)
        result["authenticated"] = True
        result["username"] = info.get("name") or info.get("fullname")
        result["organizations"] = [
            x.get("name") for x in (info.get("orgs") or []) if x.get("name")
        ]
        result["message"] = f'Connected as {result["username"] or "Hugging Face user"}.'
    except Exception as exc:
        result["message"] = f"Token found, but Hub login check failed: {exc}"
    return result


def _repo_url(repo_id: str, repo_type: str) -> str:
    if repo_type == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}"
    return f"https://huggingface.co/{repo_id}"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def push_dataset(dataset, *, repo_id: str, metadata: dict, private: bool = True, token: str | None = None) -> dict:
    token = hub_token(required=True, token=token)
    repo_id = str(repo_id or "").strip()
    if "/" not in repo_id:
        raise ValueError("Dataset Repo ID must be in `username-or-org/repo-name` format.")

    # Dataset / DatasetDict implements push_to_hub directly.
    try:
        dataset.push_to_hub(repo_id, private=bool(private), token=token)
    except AttributeError as exc:
        raise RuntimeError(
            "This prepared object cannot be pushed as a Hugging Face Dataset. "
            "Push before converting it to a DataLoader."
        ) from exc

    HfApi, _, _, _ = _hub()
    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="mlbricks_hf_dataset_") as td:
        meta_path = Path(td) / DATASET_META_FILE
        payload = copy.deepcopy(metadata or {})
        payload["format"] = "mlb-studio-dataset-v1"
        payload["hub_repo_id"] = repo_id
        _write_json(meta_path, payload)
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo=DATASET_META_FILE,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message="Add MLB Studio dataset metadata",
        )

    return {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "url": _repo_url(repo_id, "dataset"),
    }


def load_dataset(repo_id: str, *, revision: str | None = None, token: str | None = None):
    token = hub_token(required=False, token=token)
    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise ValueError("Dataset Repo ID is required.")
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Loading Hub datasets needs the `datasets` package. "
            "Install it with: pip install datasets"
        ) from exc

    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    dataset = hf_load_dataset(repo_id, **kwargs)

    metadata = None
    _, _, hf_hub_download, _ = _hub()
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=DATASET_META_FILE,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        metadata = None

    return dataset, metadata, {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "url": _repo_url(repo_id, "dataset"),
        "revision": revision or "main",
    }


def _model_readme(name: str, model_entry: dict, has_weights: bool) -> str:
    modality = ((model_entry or {}).get("requirements") or {}).get("modality", "unknown")
    return f"""---
library_name: mlbricks
tags:
- mlbricks
- pytorch
- mlb-studio
---

# {name}

This repository was exported from **MLB Studio V1.0**.

- Input modality: `{modality}`
- Weights included: `{str(bool(has_weights)).lower()}`
- Builder package metadata: `{MODEL_META_FILE}`

Load this repository from MLB Studio's **Hugging Face** panel to restore
the architecture and, when present, the complete MLBricks model artifact.
"""


def push_model(
    *,
    repo_id: str,
    package: dict,
    checkpoint_path: str | None = None,
    tokenizer=None,
    private: bool = True,
    token: str | None = None,
) -> dict:
    token = hub_token(required=True, token=token)
    repo_id = str(repo_id or "").strip()
    if "/" not in repo_id:
        raise ValueError("Model Repo ID must be in `username-or-org/repo-name` format.")

    HfApi, _, _, _ = _hub()
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=bool(private),
        exist_ok=True,
        token=token,
    )

    with tempfile.TemporaryDirectory(prefix="mlbricks_hf_model_") as td:
        folder = Path(td)
        payload = copy.deepcopy(package)
        payload["format"] = "mlb-studio-model-v1"
        payload["builder_version"] = "1.0.0"
        payload["hub_repo_id"] = repo_id

        has_weights = bool(checkpoint_path and Path(checkpoint_path).exists())
        if has_weights:
            source_path = Path(checkpoint_path)
            if source_path.is_dir() and (source_path / "model.pt").exists():
                # Current MLBricks lifecycle format. Upload the complete
                # self-describing artifact, not only its model.pt member.
                artifact_dir = folder / "model_artifact"
                shutil.copytree(source_path, artifact_dir, dirs_exist_ok=True)
                payload["model_artifact_dir"] = "model_artifact"
            elif source_path.is_file():
                # Backward compatibility with legacy Builder .pt checkpoints.
                weights = folder / "weights"
                weights.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, weights / "last.pt")
                payload["checkpoint_file"] = "weights/last.pt"
            else:
                has_weights = False

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tok_dir = folder / "tokenizer"
            tok_dir.mkdir(parents=True, exist_ok=True)
            try:
                tokenizer.save_pretrained(str(tok_dir))
                payload["tokenizer_dir"] = "tokenizer"
            except Exception:
                pass

        _write_json(folder / MODEL_META_FILE, payload)
        name = ((package or {}).get("model_entry") or {}).get("name") or repo_id.split("/")[-1]
        (folder / "README.md").write_text(
            _model_readme(name, (package or {}).get("model_entry") or {}, has_weights),
            encoding="utf-8",
        )

        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Push model from MLB Studio",
        )

    return {
        "repo_id": repo_id,
        "repo_type": "model",
        "url": _repo_url(repo_id, "model"),
        "weights_uploaded": has_weights,
    }


def load_model(repo_id: str, *, revision: str | None = None, token: str | None = None) -> tuple[dict, Path, dict]:
    token = hub_token(required=False, token=token)
    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise ValueError("Model Repo ID is required.")

    _, _, _, snapshot_download = _hub()
    folder = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
    ))
    meta_path = folder / MODEL_META_FILE
    if not meta_path.exists():
        raise RuntimeError(
            f"{repo_id!r} is not an MLB Studio model repository "
            f"({MODEL_META_FILE} was not found)."
        )
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    return payload, folder, {
        "repo_id": repo_id,
        "repo_type": "model",
        "url": _repo_url(repo_id, "model"),
        "revision": revision or "main",
    }


def push_project(*, repo_id: str, state: dict, private: bool = True, token: str | None = None) -> dict:
    token = hub_token(required=True, token=token)
    repo_id = str(repo_id or "").strip()
    if "/" not in repo_id:
        raise ValueError("Project Repo ID must be in `username-or-org/repo-name` format.")

    HfApi, _, _, _ = _hub()
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=bool(private),
        exist_ok=True,
        token=token,
    )

    clean = copy.deepcopy(state)
    clean.pop("_runtime_command", None)
    with tempfile.TemporaryDirectory(prefix="mlbricks_hf_project_") as td:
        folder = Path(td)
        _write_json(folder / PROJECT_META_FILE, {
            "format": "mlb-studio-project-v1",
            "builder_version": "1.0.0",
            "state": clean,
        })
        (folder / "README.md").write_text(
            "# MLB Studio V1.0 Project\n\n"
            f"This repository contains `{PROJECT_META_FILE}` and can be restored "
            "from MLB Studio's Hugging Face panel.\n",
            encoding="utf-8",
        )
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Push MLB Studio project",
        )
    return {
        "repo_id": repo_id,
        "repo_type": "project",
        "url": _repo_url(repo_id, "model"),
    }


def load_project(repo_id: str, *, revision: str | None = None, token: str | None = None) -> tuple[dict, dict]:
    token = hub_token(required=False, token=token)
    repo_id = str(repo_id or "").strip()
    if not repo_id:
        raise ValueError("Project Repo ID is required.")
    _, _, hf_hub_download, _ = _hub()
    path = hf_hub_download(
        repo_id=repo_id,
        filename=PROJECT_META_FILE,
        repo_type="model",
        revision=revision,
        token=token,
    )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != "mlb-studio-project-v1":
        raise RuntimeError("The repository does not contain a compatible MLB Studio project.")
    return payload["state"], {
        "repo_id": repo_id,
        "repo_type": "project",
        "url": _repo_url(repo_id, "model"),
        "revision": revision or "main",
    }
