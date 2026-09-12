from __future__ import annotations

from pathlib import Path
import io
import os
import re
import tempfile
import zipfile
import unicodedata
from urllib.parse import urlparse, quote
from urllib.request import Request, urlopen
import json
import math
import random
from typing import Any


def _datasets():
    try:
        import datasets
        return datasets
    except ImportError as exc:
        raise ImportError(
            "Text/data features need Hugging Face datasets. "
            "Install with: pip install 'mlbricks-studio[data]' "
            "or pip install datasets kagglehub transformers pandas pyarrow"
        ) from exc


def _limit_rows(dataset, max_rows: int | None):
    if not max_rows:
        return dataset
    max_rows = int(max_rows)
    if max_rows <= 0:
        return dataset
    if hasattr(dataset, "select"):
        return dataset.select(range(min(max_rows, len(dataset))))
    return dataset


def _loader_for(name: str, format: str = "auto") -> str:
    fmt = (format or "auto").lower().strip()
    if fmt != "auto":
        if fmt == "jsonl":
            return "json"
        if fmt in {"txt", "text"}:
            return "text"
        return fmt

    suffix = Path(urlparse(str(name)).path).suffix.lower()
    if suffix in {".txt", ".text"}:
        return "text"
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(
        f"Cannot infer dataset format from {name!r}. "
        "Choose txt, csv, json, jsonl, or parquet explicitly."
    )


def _emit_load_progress(callback, percent: float, message: str, **extra):
    if not callback:
        return
    payload = {"percent": max(0.0, min(100.0, float(percent))), "message": str(message)}
    payload.update(extra)
    callback(payload)


def _load_hf_parquet_api_prefix(
    ds_module,
    dataset_id: str,
    *,
    config: str | None,
    split: str,
    max_rows: int | None,
    token: str | None = None,
    progress_callback=None,
):
    """Load a small Hub prefix through the Dataset Viewer Parquet API.

    Workshop quickstarts use this fast path so Studio does not have to resolve
    an entire multi-GB repository before reading the first few thousand rows.
    The Dataset Viewer API returns resolved Parquet URLs grouped by
    configuration and split. ``datasets`` then streams those URLs and stops
    once ``max_rows`` has been materialized.

    ``None`` means the fast path is not applicable. Network/API errors are
    allowed to bubble to the caller, which falls back to normal load_dataset.
    """
    limit = int(max_rows or 0)
    if limit <= 0:
        return None

    dataset_id = str(dataset_id or "").strip()
    if not dataset_id:
        return None

    _emit_load_progress(
        progress_callback,
        1,
        f"Resolving fast stream for {dataset_id}...",
        dataset_id=dataset_id,
        fast_path="parquet_api",
    )
    endpoint = (
        "https://datasets-server.huggingface.co/parquet?dataset="
        + quote(dataset_id, safe="")
    )
    headers = {
        "Accept": "application/json",
        "User-Agent": "MLBricks-Studio/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(endpoint, headers=headers)
    with urlopen(request, timeout=20) as response:
        payload = json.loads(response.read().decode("utf-8"))

    parquet_files = list(payload.get("parquet_files") or [])
    if not parquet_files:
        return None

    requested_config = str(config or "").strip()
    requested_split = str(split or "train").strip()
    candidates = [
        item
        for item in parquet_files
        if str(item.get("split") or "") == requested_split and item.get("url")
    ]

    if requested_config:
        candidates = [
            item
            for item in candidates
            if str(item.get("config") or "") == requested_config
        ]
    elif candidates:
        configs = {str(item.get("config") or "") for item in candidates}
        if "default" in configs:
            candidates = [
                item
                for item in candidates
                if str(item.get("config") or "") == "default"
            ]
        elif len(configs) == 1:
            only = next(iter(configs))
            candidates = [
                item
                for item in candidates
                if str(item.get("config") or "") == only
            ]
        else:
            return None

    if not candidates:
        return None

    urls = [str(item["url"]) for item in candidates]
    _emit_load_progress(
        progress_callback,
        3,
        f"Fast stream ready ({len(urls)} parquet shard{'s' if len(urls) != 1 else ''}).",
        dataset_id=dataset_id,
        fast_path="parquet_api",
        shard_count=len(urls),
    )
    stream = ds_module.load_dataset(
        "parquet",
        data_files={"train": urls},
        split="train",
        streaming=True,
    )
    return _materialize_streaming_dataset(
        ds_module, stream, limit, progress_callback
    )


def _materialize_streaming_dataset(ds_module, dataset, max_rows: int, progress_callback=None):
    """Materialize only the requested prefix of an IterableDataset.

    Workshop quickstarts use streaming so large Hub repositories do not download
    their complete parquet corpus just to select the first few thousand rows.
    The returned object is a normal Dataset so downstream cleaning/splitting and
    tokenization continue to work unchanged.
    """
    limit = max(0, int(max_rows or 0))
    if limit <= 0:
        return dataset

    rows = []
    _emit_load_progress(progress_callback, 2, f"Streaming first {limit:,} rows…", rows_loaded=0, rows_total=limit)
    report_every = max(25, min(250, limit // 100 if limit >= 100 else 1))
    for index, row in enumerate(dataset.take(limit), start=1):
        rows.append(row)
        if index == 1 or index == limit or index % report_every == 0:
            pct = 5 + (index / max(limit, 1)) * 90
            _emit_load_progress(
                progress_callback,
                pct,
                f"Loading rows {index:,} / {limit:,}…",
                rows_loaded=index,
                rows_total=limit,
            )
    _emit_load_progress(progress_callback, 97, f"Preparing {len(rows):,} rows for Studio…", rows_loaded=len(rows), rows_total=limit)
    if not rows:
        # Dataset.from_list([]) cannot infer useful features on some versions.
        return ds_module.Dataset.from_dict({})
    return ds_module.Dataset.from_list(rows)


def load_huggingface_dataset(
    dataset_id: str,
    *,
    config: str | None = None,
    split: str = "train",
    text_column: str = "text",
    streaming: bool = False,
    max_rows: int | None = None,
    token: str | None = None,
    fallback_dataset_id: str | None = None,
    fallback_config: str | None = None,
    fallback_split: str | None = None,
    fallback_text_column: str | None = None,
    prefer_parquet_api: bool = False,
    progress_callback=None,
):
    """Load a dataset from the Hugging Face Hub.

    Workshop presets may include a public upstream fallback. This lets Studio keep
    using an MLBricks-maintained mirror when it is populated while still working
    if that mirror exists but has not uploaded data files yet. Authentication,
    when required, is read from the saved credential profile / normal HF login.

    When ``streaming`` and ``max_rows`` are both enabled, only that prefix is
    materialized into a normal Dataset. This prevents a 10k-row quickstart from
    downloading an entire multi-GB repository before ``select()`` can run.
    """
    ds = _datasets()

    primary = {
        "dataset_id": str(dataset_id or "").strip(),
        "config": (config or "").strip() or None,
        "split": split or "train",
        "text_column": text_column or "text",
    }
    fallback = None
    if str(fallback_dataset_id or "").strip():
        fallback = {
            "dataset_id": str(fallback_dataset_id).strip(),
            "config": (fallback_config or "").strip() or None,
            "split": fallback_split or primary["split"],
            "text_column": fallback_text_column or primary["text_column"],
        }

    def classify_and_raise(exc, target_id):
        text = f"{type(exc).__name__}: {exc}"
        lowered = text.lower()
        if "gatedrepo" in lowered or "gated repo" in lowered or ("403" in lowered and "access" in lowered):
            raise PermissionError(
                f"Hugging Face access approval is required for {target_id!r}. "
                "Your account may be authenticated but not approved for this gated dataset. "
                "Accept/request access on the dataset page, then retry with the saved credential profile."
            ) from exc
        if any(marker in lowered for marker in ("401", "unauthorized", "authentication", "invalid token", "token is required")):
            raise PermissionError(
                f"Hugging Face authentication is required for {target_id!r}. "
                "Open Cloud & Repositories, save a Hugging Face credential, and select its Credential Profile on this source node."
            ) from exc
        if "403" in lowered or "forbidden" in lowered:
            raise PermissionError(
                f"Hugging Face denied access to {target_id!r}. Verify that the saved token has permission to this private/gated repository."
            ) from exc
        raise exc

    def load_target(target):
        if prefer_parquet_api and max_rows and int(max_rows) > 0:
            try:
                fast = _load_hf_parquet_api_prefix(
                    ds,
                    target["dataset_id"],
                    config=target["config"],
                    split=target["split"],
                    max_rows=int(max_rows),
                    token=token or None,
                    progress_callback=progress_callback,
                )
                if fast is not None:
                    return fast
            except Exception as fast_exc:
                _emit_load_progress(
                    progress_callback,
                    2,
                    f"Fast stream unavailable; retrying normal Hugging Face loader ({type(fast_exc).__name__}).",
                    dataset_id=target["dataset_id"],
                    fast_path_fallback=True,
                )

        _emit_load_progress(progress_callback, 1, f"Connecting to {target['dataset_id']}…", dataset_id=target["dataset_id"])
        data = ds.load_dataset(
            target["dataset_id"],
            target["config"],
            split=target["split"],
            streaming=bool(streaming),
            token=token or None,
        )
        if streaming and max_rows and int(max_rows) > 0:
            data = _materialize_streaming_dataset(ds, data, int(max_rows), progress_callback)
        elif not streaming:
            data = _limit_rows(data, max_rows)
        return data

    active = primary
    try:
        data = load_target(primary)
    except Exception as primary_exc:
        # Permission failures must remain explicit. For Workshop mirrors, missing
        # or empty repositories are safe to replace with the declared upstream.
        lowered = f"{type(primary_exc).__name__}: {primary_exc}".lower()
        permission_like = any(x in lowered for x in ("401", "403", "unauthorized", "forbidden", "gatedrepo", "gated repo", "invalid token"))
        if fallback and not permission_like:
            _emit_load_progress(
                progress_callback,
                2,
                f"MLBricks mirror unavailable; using upstream {fallback['dataset_id']}…",
                dataset_id=fallback["dataset_id"],
                fallback=True,
            )
            active = fallback
            try:
                data = load_target(fallback)
            except Exception as fallback_exc:
                classify_and_raise(fallback_exc, fallback["dataset_id"])
        else:
            classify_and_raise(primary_exc, primary["dataset_id"])

    # A normalized MLBricks mirror can expose ``text`` while its upstream uses a
    # different text field (for example UltraChat's ``prompt``). Normalize the
    # fallback column back to the node's requested column so every downstream
    # component keeps the same API contract.
    requested_column = primary["text_column"]
    active_column = active["text_column"]
    columns = getattr(data, "column_names", None)
    if requested_column and columns is not None and requested_column not in columns:
        if active_column and active_column in columns and hasattr(data, "rename_column"):
            data = data.rename_column(active_column, requested_column)
            columns = getattr(data, "column_names", columns)
        else:
            raise KeyError(
                f"Text column {requested_column!r} not found. Available columns: {columns}"
            )

    _emit_load_progress(progress_callback, 100, f"Loaded {active['dataset_id']}.", dataset_id=active["dataset_id"], fallback=(active is fallback))
    return data

def load_url_dataset(
    url: str,
    *,
    format: str = "auto",
    text_column: str = "text",
    max_rows: int | None = None,
):
    """Load a text/CSV/JSON/JSONL/Parquet file from an HTTP(S) URL."""
    ds = _datasets()
    loader = _loader_for(url, format)
    data = ds.load_dataset(loader, data_files=url, split="train")
    data = _limit_rows(data, max_rows)
    if text_column and text_column not in data.column_names:
        raise KeyError(
            f"Text column {text_column!r} not found. Available columns: {data.column_names}"
        )
    return data


def load_local_dataset(
    path: str | Path,
    *,
    format: str = "auto",
    text_column: str = "text",
    max_rows: int | None = None,
):
    """Load a local text/CSV/JSON/JSONL/Parquet file."""
    ds = _datasets()
    path = str(path)
    loader = _loader_for(path, format)
    data = ds.load_dataset(loader, data_files=path, split="train")
    data = _limit_rows(data, max_rows)
    if text_column and text_column not in data.column_names:
        raise KeyError(
            f"Text column {text_column!r} not found. Available columns: {data.column_names}"
        )
    return data


def load_kaggle_dataset(
    dataset_handle: str,
    *,
    file_pattern: str = "*",
    format: str = "auto",
    text_column: str = "text",
    max_rows: int | None = None,
):
    """Download a Kaggle dataset using kagglehub and load matching data files.

    In a Kaggle notebook, normal Kaggle credentials/session are used. No
    credentials are stored in the Builder design.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Kaggle dataset download needs kagglehub. "
            "Install with: pip install kagglehub"
        ) from exc

    root = Path(kagglehub.dataset_download(dataset_handle))
    matches = sorted(p for p in root.rglob(file_pattern or "*") if p.is_file())
    if not matches:
        raise FileNotFoundError(
            f"No files matching {file_pattern!r} in Kaggle dataset {dataset_handle!r}"
        )

    if format == "auto":
        # Pick the first supported file and use other files of the same loader.
        supported = []
        for p in matches:
            try:
                supported.append((p, _loader_for(p.name, "auto")))
            except ValueError:
                pass
        if not supported:
            raise ValueError(
                "No supported text/CSV/JSON/JSONL/Parquet files found. "
                "Change File Pattern or Format."
            )
        loader = supported[0][1]
        files = [str(p) for p, kind in supported if kind == loader]
    else:
        loader = _loader_for(matches[0].name, format)
        files = [str(p) for p in matches]

    ds = _datasets()
    data = ds.load_dataset(loader, data_files=files, split="train")
    data = _limit_rows(data, max_rows)
    if text_column and text_column not in data.column_names:
        raise KeyError(
            f"Text column {text_column!r} not found. Available columns: {data.column_names}"
        )
    return data


def process_text_dataset(
    dataset,
    *,
    text_column: str = "text",
    lowercase: bool = False,
    strip: bool = True,
    normalize_whitespace: bool = True,
    unicode_nfkc: bool = True,
    remove_empty: bool = True,
    min_chars: int = 1,
    max_chars: int | None = None,
):
    """Apply safe, deterministic text cleanup to Dataset or DatasetDict."""
    def clean_value(value: Any) -> str:
        text = "" if value is None else str(value)
        if unicode_nfkc:
            text = unicodedata.normalize("NFKC", text)
        if strip:
            text = text.strip()
        if normalize_whitespace:
            text = re.sub(r"\s+", " ", text)
        if lowercase:
            text = text.lower()
        if max_chars and int(max_chars) > 0:
            text = text[: int(max_chars)]
        return text

    def clean_one(example):
        example[text_column] = clean_value(example.get(text_column))
        return example

    def keep(example):
        value = example.get(text_column, "")
        if remove_empty and not value:
            return False
        return len(value) >= max(0, int(min_chars or 0))

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        # DatasetDict-like
        return dataset.__class__({
            name: split.map(clean_one).filter(keep)
            for name, split in dataset.items()
        })

    return dataset.map(clean_one).filter(keep)


def train_test_split(
    dataset,
    *,
    train_size: float = 0.9,
    test_size: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
):
    """Create train/test splits from a Dataset.

    If a DatasetDict is supplied, its train split is used as the source.
    """
    ds = _datasets()
    source = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset

    train_size = float(train_size)
    test_size = float(test_size)
    if not (0 < train_size < 1):
        raise ValueError("train_size must be between 0 and 1.")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    if train_size + test_size > 1.000001:
        raise ValueError("train_size + test_size cannot be greater than 1.")

    split = source.train_test_split(
        train_size=train_size,
        test_size=test_size,
        seed=int(seed),
        shuffle=bool(shuffle),
    )
    return ds.DatasetDict({"train": split["train"], "test": split["test"]})


def tokenize_text_dataset(
    dataset,
    *,
    tokenizer_name: str = "gpt2",
    text_column: str = "text",
    context_length: int = 512,
    truncation: bool = True,
    padding: str | bool = False,
    add_special_tokens: bool = True,
):
    """Tokenize Dataset or DatasetDict with a Hugging Face tokenizer."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Tokenization needs transformers. It is included with MLBricks Studio; reinstall/upgrade mlbricks-studio and restart the notebook kernel."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if isinstance(padding, str):
        lowered = padding.lower()
        if lowered in {"false", "none", "no"}:
            padding = False
        elif lowered in {"true", "yes"}:
            padding = True

    def encode(batch):
        return tokenizer(
            batch[text_column],
            max_length=int(context_length),
            truncation=bool(truncation),
            padding=padding,
            add_special_tokens=bool(add_special_tokens),
        )

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        return dataset.__class__({
            name: split.map(encode, batched=True)
            for name, split in dataset.items()
        })

    return dataset.map(encode, batched=True)



def prepare_text_input(
    *,
    source: str = "manual",
    manual_text: str = "Once upon a time",
    dataset_id: str = "roneneldan/TinyStories",
    config: str | None = None,
    split: str = "train",
    dataset_handle: str = "",
    file_pattern: str = "*",
    url: str = "",
    path: str = "",
    format: str = "auto",
    text_column: str = "text",
    streaming: bool = False,
    max_rows: int | None = None,
    clean_text: bool = True,
    lowercase: bool = False,
    strip: bool = True,
    normalize_whitespace: bool = True,
    unicode_nfkc: bool = True,
    remove_empty: bool = True,
    min_chars: int = 1,
    max_chars: int | None = None,
    make_split: bool = True,
    train_size: float = 0.9,
    test_size: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
    tokenize: bool = True,
    tokenizer_name: str = "gpt2",
    context_length: int = 512,
    truncation: bool = True,
    padding: str | bool = False,
    add_special_tokens: bool = True,
):
    """Run the complete Text Input pipeline selected in MLB Studio."""
    source = str(source or "manual").strip().lower()

    if source == "manual":
        ds = _datasets()
        raw = ds.Dataset.from_dict({text_column: [str(manual_text)]})
    elif source in {"huggingface", "hugging_face", "hf"}:
        if streaming:
            raise ValueError(
                "Unified Text Input currently needs Streaming = false when "
                "cleaning, splitting, or tokenization is enabled."
            )
        raw = load_huggingface_dataset(
            dataset_id,
            config=config,
            split=split,
            text_column=text_column,
            streaming=False,
            max_rows=max_rows,
        )
    elif source == "kaggle":
        raw = load_kaggle_dataset(
            dataset_handle,
            file_pattern=file_pattern,
            format=format,
            text_column=text_column,
            max_rows=max_rows,
        )
    elif source in {"url", "link"}:
        raw = load_url_dataset(
            url,
            format=format,
            text_column=text_column,
            max_rows=max_rows,
        )
    elif source in {"local", "file"}:
        raw = load_local_dataset(
            path,
            format=format,
            text_column=text_column,
            max_rows=max_rows,
        )
    else:
        raise ValueError(
            "source must be manual, huggingface, kaggle, url, or local"
        )

    current = raw

    if clean_text:
        current = process_text_dataset(
            current,
            text_column=text_column,
            lowercase=lowercase,
            strip=strip,
            normalize_whitespace=normalize_whitespace,
            unicode_nfkc=unicode_nfkc,
            remove_empty=remove_empty,
            min_chars=min_chars,
            max_chars=max_chars,
        )

    if make_split:
        current = train_test_split(
            current,
            train_size=train_size,
            test_size=test_size,
            seed=seed,
            shuffle=shuffle,
        )

    if tokenize:
        current = tokenize_text_dataset(
            current,
            tokenizer_name=tokenizer_name,
            text_column=text_column,
            context_length=context_length,
            truncation=truncation,
            padding=padding,
            add_special_tokens=add_special_tokens,
        )

    return {
        "dataset": current,
        "raw_dataset": raw,
        "source": source,
        "tokenizer_name": tokenizer_name if tokenize else None,
    }



COCO128_DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO80_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _safe_extract_zip(zf: zipfile.ZipFile, target: Path) -> None:
    """Extract a trusted dataset zip while still preventing path traversal."""
    target = target.resolve()
    for member in zf.infolist():
        candidate = (target / member.filename).resolve()
        if os.path.commonpath([str(target), str(candidate)]) != str(target):
            raise ValueError(f"Unsafe path in dataset archive: {member.filename!r}")
    zf.extractall(target)


def load_coco128_cloud_dataset(
    *,
    download_url: str = COCO128_DOWNLOAD_URL,
    max_images: int | None = None,
    progress_callback=None,
):
    """Download COCO128 into a temporary directory and return an in-memory Dataset.

    Nothing is cached or installed into the Studio repository/data directory.  The
    source ZIP and extracted files live only inside ``TemporaryDirectory`` while
    they are parsed.  Image bytes are copied into the returned Arrow dataset, so
    the temporary directory can be deleted before this function returns.

    The returned schema is the normal Studio detection contract:
    ``image`` (decoded by ``datasets.Image``), ``boxes`` (pixel xywh), and
    ``class_ids`` (a sequence of COCO ``ClassLabel`` values).
    """
    ds = _datasets()
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("COCO128 loading needs Pillow: pip install pillow") from exc

    url = str(download_url or COCO128_DOWNLOAD_URL).strip()
    if not url.lower().startswith(("https://", "http://")):
        raise ValueError("COCO128 download URL must use http:// or https://")
    limit = int(max_images or 0)
    if limit < 0:
        raise ValueError("max_images must be 0 (all) or a positive integer.")

    headers = {"User-Agent": "MLBricks-Studio/1.0", "Accept": "application/zip,application/octet-stream,*/*"}
    with tempfile.TemporaryDirectory(prefix="mlbricks-coco128-") as td:
        root = Path(td)
        archive = root / "coco128.zip"
        _emit_load_progress(progress_callback, 1, "Connecting to COCO128 cloud source…", dataset_id="COCO128", storage="temporary")
        request = Request(url, headers=headers)
        with urlopen(request, timeout=60) as response, archive.open("wb") as fh:
            total = int(response.headers.get("Content-Length") or 0)
            loaded = 0
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                fh.write(chunk)
                loaded += len(chunk)
                if total:
                    pct = 2 + (loaded / total) * 48
                    _emit_load_progress(
                        progress_callback, pct,
                        f"Downloading COCO128… {loaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB",
                        bytes_loaded=loaded, bytes_total=total, dataset_id="COCO128", storage="temporary",
                    )
                else:
                    _emit_load_progress(progress_callback, 20, f"Downloading COCO128… {loaded / (1024*1024):.1f} MB", bytes_loaded=loaded, dataset_id="COCO128", storage="temporary")

        _emit_load_progress(progress_callback, 52, "Extracting COCO128 in temporary session storage…", dataset_id="COCO128", storage="temporary")
        with zipfile.ZipFile(archive, "r") as zf:
            _safe_extract_zip(zf, root)

        dataset_root = root / "coco128"
        if not dataset_root.is_dir():
            candidates = [x for x in root.rglob("coco128") if x.is_dir()]
            if candidates:
                dataset_root = candidates[0]
        image_dir = dataset_root / "images" / "train2017"
        label_dir = dataset_root / "labels" / "train2017"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise RuntimeError("COCO128 archive did not contain images/train2017 and labels/train2017.")

        image_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if limit > 0:
            image_paths = image_paths[:limit]
        if not image_paths:
            raise RuntimeError("COCO128 did not contain any supported images.")

        images, boxes, class_ids, image_ids = [], [], [], []
        total_images = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            raw = image_path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                width, height = image.size
            sample_boxes, sample_classes = [], []
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.is_file():
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(float(parts[0]))
                    if class_id < 0 or class_id >= len(COCO80_CLASS_NAMES):
                        raise ValueError(f"COCO128 class id {class_id} is outside the expected 0..79 range.")
                    cx, cy, nw, nh = [float(v) for v in parts[1:5]]
                    bw, bh = nw * width, nh * height
                    x = (cx - nw * 0.5) * width
                    y = (cy - nh * 0.5) * height
                    sample_boxes.append([float(x), float(y), float(bw), float(bh)])
                    sample_classes.append(class_id)
            images.append({"bytes": raw, "path": None})
            boxes.append(sample_boxes)
            class_ids.append(sample_classes)
            image_ids.append(image_path.stem)
            if index == 1 or index == total_images or index % 16 == 0:
                pct = 56 + (index / total_images) * 42
                _emit_load_progress(
                    progress_callback, pct,
                    f"Preparing COCO128 images {index} / {total_images}…",
                    rows_loaded=index, rows_total=total_images, dataset_id="COCO128", storage="memory",
                )

        features = ds.Features({
            "image": ds.Image(),
            "boxes": ds.Sequence(ds.Sequence(ds.Value("float32"), length=4)),
            "class_ids": ds.Sequence(ds.ClassLabel(names=list(COCO80_CLASS_NAMES))),
            "image_id": ds.Value("string"),
        })
        dataset = ds.Dataset.from_dict(
            {"image": images, "boxes": boxes, "class_ids": class_ids, "image_id": image_ids},
            features=features,
        )
        _emit_load_progress(
            progress_callback, 100,
            f"COCO128 ready in memory · {len(dataset)} images · {len(COCO80_CLASS_NAMES)} classes.",
            rows_loaded=len(dataset), rows_total=len(dataset), dataset_id="COCO128", storage="memory",
        )
        return dataset



def generate_demo_dataset(
    demo_type: str = "tabular_regression",
    *,
    samples: int = 512,
    seed: int = 42,
    sequence_length: int = 32,
    feature_count: int = 8,
    classes: int = 3,
):
    """Generate deterministic, offline demo data for Studio Gallery templates.

    The generator intentionally uses simple synthetic structures so Gallery
    examples can be opened and executed without downloading external datasets.
    It returns a normal Hugging Face ``Dataset`` so the existing split,
    preprocessing, batching, and Prepared Dataset nodes keep working unchanged.
    """
    ds = _datasets()
    kind = str(demo_type or "tabular_regression").strip().lower()
    n = max(8, int(samples or 512))
    seq = max(4, int(sequence_length or 32))
    feats = max(1, int(feature_count or 8))
    n_classes = max(2, int(classes or 3))
    rng = random.Random(int(seed or 42))

    def noise(scale=1.0):
        return (rng.random() * 2.0 - 1.0) * scale

    def wave(length, freq=1.0, phase=0.0, noise_scale=0.0):
        denom = max(1, length - 1)
        return [
            math.sin(2.0 * math.pi * freq * (i / denom) + phase) + noise(noise_scale)
            for i in range(length)
        ]

    def image(label=0, size=16):
        # Nested numeric image tensor: [H, W]. Distinct deterministic patterns
        # make it useful for educational CNN/classification examples.
        out = []
        for y in range(size):
            row = []
            for x in range(size):
                if label % 3 == 0:
                    base = 1.0 if abs(x - size // 2) <= 1 else 0.0
                elif label % 3 == 1:
                    base = 1.0 if abs(y - size // 2) <= 1 else 0.0
                else:
                    base = 1.0 if abs(x - y) <= 1 else 0.0
                row.append(max(0.0, min(1.0, base + noise(0.05))))
            out.append(row)
        return out

    if kind in {"tabular_regression", "neuron_regression"}:
        fc = 1 if kind == "neuron_regression" else max(2, feats)
        cols = {f"feature_{j+1}": [] for j in range(fc)}
        targets = []
        weights = [1.25 + 0.4 * j for j in range(fc)]
        for _ in range(n):
            xs = [noise(2.0) for _ in range(fc)]
            for j, value in enumerate(xs):
                cols[f"feature_{j+1}"].append(value)
            targets.append(sum(w * x for w, x in zip(weights, xs)) + 0.35 + noise(0.12))
        cols["target"] = targets
        return ds.Dataset.from_dict(cols)

    if kind in {"binary_classification", "multiclass_classification", "tabular_classification", "high_dimensional"}:
        fc = 16 if kind == "high_dimensional" else (max(8, feats) if kind == "tabular_classification" else max(4, feats))
        cls = 2 if kind == "binary_classification" else n_classes
        cols = {f"feature_{j+1}": [] for j in range(fc)}
        labels = []
        for i in range(n):
            label = i % cls
            center = (label - (cls - 1) / 2.0) * 1.4
            xs = [center + (j % 3) * 0.15 + noise(0.7) for j in range(fc)]
            for j, value in enumerate(xs):
                cols[f"feature_{j+1}"].append(value)
            labels.append(label)
        cols["label"] = labels
        return ds.Dataset.from_dict(cols)

    if kind == "clustering":
        xs, ys = [], []
        centers = [(-2.0, -1.5), (2.0, -1.0), (0.0, 2.2)]
        for i in range(n):
            cx, cy = centers[i % len(centers)]
            xs.append(cx + noise(0.55)); ys.append(cy + noise(0.55))
        return ds.Dataset.from_dict({"x": xs, "y": ys})

    if kind == "sequence_classification":
        sequences, labels = [], []
        for i in range(n):
            label = i % 2
            freq = 1.0 if label == 0 else 3.0
            sequences.append(wave(seq, freq=freq, phase=rng.random() * math.pi, noise_scale=0.08))
            labels.append(label)
        return ds.Dataset.from_dict({"sequence": sequences, "label": labels})

    if kind in {"image_classification", "image_reconstruction", "image_jepa"}:
        images, labels = [], []
        for i in range(n):
            label = i % n_classes
            images.append(image(label))
            labels.append(label)
        payload = {"image": images}
        if kind == "image_classification": payload["label"] = labels
        if kind == "image_reconstruction": payload["target_image"] = [list(map(list, im)) for im in images]
        return ds.Dataset.from_dict(payload)

    if kind == "object_detection":
        images, boxes, class_ids = [], [], []
        size=16
        for i in range(n):
            canvas=[[max(0.0,noise(0.02)) for _ in range(size)] for _ in range(size)]
            sample_boxes=[];sample_classes=[]
            object_count=1+(i%3)
            specs=[
                (1+(i%3),1+((i//3)%3),4,4),
                (9-((i//2)%2),2+(i%4),5,3),
                (4+(i%3),10-((i//4)%2),3,4),
            ]
            for j in range(object_count):
                x0,y0,bw,bh=specs[j]
                cls=(i+j)%n_classes
                x0=max(0,min(size-bw,x0));y0=max(0,min(size-bh,y0))
                intensity=0.45+0.25*(cls%3)
                for yy in range(y0,y0+bh):
                    for xx in range(x0,x0+bw):
                        canvas[yy][xx]=min(1.0,intensity+noise(0.03))
                sample_boxes.append([float(x0),float(y0),float(bw),float(bh)])
                sample_classes.append(cls)
            images.append(canvas);boxes.append(sample_boxes);class_ids.append(sample_classes)
        return ds.Dataset.from_dict({"image": images, "boxes": boxes, "class_ids": class_ids})

    if kind in {"text_corpus", "text_jepa"}:
        subjects = ["robot", "student", "researcher", "model", "sensor", "camera"]
        verbs = ["learns", "predicts", "observes", "compares", "builds", "measures"]
        objects = ["patterns", "future states", "signals", "images", "language", "representations"]
        texts = []
        for i in range(n):
            texts.append(f"The {subjects[i%len(subjects)]} {verbs[(i//2)%len(verbs)]} {objects[(i//3)%len(objects)]} in MLBricks Studio.")
        return ds.Dataset.from_dict({"text": texts})

    if kind == "video_jepa":
        videos = []
        frames = min(8, max(3, seq // 4))
        for i in range(n):
            clip = []
            label = i % n_classes
            for t in range(frames):
                # Move the base visual pattern slightly through time.
                frame = image((label + t) % n_classes, size=12)
                clip.append(frame)
            videos.append(clip)
        return ds.Dataset.from_dict({"video": videos})

    if kind in {"audio_jepa", "speech_transcript", "multispeaker_speech", "music_caption", "sound_caption"}:
        length = max(64, seq * 8)
        audios, texts, speakers = [], [], []
        for i in range(n):
            freq = 1.0 + (i % 5)
            audios.append(wave(length, freq=freq, phase=0.2 * i, noise_scale=0.02))
            texts.append(f"Synthetic demo sample {i % 12} for audio model training")
            speakers.append(i % 4)
        if kind == "speech_transcript": return ds.Dataset.from_dict({"audio": audios, "text": texts})
        if kind == "multispeaker_speech":
            references = [wave(length, freq=1.0 + (spk % 4) * 0.65, phase=0.7 + 0.1 * i, noise_scale=0.015) for i, spk in enumerate(speakers)]
            return ds.Dataset.from_dict({"audio": audios, "text": texts, "speaker_id": speakers, "reference_audio": references})
        if kind == "music_caption": return ds.Dataset.from_dict({"audio": audios, "caption": [f"demo instrumental pattern {i%6}" for i in range(n)]})
        if kind == "sound_caption": return ds.Dataset.from_dict({"audio": audios, "caption": [f"synthetic sound effect {i%6}" for i in range(n)]})
        return ds.Dataset.from_dict({"audio": audios})

    if kind in {"signal_jepa", "signal_classification", "anomaly_detection", "spectral_signal"}:
        length = max(64, seq * 2)
        signals, labels, spectra = [], [], []
        for i in range(n):
            label = i % 3
            sig = wave(length, freq=1.0 + label, phase=0.1 * i, noise_scale=0.04)
            if kind == "anomaly_detection" and i % 5 == 0:
                sig[length // 2] += 3.0
                label = 1
            elif kind == "anomaly_detection":
                label = 0
            signals.append(sig)
            labels.append(label)
            if kind == "spectral_signal":
                # Small pedagogical DFT magnitude prefix; enough for the Data
                # Inspector without pulling in numpy/scipy.
                mags = []
                bins = min(16, length // 2)
                for k in range(bins):
                    re = sum(v * math.cos(2*math.pi*k*j/length) for j, v in enumerate(sig))
                    im = -sum(v * math.sin(2*math.pi*k*j/length) for j, v in enumerate(sig))
                    mags.append((re*re + im*im) ** 0.5 / length)
                spectra.append(mags)
        payload = {"signal": signals}
        if kind in {"signal_classification", "anomaly_detection"}: payload["label"] = labels
        if kind == "spectral_signal":
            payload["spectrum"] = spectra
            payload["target"] = spectra
        return ds.Dataset.from_dict(payload)

    if kind == "long_signal":
        contexts, targets = [], []
        length = max(128, seq * 8)
        horizon = max(4, min(16, seq // 2))
        for i in range(n):
            full = wave(length + horizon, freq=0.5 + (i % 5) * 0.25, phase=0.05 * i, noise_scale=0.02)
            contexts.append(full[:length])
            targets.append(full[length:])
        return ds.Dataset.from_dict({"signal": contexts, "target": targets})

    if kind == "timeseries_forecast":
        contexts, targets = [], []
        horizon = max(4, seq // 4)
        for i in range(n):
            full = wave(seq + horizon, freq=1.0 + (i % 4) * 0.25, phase=0.07 * i, noise_scale=0.03)
            contexts.append(full[:seq]); targets.append(full[seq:])
        return ds.Dataset.from_dict({"context": contexts, "target": targets})

    if kind == "signal_denoise":
        noisy, clean = [], []
        length = max(64, seq * 2)
        for i in range(n):
            base = wave(length, freq=1.0 + (i % 3), phase=0.1 * i, noise_scale=0.0)
            clean.append(base)
            noisy.append([v + noise(0.2) for v in base])
        return ds.Dataset.from_dict({"noisy_signal": noisy, "clean_signal": clean})

    if kind == "sensor_fusion":
        a, b, c, labels = [], [], [], []
        length = max(16, seq)
        for i in range(n):
            label = i % 3
            a.append(wave(length, 1.0 + label, 0.0, 0.03))
            b.append(wave(length, 1.0 + label, 0.7, 0.03))
            c.append(wave(length, 0.5 + label, 1.2, 0.03))
            labels.append(label)
        return ds.Dataset.from_dict({"sensor_a": a, "sensor_b": b, "sensor_c": c, "label": labels})

    if kind == "rf_iq":
        i_vals, q_vals, labels = [], [], []
        length = max(32, seq * 2)
        for idx in range(n):
            label = idx % n_classes
            freq = 1.0 + label
            phase = 0.25 * idx
            i_sig = [math.cos(2*math.pi*freq*j/length + phase) + noise(0.03) for j in range(length)]
            q_sig = [math.sin(2*math.pi*freq*j/length + phase) + noise(0.03) for j in range(length)]
            i_vals.append(i_sig); q_vals.append(q_sig); labels.append(label)
        return ds.Dataset.from_dict({"i": i_vals, "q": q_vals, "label": labels})

    if kind == "multimodal_image_text":
        images, texts = [], []
        for i in range(n):
            label = i % n_classes
            images.append(image(label))
            texts.append(["vertical pattern", "horizontal pattern", "diagonal pattern"][label % 3])
        return ds.Dataset.from_dict({"image": images, "text": texts})

    if kind == "sensor_vision":
        images, sensors, labels = [], [], []
        length = max(16, seq)
        for i in range(n):
            label = i % n_classes
            images.append(image(label))
            sensors.append(wave(length, 1.0 + label, 0.1 * i, 0.03))
            labels.append(label)
        return ds.Dataset.from_dict({"image": images, "sensor": sensors, "label": labels})

    raise ValueError(f"Unknown Studio demo dataset type: {demo_type!r}")

def load_manual_text_dataset(
    text: str,
    *,
    text_column: str = "text",
    one_line_per_sample: bool = True,
):
    """Create a Hugging Face Dataset from text pasted into Builder."""
    ds = _datasets()
    raw = str(text or "")
    if one_line_per_sample:
        samples = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        samples = [raw]
    if not samples:
        samples = [""]
    return ds.Dataset.from_dict({text_column: samples})


def train_validation_test_split(
    dataset,
    *,
    train_size: float = 0.90,
    validation_size: float = 0.05,
    test_size: float = 0.05,
    seed: int = 42,
    shuffle: bool = True,
):
    """Create train/validation/test splits whose proportions sum to 1."""
    ds = _datasets()
    source = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset

    train_size = float(train_size)
    validation_size = float(validation_size)
    test_size = float(test_size)
    total = train_size + validation_size + test_size
    if any(x < 0 for x in (train_size, validation_size, test_size)):
        raise ValueError("Split sizes cannot be negative.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + validation_size + test_size must equal 1.0.")
    if train_size <= 0:
        raise ValueError("train_size must be greater than 0.")

    if validation_size == 0 and test_size == 0:
        return ds.DatasetDict({"train": source})

    first_holdout = validation_size + test_size
    first = source.train_test_split(
        test_size=first_holdout,
        seed=int(seed),
        shuffle=bool(shuffle),
    )
    result = {"train": first["train"]}

    if validation_size == 0:
        result["test"] = first["test"]
    elif test_size == 0:
        result["validation"] = first["test"]
    else:
        relative_test = test_size / first_holdout
        second = first["test"].train_test_split(
            test_size=relative_test,
            seed=int(seed) + 1,
            shuffle=bool(shuffle),
        )
        result["validation"] = second["train"]
        result["test"] = second["test"]

    return ds.DatasetDict(result)


def _coerce_pil_image(value, *, mode="RGB"):
    """Convert PIL/numpy/list image values to a PIL image without extra deps."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image processing needs Pillow: pip install pillow") from exc
    import numpy as np

    if hasattr(value, "convert") and hasattr(value, "size"):
        return value.convert(mode)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim not in {2, 3}:
        raise ValueError(f"Image values must be HxW, HxWxC, or CxHxW; received shape {arr.shape}.")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        finite = arr[np.isfinite(arr)]
        peak = float(finite.max()) if finite.size else 0.0
        floor = float(finite.min()) if finite.size else 0.0
        if floor >= 0.0 and peak <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return Image.fromarray(arr).convert(mode)


def _tensor_ready_image(image, *, normalize=True):
    """Return numeric image data in Studio's tensor-friendly CHW/HW layout."""
    import numpy as np
    arr = np.asarray(image, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    if arr.ndim == 3:
        arr = np.moveaxis(arr, -1, 0)
    return arr.tolist()



def prepare_jepa_dataset(
    dataset,
    *,
    modality: str = "image",
    input_column: str | None = None,
    output_column: str = "jepa_input",
    sequence_length: int = 64,
    image_size: int = 16,
    normalize: bool = True,
):
    """Prepare a Dataset/DatasetDict for the universal educational JEPA trainer.

    The transformation is intentionally transparent and deterministic:
    - text is encoded with a tiny byte vocabulary (0 = pad/mask, 1..256 = byte+1)
    - image is converted to CHW numeric data
    - video becomes TCHW numeric data
    - audio/signal are padded or truncated 1-D float windows

    Random context/target masking remains a visible *model* component (JEPA Mask),
    so the Data Graph prepares samples without hiding the learning objective.
    """
    import numpy as np

    mode = str(modality or "image").strip().lower()
    if mode not in {"image", "video", "text", "audio", "signal"}:
        raise ValueError("JEPA Preparation modality must be image, video, text, audio, or signal.")
    default_columns = {"image":"image", "video":"video", "text":"text", "audio":"audio", "signal":"signal"}
    input_column = str(input_column or default_columns[mode])
    output_column = str(output_column or "jepa_input")
    seq = max(4, int(sequence_length or 64))
    size = max(4, int(image_size or 16))

    def _pad_1d(values):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.size >= seq:
            arr = arr[:seq]
        else:
            arr = np.pad(arr, (0, seq-arr.size), mode="constant")
        if normalize and arr.size:
            peak = float(np.max(np.abs(arr)))
            if peak > 1e-8:
                arr = arr / peak
        return arr.tolist()

    def transform(example):
        value = example[input_column]
        if mode == "text":
            raw = str(value).encode("utf-8", errors="replace")[:seq]
            ids = [int(b) + 1 for b in raw]
            ids += [0] * (seq - len(ids))
            example[output_column] = ids
            return example

        if mode == "image":
            image = _coerce_pil_image(value, mode="L")
            from PIL import Image
            image = image.resize((size, size), Image.Resampling.BILINEAR)
            arr = np.asarray(image, dtype=np.float32)
            if normalize and arr.max(initial=0.0) > 1.0:
                arr = arr / 255.0
            example[output_column] = arr[None, ...].tolist()
            return example

        if mode == "video":
            frames = list(value or [])[:seq]
            prepared = []
            from PIL import Image
            for frame in frames:
                image = _coerce_pil_image(frame, mode="L").resize((size, size), Image.Resampling.BILINEAR)
                arr = np.asarray(image, dtype=np.float32)
                if normalize and arr.max(initial=0.0) > 1.0:
                    arr = arr / 255.0
                prepared.append(arr[None, ...].tolist())
            if not prepared:
                prepared = [[[ [0.0 for _ in range(size)] for _ in range(size) ]]]
            while len(prepared) < seq:
                prepared.append(prepared[-1])
            example[output_column] = prepared[:seq]
            return example

        # Hugging Face Audio values may be decoded dictionaries; Studio demo
        # audio and signal values are ordinary numeric lists.
        if isinstance(value, dict) and "array" in value:
            value = value["array"]
        example[output_column] = _pad_1d(value)
        return example

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        return dataset.__class__({name: split.map(transform) for name, split in dataset.items()})
    return dataset.map(transform)


def process_image_dataset(
    dataset,
    *,
    image_column: str = "image",
    width: int = 224,
    height: int = 224,
    mode: str = "RGB",
    center_crop: bool = False,
    tensor_ready: bool = False,
    normalize: bool = True,
):
    """Resize/crop images in a Dataset or DatasetDict.

    ``tensor_ready=True`` converts the result to nested numeric values in HW
    (grayscale) or CHW (color) layout. This lets the generic Studio trainer
    consume both Hub/PIL images and generated list-based demo images through
    the same public data component.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image processing needs Pillow: pip install pillow") from exc

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    def transform(example):
        image = _coerce_pil_image(example[image_column], mode=mode)
        if center_crop:
            w, h = image.size
            target_ratio = width / height
            current_ratio = w / h if h else target_ratio
            if current_ratio > target_ratio:
                new_w = max(1, int(h * target_ratio))
                left = max(0, (w - new_w) // 2)
                image = image.crop((left, 0, left + new_w, h))
            else:
                new_h = max(1, int(w / target_ratio))
                top = max(0, (h - new_h) // 2)
                image = image.crop((0, top, w, top + new_h))
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        example[image_column] = _tensor_ready_image(image, normalize=normalize) if tensor_ready else image
        return example

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        return dataset.__class__({name: split.map(transform) for name, split in dataset.items()})
    return dataset.map(transform)


def process_detection_dataset(
    dataset,
    *,
    image_column: str = "image",
    boxes_column: str = "boxes",
    classes_column: str = "class_ids",
    width: int = 16,
    height: int = 16,
    mode: str = "L",
    box_format: str = "xywh",
    normalize_images: bool = True,
):
    """Prepare image + bounding-box datasets for the educational detector.

    Boxes stay in pixel coordinates after resizing so the model trainer can
    normalize them against the actual tensor shape. The public detection
    contract uses ``xywh`` boxes and supports one or more boxes per sample.
    A deterministic ``label`` field (largest annotated object) is also emitted
    so the same prepared cloud dataset can train image classifiers while
    detectors retain the complete multi-object annotations.
    """
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Detection processing needs Pillow: pip install pillow") from exc

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")
    fmt = str(box_format or "xywh").strip().lower()
    if fmt != "xywh":
        raise ValueError("Studio detection processing currently supports box_format='xywh'.")

    def transform(example):
        image = _coerce_pil_image(example[image_column], mode=mode)
        old_w, old_h = image.size
        sx = width / max(float(old_w), 1.0)
        sy = height / max(float(old_h), 1.0)
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        boxes = example.get(boxes_column) or []
        scaled = []
        for raw in boxes:
            if raw is None or len(raw) < 4:
                continue
            x, y, w, h = [float(v) for v in raw[:4]]
            scaled.append([x * sx, y * sy, w * sx, h * sy])
        example[image_column] = _tensor_ready_image(resized, normalize=normalize_images)
        example[boxes_column] = scaled
        # Keep class ids explicit and numeric even when a source uses tuples.
        classes = [int(v) for v in (example.get(classes_column) or [])]
        example[classes_column] = classes

        # A detection dataset can also train ordinary image classifiers without
        # creating a second copy of the source data.  Expose one deterministic
        # image-level target using the class of the largest annotated object.
        # Detection models continue to use the complete boxes/class_ids arrays;
        # reconstruction and self-supervised models simply ignore this field.
        paired = [(box, cls) for box, cls in zip(scaled, classes) if len(box) >= 4]
        if paired:
            _, primary_class = max(paired, key=lambda item: max(float(item[0][2]), 0.0) * max(float(item[0][3]), 0.0))
            example["label"] = int(primary_class)
        else:
            example["label"] = -1
        return example

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        return dataset.__class__({name: split.map(transform) for name, split in dataset.items()})
    return dataset.map(transform)


def process_signal_dataset(
    dataset,
    *,
    signal_columns: str | list[str] = "signal",
    output_column: str = "signal",
    target_column: str | None = None,
    target_output_column: str = "target",
    normalize: bool = False,
    pad_length: int = 0,
):
    """Map arbitrary numeric signal columns into Studio's canonical signal field.

    One source column produces ``[T]`` per sample. Multiple source columns are
    stacked as ``[C,T]`` so Conv1D-based custom models can consume sensor fusion
    and RF/IQ data without hidden model-side preprocessing. An optional target
    column can be copied to a canonical target field for forecasting, denoising,
    spectral reconstruction, and other regression tasks.
    """
    import numpy as np

    if isinstance(signal_columns, str):
        columns = [part.strip() for part in signal_columns.split(",") if part.strip()]
    else:
        columns = [str(part).strip() for part in (signal_columns or []) if str(part).strip()]
    if not columns:
        raise ValueError("Signal Schema Mapper needs at least one signal column.")
    output_column = str(output_column or "signal").strip() or "signal"
    target_column = str(target_column or "").strip() or None
    target_output_column = str(target_output_column or "target").strip() or "target"
    pad_length = max(0, int(pad_length or 0))

    def prepare_vector(value):
        if isinstance(value, dict) and "array" in value:
            value = value["array"]
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if pad_length:
            if arr.size >= pad_length:
                arr = arr[:pad_length]
            else:
                arr = np.pad(arr, (0, pad_length - arr.size), mode="constant")
        if normalize and arr.size:
            peak = float(np.max(np.abs(arr)))
            if peak > 1e-8:
                arr = arr / peak
        return arr.tolist()

    def transform(example):
        missing = [name for name in columns if name not in example]
        if missing:
            raise KeyError(f"Signal column(s) not found: {', '.join(missing)}")
        prepared = [prepare_vector(example[name]) for name in columns]
        example[output_column] = prepared[0] if len(prepared) == 1 else prepared
        if target_column is not None:
            if target_column not in example:
                raise KeyError(f"Signal target column {target_column!r} not found.")
            value = example[target_column]
            if isinstance(value, (list, tuple)) or hasattr(value, "shape"):
                try:
                    value = np.asarray(value, dtype=np.float32).tolist()
                except Exception:
                    pass
            example[target_output_column] = value
        return example

    def map_one(split):
        available = set(getattr(split, "column_names", []) or [])
        if available:
            missing = [name for name in columns if name not in available]
            if missing:
                raise KeyError(f"Signal column(s) not found: {', '.join(missing)}")
            if target_column is not None and target_column not in available:
                raise KeyError(f"Signal target column {target_column!r} not found.")
        return split.map(transform)

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        return dataset.__class__({name: map_one(split) for name, split in dataset.items()})
    return map_one(dataset)


def process_audio_dataset(
    dataset,
    *,
    audio_column: str = "audio",
    sample_rate: int = 16000,
    normalize: bool = True,
    trim_silence: bool = False,
    silence_threshold: float = 0.01,
):
    """Resample through datasets.Audio, then optionally normalize/trim arrays."""
    import numpy as np
    ds = _datasets()
    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    def cast_one(split):
        if audio_column not in split.column_names:
            raise KeyError(f"Audio column {audio_column!r} not found: {split.column_names}")
        split = split.cast_column(audio_column, ds.Audio(sampling_rate=sample_rate))

        def transform(example):
            audio = example[audio_column]
            arr = np.asarray(audio["array"], dtype=np.float32)
            if trim_silence and arr.size:
                idx = np.flatnonzero(np.abs(arr) >= float(silence_threshold))
                if idx.size:
                    arr = arr[idx[0]:idx[-1] + 1]
            if normalize and arr.size:
                peak = float(np.max(np.abs(arr)))
                if peak > 0:
                    arr = arr / peak
            example[audio_column] = {
                "array": arr,
                "sampling_rate": sample_rate,
            }
            return example

        return split.map(transform)

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        return dataset.__class__({name: cast_one(split) for name, split in dataset.items()})
    return cast_one(dataset)


def make_torch_dataloader(
    dataset,
    *,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 2,
    drop_last: bool = False,
):
    """Create a torch DataLoader from a Dataset or its train split."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Batch/DataLoader needs PyTorch.") from exc

    source = dataset["train"] if hasattr(dataset, "keys") and "train" in dataset else dataset
    return DataLoader(
        source,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
    )


def prepared_dataset_output(
    dataset,
    *,
    save_to_disk: bool = False,
    path: str = "mlbricks_workspace/data/prepared_dataset",
):
    """Return prepared data and optionally persist Dataset/DatasetDict objects."""
    if save_to_disk:
        if not hasattr(dataset, "save_to_disk"):
            raise TypeError(
                "Save To Disk requires a Dataset/DatasetDict. "
                "Place Prepared Dataset before Batch/DataLoader, or disable Save To Disk."
            )
        target = Path(path).expanduser()
        if target.exists():
            raise FileExistsError(
                f"Prepared dataset path already exists: {target}. "
                "Choose another Dataset Name or Save Path; existing data is never overwritten."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(target))
    return dataset


__all__ = [
    "prepared_dataset_output",
    "make_torch_dataloader",
    "process_audio_dataset",
    "process_image_dataset",
    "process_signal_dataset",
    "process_detection_dataset",
    "train_validation_test_split",
    "load_manual_text_dataset",
    "prepare_text_input",
    "load_huggingface_dataset",
    "load_kaggle_dataset",
    "load_url_dataset",
    "load_coco128_cloud_dataset",
    "COCO128_DOWNLOAD_URL",
    "COCO80_CLASS_NAMES",
    "load_local_dataset",
    "process_text_dataset",
    "train_test_split",
    "tokenize_text_dataset",
]
