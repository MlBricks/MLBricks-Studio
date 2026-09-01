from __future__ import annotations

from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse
from typing import Any


def _datasets():
    try:
        import datasets
        return datasets
    except ImportError as exc:
        raise ImportError(
            "Text/data features need Hugging Face datasets. "
            "Install with: pip install 'mlb-studio[data]' "
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


def load_huggingface_dataset(
    dataset_id: str,
    *,
    config: str | None = None,
    split: str = "train",
    text_column: str = "text",
    streaming: bool = False,
    max_rows: int | None = None,
):
    """Load a dataset from the Hugging Face Hub.

    Authentication, when required, is read by the datasets library from the
    normal Hugging Face environment/login. Tokens are intentionally not stored
    in Builder design files.
    """
    ds = _datasets()
    config = (config or "").strip() or None
    data = ds.load_dataset(
        dataset_id,
        config,
        split=split or "train",
        streaming=bool(streaming),
    )
    if not streaming:
        data = _limit_rows(data, max_rows)
    if text_column and hasattr(data, "column_names") and text_column not in data.column_names:
        raise KeyError(
            f"Text column {text_column!r} not found. Available columns: {data.column_names}"
        )
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
            "Tokenization needs transformers. Install with: pip install transformers"
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


def process_image_dataset(
    dataset,
    *,
    image_column: str = "image",
    width: int = 224,
    height: int = 224,
    mode: str = "RGB",
    center_crop: bool = False,
):
    """Resize/crop PIL-compatible images in a Dataset or DatasetDict."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Image processing needs Pillow: pip install pillow") from exc

    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive.")

    def transform(example):
        image = example[image_column]
        if hasattr(image, "convert"):
            image = image.convert(mode)
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
        example[image_column] = image.resize((width, height), Image.Resampling.BILINEAR)
        return example

    if hasattr(dataset, "items") and not hasattr(dataset, "column_names"):
        return dataset.__class__({name: split.map(transform) for name, split in dataset.items()})
    return dataset.map(transform)


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
    path: str = "mlbricks/data/prepared_dataset",
):
    """Return prepared data and optionally persist Dataset/DatasetDict objects."""
    if save_to_disk:
        if not hasattr(dataset, "save_to_disk"):
            raise TypeError(
                "Save To Disk requires a Dataset/DatasetDict. "
                "Place Prepared Dataset before Batch/DataLoader, or disable Save To Disk."
            )
        dataset.save_to_disk(str(path))
    return dataset


__all__ = [
    "prepared_dataset_output",
    "make_torch_dataloader",
    "process_audio_dataset",
    "process_image_dataset",
    "train_validation_test_split",
    "load_manual_text_dataset",
    "prepare_text_input",
    "load_huggingface_dataset",
    "load_kaggle_dataset",
    "load_url_dataset",
    "load_local_dataset",
    "process_text_dataset",
    "train_test_split",
    "tokenize_text_dataset",
]
