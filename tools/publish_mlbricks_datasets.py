#!/usr/bin/env python3
"""Publish the MLBricks-maintained Hugging Face datasets used by Studio Gallery.

The script intentionally uses two strategies:

* Server-side mirrors for datasets that are practical to duplicate as-is.
* Curated/normalized repositories for sources where Studio only needs one
  language/subset or one stable training schema.

Authentication is read from HF_TOKEN or the normal ``hf auth login`` cache.
Never hard-code a Hugging Face token into this file.

Examples
--------
Create the two server-side mirrors first (fast):

    python tools/publish_mlbricks_datasets.py --dataset tinystories
    python tools/publish_mlbricks_datasets.py --dataset cosmopedia

Build a 1B-token curated edition (long-running; Kaggle/Colab recommended):

    python tools/publish_mlbricks_datasets.py --dataset wikipedia
    python tools/publish_mlbricks_datasets.py --dataset fineweb-edu
    python tools/publish_mlbricks_datasets.py --dataset openwebmath

Normalize UltraChat into a common ``text`` column:

    python tools/publish_mlbricks_datasets.py --dataset ultrachat

Run everything sequentially:

    python tools/publish_mlbricks_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator


ONE_BILLION = 1_000_000_000
DEFAULT_SHARD_ROWS = 20_000
DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True)
class CuratedSpec:
    key: str
    destination: str
    source: str
    source_config: str | None
    source_split: str
    pretty_name: str
    license_frontmatter: str
    license_text: str
    caveat: str
    extractor: Callable[[dict], tuple[str, str]]


def _require_packages():
    try:
        import datasets  # noqa: F401
        import pyarrow  # noqa: F401
        import huggingface_hub  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing publishing dependencies. Install them with:\n"
            "  python -m pip install -U huggingface_hub datasets transformers pyarrow\n"
            f"Original import error: {exc}"
        ) from exc


def _text_and_ref(row: dict) -> tuple[str, str]:
    text = str(row.get("text") or "").strip()
    ref = str(row.get("url") or row.get("id") or row.get("source") or "")
    return text, ref


def _ultrachat_text_and_ref(row: dict) -> tuple[str, str]:
    messages = row.get("messages") or []
    rendered: list[str] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        label = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
        }.get(role, role.title() or "Message")
        rendered.append(f"{label}: {content}")
    text = "\n\n".join(rendered).strip()
    return text, str(row.get("prompt_id") or "")


CURATED_SPECS = {
    "wikipedia": CuratedSpec(
        key="wikipedia",
        destination="wikipedia-en-1b",
        source="wikimedia/wikipedia",
        source_config="20231101.en",
        source_split="train",
        pretty_name="MLBricks Wikipedia EN 1B",
        license_frontmatter="license:\n- cc-by-sa-3.0\n- gfdl",
        license_text="CC BY-SA 3.0 + GNU Free Documentation License (GFDL)",
        caveat=(
            "Wikipedia attribution and share-alike requirements remain in force. "
            "MLBricks does not claim ownership of the underlying articles."
        ),
        extractor=_text_and_ref,
    ),
    "fineweb-edu": CuratedSpec(
        key="fineweb-edu",
        destination="fineweb-edu-1b",
        source="HuggingFaceFW/fineweb-edu",
        source_config="sample-10BT",
        source_split="train",
        pretty_name="MLBricks FineWeb-Edu 1B",
        license_frontmatter="license: odc-by",
        license_text="ODC-By 1.0",
        caveat=(
            "The upstream dataset is subject to Common Crawl Terms of Use and "
            "the rights of underlying webpages are not changed by this curated edition."
        ),
        extractor=_text_and_ref,
    ),
    "openwebmath": CuratedSpec(
        key="openwebmath",
        destination="openwebmath-1b",
        source="open-web-math/open-web-math",
        source_config=None,
        source_split="train",
        pretty_name="MLBricks OpenWebMath 1B",
        license_frontmatter="license: odc-by",
        license_text="ODC-By 1.0",
        caveat=(
            "The upstream dataset is subject to Common Crawl Terms of Use and "
            "does not alter licenses or rights of the underlying web content."
        ),
        extractor=_text_and_ref,
    ),
}


DIRECT_MIRRORS = {
    "tinystories": {
        "source": "roneneldan/TinyStories",
        "destination": "tinystories",
        "license": "CDLA-Sharing-1.0",
        "note": "Full server-side mirror for stable MLBricks Studio presets.",
    },
    "cosmopedia": {
        "source": "HuggingFaceTB/cosmopedia",
        "destination": "cosmopedia",
        "license": "Apache-2.0",
        "note": "Full server-side mirror; Studio defaults to the OpenStax config.",
    },
}


def _repo_exists(api, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        return True
    except Exception:
        return False


def _insert_mirror_banner(readme: str, banner: str) -> str:
    if readme.startswith("---"):
        end = readme.find("\n---", 3)
        if end != -1:
            end += len("\n---")
            return readme[:end] + "\n\n" + banner + "\n\n" + readme[end:].lstrip("\n")
    return banner + "\n\n" + readme


def _upload_text(api, repo_id: str, path_in_repo: str, text: str, commit_message: str):
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".md", delete=False) as f:
        f.write(text)
        temp_name = f.name
    try:
        api.upload_file(
            path_or_fileobj=temp_name,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    finally:
        Path(temp_name).unlink(missing_ok=True)


def publish_direct_mirror(api, org: str, key: str, private: bool = False) -> str:
    from huggingface_hub import hf_hub_download

    spec = DIRECT_MIRRORS[key]
    destination = f"{org}/{spec['destination']}"
    if _repo_exists(api, destination):
        print(f"[skip] {destination} already exists")
        return destination

    print(f"[mirror] {spec['source']} -> {destination}")
    api.duplicate_repo(
        from_id=spec["source"],
        to_id=destination,
        repo_type="dataset",
        private=private,
    )

    source_info = api.dataset_info(spec["source"])
    source_sha = getattr(source_info, "sha", None) or "unknown"
    original_readme_path = hf_hub_download(
        repo_id=spec["source"], filename="README.md", repo_type="dataset"
    )
    original = Path(original_readme_path).read_text(encoding="utf-8")
    banner = (
        "> **MLBricks maintained mirror**  \n"
        f"> Upstream: `{spec['source']}` @ `{source_sha}`  \n"
        f"> License: **{spec['license']}**  \n"
        f"> {spec['note']}  \n"
        "> MLBricks preserves upstream attribution and does not claim ownership of the underlying dataset."
    )
    readme = _insert_mirror_banner(original, banner)
    _upload_text(api, destination, "README.md", readme, "Add MLBricks mirror provenance")

    manifest = {
        "mlbricks_dataset_format": 1,
        "strategy": "server-side-mirror",
        "upstream_repo": spec["source"],
        "upstream_revision": source_sha,
        "destination_repo": destination,
        "license": spec["license"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
        json.dump(manifest, f, indent=2)
        manifest_path = f.name
    try:
        api.upload_file(
            path_or_fileobj=manifest_path,
            path_in_repo="mlbricks_manifest.json",
            repo_id=destination,
            repo_type="dataset",
            commit_message="Add MLBricks provenance manifest",
        )
    finally:
        Path(manifest_path).unlink(missing_ok=True)
    print(f"[done] {destination}")
    return destination


def _batched(iterable: Iterable[tuple[str, str]], size: int) -> Iterator[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _source_rows(spec: CuratedSpec):
    from datasets import load_dataset

    print(
        f"[stream] {spec.source}"
        + (f" / {spec.source_config}" if spec.source_config else "")
        + f" / {spec.source_split}"
    )
    ds = load_dataset(
        spec.source,
        spec.source_config,
        split=spec.source_split,
        streaming=True,
    )
    for row in ds:
        text, ref = spec.extractor(row)
        if text:
            yield text, ref


def _write_shard(rows: list[dict], out_dir: Path, split_name: str, shard_index: int) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = out_dir / f"{split_name}-{shard_index:05d}.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path, compression="zstd")
    return path


def build_token_limited_split(
    spec: CuratedSpec,
    out_dir: Path,
    *,
    target_tokens: int,
    split_name: str = "train",
    shard_rows: int = DEFAULT_SHARD_ROWS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    rows_buffer: list[dict] = []
    shard_index = 0
    total_rows = 0
    total_tokens = 0

    for batch in _batched(_source_rows(spec), batch_size):
        texts = [item[0] for item in batch]
        refs = [item[1] for item in batch]
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False)

        for text, ref, token_ids in zip(texts, refs, encoded["input_ids"]):
            remaining = target_tokens - total_tokens
            if remaining <= 0:
                break
            if len(token_ids) > remaining:
                if remaining < 8:
                    total_tokens = target_tokens
                    break
                token_ids = token_ids[:remaining]
                text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

            rows_buffer.append(
                {
                    "text": text,
                    "source_dataset": spec.source,
                    "source_config": spec.source_config or "",
                    "source_split": spec.source_split,
                    "source_ref": ref,
                }
            )
            total_rows += 1
            total_tokens += len(token_ids)

            if len(rows_buffer) >= shard_rows:
                path = _write_shard(rows_buffer, out_dir, split_name, shard_index)
                print(
                    f"[shard] {path.name} · rows={total_rows:,} · "
                    f"tokens={total_tokens:,}/{target_tokens:,}"
                )
                shard_index += 1
                rows_buffer = []

            if total_tokens >= target_tokens:
                break
        if total_tokens >= target_tokens:
            break

    if rows_buffer:
        path = _write_shard(rows_buffer, out_dir, split_name, shard_index)
        print(f"[shard] {path.name} · final")

    if total_tokens < target_tokens:
        print(
            f"[warning] source ended at {total_tokens:,} tokens, "
            f"below requested {target_tokens:,}"
        )

    return {"rows": total_rows, "tokens": total_tokens, "split": split_name}


def build_full_normalized_split(
    *,
    source: str,
    source_config: str | None,
    source_split: str,
    destination_split: str,
    extractor: Callable[[dict], tuple[str, str]],
    out_dir: Path,
    shard_rows: int = DEFAULT_SHARD_ROWS,
) -> dict:
    from datasets import load_dataset

    ds = load_dataset(source, source_config, split=source_split, streaming=True)
    rows_buffer: list[dict] = []
    shard_index = 0
    total_rows = 0

    for row in ds:
        text, ref = extractor(row)
        if not text:
            continue
        rows_buffer.append(
            {
                "text": text,
                "source_dataset": source,
                "source_config": source_config or "",
                "source_split": source_split,
                "source_ref": ref,
            }
        )
        total_rows += 1
        if len(rows_buffer) >= shard_rows:
            path = _write_shard(rows_buffer, out_dir, destination_split, shard_index)
            print(f"[shard] {path.name} · rows={total_rows:,}")
            shard_index += 1
            rows_buffer = []

    if rows_buffer:
        path = _write_shard(rows_buffer, out_dir, destination_split, shard_index)
        print(f"[shard] {path.name} · final")

    return {"rows": total_rows, "split": destination_split}


def curated_readme(spec: CuratedSpec, destination: str, source_sha: str, stats: dict) -> str:
    token_count = int(stats.get("tokens") or 0)
    row_count = int(stats.get("rows") or 0)
    return f"""---
{spec.license_frontmatter}
language:
- en
task_categories:
- text-generation
pretty_name: {spec.pretty_name}
---

# {spec.pretty_name}

> **MLBricks curated dataset**  
> Upstream: `{spec.source}` @ `{source_sha}`  
> Upstream config: `{spec.source_config or 'default'}`  
> Upstream split: `{spec.source_split}`  
> License: **{spec.license_text}**

This repository is maintained by MLBricks as a stable, reproducible Studio data preset.
MLBricks does **not** claim ownership of the underlying source material.

## Edition

- Destination: `{destination}`
- Rows: **{row_count:,}**
- GPT-2 tokens: **{token_count:,}**
- Normalized training column: `text`
- Additional provenance columns: `source_dataset`, `source_config`, `source_split`, `source_ref`

## License and source-rights notice

{spec.caveat}

Users remain responsible for complying with the upstream license, attribution requirements,
terms of use, and any rights that apply to underlying source content.

## MLBricks Studio

Studio Gallery opens this repository with a 10,000-row quickstart limit by default.
Set **Max Rows = 0** in the Hugging Face Dataset component when you intentionally want the
entire edition.
"""


def publish_curated(api, org: str, key: str, build_root: Path, target_tokens: int, private: bool) -> str:
    spec = CURATED_SPECS[key]
    destination = f"{org}/{spec.destination}"
    if _repo_exists(api, destination):
        print(f"[skip] {destination} already exists")
        return destination

    repo_build = build_root / spec.destination
    data_dir = repo_build / "data"
    if repo_build.exists():
        shutil.rmtree(repo_build)
    data_dir.mkdir(parents=True, exist_ok=True)

    source_info = api.dataset_info(spec.source)
    source_sha = getattr(source_info, "sha", None) or "unknown"
    stats = build_token_limited_split(
        spec,
        data_dir,
        target_tokens=target_tokens,
    )

    readme = curated_readme(spec, destination, source_sha, stats)
    (repo_build / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "mlbricks_dataset_format": 1,
        "strategy": "curated-token-limited",
        "upstream_repo": spec.source,
        "upstream_config": spec.source_config,
        "upstream_split": spec.source_split,
        "upstream_revision": source_sha,
        "destination_repo": destination,
        "tokenizer": "gpt2",
        "target_tokens": target_tokens,
        "actual_tokens": stats["tokens"],
        "rows": stats["rows"],
        "license": spec.license_text,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (repo_build / "mlbricks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[create] {destination}")
    api.create_repo(repo_id=destination, repo_type="dataset", private=private, exist_ok=False)
    print(f"[upload] {repo_build}")
    api.upload_folder(
        folder_path=str(repo_build),
        repo_id=destination,
        repo_type="dataset",
        commit_message=f"Publish {spec.pretty_name}",
    )
    print(f"[done] {destination}")
    return destination


def publish_ultrachat(api, org: str, build_root: Path, private: bool) -> str:
    source = "HuggingFaceH4/ultrachat_200k"
    destination = f"{org}/ultrachat-200k"
    if _repo_exists(api, destination):
        print(f"[skip] {destination} already exists")
        return destination

    repo_build = build_root / "ultrachat-200k"
    data_dir = repo_build / "data"
    if repo_build.exists():
        shutil.rmtree(repo_build)
    data_dir.mkdir(parents=True, exist_ok=True)

    train = build_full_normalized_split(
        source=source,
        source_config=None,
        source_split="train_sft",
        destination_split="train",
        extractor=_ultrachat_text_and_ref,
        out_dir=data_dir,
    )
    validation = build_full_normalized_split(
        source=source,
        source_config=None,
        source_split="test_sft",
        destination_split="validation",
        extractor=_ultrachat_text_and_ref,
        out_dir=data_dir,
    )
    source_info = api.dataset_info(source)
    source_sha = getattr(source_info, "sha", None) or "unknown"

    readme = f"""---
license: mit
language:
- en
task_categories:
- conversational
- text-generation
pretty_name: MLBricks UltraChat 200K
---

# MLBricks UltraChat 200K

> **MLBricks normalized mirror**  
> Upstream: `{source}` @ `{source_sha}`  
> License: **MIT**

This repository keeps the upstream SFT conversations while adding a normalized `text`
column for MLBricks Studio. Each conversation is rendered with explicit `User:`,
`Assistant:`, and `System:` role labels. The upstream prompt id is retained in
`source_ref`.

MLBricks does not claim ownership of the original dataset.

## Splits

- `train`: {train['rows']:,} conversations (upstream `train_sft`)
- `validation`: {validation['rows']:,} conversations (upstream `test_sft`)

## Columns

- `text`
- `source_dataset`
- `source_config`
- `source_split`
- `source_ref`
"""
    (repo_build / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "mlbricks_dataset_format": 1,
        "strategy": "normalized-mirror",
        "upstream_repo": source,
        "upstream_revision": source_sha,
        "destination_repo": destination,
        "splits": {"train": train["rows"], "validation": validation["rows"]},
        "license": "MIT",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (repo_build / "mlbricks_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    api.create_repo(repo_id=destination, repo_type="dataset", private=private, exist_ok=False)
    api.upload_folder(
        folder_path=str(repo_build),
        repo_id=destination,
        repo_type="dataset",
        commit_message="Publish MLBricks normalized UltraChat 200K",
    )
    print(f"[done] {destination}")
    return destination


def print_plan(org: str, target_tokens: int):
    print("MLBricks Hugging Face dataset publication plan")
    print("=" * 58)
    print(f"Organization: {org}")
    print(f"Curated token target: {target_tokens:,} GPT-2 tokens")
    print()
    print(f"1. roneneldan/TinyStories        -> {org}/tinystories            (server-side mirror)")
    print(f"2. wikimedia/wikipedia EN       -> {org}/wikipedia-en-1b        (curated)")
    print(f"3. HuggingFaceTB/cosmopedia     -> {org}/cosmopedia             (server-side mirror)")
    print(f"4. HuggingFaceFW/fineweb-edu    -> {org}/fineweb-edu-1b         (curated)")
    print(f"5. open-web-math/open-web-math  -> {org}/openwebmath-1b         (curated)")
    print(f"6. HuggingFaceH4/ultrachat_200k -> {org}/ultrachat-200k         (normalized mirror)")


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        default="all",
        choices=["all", "tinystories", "wikipedia", "cosmopedia", "fineweb-edu", "openwebmath", "ultrachat"],
        help="Dataset to publish. 'all' runs them sequentially.",
    )
    p.add_argument("--org", default="MlBricks", help="Destination Hugging Face organization")
    p.add_argument("--target-tokens", type=int, default=ONE_BILLION, help="Target GPT-2 tokens for curated 1B editions")
    p.add_argument("--build-dir", default="mlbricks_hf_build", help="Local build/cache directory for curated repositories")
    p.add_argument("--private", action="store_true", help="Create destination repositories as private")
    p.add_argument("--keep-build", action="store_true", help="Keep local curated parquet files after successful upload")
    p.add_argument("--dry-run", action="store_true", help="Print the publication plan without accessing Hugging Face")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.target_tokens <= 0:
        raise SystemExit("--target-tokens must be greater than zero")

    print_plan(args.org, args.target_tokens)
    if args.dry_run:
        return 0

    _require_packages()
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    who = api.whoami()
    print(f"Authenticated as: {who.get('name') or who.get('fullname') or 'unknown'}")

    build_root = Path(args.build_dir).expanduser().resolve()
    build_root.mkdir(parents=True, exist_ok=True)

    keys = (
        ["tinystories", "wikipedia", "cosmopedia", "fineweb-edu", "openwebmath", "ultrachat"]
        if args.dataset == "all"
        else [args.dataset]
    )

    try:
        for key in keys:
            print("\n" + "=" * 72)
            print(f"Publishing: {key}")
            print("=" * 72)
            if key in DIRECT_MIRRORS:
                publish_direct_mirror(api, args.org, key, private=args.private)
            elif key == "ultrachat":
                publish_ultrachat(api, args.org, build_root, args.private)
            else:
                publish_curated(api, args.org, key, build_root, args.target_tokens, args.private)
    finally:
        if not args.keep_build:
            # Only remove generated local data; remote Hub repositories are untouched.
            shutil.rmtree(build_root, ignore_errors=True)

    print("\nAll requested datasets completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
