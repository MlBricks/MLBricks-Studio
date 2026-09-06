from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
import threading


class CloudProviderError(RuntimeError):
    pass


def normalize_github_repo(repo: str) -> str:
    """Return a canonical ``owner/repository`` identifier.

    Studio accepts the short GitHub form as well as common HTTPS/SSH repo URLs.
    Extra URL path segments (``/tree/...``, ``/blob/...``) are ignored because the
    branch/file path is configured separately in the Cloud workspace.
    """
    value = str(repo or "").strip()
    if not value:
        raise ValueError("GitHub repository must be `owner/repository`.")

    if value.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(value)
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            raise ValueError("GitHub repository URL must point to github.com.")
        parts = [urllib.parse.unquote(part) for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("GitHub repository must be `owner/repository`.")
        owner, name = parts[0], parts[1]
    elif value.startswith("git@github.com:"):
        path = value.split(":", 1)[1]
        parts = [part for part in path.split("/") if part]
        if len(parts) < 2:
            raise ValueError("GitHub repository must be `owner/repository`.")
        owner, name = parts[0], parts[1]
    else:
        # Do not silently accept branch/file suffixes in the short form.
        parts = [part for part in value.strip("/").split("/") if part]
        if len(parts) != 2:
            raise ValueError("GitHub repository must be `owner/repository`.")
        owner, name = parts

    if name.endswith(".git"):
        name = name[:-4]
    if not owner or not name:
        raise ValueError("GitHub repository must be `owner/repository`.")
    return f"{owner}/{name}"


def _require_github_object(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        raise CloudProviderError(
            f"GitHub {context} resolved to a directory/list instead of a file object."
        )
    raise CloudProviderError(f"GitHub returned an unexpected response for {context}.")


def _json_request(url: str, *, method: str = "GET", token: str | None = None, payload=None):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "MLBricks-Studio",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            raw = response.read()
            return json.loads(raw.decode("utf-8")) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("message", detail)
        except Exception:
            pass
        raise CloudProviderError(f"GitHub HTTP {exc.code}: {detail}") from exc


def github_status(*, token: str) -> dict[str, Any]:
    if not token:
        return {"ok": False, "message": "GitHub token is required."}
    info = _require_github_object(
        _json_request("https://api.github.com/user", token=token),
        context="user lookup",
    )
    login = info.get("login")
    organizations: list[str] = []
    try:
        org_rows = _json_request("https://api.github.com/user/orgs", token=token)
        if isinstance(org_rows, list):
            organizations = [
                str(row.get("login") or "").strip()
                for row in org_rows
                if isinstance(row, dict) and row.get("login")
            ]
    except Exception:
        # Organization enumeration is supplementary; a valid /user response is
        # enough to establish the connection.
        pass
    return {
        "ok": True,
        "username": login,
        "organizations": organizations,
        "message": f"Connected to GitHub as {login}." if login else "GitHub token accepted.",
    }



def github_list_studio_bundles(
    *,
    repo: str,
    branch: str = "main",
    token: str | None = None,
    base_path: str = "",
) -> list[str]:
    """Return MLBricks Studio bundle paths available in a GitHub repository.

    This is used by the Load flow when the user leaves File Path blank or
    points at a directory.  It keeps GitHub convenient for Studio projects
    without guessing arbitrary non-MLBricks files.
    """
    repo = normalize_github_repo(repo)
    branch = str(branch or "main").strip() or "main"
    prefix = str(base_path or "").strip().strip("/")
    encoded_branch = urllib.parse.quote(branch, safe="")
    url = f"https://api.github.com/repos/{repo}/git/trees/{encoded_branch}?recursive=1"
    tree_info = _require_github_object(
        _json_request(url, token=token),
        context="repository tree",
    )
    rows = tree_info.get("tree") or []
    if not isinstance(rows, list):
        raise CloudProviderError("GitHub repository tree did not contain a valid file list.")

    matches: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("type") != "blob":
            continue
        path = str(row.get("path") or "").strip().lstrip("/")
        if not path.lower().endswith(".mlbricks.zip"):
            continue
        if prefix and path != prefix and not path.startswith(prefix + "/"):
            continue
        matches.append(path)
    return sorted(set(matches), key=lambda value: value.lower())


def _select_github_bundle_path(
    *,
    repo: str,
    branch: str,
    token: str | None,
    base_path: str = "",
) -> str:
    candidates = github_list_studio_bundles(
        repo=repo, branch=branch, token=token, base_path=base_path
    )
    scope = f" under `{base_path.strip().strip('/')}`" if str(base_path or "").strip().strip("/") else ""
    if not candidates:
        raise CloudProviderError(
            "No MLBricks Studio bundle (`*.mlbricks.zip`) was found" + scope +
            ". Push Studio content first or enter the complete GitHub File Path."
        )
    if len(candidates) > 1:
        preview = "\n".join(f"- {item}" for item in candidates[:10])
        more = f"\n- … and {len(candidates) - 10} more" if len(candidates) > 10 else ""
        raise CloudProviderError(
            "Multiple MLBricks Studio bundles were found" + scope +
            ". Enter the File Path for the one you want to load:\n" + preview + more
        )
    return candidates[0]

def github_upload(
    local_path: str | Path,
    *,
    repo: str,
    path_in_repo: str,
    branch: str = "main",
    token: str,
    commit_message: str = "Upload from MLB Studio",
    progress_callback=None,
) -> dict:
    if not token:
        raise CloudProviderError("GitHub token is required for upload.")
    repo = normalize_github_repo(repo)
    source_path = Path(local_path)
    path_in_repo = str(path_in_repo or "").strip().lstrip("/")
    if not path_in_repo:
        raise ValueError("GitHub file path is required.")
    if path_in_repo.endswith("/"):
        path_in_repo += source_path.name

    def content_url(remote_path: str) -> str:
        encoded = urllib.parse.quote(remote_path, safe="/")
        return f"https://api.github.com/repos/{repo}/contents/{encoded}"

    url = content_url(path_in_repo)
    sha = None
    try:
        existing = _json_request(
            url + "?" + urllib.parse.urlencode({"ref": branch}),
            token=token,
        )
        if isinstance(existing, list):
            # The user supplied an existing directory (for example ``studio``)
            # instead of a complete file path. Upload the Studio bundle inside
            # that directory rather than crashing on ``list.get``.
            path_in_repo = f"{path_in_repo.rstrip('/')}/{source_path.name}"
            url = content_url(path_in_repo)
            try:
                existing = _json_request(
                    url + "?" + urllib.parse.urlencode({"ref": branch}),
                    token=token,
                )
            except CloudProviderError as exc:
                if "HTTP 404" in str(exc):
                    existing = {}
                else:
                    raise
        existing = _require_github_object(existing, context="upload target")
        sha = existing.get("sha")
    except CloudProviderError as exc:
        if "HTTP 404" not in str(exc):
            raise

    total = source_path.stat().st_size
    if progress_callback:
        progress_callback(0, total)
    content = base64.b64encode(source_path.read_bytes()).decode("ascii")
    payload = {
        "message": commit_message,
        "content": content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha
    result = _require_github_object(
        _json_request(url, method="PUT", token=token, payload=payload),
        context="upload result",
    )
    if progress_callback:
        progress_callback(total, total)
    result_content = result.get("content")
    html_url = result_content.get("html_url") if isinstance(result_content, dict) else None
    return {
        "provider": "github",
        "repo": repo,
        "path": path_in_repo,
        "branch": branch,
        "url": html_url or f"https://github.com/{repo}/blob/{branch}/{path_in_repo}",
    }


def github_download(
    destination: str | Path,
    *,
    repo: str,
    path_in_repo: str,
    branch: str = "main",
    token: str | None = None,
    progress_callback=None,
) -> dict:
    repo = normalize_github_repo(repo)
    branch = str(branch or "main").strip() or "main"
    path_in_repo = str(path_in_repo or "").strip().lstrip("/")

    # A blank path means "find my Studio bundle".  This matches Push, which
    # defaults to mlbricks/<name>.mlbricks.zip, and avoids forcing users to
    # remember the generated bundle filename.
    if not path_in_repo:
        path_in_repo = _select_github_bundle_path(
            repo=repo, branch=branch, token=token, base_path=""
        )

    encoded_path = urllib.parse.quote(path_in_repo, safe="/")
    url = f"https://api.github.com/repos/{repo}/contents/{encoded_path}?" + urllib.parse.urlencode({"ref": branch})
    raw_info = _json_request(url, token=token)
    if isinstance(raw_info, list):
        # If a directory was entered, auto-select the bundle when there is only
        # one beneath it.  If there are several, return their paths so the user
        # can choose explicitly instead of failing with an opaque list error.
        path_in_repo = _select_github_bundle_path(
            repo=repo, branch=branch, token=token, base_path=path_in_repo
        )
        encoded_path = urllib.parse.quote(path_in_repo, safe="/")
        url = f"https://api.github.com/repos/{repo}/contents/{encoded_path}?" + urllib.parse.urlencode({"ref": branch})
        raw_info = _json_request(url, token=token)
    info = _require_github_object(raw_info, context="download target")
    download_url = info.get("download_url")
    destination = Path(destination)
    if not download_url:
        content = info.get("content")
        if not content:
            raise CloudProviderError("GitHub did not return downloadable file content.")
        data = base64.b64decode(content)
        if progress_callback:
            progress_callback(0, len(data))
        destination.write_bytes(data)
        if progress_callback:
            progress_callback(len(data), len(data))
    else:
        headers = {"User-Agent": "MLBricks-Studio"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(download_url, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as response, destination.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            if progress_callback:
                progress_callback(0, total or None)
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress_callback:
                    progress_callback(done, total or None)
    return {
        "provider": "github",
        "repo": repo,
        "path": path_in_repo,
        "branch": branch,
        "url": info.get("html_url") or f"https://github.com/{repo}/blob/{branch}/{path_in_repo}",
    }


def s3_client(credentials: dict):
    try:
        import boto3
    except ImportError as exc:
        raise CloudProviderError(
            "AWS S3 support needs boto3. Install: pip install boto3"
        ) from exc
    kwargs = {}
    if credentials.get("access_key"):
        kwargs["aws_access_key_id"] = credentials["access_key"]
    if credentials.get("secret_key"):
        kwargs["aws_secret_access_key"] = credentials["secret_key"]
    if credentials.get("session_token"):
        kwargs["aws_session_token"] = credentials["session_token"]
    if credentials.get("region"):
        kwargs["region_name"] = credentials["region"]
    return boto3.client("s3", **kwargs)


def s3_status(credentials: dict) -> dict:
    try:
        client = s3_client(credentials)
        ident = client._request_signer._credentials
        # Avoid a network STS dependency just to display credential presence.
        return {
            "ok": bool(ident),
            "message": "AWS S3 credentials/configuration available." if ident else "AWS credentials were not found.",
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def s3_upload(local_path, *, bucket: str, object_key: str, credentials: dict, progress_callback=None) -> dict:
    if not bucket or not object_key:
        raise ValueError("AWS Bucket and Object Key are required.")
    client = s3_client(credentials)
    source = Path(local_path)
    total = source.stat().st_size
    lock = threading.Lock()
    done = 0
    def callback(amount):
        nonlocal done
        with lock:
            done += int(amount or 0)
            if progress_callback:
                progress_callback(done, total)
    if progress_callback:
        progress_callback(0, total)
    client.upload_file(str(source), bucket, object_key, Callback=callback)
    if progress_callback:
        progress_callback(total, total)
    return {
        "provider": "aws",
        "bucket": bucket,
        "path": object_key,
        "url": f"s3://{bucket}/{object_key}",
    }


def s3_download(destination, *, bucket: str, object_key: str, credentials: dict, progress_callback=None) -> dict:
    if not bucket or not object_key:
        raise ValueError("AWS Bucket and Object Key are required.")
    client = s3_client(credentials)
    try:
        total = int((client.head_object(Bucket=bucket, Key=object_key) or {}).get("ContentLength") or 0)
    except Exception:
        total = 0
    lock = threading.Lock()
    done = 0
    def callback(amount):
        nonlocal done
        with lock:
            done += int(amount or 0)
            if progress_callback:
                progress_callback(done, total or None)
    if progress_callback:
        progress_callback(0, total or None)
    client.download_file(bucket, object_key, str(destination), Callback=callback)
    if progress_callback and total:
        progress_callback(total, total)
    return {
        "provider": "aws",
        "bucket": bucket,
        "path": object_key,
        "url": f"s3://{bucket}/{object_key}",
    }


def _gcs_client(credentials: dict):
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
    except ImportError as exc:
        raise CloudProviderError(
            "Google Cloud Storage support needs google-cloud-storage and google-auth. "
            "Install: pip install google-cloud-storage google-auth"
        ) from exc

    raw = (credentials.get("service_account_json") or "").strip()
    if raw:
        try:
            info = json.loads(raw)
        except Exception as exc:
            raise CloudProviderError("Google Service Account JSON is invalid JSON.") from exc
        creds = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=info.get("project_id"), credentials=creds)
    return storage.Client()


def gcs_status(credentials: dict) -> dict:
    try:
        client = _gcs_client(credentials)
        return {
            "ok": True,
            "project": client.project or "default",
            "message": f"Google Cloud Storage client ready for project {client.project or 'default'}."
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def gcs_upload(local_path, *, bucket: str, object_name: str, credentials: dict, progress_callback=None) -> dict:
    if not bucket or not object_name:
        raise ValueError("GCS Bucket and Object Name are required.")
    client = _gcs_client(credentials)
    blob = client.bucket(bucket).blob(object_name)
    total = Path(local_path).stat().st_size
    if progress_callback:
        progress_callback(0, total)
    blob.upload_from_filename(str(local_path))
    if progress_callback:
        progress_callback(total, total)
    return {
        "provider": "gcp",
        "bucket": bucket,
        "path": object_name,
        "url": f"gs://{bucket}/{object_name}",
    }


def gcs_download(destination, *, bucket: str, object_name: str, credentials: dict, progress_callback=None) -> dict:
    if not bucket or not object_name:
        raise ValueError("GCS Bucket and Object Name are required.")
    client = _gcs_client(credentials)
    blob = client.bucket(bucket).blob(object_name)
    try:
        blob.reload()
        total = int(blob.size or 0)
    except Exception:
        total = 0
    if progress_callback:
        progress_callback(0, total or None)
    blob.download_to_filename(str(destination))
    if progress_callback:
        final_size = total or (Path(destination).stat().st_size if Path(destination).exists() else 0)
        progress_callback(final_size, final_size or None)
    return {
        "provider": "gcp",
        "bucket": bucket,
        "path": object_name,
        "url": f"gs://{bucket}/{object_name}",
    }


def _azure_service(credentials: dict):
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise CloudProviderError(
            "Azure Blob support needs azure-storage-blob. "
            "Install: pip install azure-storage-blob"
        ) from exc
    connection_string = (credentials.get("connection_string") or "").strip()
    if not connection_string:
        raise CloudProviderError("Azure Connection String is required.")
    return BlobServiceClient.from_connection_string(connection_string)


def azure_status(credentials: dict) -> dict:
    try:
        service = _azure_service(credentials)
        # Construction validates the connection string without exposing it.
        return {
            "ok": True,
            "account": service.account_name,
            "message": f"Azure Blob client ready for account {service.account_name}.",
        }
    except Exception as exc:
        return {"ok": False, "message": str(exc)}


def azure_upload(local_path, *, container: str, blob_name: str, credentials: dict, progress_callback=None) -> dict:
    if not container or not blob_name:
        raise ValueError("Azure Container and Blob Name are required.")
    service = _azure_service(credentials)
    client = service.get_blob_client(container=container, blob=blob_name)
    total = Path(local_path).stat().st_size
    if progress_callback:
        progress_callback(0, total)
    def hook(current, total_bytes):
        if progress_callback:
            progress_callback(int(current or 0), int(total_bytes or total or 0) or None)
    with open(local_path, "rb") as handle:
        try:
            client.upload_blob(handle, overwrite=True, progress_hook=hook)
        except TypeError:
            client.upload_blob(handle, overwrite=True)
    if progress_callback:
        progress_callback(total, total)
    return {
        "provider": "azure",
        "container": container,
        "path": blob_name,
        "url": client.url,
    }


def azure_download(destination, *, container: str, blob_name: str, credentials: dict, progress_callback=None) -> dict:
    if not container or not blob_name:
        raise ValueError("Azure Container and Blob Name are required.")
    service = _azure_service(credentials)
    client = service.get_blob_client(container=container, blob=blob_name)
    def hook(current, total_bytes):
        if progress_callback:
            progress_callback(int(current or 0), int(total_bytes or 0) or None)
    try:
        downloader = client.download_blob(progress_hook=hook)
    except TypeError:
        downloader = client.download_blob()
    if progress_callback:
        progress_callback(0, int(getattr(getattr(downloader, "properties", None), "size", 0) or 0) or None)
    data = downloader.readall()
    Path(destination).write_bytes(data)
    if progress_callback:
        progress_callback(len(data), len(data))
    return {
        "provider": "azure",
        "container": container,
        "path": blob_name,
        "url": client.url,
    }
