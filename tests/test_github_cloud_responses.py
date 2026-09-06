from pathlib import Path

import pytest

from mlb_studio import cloud


def test_normalize_github_repo_accepts_short_https_and_ssh():
    assert cloud.normalize_github_repo("maxzameer/mlbs") == "maxzameer/mlbs"
    assert cloud.normalize_github_repo("https://github.com/maxzameer/mlbs") == "maxzameer/mlbs"
    assert cloud.normalize_github_repo("https://github.com/maxzameer/mlbs.git") == "maxzameer/mlbs"
    assert cloud.normalize_github_repo("git@github.com:maxzameer/mlbs.git") == "maxzameer/mlbs"


def test_github_upload_existing_directory_appends_bundle_filename(monkeypatch, tmp_path):
    source = tmp_path / "demo.mlbricks.zip"
    source.write_bytes(b"demo")
    calls = []

    def fake_request(url, *, method="GET", token=None, payload=None):
        calls.append((url, method, payload))
        if method == "PUT":
            return {"content": {"html_url": "https://github.com/maxzameer/mlbs/blob/main/studio/demo.mlbricks.zip"}}
        if "/contents/studio/demo.mlbricks.zip" in url:
            raise cloud.CloudProviderError("GitHub HTTP 404: Not Found")
        if "/contents/studio" in url:
            return [{"name": "existing.txt", "type": "file"}]
        raise AssertionError(url)

    monkeypatch.setattr(cloud, "_json_request", fake_request)
    result = cloud.github_upload(
        source,
        repo="https://github.com/maxzameer/mlbs",
        path_in_repo="studio",
        branch="main",
        token="token",
    )

    assert result["repo"] == "maxzameer/mlbs"
    assert result["path"] == "studio/demo.mlbricks.zip"
    assert any(method == "PUT" and "/contents/studio/demo.mlbricks.zip" in url for url, method, _ in calls)


def test_github_download_directory_multiple_bundles_returns_choices(monkeypatch, tmp_path):
    def fake_request(url, *, method="GET", token=None, payload=None):
        if "/contents/studio?" in url:
            return [{"name": "a.mlbricks.zip", "type": "file"}, {"name": "b.mlbricks.zip", "type": "file"}]
        if "/git/trees/main?recursive=1" in url:
            return {
                "tree": [
                    {"path": "studio/a.mlbricks.zip", "type": "blob"},
                    {"path": "studio/b.mlbricks.zip", "type": "blob"},
                ]
            }
        raise AssertionError(url)

    monkeypatch.setattr(cloud, "_json_request", fake_request)
    with pytest.raises(cloud.CloudProviderError, match="Multiple MLBricks Studio bundles") as exc:
        cloud.github_download(
            tmp_path / "bundle.zip",
            repo="maxzameer/mlbs",
            path_in_repo="studio",
            branch="main",
            token="token",
        )
    assert "studio/a.mlbricks.zip" in str(exc.value)
    assert "studio/b.mlbricks.zip" in str(exc.value)


def test_github_status_handles_org_list_without_list_get(monkeypatch):
    responses = iter([
        {"login": "maxzameer"},
        [{"login": "MLBricks"}, {"login": "OtherOrg"}],
    ])
    monkeypatch.setattr(cloud, "_json_request", lambda *args, **kwargs: next(responses))
    status = cloud.github_status(token="token")
    assert status["username"] == "maxzameer"
    assert status["organizations"] == ["MLBricks", "OtherOrg"]


def test_github_download_blank_path_auto_detects_single_bundle(monkeypatch, tmp_path):
    calls = []

    def fake_request(url, *, method="GET", token=None, payload=None):
        calls.append(url)
        if "/git/trees/main?recursive=1" in url:
            return {
                "tree": [
                    {"path": "README.md", "type": "blob"},
                    {"path": "mlbricks/demo.mlbricks.zip", "type": "blob"},
                ]
            }
        if "/contents/mlbricks/demo.mlbricks.zip" in url:
            return {"content": "ZGVtbw==", "html_url": "https://github.com/maxzameer/mlbs/blob/main/mlbricks/demo.mlbricks.zip"}
        raise AssertionError(url)

    monkeypatch.setattr(cloud, "_json_request", fake_request)
    destination = tmp_path / "bundle.zip"
    result = cloud.github_download(
        destination,
        repo="maxzameer/mlbs",
        path_in_repo="",
        branch="main",
        token="token",
    )
    assert destination.read_bytes() == b"demo"
    assert result["path"] == "mlbricks/demo.mlbricks.zip"
    assert any("/git/trees/main?recursive=1" in url for url in calls)


def test_github_download_directory_auto_detects_single_nested_bundle(monkeypatch, tmp_path):
    def fake_request(url, *, method="GET", token=None, payload=None):
        if "/contents/studio?" in url:
            return [{"name": "nested", "type": "dir"}]
        if "/git/trees/main?recursive=1" in url:
            return {
                "tree": [
                    {"path": "studio/nested/model.mlbricks.zip", "type": "blob"},
                    {"path": "other/ignore.mlbricks.zip", "type": "blob"},
                ]
            }
        if "/contents/studio/nested/model.mlbricks.zip" in url:
            return {"content": "bW9kZWw=", "html_url": "https://github.com/maxzameer/mlbs/blob/main/studio/nested/model.mlbricks.zip"}
        raise AssertionError(url)

    monkeypatch.setattr(cloud, "_json_request", fake_request)
    destination = tmp_path / "bundle.zip"
    result = cloud.github_download(
        destination,
        repo="maxzameer/mlbs",
        path_in_repo="studio",
        branch="main",
        token="token",
    )
    assert destination.read_bytes() == b"model"
    assert result["path"] == "studio/nested/model.mlbricks.zip"


def test_github_download_blank_path_multiple_bundles_lists_choices(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cloud,
        "_json_request",
        lambda *args, **kwargs: {
            "tree": [
                {"path": "mlbricks/a.mlbricks.zip", "type": "blob"},
                {"path": "mlbricks/b.mlbricks.zip", "type": "blob"},
            ]
        },
    )
    with pytest.raises(cloud.CloudProviderError, match="Multiple MLBricks Studio bundles") as exc:
        cloud.github_download(
            tmp_path / "bundle.zip",
            repo="maxzameer/mlbs",
            path_in_repo="",
            branch="main",
            token="token",
        )
    assert "mlbricks/a.mlbricks.zip" in str(exc.value)
    assert "mlbricks/b.mlbricks.zip" in str(exc.value)
