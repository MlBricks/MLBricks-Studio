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


def test_github_download_directory_returns_clear_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cloud,
        "_json_request",
        lambda *args, **kwargs: [{"name": "bundle.mlbricks.zip", "type": "file"}],
    )
    with pytest.raises(cloud.CloudProviderError, match="points to a directory"):
        cloud.github_download(
            tmp_path / "bundle.zip",
            repo="maxzameer/mlbs",
            path_in_repo="studio",
            branch="main",
            token="token",
        )


def test_github_status_handles_org_list_without_list_get(monkeypatch):
    responses = iter([
        {"login": "maxzameer"},
        [{"login": "MLBricks"}, {"login": "OtherOrg"}],
    ])
    monkeypatch.setattr(cloud, "_json_request", lambda *args, **kwargs: next(responses))
    status = cloud.github_status(token="token")
    assert status["username"] == "maxzameer"
    assert status["organizations"] == ["MLBricks", "OtherOrg"]
