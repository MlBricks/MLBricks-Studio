from mlb_studio import hub


def test_resolve_repo_id_uses_canonical_org_casing(monkeypatch):
    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self, token=None):
            return {
                "name": "demo-user",
                "orgs": [{"name": "MLBricks"}, {"name": "OtherOrg"}],
            }

    monkeypatch.setattr(hub, "hub_token", lambda required, token=None: token or "hf_test")
    monkeypatch.setattr(hub, "_hub", lambda: (FakeApi, lambda: None, None, None))

    assert hub.resolve_repo_id("mlbricks/demoModel", token="hf_test") == "MLBricks/demoModel"
    assert hub.resolve_repo_id("DEMO-USER/repo", token="hf_test") == "demo-user/repo"
    assert hub.resolve_repo_id("unknown/repo", token="hf_test") == "unknown/repo"
