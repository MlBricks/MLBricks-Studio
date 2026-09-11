from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step10n_gallery_has_one_flat_navigation_row():
    expected = '[["core","Core"],["models","Models"],["mine","My Models"],["components","Components"],["data","Data"],["drafts","Drafts"]]'
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert expected in text
        assert 'modelNav.className="mlb-model-gallery-nav"' not in text
        assert 'groupBar.className="mlb-model-group-tabs"' not in text
        assert 'navTools.className="mlb-gallery-flat-nav-tools"' in text
        assert '["core","models","mine"].includes(galleryWorkspace.tab)' in text


def test_step10n_gallery_actions_move_into_header_and_filters_stay_contextual():
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'headTools.className="mlb-gallery-head-tools"' in text
        assert 'headTools.append(galleryActions,close)' in text
        assert 'modelNavTools=navTools' in text
        assert 'openGallery(state.active_workspace==="data"?"data":"core")' in text


def test_step10n_css_flattens_gallery_hierarchy():
    for rel in ("frontend/src/builder.css", "src/mlb_studio/static/builder.css"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert '.mlb-gallery-flat-tabs{' in text
        assert '.mlb-gallery-flat-nav-tools{' in text
        assert '.mlb-model-gallery-nav,.mlb-model-group-tabs{display:none!important}' in text
