from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step10o_file_actions_live_on_builder_toolbar_not_gallery_header():
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'btn("⇧ Load","mlb-tool mlb-builder-file-action")' in text
        assert 'btn("⇩ Export","mlb-tool mlb-builder-file-action")' in text
        assert 'btn("Bundle","mlb-tool mlb-builder-file-action")' in text
        assert 'btn(saveLabel,"mlb-tool mlb-builder-save-action")' in text
        assert 'headTools.appendChild(close)' in text
        assert 'galleryActions.className="mlb-gallery-page-actions"' not in text
        assert 'btn("⇧ Load","mlb-gallery-action mlb-gallery-file-action")' not in text
        assert 'btn("⇩ Export","mlb-gallery-action mlb-gallery-file-action")' not in text


def test_step10o_save_button_remains_contextual_to_active_builder():
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'const saveLabel="Save";' in text
        assert 'state.active_workspace==="data"?"Save the current Data Builder pipeline to My Data":"Save the current Model Builder graph to My Models"' in text
        assert 'saveAction.addEventListener("click",saveCurrentToGallery)' in text


def test_step10o_builder_save_action_has_toolbar_styling():
    for rel in ("frontend/src/builder.css", "src/mlb_studio/static/builder.css"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert '.mlb-toolbar>.mlb-builder-save-action{' in text
