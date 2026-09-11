from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_step10k_uses_single_clean_model_gallery_nav_row():
    for rel in ("frontend/src/legacy-builder.js", "src/mlb_studio/static/builder.js"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert 'navTools.className="mlb-gallery-flat-nav-tools"' in text
        assert 'filterSelect.className="mlb-model-gallery-filter-select"' in text
        assert 'filterLabel.textContent="CORE AREA"' not in text
        assert 'filterLabel.textContent="MODEL FAMILY"' not in text
        assert 'visibleCore.length+" models"' in text
        assert '"All Core",\n            ""' in text
        assert '"All Models",\n            ""' in text


def test_step10k_css_removes_nested_pill_container_look():
    for rel in ("frontend/src/builder.css", "src/mlb_studio/static/builder.css"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert ".mlb-gallery-flat-tabs{" in text
        assert ".mlb-gallery-flat-nav-tools{" in text
        assert ".mlb-model-gallery-filter-select{" in text
        assert "border-bottom:2px solid transparent" in text
        assert ".mlb-gallery-flat-tabs .mlb-central-gallery-tab.active:after" in text
