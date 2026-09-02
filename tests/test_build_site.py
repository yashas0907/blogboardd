"""Tests for the self-healing site builder (scripts/build_site.py)."""
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import build_site  # noqa: E402


@pytest.fixture
def web_sandbox(tmp_path: Path, monkeypatch):
    """Point build_site at a temp 'web' dir with a controlled article layout."""
    (tmp_path / "blogs" / "ml").mkdir(parents=True)
    (tmp_path / "js").mkdir(parents=True)
    monkeypatch.setattr(build_site, "WEB", tmp_path)
    return tmp_path


def _write_md(web: Path, rel: str, content: str):
    p = web / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


class TestSelfHeal:
    def test_creates_missing_registry(self, web_sandbox):
        _write_md(web_sandbox, "blogs/ml/a.md", "# Title A\n\nSome intro text.")
        build_site.self_heal_registries()
        reg = json.loads((web_sandbox / "blogs/ml/articles.json").read_text(encoding="utf-8"))
        assert len(reg) == 1
        assert reg[0]["title"] == "Title A"
        assert reg[0]["file"] == "blogs/ml/a.md"
        assert reg[0]["category"] == "ml"

    def test_rebuilds_corrupt_registry(self, web_sandbox):
        (web_sandbox / "blogs/ml/articles.json").write_text("not json {{{", encoding="utf-8")
        _write_md(web_sandbox, "blogs/ml/a.md", "# Fixed\n\nContent.")
        build_site.self_heal_registries()
        reg = json.loads((web_sandbox / "blogs/ml/articles.json").read_text(encoding="utf-8"))
        assert len(reg) == 1 and reg[0]["title"] == "Fixed"

    def test_registers_orphan_md(self, web_sandbox):
        good = [{"id": "blogs/ml/kept.md", "file": "blogs/ml/kept.md",
                 "title": "Kept", "category": "ml", "date": "2026-01-01"}]
        (web_sandbox / "blogs/ml/articles.json").write_text(
            json.dumps(good), encoding="utf-8")
        _write_md(web_sandbox, "blogs/ml/kept.md", "# Kept")
        _write_md(web_sandbox, "blogs/ml/orphan.md", "# Orphan Article\n\nOrphan body.")
        healed = build_site.self_heal_registries()
        assert "blogs/ml/orphan.md" in healed
        reg = json.loads((web_sandbox / "blogs/ml/articles.json").read_text(encoding="utf-8"))
        assert len(reg) == 2

    def test_purges_dead_entries(self, web_sandbox):
        stale = [
            {"id": "blogs/ml/alive.md", "file": "blogs/ml/alive.md",
             "title": "Alive", "category": "ml", "date": "2026-01-01"},
            {"id": "blogs/ml/gone.md", "file": "blogs/ml/gone.md",
             "title": "Gone", "category": "ml", "date": "2026-01-01"},
        ]
        (web_sandbox / "blogs/ml/articles.json").write_text(
            json.dumps(stale), encoding="utf-8")
        _write_md(web_sandbox, "blogs/ml/alive.md", "# Alive")
        build_site.self_heal_registries()
        reg = json.loads((web_sandbox / "blogs/ml/articles.json").read_text(encoding="utf-8"))
        assert [a["title"] for a in reg] == ["Alive"]

    def test_derives_read_time_and_description(self, web_sandbox):
        words = " ".join(["word"] * 450)  # 450 words → ceil(450/200) = 3 min
        _write_md(web_sandbox, "blogs/ml/big.md", f"# Big\n\n{words}")
        build_site.self_heal_registries()
        reg = json.loads((web_sandbox / "blogs/ml/articles.json").read_text(encoding="utf-8"))
        assert reg[0]["readTime"] == "3 min"
        assert reg[0]["description"].startswith("word")


class TestBuildSiteData:
    def test_bakes_all_articles_and_content(self, web_sandbox):
        _write_md(web_sandbox, "blogs/ml/a.md", "# A\n\nBody A")
        _write_md(web_sandbox, "blogs/nlp/b.md", "# B\n\nBody B")
        build_site.self_heal_registries()
        out = build_site.build_site_data()
        js = out.read_text(encoding="utf-8")
        assert "window.SITE_DATA" in js
        assert "window.SITE_CONTENT" in js
        assert "Body A" in js and "Body B" in js
        # Parse the embedded JSON back out
        import re
        m = re.search(r"window\.SITE_DATA = (\{.*?\});\n", js, re.DOTALL)
        data = json.loads(m.group(1))
        assert data["categories"]["ml"][0]["title"] == "A"
        m2 = re.search(r"window\.SITE_CONTENT = (\{.*?\});\n", js, re.DOTALL)
        content = json.loads(m2.group(1))
        assert content["blogs/ml/a.md"] == "# A\n\nBody A"

    def test_bakery_survives_unicode(self, web_sandbox):
        _write_md(web_sandbox, "blogs/ml/uni.md", "# Título — 中文 🚀\n\nCuerpo")
        build_site.self_heal_registries()
        out = build_site.build_site_data()
        js = out.read_text(encoding="utf-8")
        assert "Título" in js and "🚀" in js

    def test_missing_category_yields_empty(self, web_sandbox):
        # No files at all — build must still emit valid JS
        out = build_site.build_site_data()
        js = out.read_text(encoding="utf-8")
        import re
        m = re.search(r"window\.SITE_DATA = (\{.*?\});\n", js, re.DOTALL)
        data = json.loads(m.group(1))
        assert data["categories"]["ml"] == []
        assert data["categories"]["genai"] == []
