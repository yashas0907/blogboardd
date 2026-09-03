"""Tests for local + R2 storage backends and the storage factory."""

from pathlib import Path

from blogboard.services.local_storage import LocalStorageService


def test_local_put_get_roundtrip(tmp_path: Path):
    s = LocalStorageService(root=str(tmp_path))
    assert s.put_object("blogs/ml/hello.md", "# Hello", "text/markdown") is True
    assert s.get_object("blogs/ml/hello.md") == "# Hello"
    assert s.get_object("blogs/ml/missing.md") is None


def test_local_path_traversal_blocked(tmp_path: Path):
    s = LocalStorageService(root=str(tmp_path))
    # ".." segments must be neutralised, never escape the root
    s.put_object("blogs/../../evil.md", "nope")
    assert (tmp_path.parent.parent / "evil.md").exists() is False
    # The file lands inside the root instead
    assert s.get_object("blogs/evil.md") == "nope"


def test_local_registry_roundtrip(tmp_path: Path):
    s = LocalStorageService(root=str(tmp_path))
    articles = [{"id": "blogs/ml/a.md", "title": "A", "date": "2026-01-01"}]
    assert s.save_articles_json("ml", articles) is True
    loaded = s.get_articles_json("ml")
    assert loaded == articles


def test_local_invalid_json_returns_empty(tmp_path: Path):
    s = LocalStorageService(root=str(tmp_path))
    s.put_object("blogs/ml/articles.json", "not json {{{")
    assert s.get_articles_json("ml") == []


def test_local_list_objects(tmp_path: Path):
    s = LocalStorageService(root=str(tmp_path))
    s.put_object("blogs/ml/a.md", "a")
    s.put_object("blogs/ml/b.md", "b")
    s.put_object("blogs/dl/c.md", "c")
    keys = s.list_objects("blogs/ml")
    assert sorted(keys) == ["blogs/ml/a.md", "blogs/ml/b.md"]
    assert len(s.list_objects()) == 3


def test_get_recent_history_sorted(tmp_path: Path):
    s = LocalStorageService(root=str(tmp_path))
    s.save_articles_json(
        "ml",
        [
            {"id": "1", "title": "old", "topic": "t1", "date": "2026-01-01"},
            {"id": "2", "title": "new", "topic": "t2", "date": "2026-02-01"},
            {"id": "3", "title": "mid", "topic": "t3", "date": "2026-01-15"},
        ],
    )
    history = s.get_recent_history("ml", limit=2)
    assert [h["title"] for h in history] == ["new", "mid"]


def test_get_all_domains_last_updated(tmp_path: Path):
    s = LocalStorageService(root=str(tmp_path))
    s.save_articles_json("ml", [{"date": "2026-03-01"}])
    s.save_articles_json("nlp", [{"date": "2026-01-01"}, {"date": "2026-05-01"}])
    latest = s.get_all_domains_last_updated()
    assert latest["ml"] == "2026-03-01"
    assert latest["nlp"] == "2026-05-01"
    assert latest.get("cv") == "Never"


def test_factory_returns_local_when_configured(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    # Reimport settings machinery with a fresh env
    import importlib

    import blogboard.config.settings as settings_mod

    importlib.reload(settings_mod)
    import blogboard.services.storage as storage_mod

    importlib.reload(storage_mod)
    # local_storage reads app_settings.local_storage_root at init; point it at tmp
    monkeypatch.setattr(settings_mod.app_settings, "local_storage_root", str(tmp_path))
    storage = storage_mod.get_storage()
    assert isinstance(storage, LocalStorageService)


def test_factory_rejects_unknown_backend(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("blogboard.services.storage.app_settings.storage_backend", "ftp")
    from blogboard.services.storage import get_storage

    try:
        get_storage()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "STORAGE_BACKEND" in str(e)
