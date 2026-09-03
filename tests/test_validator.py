"""Tests for the validator agent's JSON extraction and save logic."""

from pathlib import Path
from unittest.mock import patch

from blogboard.agents.validator_agent.agent import _extract_json, validator_node


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"approved": true}') == {"approved": True}

    def test_fenced_json(self):
        raw = '```json\n{"approved": false}\n```'
        assert _extract_json(raw) == {"approved": False}

    def test_json_with_prose_around(self):
        raw = 'Here you go:\n{"approved": true, "title": "T"}\nHope that helps!'
        assert _extract_json(raw)["approved"] is True

    def test_invalid_returns_empty(self):
        assert _extract_json("total garbage") == {}

    def test_fenced_invalid_returns_empty(self):
        assert _extract_json("```json\nbroken\n```") == {}


class TestValidatorDryRun:
    def test_dry_run_short_circuits(self):
        state = {"dry_run": True, "topic": "t", "content": "c", "domain": "ml"}
        result = validator_node(state)
        assert result["revision_needed"] is False
        assert result["slug"] == "dry-run-generated"


class TestValidatorSaveFlow:
    def _run_approved(self, tmp_path: Path, state: dict, llm_payload: str):
        """Patch the LLM + storage, invoke validator_node, return storage calls."""
        from blogboard.services.local_storage import LocalStorageService

        storage = LocalStorageService(root=str(tmp_path))

        class FakeRes:
            content = llm_payload

        fake_llm = type(
            "FakeLLM", (), {"llm": type("I", (), {"invoke": staticmethod(lambda p: FakeRes())})}
        )()
        with (
            patch("blogboard.agents.validator_agent.agent.LLMAgentService", return_value=fake_llm),
            patch("blogboard.agents.validator_agent.agent.get_storage", return_value=storage),
            patch("blogboard.agents.validator_agent.agent.fetch_cover_image", return_value=None),
        ):
            return validator_node(state), storage

    def test_approved_article_saved_and_registered(self, tmp_path: Path):
        state = {
            "topic": "Transformers",
            "content": "Long enough content.",
            "domain": "nlp",
            "date": "2026-09-01",
            "read_time": "4 min",
        }
        payload = '{"approved": true, "title": "T", "description": "D", "slug": "transformers"}'
        result, storage = self._run_approved(tmp_path, state, payload)

        assert result["revision_needed"] is False
        assert result["md_path"] == "blogs/nlp/transformers.md"
        assert storage.get_object("blogs/nlp/transformers.md") == "Long enough content."
        registered = storage.get_articles_json("nlp")
        assert len(registered) == 1 and registered[0]["title"] == "T"

    def test_empty_slug_gets_fallback(self, tmp_path: Path):
        """LLM returning 'slug': '' must NOT produce 'blogs/nlp/.md'."""
        state = {
            "topic": "My Cool Topic",
            "content": "content",
            "domain": "nlp",
            "date": "2026-09-01",
        }
        payload = '{"approved": true, "title": "My Cool Topic", "description": "", "slug": ""}'
        result, storage = self._run_approved(tmp_path, state, payload)
        assert result["slug"] != ""
        assert ".md" not in result["slug"]
        # The saved file is reachable
        assert storage.get_object(f"blogs/nlp/{result['slug']}.md") == "content"

    def test_upload_failure_does_not_register(self, tmp_path: Path):
        from blogboard.services.local_storage import LocalStorageService

        storage = LocalStorageService(root=str(tmp_path))

        class FailingStorage(LocalStorageService):
            def put_object(self, key, data, content_type="text/plain"):
                print(f"  [TEST] Simulating upload failure for {key}")
                return False

        class FakeRes:
            content = '{"approved": true, "title": "T", "description": "D", "slug": "s"}'

        fake_llm = type(
            "FakeLLM", (), {"llm": type("I", (), {"invoke": staticmethod(lambda p: FakeRes())})}
        )()
        state = {"topic": "T", "content": "c", "domain": "ml", "date": "2026-09-01"}
        with (
            patch("blogboard.agents.validator_agent.agent.LLMAgentService", return_value=fake_llm),
            patch(
                "blogboard.agents.validator_agent.agent.get_storage",
                return_value=FailingStorage(root=str(tmp_path)),
            ),
            patch("blogboard.agents.validator_agent.agent.fetch_cover_image", return_value=None),
        ):
            result = validator_node(state)

        # Registry must NOT contain a dead entry when upload failed
        assert result["revision_needed"] is True
        assert storage.get_articles_json("ml") == []

    def test_rejected_draft_loops_back(self, tmp_path: Path):
        state = {
            "topic": "T",
            "content": "c",
            "domain": "ml",
            "date": "2026-09-01",
            "revision_count": 0,
        }
        payload = '{"approved": false, "feedback": "Too shallow", "title": "", "slug": ""}'
        result, _ = self._run_approved(tmp_path, state, payload)
        assert result["revision_needed"] is True
        assert result["validator_feedback"] == "Too shallow"
        assert result["revision_count"] == 1

    def test_max_revisions_forces_approval(self, tmp_path: Path):
        state = {
            "topic": "T",
            "content": "c",
            "domain": "ml",
            "date": "2026-09-01",
            "revision_count": 3,
        }
        payload = '{"approved": false, "feedback": "Still bad", "title": "T", "slug": "t"}'
        result, _ = self._run_approved(tmp_path, state, payload)
        # Forced approval → proceeds to save
        assert result["revision_needed"] is False
