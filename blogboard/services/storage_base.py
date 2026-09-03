from abc import ABC, abstractmethod
from typing import Any


class StorageService(ABC):
    """Interface for all storage backends (R2, local filesystem, ...)."""

    @abstractmethod
    def get_object(self, key: str) -> str | None:
        """Fetch raw string data. Returns None if the key does not exist."""

    @abstractmethod
    def put_object(self, key: str, data: str, content_type: str = "text/plain") -> bool:
        """Upload string data. Returns True on success."""

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[str]:
        """List keys under a prefix."""

    # ── Shared JSON / registry helpers ──────────────────────────────────────

    def get_json(self, key: str) -> list[dict[str, Any]] | None:
        data = self.get_object(key)
        if data:
            try:
                parsed = json.loads(data)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                print(f"[WARN] Failed to decode JSON from {key}. Starting fresh.")
        return []

    def get_articles_json(self, domain: str) -> list[dict[str, Any]]:
        return self.get_json(f"blogs/{domain}/articles.json")

    def save_articles_json(self, domain: str, articles: list[dict[str, Any]]) -> bool:
        json_str = json.dumps(articles, indent=2, ensure_ascii=False)
        return self.put_object(
            f"blogs/{domain}/articles.json", json_str, content_type="application/json"
        )

    def get_recent_history(self, domain: str, limit: int = 3) -> list[dict[str, Any]]:
        articles = self.get_articles_json(domain)
        sorted_articles = sorted(articles, key=lambda x: x.get("date", ""), reverse=True)
        recent = sorted_articles[:limit]
        return [
            {
                "title": a.get("title"),
                "topic": a.get("topic"),
                "subtopics": a.get("subtopics", ""),
            }
            for a in recent
        ]

    def get_all_domains_last_updated(self) -> dict[str, str]:
        from blogboard.config.settings import app_settings

        latest_dates: dict[str, str] = {}
        for domain_slug in app_settings.tags.model_dump().keys():
            articles = self.get_articles_json(domain_slug)
            if not articles:
                latest_dates[domain_slug] = "Never"
            else:
                sorted_articles = sorted(articles, key=lambda x: x.get("date", ""), reverse=True)
                latest_dates[domain_slug] = sorted_articles[0].get("date", "Unknown")
        return latest_dates

    def get_all_articles(self) -> list[dict[str, Any]]:
        """Every published article across every domain, newest first."""
        from blogboard.config.settings import app_settings

        all_articles: list[dict[str, Any]] = []
        for domain_slug in app_settings.tags.model_dump().keys():
            all_articles.extend(self.get_articles_json(domain_slug))
        all_articles.sort(key=lambda x: x.get("date", ""), reverse=True)
        return all_articles


import json
