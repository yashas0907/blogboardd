"""BlogBoard Admin API — health, stats, article listing, generation trigger."""

from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from blogboard.config.settings import app_settings
from blogboard.graph.graph import graph
from blogboard.services.storage import get_storage

app = FastAPI(
    title="BlogBoard Admin API",
    version="0.1.0",
    description="Inspect and trigger the BlogBoard multi-agent article pipeline.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_token(x_admin_token: str | None = Header(default=None)) -> None:
    """Auth dependency: enforces the token only when one is configured."""
    expected = app_settings.admin_token
    if expected and x_admin_token != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Token header")


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "llm_configured": app_settings.is_llm_configured(),
        "storage_backend": (app_settings.storage_backend or "r2").lower(),
        "r2_configured": app_settings.is_r2_configured(),
    }


@app.get("/api/stats")
def stats() -> dict[str, Any]:
    storage = get_storage()
    all_articles = storage.get_all_articles()
    by_domain: dict[str, int] = {}
    for a in all_articles:
        by_domain[a.get("category", "?")] = by_domain.get(a.get("category", "?"), 0) + 1
    return {
        "total_articles": len(all_articles),
        "by_domain": by_domain,
        "latest_date": all_articles[0]["date"] if all_articles else None,
    }


@app.get("/api/articles")
def articles(domain: str | None = None, limit: int = 50) -> dict[str, Any]:
    storage = get_storage()
    all_articles = storage.get_all_articles()
    if domain:
        all_articles = [a for a in all_articles if a.get("category") == domain]
    return {"count": len(all_articles[:limit]), "articles": all_articles[:limit]}


@app.post("/api/generate")
def generate(
    domain: str | None = None,
    dry_run: bool = False,
    _token: None = Depends(require_token),
) -> dict[str, Any]:
    """Trigger the generation pipeline. Protected by X-Admin-Token when configured."""
    initial_state: dict[str, Any] = {"dry_run": dry_run}
    if domain:
        initial_state["domain"] = domain

    config = {"configurable": {"thread_id": "blogboard-api"}}
    final_state = graph.invoke(initial_state, config=config)
    return {
        "title": final_state.get("title"),
        "domain": final_state.get("domain"),
        "md_path": final_state.get("md_path"),
        "read_time": final_state.get("read_time"),
    }
