# BlogBoard — Autonomous Multi-Agent AI Blog Generator

> **Self-writing blog platform.** A LangGraph multi-agent pipeline researches, drafts,
> reviews, revises, illustrates, and publishes deep-dive AI/ML articles — automatically,
> on a schedule, with zero human intervention.

[![CI](https://github.com/KalyanM45/Multi-Agentic-Blog-Generation/actions/workflows/ci.yml/badge.svg)](https://github.com/KalyanM45/Multi-Agentic-Blog-Generation/actions/workflows/ci.yml)
[![Generate](https://github.com/KalyanM45/Multi-Agentic-Blog-Generation/actions/workflows/generate.yml/badge.svg)](https://github.com/KalyanM45/Multi-Agentic-Blog-Generation/actions/workflows/generate.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## What it does

1. **Scheduler** picks the least-recently-updated domain
2. **TutorialAgent / NewsAgent** selects a novel topic (using published-article history
   to avoid repeats), researches live news via Tavily + The Guardian APIs, and drafts a
   Markdown article
3. **ValidatorAgent** reviews the draft like a strict editor — rejects shallow drafts with
   actionable feedback and loops the writer back (max 3 revisions)
4. On approval it **publishes**: article + registry + **cover image from Unsplash** +
   **RSS feed** + **sitemap.xml** — atomically (registry is never updated if an upload fails)
5. The **static frontend** renders everything client-side: markdown, syntax highlighting,
   table of contents, reading progress, related posts, comments, dark/light themes

## Quick start (60 seconds, zero API keys)

```bash
git clone https://github.com/KalyanM45/Multi-Agentic-Blog-Generation.git
cd Multi-Agentic-Blog-Generation
uv venv && uv sync

# Preview the whole pipeline without any keys or file writes:
uv run python -m blogboard.run --dry-run

# Serve the frontend (uses bundled sample articles):
uv run python -m http.server 8000 --directory blogboard/web
# → http://localhost:8000
```

The default storage backend is **local** (`blogboard/web/blogs/`) — no cloud account needed.

## Architecture

```
                    ┌────────────────────┐
   domain=ainews ──►│     NewsAgent      │──┐        Research: Tavily + Guardian
                    └────────────────────┘  │              (ReAct tools)
                    ┌────────────────────┐  ├──► Draft
   domain=anything ─►│  TutorialAgent    │──┤        Topic history from storage
                    └────────────────────┘  │              (avoids repeats)
                                            ▼
                    ┌────────────────────┐
                    │   ValidatorAgent   │  reject + feedback (≤ 3 revisions)
                    │  (editor + SEO)   │──────► loops back to the writer
                    └─────────┬──────────┘
                              │ approve
                              ▼
        article.md + articles.json + cover image + rss.xml + sitemap.xml
```

| Component | Role |
|---|---|
| `blogboard/graph/` | LangGraph state machine + routing/revision logic |
| `blogboard/agents/` | TutorialAgent, NewsAgent, ValidatorAgent (+ prompts) |
| `blogboard/services/llm.py` | Groq ChatGroq wrapper + ReAct agent factory |
| `blogboard/services/storage.py` | Storage factory: **R2** (S3 API) or **local** filesystem |
| `blogboard/services/site_services.py` | RSS 2.0 + sitemap.xml generation |
| `blogboard/tools/` | Tavily, Guardian, Unsplash API tools |
| `blogboard/api/app.py` | FastAPI admin API (health/stats/articles/trigger) |
| `blogboard/web/` | Static frontend (vanilla JS, no build step) |
| `.github/workflows/` | CI (tests) + daily 8AM-IST auto-publishing |

## Configuration

Copy `.env.example` → `.env`:

```dotenv
LLM__API_KEY="gsk_..."          # required for real generation (console.groq.com)
STORAGE_BACKEND="local"         # "local" or "r2"
R2__ACCOUNT_ID=""               # only needed when STORAGE_BACKEND=r2
R2__ACCESS_KEY_ID=""
R2__SECRET_ACCESS_KEY=""
R2__BUCKET_NAME=""
CONTENT__TAVILY_API_KEY=""      # optional — news research
CONTENT__GUARDIAN_API_KEY=""    # optional — news research
CONTENT__UNSPLASH_API_KEY=""    # optional — cover images
SITE_URL="https://your-site"    # used in RSS/sitemap links
ADMIN_TOKEN=""                  # optional — protects POST /api/generate
```

Everything except `LLM__API_KEY` is optional; missing pieces degrade gracefully
(no key → that feature is skipped, pipeline never crashes).

## Usage

```bash
# Daily scheduled article (auto-picks domain):
uv run python -m blogboard.run

# Force a domain:
uv run python -m blogboard.run --domain nlp

# Weekly news roundup:
uv run python -m blogboard.run --ainews

# Preview without keys/writes:
uv run python -m blogboard.run --dry-run

# Admin API:
uv run uvicorn blogboard.api.app:app --port 8001
#   GET  /api/health      GET /api/stats      GET /api/articles
#   POST /api/generate?domain=ml&dry_run=false   (X-Admin-Token header)

# Tests:
uv run pytest tests/ -v
```

## Publishing schedule (GitHub Actions)

`generate.yml` runs daily at 08:00 IST and maps weekday → domain:

Mon **ML** · Tue **DL** · Wed **Stats** · Thu **NLP** · Fri **CV** · Sat **GenAI** · Sun **AI News**

Add these repo secrets: `LLM_API_KEY`, `TAVILY_API_KEY`, `GUARDIAN_API_KEY`,
`UNSPLASH_API_KEY`, `SITE_URL` (+ R2 credentials if using R2). Generated articles
are committed automatically by the workflow.

## Testing

30 tests cover: storage backends (incl. path-traversal security), validator JSON
parsing + save flow (incl. empty-slug + failed-upload edge cases), graph routing,
revision loops, RSS/sitemap generation, and the admin API.

```bash
uv run pytest tests/ -v
```

## Contributing

1. Fork → branch (`feat/your-idea`) → commit → PR
2. Run `uv run pytest tests/` before submitting
3. Bug reports and feature ideas → issues welcome

## License

MIT — see [LICENSE](LICENSE).
