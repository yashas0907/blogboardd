import json
import re
from datetime import datetime
from blogboard.graph.state import BlogState
from blogboard.services.llm import LLMAgentService
from blogboard.services.storage import get_storage
from blogboard.services.prompt_manager import prompt_manager
from blogboard.services import site_services
from blogboard.tools.unsplash_search import fetch_cover_image
from .prompts import VALIDATOR_PROMPT

MAX_REVISIONS = 3


def _extract_json(raw: str) -> dict:
    """Tolerant JSON extraction: strips code fences and finds the JSON object."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        # Fall back to grabbing the outermost {...} block
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def validator_node(state: BlogState) -> BlogState:
    print("  => [ValidatorAgent] Running...")

    if state.get("dry_run"):
        print("  [DRY RUN] Simulating Approval and Metadata Gen.")
        return {
            **state,
            "revision_needed": False,
            "title": "Dry Run Generated Title",
            "slug": "dry-run-generated",
            "md_path": "local://dry-run",
            "description": "Dry run generic description.",
        }

    current_revision = state.get("revision_count", 0)
    topic = state.get("topic")
    content = state.get("content", "")
    domain = state.get("domain")
    date = state.get("date", datetime.now().strftime("%Y-%m-%d"))

    prompt = prompt_manager.get_prompt(
        prompt_name="Validator_Prompt",
        fallback_prompt=VALIDATOR_PROMPT,
        topic=topic,
        content=content
    )

    llm_service = LLMAgentService(temperature=0.1)
    res = llm_service.llm.invoke(prompt)

    data = _extract_json(res.content)
    approved = data.get("approved", True)
    feedback = data.get("feedback", "")

    # Robust fallbacks: an LLM may return "key": "" (empty string) which must
    # NOT be treated as a valid value — default only applies to falsy values.
    title = data.get("title") or (topic[:70] if topic else "Untitled Draft")
    description = data.get("description") or (f"A blog post about {title}")
    slug_value = data.get("slug") or title.lower().replace(" ", "-")

    if not data:
        print("  [WARN] Validator failed to return JSON. Forcing approval fallback.")
        approved = True
        feedback = ""

    if not isinstance(approved, bool):
        approved = str(approved).lower() in ("true", "yes", "1")

    if not approved and current_revision >= MAX_REVISIONS:
        print("  [WARN] Max revisions reached. Forcing approval.")
        approved = True

    revision_needed = not approved

    if revision_needed:
        print(f"  [AGENT] Draft REJECTED. Feedback: {feedback}")
        return {
            **state,
            "revision_needed": True,
            "validator_feedback": feedback,
            "revision_count": current_revision + 1
        }

    print("  [AGENT] Draft APPROVED! Generating Metadata and Saving...")

    # ── Persist: upload first, then update the registry only on success ──
    slug_value = re.sub(r"[^\w\s-]", "", slug_value).strip().replace(" ", "-")
    if not slug_value:  # last-resort guard against empty slugs
        slug_value = f"article-{date}"

    md_relative = f"blogs/{domain}/{slug_value}.md"
    storage = get_storage()

    if not storage.put_object(md_relative, content, content_type="text/markdown"):
        print("  [ERROR] Article upload failed — aborting save (registry NOT updated).")
        return {
            **state,
            "revision_needed": True,
            "validator_feedback": "Internal storage error while saving — regenerate the article.",
            "revision_count": current_revision + 1,
        }

    articles = storage.get_articles_json(domain)
    articles = [a for a in articles if a.get("id") != md_relative]

    cover_image = None
    try:
        cover_image = fetch_cover_image(topic or title, domain or "technology")
    except Exception as e:
        print(f"  [WARN] Cover image fetch failed (continuing without): {e}")

    articles.append({
        "id": md_relative,
        "category": domain,
        "topic": topic,
        "subtopics": state.get("subtopics", ""),
        "title": title,
        "description": description,
        "date": date,
        "tags": [domain],
        "readTime": state.get("read_time", "5 min"),
        "file": md_relative,
        "coverImage": cover_image or "",
    })

    articles = sorted(articles, key=lambda x: x["date"], reverse=True)
    storage.save_articles_json(domain, articles)

    # ── Site-wide artifacts: RSS + sitemap + baked site data (best-effort) ──
    try:
        all_articles = storage.get_all_articles()
        storage.put_object("rss.xml", site_services.generate_rss(all_articles), "application/rss+xml")
        storage.put_object("sitemap.xml", site_services.generate_sitemap(all_articles), "application/xml")
    except Exception as e:
        print(f"  [WARN] RSS/sitemap regeneration failed (non-fatal): {e}")

    try:
        if (storage.__class__.__name__ == "LocalStorageService"):
            import importlib
            import scripts.build_site as build_site
            importlib.reload(build_site).build_all()
        else:  # R2: bake locally anyway so the repo copy stays fresh
            import scripts.build_site as build_site
            build_site.build_all()
    except Exception as e:
        print(f"  [WARN] site-data.js rebuild failed (non-fatal): {e}")

    return {
        **state,
        "revision_needed": False,
        "title": title,
        "description": description,
        "slug": slug_value,
        "md_path": md_relative,
    }
