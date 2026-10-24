"""
build_site.py — NeuraPress site builder.

Self-heals registries (registers orphan .md files), then bakes the ENTIRE
site (article metadata + full markdown content) into js/site-data.js so the
frontend works with zero fetches: file://, offline, any static host.

Run standalone:  uv run python scripts/build_site.py
Called automatically by the validator after every publish.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "neurapress" / "web"

CATEGORIES = ["ml", "dl", "statistics", "nlp", "cv", "genai", "ainews"]

LABELS = {
    "ml": "Machine Learning",
    "dl": "Deep Learning",
    "statistics": "Statistics for AI",
    "nlp": "Natural Language Processing",
    "cv": "Computer Vision",
    "genai": "Generative AI",
    "ainews": "AI News",
}

WORDS_PER_MINUTE = 200

# LLMs sometimes emit exotic Unicode that renders as '?' in some fonts.
# Normalize the offenders to plain ASCII equivalents during the site bake.
CHAR_FIXES = {
    "\u202f": " ",   # narrow no-break space
    "\u00a0": " ",   # non-breaking space
    "\u2011": "-",   # non-breaking hyphen
    "\u2010": "-",   # hyphen (unicode)
    "\u200b": "",    # zero-width space
    "\ufeff": "",    # BOM / zero-width no-break space
}


def normalize_text(text: str) -> str:
    for bad, good in CHAR_FIXES.items():
        text = text.replace(bad, good)
    return text


def _read_time(text: str) -> str:
    return f"{max(1, -(-len(text.split()) // WORDS_PER_MINUTE))} min"


def _derive_meta(md_path: Path, cat: str) -> dict:
    """Derive registry metadata for an orphan article from its content."""
    text = md_path.read_text(encoding="utf-8")
    title = None
    first_para = None
    for line in text.splitlines():
        stripped = line.strip()
        if not title and stripped.startswith("# "):
            title = stripped[2:].strip()
        elif (
            not first_para
            and stripped
            and not stripped.startswith(("#", "```", "-", "*", "|", ">", "[!", "---"))
        ):
            first_para = stripped
        if title and first_para:
            break
    title = title or md_path.stem.replace("-", " ").title()
    first_para = first_para or f"An article about {title}."
    rel = md_path.relative_to(WEB).as_posix()
    mtime = datetime.fromtimestamp(md_path.stat().st_mtime, tz=UTC)
    return {
        "id": rel,
        "category": cat,
        "topic": title,
        "subtopics": "",
        "title": title,
        "description": first_para[:200],
        "date": mtime.strftime("%Y-%m-%d"),
        "tags": [cat],
        "readTime": _read_time(text),
        "file": rel,
        "coverImage": "",
    }


def self_heal_registries() -> list:
    """Register orphan .md files into their category registries. Returns healed IDs."""
    healed = []
    for cat in CATEGORIES:
        cat_dir = WEB / "blogs" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        reg_path = cat_dir / "articles.json"

        # Load (or create) a valid registry
        articles = []
        if reg_path.exists():
            try:
                loaded = json.loads(reg_path.read_text(encoding="utf-8"))
                if isinstance(loaded, list):
                    articles = loaded
            except (json.JSONDecodeError, UnicodeDecodeError):
                print(f"  [HEAL] {cat}: corrupt registry — rebuilding")
        else:
            print(f"  [HEAL] {cat}: missing registry — creating")

        registered = {a.get("file") for a in articles}

        # Purge dead entries (file no longer exists)
        alive = []
        for a in articles:
            f = a.get("file", "")
            if (WEB / f).is_file():
                alive.append(a)
            else:
                print(f"  [HEAL] {cat}: removing dead entry {f}")
        articles = alive

        # Register orphans
        for md in sorted(cat_dir.glob("*.md")):
            rel = md.relative_to(WEB).as_posix()
            if rel not in registered:
                print(f"  [HEAL] {cat}: registering orphan {md.name}")
                articles.append(_derive_meta(md, cat))
                healed.append(rel)

        articles.sort(key=lambda x: x.get("date", ""), reverse=True)
        reg_path.write_text(json.dumps(articles, indent=2, ensure_ascii=False), encoding="utf-8")
    return healed


def build_site_data() -> Path:
    """Bake all registries + article contents into js/site-data.js."""
    categories = {}
    contents = {}

    for cat in CATEGORIES:
        reg_path = WEB / "blogs" / cat / "articles.json"
        try:
            articles = json.loads(reg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            articles = []
        categories[cat] = articles
        for a in articles:
            md_path = WEB / a.get("file", "")
            if md_path.is_file():
                try:
                    contents[a["id"]] = normalize_text(md_path.read_text(encoding="utf-8"))
                except OSError:
                    pass

    payload = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "categories": categories,
    }

    # Write as a JS file: metadata + a separate CONTENT map (markdown sources)
    def js_str(obj) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=1)

    js = (
        "// AUTO-GENERATED by scripts/build_site.py — do not edit by hand.\n"
        "// Contains every article's metadata and full content. Regenerated on each publish.\n"
        f"window.SITE_DATA = {js_str(payload)};\n"
        f"window.SITE_CONTENT = {js_str(contents)};\n"
    )

    out = WEB / "js" / "site-data.js"
    out.write_text(js, encoding="utf-8")
    total = sum(len(v) for v in categories.values())
    print(
        f"  [BUILD] site-data.js: {total} articles, {len(contents)} contents "
        f"({out.stat().st_size // 1024} KB)"
    )
    return out


def build_all() -> Path:
    print("  [BUILD] Self-healing registries...")
    self_heal_registries()
    print("  [BUILD] Baking site data...")
    out = build_site_data()
    print("  [BUILD] Refreshing RSS + sitemap...")
    _refresh_feeds()
    print("  [BUILD] Generating static article pages...")
    _generate_article_pages()
    return out


def _article_page_name(article: dict) -> str:
    """articles/{cat}-{slug}.html — flat, one directory deep."""
    slug = Path(article["file"]).stem
    return f"{article['category']}-{slug}.html"


def _generate_article_pages() -> int:
    """Write a static, crawlable HTML page per article with real <title>,
    meta description and og tags — so every article is individually
    indexable by search engines (hash-fragment URLs are ignored)."""
    template = (WEB / "article.html").read_text(encoding="utf-8")
    articles_dir = WEB / "articles"
    articles_dir.mkdir(exist_ok=True)

    valid: set[str] = set()
    count = 0
    for cat in CATEGORIES:
        reg_path = WEB / "blogs" / cat / "articles.json"
        try:
            articles = json.loads(reg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for a in articles:
            html = (
                template.replace("{{ARTICLE_TITLE}}", normalize_text(a.get("title", "Article")))
                .replace(
                    "{{ARTICLE_DESCRIPTION}}", normalize_text(a.get("description", ""))[:300]
                )
                .replace("{{ARTICLE_DATE}}", a.get("date", ""))
                .replace("{{ARTICLE_ID}}", a.get("id", ""))
            )
            page_name = _article_page_name(a)
            (articles_dir / page_name).write_text(html, encoding="utf-8")
            valid.add(page_name)
            count += 1

    # Remove stale generated pages for articles that no longer exist
    for old in articles_dir.glob("*.html"):
        if old.name not in valid:
            old.unlink()
            print(f"  [BUILD] removed stale article page {old.name}")
    return count


def _article_url(article: dict, site: str) -> str:
    return f"{site}/articles/{_article_page_name(article)}"


def _refresh_feeds() -> None:
    """Regenerate rss.xml + sitemap.xml from current registries (keeps feeds
    in sync with branding/content even without a new publish). Uses the
    crawlable static article URLs for search engines."""
    import sys

    sys.path.insert(0, str(ROOT))

    from neurapress.config.settings import app_settings
    from neurapress.services.local_storage import LocalStorageService

    site = app_settings.site_url.rstrip("/")
    # Prefer the configured site URL, falling back to the deployed Pages URL
    if not site or "localhost" in site:
        site = "https://yashas0907.github.io/blogboardd"

    storage = LocalStorageService()
    all_articles = storage.get_all_articles()

    # RSS with crawlable article links
    items = []
    from xml.sax.saxutils import escape

    for a in all_articles[:50]:
        link = f"{site}/articles/{_article_page_name(a)}"
        items.append(
            "<item>\n"
            f"  <title>{escape(str(a.get('title', '')))}</title>\n"
            f"  <link>{link}</link>\n"
            f"  <guid>{link}</guid>\n"
            f"  <pubDate>{escape(str(a.get('date', '')))}</pubDate>\n"
            f"  <description>{escape(str(a.get('description', '')))}</description>\n"
            f"  <category>{escape(str(a.get('category', '')))}</category>\n"
            "</item>"
        )
    rss = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        "<channel>\n"
        "  <title>NeuraPress</title>\n"
        f"  <link>{site}</link>\n"
        "  <description>AI & ML insights - autogenerated by NeuraPress</description>\n"
        f"  <atom:link href=\"{site}/rss.xml\" rel=\"self\" type=\"application/rss+xml\"/>\n"
        + "\n".join(items) + "\n"
        "</channel>\n"
        "</rss>"
    )
    (WEB / "rss.xml").write_text(rss, encoding="utf-8")

    # Sitemap pointing at crawlable article pages
    urls = [
        f"  <url><loc>{site}/index.html</loc></url>",
        f"  <url><loc>{site}/category.html</loc></url>",
        f"  <url><loc>{site}/search.html</loc></url>",
    ]
    for a in all_articles:
        urls.append(
            f"  <url><loc>{_article_url(a, site)}</loc>"
            f"<lastmod>{escape(str(a.get('date', '')))}</lastmod></url>"
        )
    sitemap = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls) + "\n"
        "</urlset>"
    )
    (WEB / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    print(f"  [BUILD] feeds refreshed ({len(all_articles)} articles, crawlable URLs)")


if __name__ == "__main__":
    out = build_all()
    print(f"Done → {out}")
