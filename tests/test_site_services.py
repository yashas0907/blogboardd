"""Tests for RSS + sitemap generation."""

from blogboard.services import site_services

ARTICLES = [
    {
        "id": "blogs/ml/a.md",
        "title": "Alpha <Post>",
        "description": "About ML & more",
        "category": "ml",
        "date": "2026-09-01",
    },
    {
        "id": "blogs/nlp/b.md",
        "title": "Beta Post",
        "description": "NLP desc",
        "category": "nlp",
        "date": "2026-08-01",
    },
]


def test_rss_contains_items_and_escapes():
    rss = site_services.generate_rss(ARTICLES)
    assert "<rss" in rss and "<channel>" in rss
    assert "Alpha &lt;Post&gt;" in rss  # XML-escaped
    assert "About ML &amp; more" in rss
    assert rss.count("<item>") == 2


def test_sitemap_contains_all_urls():
    sitemap = site_services.generate_sitemap(ARTICLES)
    assert "<urlset" in sitemap
    assert sitemap.count("<url>") == 5  # 3 static + 2 articles
    assert "blogs/ml/a.md" in sitemap
    assert "<lastmod>2026-09-01</lastmod>" in sitemap
