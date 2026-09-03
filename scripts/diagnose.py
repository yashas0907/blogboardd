"""Diagnostic script: audit registries, orphans, and article integrity."""

import json
from pathlib import Path

web = Path("blogboard/web")

print("=== 1. ARTICLES.JSON SANITY CHECK ===")
issues = []
for cat_dir in sorted((web / "blogs").iterdir()):
    if not cat_dir.is_dir():
        continue
    reg = cat_dir / "articles.json"
    print(f"-- {cat_dir.name} --")
    if not reg.exists():
        print("   MISSING articles.json")
        issues.append(f"{cat_dir.name}: no registry")
        continue
    try:
        articles = json.loads(reg.read_text(encoding="utf-8"))
        print(f"   {len(articles)} registered")
        for a in articles:
            f = web / a["file"]
            status = "OK " if f.exists() else "DEAD"
            print(f"   [{status}] {a['file']}")
            if not f.exists():
                issues.append(f"DEAD: {a['file']}")
    except Exception as e:
        print(f"   CORRUPT: {e}")
        issues.append(f"{cat_dir.name}: corrupt")

print()
print("=== 2. ORPHAN MD FILES (on disk but not registered) ===")
for md in sorted(web.glob("blogs/*/*.md")):
    try:
        arts = json.loads((md.parent / "articles.json").read_text(encoding="utf-8"))
        ids = {a["file"] for a in arts}
        rel = str(md.relative_to(web)).replace("\\", "/")
        if rel not in ids:
            print(f"   ORPHAN: {rel}")
    except Exception:
        print(f"   {md}: registry unreadable")

print()
print("=== 3. H1 CHECK ===")
for md in sorted(web.glob("blogs/*/*.md")):
    first = None
    for line in md.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            first = line[2:].strip()
            break
    print(f"   {md.name[:50]:52} h1={str(first)[:40]}")

print()
print("=== 4. RSS/SITEMAP ===")
print("   rss.xml exists:", (web / "rss.xml").exists())
print("   sitemap.xml exists:", (web / "sitemap.xml").exists())

print()
print("ISSUES FOUND:", len(issues))
for i in issues:
    print("  -", i)
