"""Full site verification: simulates the browser's data flow end-to-end."""
import json
import re
from pathlib import Path

web = Path("blogboard/web")
ok, fail = 0, []

def check(name, cond, detail=""):
    global ok
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail.append(name)
        print(f"  FAIL  {name} {detail}")

# 1. site-data.js exists and parses
sd = (web / "js" / "site-data.js").read_text(encoding="utf-8")
m = re.search(r"window\.SITE_DATA = (\{.*?\});\n", sd, re.DOTALL)
check("site-data.js parses as JS object", m is not None)
data = json.loads(m.group(1))

m2 = re.search(r"window\.SITE_CONTENT = (\{.*?\});\n", sd, re.DOTALL)
check("SITE_CONTENT present", m2 is not None)
content = json.loads(m2.group(1))

# 2. All 7 categories present with ≥1 article
cats = data["categories"]
for cat in ["ml", "dl", "statistics", "nlp", "cv", "genai", "ainews"]:
    check(f"category '{cat}' has articles", len(cats.get(cat, [])) >= 1,
          f"(has {len(cats.get(cat, []))})")

# 3. Every article has content baked
total = sum(len(v) for v in cats.values())
check(f"all {total} articles have baked content", total == len(content))

# 4. Every article entry has required fields
required = ["id", "category", "title", "description", "date", "readTime", "file"]
bad_fields = []
for cat, arts in cats.items():
    for a in arts:
        for field in required:
            if not a.get(field):
                bad_fields.append(f"{a.get('id','?')}.{field}")
check("every article has required fields", not bad_fields, str(bad_fields[:5]))

# 5. Every .md file on disk is registered (no orphans)
orphans = []
for md in web.glob("blogs/*/*.md"):
    rel = md.relative_to(web).as_posix()
    found = any(a.get("file") == rel for arts in cats.values() for a in arts)
    if not found:
        orphans.append(rel)
check("no orphan .md files", not orphans, str(orphans))

# 6. Every registered file exists on disk (no dead links)
dead = [a.get("file") for arts in cats.values() for a in arts
        if not (web / a.get("file", "x")).is_file()]
check("no dead article links", not dead, str(dead))

# 7. HTML pages reference the right scripts in the right order
for page in ["index.html", "post.html", "category.html", "search.html"]:
    html = (web / page).read_text(encoding="utf-8")
    idx_sd = html.find("site-data.js")
    idx_bd = html.find("blogs-data.js")
    check(f"{page}: site-data.js before blogs-data.js",
          0 < idx_sd < idx_bd)

# 8. No junk titles
junk_patterns = ["Load the data", "Import necessary", "test blog post"]
for arts in cats.values():
    for a in arts:
        check(f"title sane: {a['title'][:40]}",
              not any(p.lower() in a["title"].lower() for p in junk_patterns))

# 9. RSS + sitemap exist and reference articles
rss = (web / "rss.xml").read_text(encoding="utf-8")
sm = (web / "sitemap.xml").read_text(encoding="utf-8")
check("rss.xml has items", rss.count("<item>") >= total)
check("sitemap.xml has URLs", sm.count("<url>") >= total)

# 10. Dates are valid ISO format
import datetime
for arts in cats.values():
    for a in arts:
        try:
            datetime.date.fromisoformat(a["date"])
        except ValueError:
            fail.append(f"bad date {a['id']}: {a['date']}")
check("all dates valid ISO", not [f for f in fail if "bad date" in f])

print()
print(f"{'=' * 50}")
print(f"RESULT: {ok} passed, {len(fail)} failed")
if fail:
    for f in fail:
        print("  FAILED:", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
