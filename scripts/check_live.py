"""Final live site state: per-category freshness."""
import json
import re

import requests

live = requests.get(
    "https://yashas0907.github.io/blogboardd/js/site-data.js?v=final"
).text
m = re.search(r"window\.SITE_DATA = (\{.*?\});", live, re.DOTALL)
data = json.loads(m.group(1))
for cat, arts in data["categories"].items():
    dates = sorted((a.get("date", "") for a in arts), reverse=True)
    latest = dates[0] if dates else "-"
    print(f"  {cat}: latest={latest} | articles={len(arts)}")
