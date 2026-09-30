"""Live Google Safe Browsing flag monitor for the site.

Decodes Google's own status API (the same database Chrome uses).
Run anytime:  uv run python scripts/check_flag.py
"""
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SITES = [
    "yashas0907.github.io",
    "yashas0907.github.io/blogboardd/",
]

ENDPOINT = (
    "https://transparencyreport.google.com/transparencyreport/api/v3/"
    "safebrowsing/status?site={site}"
)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def check_site(site: str) -> dict:
    """Returns {malware, unwanted, phishing, pha, flagged, since} for a site."""
    r = requests.get(ENDPOINT.format(site=site), headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code} (endpoint rate-limited — retry in a minute)"}
    raw = r.text
    m = re.search(r"\[\[\"sb\.ssr\"(.*?)\]\]", raw, re.DOTALL)
    if not m:
        return {"error": "unexpected response format"}
    fields = json.loads('["sb.ssr"' + m.group(1) + "]")[1:]
    return {
        "malware": fields[0] is True,
        "unwanted_software": fields[1] is True,
        "phishing": fields[2] is True,
        "potentially_harmful_app": fields[3] is True,
        "flagged": any(f is True for f in fields[:4]),
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def main():
    watch = "--watch" in sys.argv
    while True:
        print(f"\n=== Safe Browsing status — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
        all_clear = True
        for site in SITES:
            status = check_site(site)
            if "error" in status:
                print(f"  {site}: {status['error']}")
                all_clear = False
                continue
            if status["flagged"]:
                all_clear = False
                kinds = [k for k in ("malware", "unwanted_software", "phishing", "potentially_harmful_app") if status[k]]
                print(f"  {site}: FLAGGED ({', '.join(kinds)})")
            else:
                print(f"  {site}: CLEAR - NOT FLAGGED")

        if all_clear:
            print("\n  SITE IS FULLY CLEAR - the review went through!")
            break
        if not watch:
            print("\n  Flag still active. Reviews clear in 1-3 days. Re-run with --watch to monitor live.")
            break
        print("\n  Checking again in 5 minutes...")
        time.sleep(300)


if __name__ == "__main__":
    main()
