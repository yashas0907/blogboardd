"""One-time cleanup: strip forbidden sections from existing articles.

Removes: Author Bio, Internal Linking Suggestions, placeholder CTAs,
fake social links. Then rebuilds site-data.js.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BLOGS = ROOT / "blogboard" / "web" / "blogs"

FORBIDDEN_HEADERS = [
    "author bio",
    "internal linking suggestions",
]


def clean_article(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    original = text

    for header in FORBIDDEN_HEADERS:
        # Remove "## Author Bio" / "## Internal Linking Suggestions" sections
        # (header + everything until the next ## heading or EOF)
        pattern = re.compile(
            rf"^##+\s+{re.escape(header)}\s*\n(?:(?!^##+\s).)*",
            re.MULTILINE | re.DOTALL,
        )
        text = pattern.sub("", text)

    # Remove TOC entries pointing at removed sections
    text = re.sub(r"^\d+\.\s*\[Author Bio\]\([^)]*\)\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s*\[Internal Linking[^\]]*\]\([^)]*\)\s*\n", "", text, flags=re.MULTILINE)

    # Remove placeholder CTA bullets
    text = re.sub(
        r"^- \*\*(Download our free checklist|Join our upcoming webinar|Contact our AI Innovation Lab)\*\*.*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^\*Transform data constraints.*\*\s*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\*Ready to start your synthetic data journey\?\*\s*\n", "", text, flags=re.MULTILINE)

    # Remove "Ready to start..." header line if left dangling
    text = re.sub(r"^\*\*Ready to start your synthetic data journey\?\*\*\s*\n", "", text, flags=re.MULTILINE)

    # Generic: placeholder "(link)" / "(date & registration)" bullets
    text = re.sub(r"^.*\(link\).*$\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"^.*\(date & registration\).*$\n?", "", text, flags=re.MULTILINE)

    if text != original:
        # Collapse 3+ blank lines down to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main():
    cleaned = []
    for md in BLOGS.glob("*/*.md"):
        if clean_article(md):
            cleaned.append(md.relative_to(ROOT).as_posix())

    print(f"Cleaned {len(cleaned)} articles:")
    for c in cleaned:
        print(f"  {c}")

    import sys
    sys.path.insert(0, str(ROOT))
    from scripts.build_site import build_all

    build_all()


if __name__ == "__main__":
    main()
