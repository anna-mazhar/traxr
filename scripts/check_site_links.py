"""Internal link check over the assembled static site (make site).

Walks every HTML file in the staging dir, extracts local href/src targets,
and verifies each resolves to a file in the staging tree. External URLs and
fragments are skipped (offline gate).
"""

import re
import sys
from pathlib import Path

_LINK_RE = re.compile(r'(?:href|src)="([^"#]+)(?:#[^"]*)?"')
_SKIP_PREFIXES = ("http://", "https://", "mailto:", "data:", "//")
# Deployed at https://<user>.github.io/traxr/ — root-absolute URLs carry the
# project-pages prefix, which maps to the staging root.
_SITE_PREFIX = "/traxr/"


def targets(html_path: Path, root: Path) -> list[tuple[str, Path]]:
    found = []
    for raw in _LINK_RE.findall(html_path.read_text(errors="replace")):
        if raw.startswith(_SKIP_PREFIXES) or raw == "":
            continue
        if raw.startswith("/"):
            path = raw.removeprefix(_SITE_PREFIX).lstrip("/")
            resolved = (root / path).resolve()
        else:
            resolved = (html_path.parent / raw).resolve()
        found.append((raw, resolved))
    return found


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "staging").resolve()
    if not root.is_dir():
        print(f"check_site_links: staging dir not found: {root}", file=sys.stderr)
        return 2
    broken = []
    pages = list(root.rglob("*.html"))
    for page in pages:
        for raw, resolved in targets(page, root):
            ok = (
                resolved.exists()
                or (resolved / "index.html").exists()
                or resolved.with_suffix(".html").exists()
            )
            if not ok:
                broken.append(f"{page.relative_to(root)} -> {raw}")
    if broken:
        print(f"check_site_links: {len(broken)} broken link(s):", file=sys.stderr)
        for line in broken:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"check_site_links: PASS — {len(pages)} page(s), no broken internal links")
    return 0


if __name__ == "__main__":
    sys.exit(main())
