"""Analyzer golden check: the analyzer must reproduce the committed goldens.

Compares the traxr.metrics analyzer against the ground-truth fixtures in
tests/fixtures/analyzer_goldens/ (committed once during initial development).
"""

import sys


def main() -> int:
    print("analyzer-goldens: not yet built — lands in M1", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
