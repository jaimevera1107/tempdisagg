"""Move the entries under ``## Unreleased`` into a dated version section.

Idempotent: if a section for the version already exists, nothing changes.
Run from ``scripts/release.ps1``; it needs no arguments beyond the version.

Usage: ``python tools/open_changelog_section.py 0.9.1``
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[1] / "CHANGELOG.md"
UNRELEASED = "## Unreleased"


def main(version: str, today: str | None = None) -> int:
    """Insert ``## <version> — <date>`` under the Unreleased heading."""
    text = CHANGELOG.read_text(encoding="utf-8")
    if re.search(rf"^## {re.escape(version)}\b", text, re.MULTILINE):
        print(f"CHANGELOG.md already has a section for {version}.")
        return 0
    if UNRELEASED not in text:
        raise SystemExit(f"CHANGELOG.md has no '{UNRELEASED}' heading")
    head, _, tail = text.partition(UNRELEASED)
    if not tail.strip():
        raise SystemExit("Nothing to release: the Unreleased section is empty")
    date = today or dt.date.today().isoformat()
    body = tail.lstrip("\n")
    opened = f"{UNRELEASED}\n\n## {version} — {date}\n\n{body}"
    CHANGELOG.write_text(head + opened, encoding="utf-8")
    print(f"CHANGELOG.md: opened section {version} — {date}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(
            "Usage: python tools/open_changelog_section.py X.Y.Z [YYYY-MM-DD]"
        )
    raise SystemExit(main(*sys.argv[1:]))
