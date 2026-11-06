"""Set the package version for the next release.

The version lives in ``tempdisagg/_version.py`` only; ``pyproject.toml`` reads
it dynamically. Run while preparing a *new* version and never reuse a version
that already exists on PyPI.

Usage: ``python tools/bump_version.py 0.6.5``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "tempdisagg" / "_version.py"
VERSION_PATTERN = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
RELEASE_PATTERN = re.compile(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+)?")


def main(new_version: str) -> int:
    """Replace the version in ``_version.py``."""
    if not RELEASE_PATTERN.fullmatch(new_version):
        raise SystemExit(f"Expected a PEP 440 release version, got {new_version!r}")
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = VERSION_PATTERN.search(text)
    if match is None:
        raise SystemExit(f"No __version__ assignment found in {VERSION_FILE}")
    old_version = match.group(1)
    if old_version == new_version:
        print(f"Version already {new_version}; nothing to do.")
        return 0
    VERSION_FILE.write_text(
        VERSION_PATTERN.sub(f'__version__ = "{new_version}"', text), encoding="utf-8"
    )
    print(f"Updated release {old_version} -> {new_version}.")
    print("Next: add the CHANGELOG.md section and rerun the tests.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/bump_version.py X.Y.Z")
    raise SystemExit(main(sys.argv[1]))
