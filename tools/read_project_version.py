"""Print the package version (single source: ``tempdisagg/_version.py``)."""

import re
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parents[1] / "tempdisagg" / "_version.py"

match = re.search(
    r'^__version__ = "([^"]+)"$', VERSION_FILE.read_text(encoding="utf-8"), re.MULTILINE
)
if match is None:
    raise SystemExit(f"No __version__ found in {VERSION_FILE}")
print(match.group(1))
