"""Release checklist: version consistency, changelog entry and required files.

Run before tagging. Exit code 0 means every check passed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "LICENSE",
    "README.md",
    "CHANGELOG.md",
    "SECURITY.md",
    "CONTRIBUTING.md",
    "MIGRATION.md",
    "tempdisagg/py.typed",
    "tempdisagg/datasets/data",
    "tests/reference/generate_r_reference.R",
    ".github/workflows/tests.yml",
    ".github/workflows/publish.yml",
    "scripts/publish_release.ps1",
    "scripts/verify_pypi.ps1",
)


def check() -> None:
    """Raise ``AssertionError`` on the first failed check."""
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert "version" in project.get("dynamic", []), (
        "pyproject.toml must declare version as dynamic (single source: _version.py)"
    )
    assert "version" not in project, "pyproject.toml must not hard-code the version"
    source = (ROOT / "tempdisagg" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', source, re.MULTILINE)
    assert match, "Missing __version__ in tempdisagg/_version.py"
    version = match.group(1)
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+)?", version), (
        f"Invalid version {version!r}"
    )
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {version} " in changelog or f"## {version}\n" in changelog, (
        f"CHANGELOG.md has no section for {version}"
    )
    assert "## Unreleased\n\n### " not in changelog, (
        "CHANGELOG.md still has entries under Unreleased; move them to the version section"
    )
    for item in REQUIRED_FILES:
        assert (ROOT / item).exists(), f"Missing artifact: {item}"
    workflow = (ROOT / ".github/workflows/publish.yml").read_text(encoding="utf-8")
    assert "id-token: write" in workflow and "pypa/gh-action-pypi-publish" in workflow
    assert "environment:" in workflow and "pypi" in workflow
    print("tempdisagg release checklist: PASS")
    print(f"Version: {version}")
    print("Note: CI runs (Python matrix and R parity) must be green on the exact commit.")


if __name__ == "__main__":
    try:
        check()
    except AssertionError as exc:
        print(f"tempdisagg release checklist: FAIL — {exc}", file=sys.stderr)
        sys.exit(1)
