"""Single source of truth for the package version.

``pyproject.toml`` reads it dynamically and ``tempdisagg.__version__``
re-exports it; change it only with ``python tools/bump_version.py X.Y.Z``.
"""

__version__ = "0.6.4"
