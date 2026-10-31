"""Collection of statistical warnings raised during one estimation.

Estimators record situations through a :class:`DiagnosticLog` instead of
calling :func:`warnings.warn` directly. The model emits the collected
warnings once, after the fit, and stores their text in the result. This keeps
the warnings auditable in batch runs and makes the estimators pure functions
of their inputs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from tempdisagg.exceptions import TempDisaggWarning


@dataclass
class DiagnosticLog:
    """Ordered record of ``(category, message)`` pairs."""

    entries: list[tuple[type[TempDisaggWarning], str]] = field(default_factory=list)

    def warn(self, category: type[TempDisaggWarning], message: str) -> None:
        """Record a warning (duplicates are ignored)."""
        if (category, message) not in self.entries:
            self.entries.append((category, message))

    def emit(self, stacklevel: int = 3) -> None:
        """Issue every recorded warning through :mod:`warnings`."""
        for category, message in self.entries:
            warnings.warn(message, category, stacklevel=stacklevel)

    def messages(self) -> tuple[str, ...]:
        """Return ``"Category: message"`` strings."""
        return tuple(f"{category.__name__}: {message}" for category, message in self.entries)
