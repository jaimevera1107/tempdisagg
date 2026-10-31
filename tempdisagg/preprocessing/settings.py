"""Settings of the two preprocessing steps: completion and padding.

Completion (on by default)
    Makes the high-frequency time grid regular and complete. The residual
    models (AR(1), random walk) and the Denton difference operators assume
    consecutive, equally spaced periods, and the aggregation constraint of a
    low-frequency period needs all of its sub-periods. Completion therefore

    * adds missing sub-periods of every low-frequency period with a target;
    * adds interior gaps (whole missing periods inside the sample);
    * fills missing indicator values: linear interpolation inside the
      sample, extension (``edge_method``) at the edges.

    It never invents low-frequency targets: a period without a target stays
    unconstrained and the model estimates it. Every artificial value is
    recorded in the result.

Padding (off by default)
    Adds ``before`` / ``after`` artificial low-frequency periods around the
    observed targets, with extended indicators and (unless
    ``target_method="none"``) artificial targets. It is a heuristic against
    end-point distortions: the artificial constraints anchor the ends of the
    sample. Padded periods are removed from the returned series.

    Padding with ``target_method="none"`` (unconstrained padded periods) has
    no effect on Chow-Lin, Litterman, Fernandez or Denton-Cholette estimates
    inside the sample: those estimators do not depend on unconstrained
    periods outside it (this is verified by the tests). Only the original
    Denton, whose initial condition moves, reacts to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tempdisagg.exceptions import InputValidationError


class EdgeMethod(str, Enum):
    """How a series is extended beyond its last known value."""

    LINEAR = "linear"      # OLS trend fitted on the last observations
    CONSTANT = "constant"  # repeat the edge value


class PartialPeriodPolicy(str, Enum):
    """Treatment of a low-frequency period whose sub-periods are not all observed.

    ``complete``  the target covers the whole period: missing sub-periods are
                  added (artificial indicator values) and the constraint spans
                  the full period (behaviour of 0.2.x).
    ``partial``   the target covers only the observed sub-periods: the
                  constraint applies to them alone (missing interior
                  sub-periods are added to keep the grid regular, but left
                  outside the constraint).
    ``error``     raise an error.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    ERROR = "error"


class TargetPaddingMethod(str, Enum):
    """Artificial low-frequency targets for padded periods."""

    LINEAR = "linear"  # OLS trend of the last ``fit_window`` targets
    REPEAT = "repeat"  # repeat the edge target (behaviour of 0.2.x)
    NONE = "none"      # no target: padded periods stay unconstrained


def _check_window(fit_window: int) -> None:
    if isinstance(fit_window, bool) or not isinstance(fit_window, int) or fit_window < 2:
        raise InputValidationError("fit_window must be an integer >= 2.")


@dataclass(frozen=True)
class CompletionSettings:
    """Settings of the time-grid completion.

    Parameters
    ----------
    enabled : bool, default True
        If False, incomplete grids or missing indicator values raise
        :class:`~tempdisagg.exceptions.InputValidationError` (strict mode).
    edge_method : {"linear", "constant"}, default "linear"
        Extension used for missing indicator values before the first or after
        the last known value.
    fit_window : int, default 3
        Number of low-frequency periods whose indicator values are used to fit
        the linear edge extension.
    partial_periods : {"complete", "partial", "error"}, default "complete"
        What a target of an incompletely observed period covers (see
        :class:`PartialPeriodPolicy`).
    """

    enabled: bool = True
    edge_method: EdgeMethod | str = EdgeMethod.LINEAR
    fit_window: int = 3
    partial_periods: PartialPeriodPolicy | str = PartialPeriodPolicy.COMPLETE

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise InputValidationError("enabled must be True or False.")
        try:
            object.__setattr__(self, "partial_periods", PartialPeriodPolicy(self.partial_periods))
            object.__setattr__(self, "edge_method", EdgeMethod(self.edge_method))
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        _check_window(self.fit_window)


@dataclass(frozen=True)
class PaddingSettings:
    """Settings of the artificial padding around the observed targets.

    Parameters
    ----------
    before, after : int, default 0
        Number of **low-frequency** periods added before the first and after
        the last observed target (e.g. ``after=1`` adds one year when
        disaggregating annual data).
    indicator_method : {"linear", "constant"}, default "linear"
        Extension of the indicators into padded periods (where no real
        indicator values exist).
    target_method : {"linear", "repeat", "none"}, default "linear"
        Artificial targets of padded periods.
    fit_window : int, default 3
        Low-frequency periods used to fit the linear extensions.
    """

    before: int = 0
    after: int = 0
    indicator_method: EdgeMethod | str = EdgeMethod.LINEAR
    target_method: TargetPaddingMethod | str = TargetPaddingMethod.LINEAR
    fit_window: int = 3

    def __post_init__(self) -> None:
        for name in ("before", "after"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InputValidationError(f"{name} must be a non-negative integer.")
        try:
            object.__setattr__(self, "indicator_method", EdgeMethod(self.indicator_method))
            object.__setattr__(self, "target_method", TargetPaddingMethod(self.target_method))
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        _check_window(self.fit_window)

    @property
    def active(self) -> bool:
        """Whether any padding is requested."""
        return self.before > 0 or self.after > 0


def as_completion(value: CompletionSettings | bool | None) -> CompletionSettings:
    """Normalise ``completion=`` arguments (bool or settings)."""
    if value is None or value is True:
        return CompletionSettings()
    if value is False:
        return CompletionSettings(enabled=False)
    if isinstance(value, CompletionSettings):
        return value
    raise InputValidationError("completion must be a bool or CompletionSettings.")


def as_padding(value: PaddingSettings | int | tuple[int, int] | None) -> PaddingSettings:
    """Normalise ``padding=`` arguments (int, (before, after) or settings)."""
    if value is None:
        return PaddingSettings()
    if isinstance(value, PaddingSettings):
        return value
    if isinstance(value, bool):
        raise InputValidationError("padding must be an int, a (before, after) tuple or settings.")
    if isinstance(value, int):
        return PaddingSettings(before=value, after=value)
    if isinstance(value, tuple) and len(value) == 2:
        return PaddingSettings(before=value[0], after=value[1])
    raise InputValidationError("padding must be an int, a (before, after) tuple or settings.")
