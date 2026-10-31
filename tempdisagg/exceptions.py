"""Errors and warnings raised by tempdisagg.

Errors
------
All errors derive from :class:`TempDisaggError`, so callers can catch library
failures without masking unrelated exceptions. Each error type maps to one
cause, which keeps automated pipelines able to react differently to bad
inputs, unidentified models and numerical breakdowns.

Warnings
--------
All warnings derive from :class:`TempDisaggWarning`. Warnings never change the
estimate; they flag statistical situations a rigorous user must know about
(for example, a ``rho`` estimate sitting on the search boundary). Every
warning emitted during a fit is also stored in
``result.diagnostics.warnings`` so batch jobs can audit them later.
"""

from __future__ import annotations


# ----------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------
class TempDisaggError(Exception):
    """Base class for every error raised by tempdisagg."""


class InputValidationError(TempDisaggError, ValueError):
    """User-supplied data or parameters are invalid."""


class SingularDesignError(TempDisaggError, ValueError):
    """The low-frequency design matrix is rank deficient.

    Typical causes: perfectly collinear indicators, an all-zero indicator, an
    intercept combined with an indicator that is constant within the sample,
    or fewer low-frequency observations than coefficients.
    """


class NumericalError(TempDisaggError, ArithmeticError):
    """A linear-algebra operation failed numerically.

    Examples: a covariance matrix that is not positive definite, or a Denton
    system that is singular because the constraints do not identify the series.
    """


class InfeasibleConstraintError(TempDisaggError, ValueError):
    """The requested constraints admit no solution.

    Example: enforcing non-negativity on a group whose low-frequency value is
    negative.
    """


# ----------------------------------------------------------------------
# Warnings
# ----------------------------------------------------------------------
class TempDisaggWarning(UserWarning):
    """Base class for statistical warnings issued by tempdisagg."""


class RhoBoundaryWarning(TempDisaggWarning):
    """The estimated ``rho`` lies on the boundary of the search interval.

    The optimum may lie outside the interval; inference that treats ``rho`` as
    an interior estimate is unreliable.
    """


class RhoNotIdentifiedWarning(TempDisaggWarning):
    """The ``rho`` criterion is (numerically) flat, so ``rho`` is not identified."""


class MultipleOptimaWarning(TempDisaggWarning):
    """The ``rho`` criterion has several local optima on the search grid."""


class ExactFitWarning(TempDisaggWarning):
    """The low-frequency regression fits exactly (zero residuals).

    The estimate does not depend on ``rho``, standard errors are zero and the
    log-likelihood is unbounded; reported statistics are degenerate.
    """


class IllConditionedWarning(TempDisaggWarning):
    """A matrix used in the estimation is close to singular.

    Results satisfy the constraints but may be sensitive to small changes in
    the data (e.g. nearly collinear indicators).
    """


class IndicatorSignWarning(TempDisaggWarning):
    """The indicator changes sign under the proportional Denton criterion."""


class IneffectiveParameterWarning(TempDisaggWarning):
    """A configured parameter has no effect with the selected settings."""


class ArtificialDataWarning(TempDisaggWarning):
    """Completion or padding created artificial values (all recorded in the result)."""


class RedundantConstraintWarning(TempDisaggWarning):
    """A consistent but linearly dependent constraint was dropped."""


class MissingTargetWarning(TempDisaggWarning):
    """Some low-frequency periods have no target and are left unconstrained."""


class ExperimentalMethodWarning(TempDisaggWarning):
    """The selected method is experimental (as in the R package)."""
