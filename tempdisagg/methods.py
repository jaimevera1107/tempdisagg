"""Method specifications: every estimation method and its configuration.

Each method is a frozen dataclass whose fields are its tunable settings. The
dataclass is the single source of truth for what a method does, so the
question "does method X include an intercept, and can I change it?" is
answered by reading its fields or calling :meth:`describe`::

    >>> from tempdisagg.methods import ChowLin
    >>> ChowLin().include_intercept
    True
    >>> ChowLin(include_intercept=False, rho_method="minrss").name
    'chow-lin-minrss-ecotrim'

The R method names are presets of these classes (:data:`PRESETS`), and any
field of a preset can be overridden::

    >>> make_method("fernandez", include_intercept=False)
    Fernandez(include_intercept=False)

A field that does not exist for a method raises an error listing the valid
ones (e.g. Denton has no intercept because it estimates no regression), and a
field that exists but has no effect with the other settings (e.g.
``fixed_rho`` with ``rho_method="maxlog"``) raises an
:class:`~tempdisagg.exceptions.IneffectiveParameterWarning`.

=====================  =============================================  ======================
Class                  Model                                          R presets
=====================  =============================================  ======================
ChowLin                AR(1) residuals                                chow-lin-*, fast
Litterman              random walk + AR(1) residuals                  litterman-*
Fernandez              random-walk residuals                          fernandez
DynamicChowLin         y_t = rho y_{t-1} + x_t b + e_t (experimental) dynamic-*
OLS                    white-noise residuals                          ols
Denton                 benchmarking of one indicator                  denton, denton-cholette
Uniform                even split                                     uniform
=====================  =============================================  ======================
"""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd

from tempdisagg.benchmarking.denton import (
    DentonBenchmarker,
    DentonCriterion,
    DentonVariant,
    UniformDistributor,
)
from tempdisagg.covariance.structures import (
    AR1Covariance,
    CovarianceForm,
    DynamicStructure,
    RandomWalkAR1Covariance,
    WhiteNoiseCovariance,
)
from tempdisagg.estimation.base import Disaggregator
from tempdisagg.estimation.regression import NO_TRUNCATION, RegressionDisaggregator
from tempdisagg.estimation.rho import RhoMethod, RhoOptimizerSettings
from tempdisagg.exceptions import (
    ExperimentalMethodWarning,
    IneffectiveParameterWarning,
    InputValidationError,
)

#: ``rho`` of the ``fast`` preset (as in R).
FAST_RHO: float = 0.99999


def _doc(text: str, r_name: str = "") -> dict[str, str]:
    """Field metadata: description and the equivalent R argument."""
    return {"doc": text, "r": r_name}


def _warn_ineffective(message: str) -> None:
    warnings.warn(message, IneffectiveParameterWarning, stacklevel=4)


@runtime_checkable
class MethodSpec(Protocol):
    """Interface of every method specification."""

    @property
    def name(self) -> str:
        """Canonical (R-compatible) method name."""

    def build(self) -> Disaggregator:
        """Instantiate the estimator."""


class _SpecMixin:
    """Shared helpers for method specifications (no state)."""

    family: ClassVar[str] = ""
    experimental: ClassVar[bool] = False

    def describe(self) -> pd.DataFrame:
        """Return every setting with its value, meaning and R equivalent."""
        rows = []
        for spec_field in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, spec_field.name)
            if isinstance(value, RhoOptimizerSettings):
                for sub in dataclasses.fields(value):
                    rows.append((f"optimizer.{sub.name}", getattr(value, sub.name),
                                 f"rho search: {sub.name}", ""))
                continue
            rows.append((
                spec_field.name,
                value.value if hasattr(value, "value") else value,
                spec_field.metadata.get("doc", ""),
                spec_field.metadata.get("r", ""),
            ))
        frame = pd.DataFrame(rows, columns=["parameter", "value", "description", "R argument"])
        return frame.set_index("parameter")


def _validate_rho_settings(fixed_rho: float, truncated_rho: float) -> None:
    if not -1.0 < float(fixed_rho) < 1.0:
        raise InputValidationError("fixed_rho must lie strictly between -1 and 1.")
    if not -1.0 <= float(truncated_rho) < 1.0:
        raise InputValidationError("truncated_rho must lie in [-1, 1); -1 disables truncation.")


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise InputValidationError(f"{name} must be True or False; got {value!r}.")


def _check_rho_effectiveness(spec: Any) -> None:
    """Warn about rho settings that the chosen rho_method ignores."""
    defaults = {f.name: f.default for f in dataclasses.fields(spec)}
    if spec.rho_method is not RhoMethod.FIXED and spec.fixed_rho != defaults["fixed_rho"]:
        _warn_ineffective("fixed_rho has no effect unless rho_method='fixed'.")
    if spec.rho_method is RhoMethod.FIXED and spec.truncated_rho != defaults["truncated_rho"]:
        _warn_ineffective("truncated_rho only applies to an estimated rho.")


# ----------------------------------------------------------------------
# Regression-based methods
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ChowLin(_SpecMixin):
    """Chow and Lin (1971): regression with stationary AR(1) residuals."""

    rho_method: RhoMethod | str = field(
        default=RhoMethod.MAXLOG,
        metadata=_doc("'maxlog', 'minrss' or 'fixed'", "method suffix"))
    fixed_rho: float = field(
        default=0.5, metadata=_doc("rho when rho_method='fixed'", "fixed.rho"))
    truncated_rho: float = field(
        default=0.0, metadata=_doc("lower bound for an estimated rho; -1 disables",
                                   "truncated.rho"))
    include_intercept: bool = field(
        default=True, metadata=_doc("add a constant regressor", "formula: y ~ x vs y ~ 0 + x"))
    covariance_form: CovarianceForm | str = field(
        default=CovarianceForm.CORRELATION,
        metadata=_doc("'correlation' (Ecotrim) or 'covariance' (Quilis); affects minrss only",
                      "minrss-ecotrim / minrss-quilis"))
    optimizer: RhoOptimizerSettings = field(default_factory=RhoOptimizerSettings)

    family: ClassVar[str] = "regression"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho_method", RhoMethod(self.rho_method))
        object.__setattr__(self, "covariance_form", CovarianceForm(self.covariance_form))
        _validate_rho_settings(self.fixed_rho, self.truncated_rho)
        _validate_bool("include_intercept", self.include_intercept)
        _check_rho_effectiveness(self)
        if (self.rho_method is not RhoMethod.MINRSS
                and self.covariance_form is not CovarianceForm.CORRELATION):
            _warn_ineffective("covariance_form only affects rho_method='minrss'.")

    @property
    def name(self) -> str:
        """Canonical R-compatible name."""
        if self.rho_method is RhoMethod.FIXED:
            return "fast" if self.fixed_rho == FAST_RHO else "chow-lin-fixed"
        if self.rho_method is RhoMethod.MAXLOG:
            return "chow-lin-maxlog"
        suffix = "ecotrim" if self.covariance_form is CovarianceForm.CORRELATION else "quilis"
        return f"chow-lin-minrss-{suffix}"

    def build(self) -> Disaggregator:
        """Instantiate the GLS estimator."""
        # rho is estimated with the chosen form; statistics are reported with the
        # stationary covariance rho^|i-j| / (1 - rho^2), as in the R package.
        return RegressionDisaggregator(
            AR1Covariance(self.covariance_form), self.rho_method, self.fixed_rho,
            self.truncated_rho, self.include_intercept, self.optimizer,
            reporting_structure=AR1Covariance(CovarianceForm.COVARIANCE),
        )


@dataclass(frozen=True)
class Litterman(_SpecMixin):
    """Litterman (1983): random-walk residuals with AR(1) innovations."""

    rho_method: RhoMethod | str = field(
        default=RhoMethod.MAXLOG, metadata=_doc("'maxlog', 'minrss' or 'fixed'", "method suffix"))
    fixed_rho: float = field(
        default=0.5, metadata=_doc("rho when rho_method='fixed'", "fixed.rho"))
    truncated_rho: float = field(
        default=0.0, metadata=_doc("lower bound for an estimated rho; -1 disables",
                                   "truncated.rho"))
    include_intercept: bool = field(
        default=True, metadata=_doc("add a constant regressor", "formula: y ~ x vs y ~ 0 + x"))
    optimizer: RhoOptimizerSettings = field(default_factory=RhoOptimizerSettings)

    family: ClassVar[str] = "regression"

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho_method", RhoMethod(self.rho_method))
        _validate_rho_settings(self.fixed_rho, self.truncated_rho)
        _validate_bool("include_intercept", self.include_intercept)
        _check_rho_effectiveness(self)

    @property
    def name(self) -> str:
        """Canonical R-compatible name."""
        return f"litterman-{self.rho_method.value}"

    def build(self) -> Disaggregator:
        """Instantiate the GLS estimator."""
        return RegressionDisaggregator(
            RandomWalkAR1Covariance(), self.rho_method, self.fixed_rho,
            self.truncated_rho, self.include_intercept, self.optimizer,
        )


@dataclass(frozen=True)
class Fernandez(_SpecMixin):
    """Fernandez (1981): random-walk residuals (Litterman with rho = 0)."""

    include_intercept: bool = field(
        default=True, metadata=_doc("add a constant regressor", "formula: y ~ x vs y ~ 0 + x"))

    family: ClassVar[str] = "regression"

    def __post_init__(self) -> None:
        _validate_bool("include_intercept", self.include_intercept)

    @property
    def name(self) -> str:
        """Canonical R-compatible name."""
        return "fernandez"

    def build(self) -> Disaggregator:
        """Instantiate the GLS estimator."""
        return RegressionDisaggregator(
            RandomWalkAR1Covariance(), RhoMethod.FIXED, 0.0, NO_TRUNCATION,
            self.include_intercept, RhoOptimizerSettings(),
        )


@dataclass(frozen=True)
class DynamicChowLin(_SpecMixin):
    """Santos Silva and Cardoso (2001): dynamic Chow-Lin. Experimental."""

    rho_method: RhoMethod | str = field(
        default=RhoMethod.MAXLOG, metadata=_doc("'maxlog', 'minrss' or 'fixed'", "method suffix"))
    fixed_rho: float = field(
        default=0.5, metadata=_doc("rho when rho_method='fixed'", "fixed.rho"))
    truncated_rho: float = field(
        default=0.0, metadata=_doc("lower bound for an estimated rho; -1 disables",
                                   "truncated.rho"))
    include_intercept: bool = field(
        default=True, metadata=_doc("add a constant regressor", "formula: y ~ x vs y ~ 0 + x"))
    include_truncation_remainder: bool = field(
        default=True, metadata=_doc("add rho^t to capture the unknown initial value", ""))
    optimizer: RhoOptimizerSettings = field(default_factory=RhoOptimizerSettings)

    family: ClassVar[str] = "regression"
    experimental: ClassVar[bool] = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho_method", RhoMethod(self.rho_method))
        _validate_rho_settings(self.fixed_rho, self.truncated_rho)
        _validate_bool("include_intercept", self.include_intercept)
        _validate_bool("include_truncation_remainder", self.include_truncation_remainder)
        _check_rho_effectiveness(self)

    @property
    def name(self) -> str:
        """Canonical R-compatible name."""
        return f"dynamic-{self.rho_method.value}"

    def build(self) -> Disaggregator:
        """Instantiate the GLS estimator."""
        return RegressionDisaggregator(
            DynamicStructure(self.include_truncation_remainder), self.rho_method,
            self.fixed_rho, self.truncated_rho, self.include_intercept, self.optimizer,
        )


@dataclass(frozen=True)
class OLS(_SpecMixin):
    """OLS on the aggregated data; residuals are spread uniformly."""

    include_intercept: bool = field(
        default=True, metadata=_doc("add a constant regressor", "formula: y ~ x vs y ~ 0 + x"))

    family: ClassVar[str] = "regression"

    def __post_init__(self) -> None:
        _validate_bool("include_intercept", self.include_intercept)

    @property
    def name(self) -> str:
        """Canonical R-compatible name."""
        return "ols"

    def build(self) -> Disaggregator:
        """Instantiate the GLS estimator."""
        return RegressionDisaggregator(
            WhiteNoiseCovariance(), RhoMethod.FIXED, 0.0, NO_TRUNCATION,
            self.include_intercept, RhoOptimizerSettings(),
        )


# ----------------------------------------------------------------------
# Benchmarking methods
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Denton(_SpecMixin):
    """Denton benchmarking of a single indicator (no regression, no intercept)."""

    variant: DentonVariant | str = field(
        default=DentonVariant.CHOLETTE,
        metadata=_doc("'cholette' (no initial condition) or 'original'",
                      "denton-cholette / denton"))
    criterion: DentonCriterion | str = field(
        default=DentonCriterion.PROPORTIONAL,
        metadata=_doc("'proportional' or 'additive'", "criterion"))
    h: int = field(default=1, metadata=_doc("differencing order (0 levels, 1 changes, ...)", "h"))

    family: ClassVar[str] = "benchmarking"

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "variant", DentonVariant(self.variant))
            object.__setattr__(self, "criterion", DentonCriterion(self.criterion))
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        if isinstance(self.h, bool) or not isinstance(self.h, int) or self.h < 0:
            raise InputValidationError("h must be a non-negative integer.")

    @property
    def name(self) -> str:
        """Canonical R-compatible name."""
        return "denton-cholette" if self.variant is DentonVariant.CHOLETTE else "denton"

    def build(self) -> Disaggregator:
        """Instantiate the benchmarker."""
        return DentonBenchmarker(self.variant, self.criterion, self.h)


@dataclass(frozen=True)
class Uniform(_SpecMixin):
    """Even distribution within each low-frequency period."""

    family: ClassVar[str] = "benchmarking"

    @property
    def name(self) -> str:
        """Canonical R-compatible name."""
        return "uniform"

    def build(self) -> Disaggregator:
        """Instantiate the distributor."""
        return UniformDistributor()


# ----------------------------------------------------------------------
# Presets (R names), legacy aliases and factory
# ----------------------------------------------------------------------
def _quiet(factory: Any, **kwargs: Any) -> Any:
    """Build a preset without ineffective-parameter warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IneffectiveParameterWarning)
        return factory(**kwargs)


#: R method names mapped to their specification.
PRESETS: dict[str, MethodSpec] = {
    "chow-lin-maxlog": _quiet(ChowLin, rho_method="maxlog"),
    "chow-lin-minrss-ecotrim": _quiet(ChowLin, rho_method="minrss", covariance_form="correlation"),
    "chow-lin-minrss-quilis": _quiet(ChowLin, rho_method="minrss", covariance_form="covariance"),
    "chow-lin-fixed": _quiet(ChowLin, rho_method="fixed"),
    "fast": _quiet(ChowLin, rho_method="fixed", fixed_rho=FAST_RHO),
    "litterman-maxlog": _quiet(Litterman, rho_method="maxlog"),
    "litterman-minrss": _quiet(Litterman, rho_method="minrss"),
    "litterman-fixed": _quiet(Litterman, rho_method="fixed"),
    "fernandez": Fernandez(),
    "dynamic-maxlog": _quiet(DynamicChowLin, rho_method="maxlog"),
    "dynamic-minrss": _quiet(DynamicChowLin, rho_method="minrss"),
    "dynamic-fixed": _quiet(DynamicChowLin, rho_method="fixed"),
    "ols": OLS(),
    "denton-cholette": Denton(variant="cholette"),
    "denton": Denton(variant="original"),
    "uniform": Uniform(),
}

#: Names used by tempdisagg 0.2.x, mapped to presets.
LEGACY_ALIASES: dict[str, str] = {
    "chow-lin": "chow-lin-fixed",
    "chow-lin-opt": "chow-lin-maxlog",
    "chow-lin-ecotrim": "chow-lin-minrss-ecotrim",
    "chow-lin-quilis": "chow-lin-minrss-quilis",
    "litterman": "litterman-fixed",
    "litterman-opt": "litterman-minrss",
}

#: Methods removed after 0.2.x, with the reason shown to the user.
REMOVED_METHODS: dict[str, str] = {
    "ensemble": (
        "The ensemble was removed: every constrained method reproduces the "
        "low-frequency data exactly, so in-sample weights are not identified. "
        "Compare methods with out-of-sample validation instead."
    ),
}


def available_methods() -> list[str]:
    """Return the R-compatible names of all presets."""
    return sorted(PRESETS)


def make_method(method: str | MethodSpec, **overrides: Any) -> MethodSpec:
    """Return a method specification, applying ``overrides``.

    Parameters
    ----------
    method : str or MethodSpec
        R preset name (see :func:`available_methods`), a 0.2.x alias, or a
        specification instance.
    **overrides
        Fields of the specification to change, e.g. ``include_intercept``.

    Raises
    ------
    InputValidationError
        Unknown or removed method, or a field the method does not have.
    """
    if isinstance(method, str):
        key = method.lower().strip()
        if key in REMOVED_METHODS:
            raise InputValidationError(REMOVED_METHODS[key])
        if key in LEGACY_ALIASES:
            warnings.warn(f"Method '{key}' is deprecated; use '{LEGACY_ALIASES[key]}'.",
                          DeprecationWarning, stacklevel=3)
            key = LEGACY_ALIASES[key]
        if key not in PRESETS:
            raise InputValidationError(
                f"Unknown method '{method}'. Available: {', '.join(available_methods())}."
            )
        spec = PRESETS[key]
    elif dataclasses.is_dataclass(method) and isinstance(method, MethodSpec):
        spec = method
    else:
        raise InputValidationError("method must be a preset name or a method specification.")

    if overrides:
        valid = {f.name for f in dataclasses.fields(spec)}  # type: ignore[arg-type]
        unknown = sorted(set(overrides) - valid)
        if unknown:
            benchmarking = getattr(spec, "family", "") == "benchmarking"
            hint = (" (benchmarking methods estimate no regression)"
                    if "include_intercept" in unknown and benchmarking else "")
            raise InputValidationError(
                f"{type(spec).__name__} has no parameter(s) {unknown}{hint}. "
                f"Valid parameters: {sorted(valid) or 'none'}."
            )
        spec = dataclasses.replace(spec, **overrides)  # type: ignore[type-var]

    if getattr(spec, "experimental", False):
        warnings.warn(f"Method '{spec.name}' is experimental (as in the R package).",
                      ExperimentalMethodWarning, stacklevel=3)
    return spec
