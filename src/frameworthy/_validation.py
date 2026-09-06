"""Shared parameter validation for inference (alpha/n_resamples/method) and
for confidence-interval computation (min observations, equal-length pairs).

Consolidating these here means every call site raises the same typed
exception with the same wording for the same underlying problem, instead of
each function in `_intervals.py` re-implementing (and subtly re-wording) its
own guard.
"""

from dataclasses import dataclass

import numpy as np

from ._constants import (
    DEFAULT_ALPHA,
    DEFAULT_INFERENCE_METHOD,
    DEFAULT_N_RESAMPLES,
    InferenceMethod,
)
from ._errors import (
    InsufficientDataError,
    InvalidColumnDataError,
    InvalidParameterError,
)


def validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 0.5:
        raise InvalidParameterError(f"`alpha` must be in (0, 0.5), got {alpha}.")


def validate_n_resamples(n_resamples: int) -> None:
    if n_resamples < 1:
        raise InvalidParameterError(
            f"`n_resamples` must be positive, got {n_resamples}."
        )


def validate_method(method: InferenceMethod) -> None:
    if method not in ("analytical", "bootstrap"):
        raise InvalidParameterError(f"Unknown inference `method`: {method!r}.")


def validate_min_observations(n: int, min_count: int, *, context: str) -> None:
    """Guard confidence-interval computation against too few observations.

    Distinct from `_arrays.assert_min_count`, which guards the earlier
    column-extraction stage (and reports in terms of "usable (non-null)
    values" rather than "observations required for a CI").
    """
    if n < min_count:
        raise InsufficientDataError(
            f"At least {min_count} {context} are required, got {n}."
        )


def validate_equal_length(
    before: np.ndarray, after: np.ndarray, *, context: str
) -> None:
    """Guard a paired comparison against mismatched `before`/`after` lengths."""
    if len(before) != len(after):
        raise InvalidColumnDataError(
            f"Paired {context} requires `before` and `after` to have the "
            f"same length, got {len(before)} and {len(after)}."
        )


@dataclass(frozen=True)
class InferenceConfig:
    """Validated bundle of the inference parameters shared by every
    built-in check: `alpha`, `n_resamples`, `random_state`, and `method`.

    Validating in `__post_init__` means an invalid `alpha` or unknown
    `method` fails as soon as `.equivalent()` / `.change_greater_than()` /
    `.change_less_than()` is called, rather than after `before`/`after`
    data has already been extracted and validated, partway through CI
    computation.
    """

    alpha: float = DEFAULT_ALPHA
    n_resamples: int = DEFAULT_N_RESAMPLES
    random_state: int | np.random.Generator | None = None
    method: InferenceMethod = DEFAULT_INFERENCE_METHOD

    def __post_init__(self) -> None:
        validate_alpha(self.alpha)
        validate_n_resamples(self.n_resamples)
        validate_method(self.method)

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    @property
    def reported_n_resamples(self) -> int:
        """`n_resamples` if it was actually used (bootstrap), else `0` --
        the rule both `EquivalenceResult` and `ChangeResult` apply to their
        own `n_resamples` field.
        """
        return self.n_resamples if self.method == "bootstrap" else 0
