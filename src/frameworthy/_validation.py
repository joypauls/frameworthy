"""Shared parameter validation for inference (alpha/n_resamples/method) and
for confidence-interval computation (min observations, equal-length pairs).

Consolidating these here means every call site raises the same typed
exception with the same wording for the same underlying problem, instead of
each function in `_intervals.py` re-implementing (and subtly re-wording) its
own guard.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ._constants import (
    DEFAULT_ALPHA,
    DEFAULT_INFERENCE_METHOD,
    DEFAULT_N_RESAMPLES,
    InferenceMethod,
)
from ._errors import InvalidDataError, UsageError


def validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 0.5:
        raise UsageError(f"`alpha` must be in (0, 0.5), got {alpha}.")


def validate_n_resamples(n_resamples: int) -> None:
    if n_resamples < 1:
        raise UsageError(f"`n_resamples` must be positive, got {n_resamples}.")


def validate_method(method: InferenceMethod) -> None:
    if method not in ("analytical", "bootstrap"):
        raise UsageError(f"Unknown inference `method`: {method!r}.")


def validate_min_observations(n: int, min_count: int, *, context: str) -> None:
    """Guard confidence-interval computation against too few observations.

    Distinct from `_arrays.assert_min_count`, which guards the earlier
    column-extraction stage (and reports in terms of "usable (non-null)
    values" rather than "observations required for a CI").
    """
    if n < min_count:
        raise InvalidDataError(f"At least {min_count} {context} are required, got {n}.")


def validate_equal_length(
    before: np.ndarray, after: np.ndarray, *, context: str
) -> None:
    """Guard a paired comparison against mismatched `before`/`after` lengths."""
    if len(before) != len(after):
        raise InvalidDataError(
            f"Paired {context} requires `before` and `after` to have the "
            f"same length, got {len(before)} and {len(after)}."
        )


def validate_custom_metric(name: str, statistic_func: Callable) -> None:
    """Guard `.custom()`'s `name`/`statistic_func` arguments themselves,
    before anything is done with the data. `validate_statistic_func` below
    separately checks that `statistic_func` actually behaves as required.
    """
    if not isinstance(name, str) or not name:
        raise UsageError(f"`name` must be a non-empty string, got {name!r}.")
    if not callable(statistic_func):
        raise UsageError(f"`statistic_func` must be callable, got {statistic_func!r}.")


def validate_statistic_func(statistic_func: Callable, sample: np.ndarray) -> None:
    """Fail fast on a `statistic_func` that won't work for `.custom()`,
    before any bootstrap resampling begins, rather than letting a bad
    function surface as an opaque numpy error deep inside a
    multi-thousand-iteration bootstrap loop.

    Checks two things `statistic_func` must support:

    * Called plainly on a 1-D array, it must return something castable to
      a single `float` (this is also how `before_value`/`after_value` are
      reported).
    * Called with `axis=1` on a 2-D array, it must apply row-wise, like
      `np.mean`/`np.median` do -- this is what lets bootstrap resampling
      apply `statistic_func` to every resample at once instead of looping
      in Python. Bind extra arguments with `functools.partial` (e.g.
      `functools.partial(np.percentile, q=95)`) rather than a plain
      `lambda x: np.percentile(x, 95)`, which won't accept `axis=`.
    """
    try:
        float(statistic_func(sample))
    except Exception as exc:
        raise UsageError(
            "`statistic_func` must return a single scalar when called on "
            "a 1-D array, e.g. `statistic_func(before_values)`; calling it "
            f"that way raised {exc!r}."
        ) from exc

    batch = np.stack([sample, sample])
    try:
        statistic_func(batch, axis=1)
    except Exception as exc:
        raise UsageError(
            "`statistic_func` must accept an `axis=` keyword argument and "
            "apply row-wise, like `np.mean`/`np.median` do (this is what "
            "lets bootstrap resampling vectorize instead of looping in "
            "Python); bind extra arguments with `functools.partial` (e.g. "
            "`functools.partial(np.percentile, q=95)`) rather than a plain "
            f"lambda. Calling `statistic_func(array, axis=1)` raised {exc!r}."
        ) from exc


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
