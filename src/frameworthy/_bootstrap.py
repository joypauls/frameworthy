"""Generic bootstrap confidence interval for a difference `after - before`.

Split out from `_intervals.py`: that module owns the analytical-vs-bootstrap
dispatch (`diff_ci`) shared by every built-in metric; this module owns the
actual bootstrap mechanics, delegating the resampling itself to
`scipy.stats.bootstrap` rather than maintaining a hand-rolled percentile
bootstrap.

`method="BCa"` (bias-corrected and accelerated) is used by default rather
than a plain percentile bootstrap: it corrects for both median bias and
skewness in the bootstrap distribution, which generally gives better
coverage than the percentile method without requiring any extra input from
the caller. See Efron & Tibshirani (1993), *An Introduction to the
Bootstrap*.

Note this is a different (and separate) concern from the Wasserstein/
distribution-check bootstrap in `_distribution.py`: that one specifically
needs an m-out-of-n/subsampling strategy because the Wasserstein distance
statistic is non-smooth at its zero boundary, which `scipy.stats.bootstrap`
doesn't implement, so it isn't routed through this module. `bootstrap_diff_ci`
here is for ordinary "difference of a statistic" claims like `.mean()`/
`.median()`/`.custom()`.
"""

import warnings
from collections.abc import Callable

import numpy as np
from scipy import stats

from ._constants import Interval
from ._validation import (
    validate_alpha,
    validate_equal_length,
    validate_min_observations,
    validate_n_resamples,
)

StatisticFunc = Callable[..., np.ndarray]


def bootstrap_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    statistic_func: StatisticFunc = np.mean,
) -> Interval:
    """
    Bootstrap the difference `after - before` and its confidence interval.

    Valid for any bounded array, including the 0/1 values `.rate()` checks
    resample directly (hence the generic name, rather than `*_mean_*`: this
    is the shared bootstrap fallback used by `diff_ci` for every metric,
    including `.mean()`/`.rate()`/`.median()` checks).

    `statistic_func` (default `np.mean`) is the statistic whose difference
    is bootstrapped; it must accept an `axis=` kwarg (e.g. `np.mean`/
    `np.median`) so it can be applied to every row of a resample matrix at
    once.

    If `paired`, `before` and `after` must be the same length and correspond
    element-wise; both sides are resampled by the same drawn indices,
    preserving before/after correlation (passed through to
    `scipy.stats.bootstrap` as `paired=True`). Otherwise, `before` and
    `after` are resampled independently, which is valid for unpaired/
    independent samples.

    Returns `(observed_diff, ci_low, ci_high)` where `observed_diff` is the
    plain (not bias-corrected) `statistic_func(after) - statistic_func(before)`
    and the interval is a `(1 - 2 * alpha)` BCa bootstrap CI.

    If the bootstrap distribution is degenerate (e.g. `statistic_func` is
    exactly constant across every resample -- this happens for a perfectly
    constant paired shift, or when `before`/`after` are identical), BCa's
    bias-correction/acceleration terms are undefined and `scipy.stats.
    bootstrap` returns a `nan` interval. Rather than propagating that `nan`,
    this falls back to a plain percentile interval of the same bootstrap
    distribution, which degrades gracefully to a zero-width interval at
    `observed_diff` in that degenerate case.
    """
    validate_alpha(alpha)
    validate_n_resamples(n_resamples)

    if paired:
        validate_equal_length(before, after, context="bootstrap")
        validate_min_observations(
            len(before), 2, context="paired observations to bootstrap"
        )
    else:
        validate_min_observations(
            min(len(before), len(after)),
            2,
            context="observations per side to bootstrap",
        )

    observed = float(statistic_func(after) - statistic_func(before))

    def statistic(b: np.ndarray, a: np.ndarray, axis: int | None = None) -> np.ndarray:
        return statistic_func(a, axis=axis) - statistic_func(b, axis=axis)

    with warnings.catch_warnings():
        # degenerate bootstrap distributions are handled explicitly below,
        # so scipy's warning about them would just be noise to the caller
        warnings.filterwarnings("ignore", category=stats.DegenerateDataWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = stats.bootstrap(
            (before, after),
            statistic,
            paired=paired,
            n_resamples=n_resamples,
            confidence_level=1 - 2 * alpha,
            method="BCa",
            rng=rng,
            vectorized=True,
        )

    ci_low, ci_high = result.confidence_interval
    if np.isnan(ci_low) or np.isnan(ci_high):
        ci_low, ci_high = np.percentile(
            result.bootstrap_distribution, [100 * alpha, 100 * (1 - alpha)]
        )

    return Interval(observed, float(ci_low), float(ci_high))
