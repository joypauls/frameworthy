"""Bootstrap confidence intervals for two-sample statistics.

Split out from `_intervals.py`/`_distribution.py`: those modules own
*what* is being estimated (a difference `after - before`, or a Wasserstein
distance) and any domain-specific validation/dispatch; this module owns
the actual bootstrap mechanics, so there's exactly one place that knows how
to resample. It provides two independent strategies, each suited to a
different kind of statistic:

* `bootstrap_diff_ci`: an ordinary bootstrap CI for the difference of a
  statistic (e.g. `.mean()`/`.median()`/`.custom()`), delegating the
  resampling itself to `scipy.stats.bootstrap` with `method="BCa"`
  (bias-corrected and accelerated) rather than maintaining a hand-rolled
  percentile bootstrap. BCa corrects for both median bias and skewness in
  the bootstrap distribution, which generally gives better coverage than
  the percentile method without requiring any extra input from the caller.
  See Efron & Tibshirani (1993), *An Introduction to the Bootstrap*.

* `subsample_bootstrap_ci`: an m-out-of-n/subsampling bootstrap (Politis &
  Romano, 1994) for statistics that are non-negative and/or non-smooth at
  a boundary of their parameter space (e.g. a distance that's exactly 0
  when two distributions are identical), where the ordinary n-out-of-n
  bootstrap `scipy.stats.bootstrap` implements is known to be
  *inconsistent*. Used by `_distribution.py`'s Wasserstein distance check;
  see that module for why.
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
TwoSampleStatisticFunc = Callable[[np.ndarray, np.ndarray], float]

# subsample size grows like n**_SUBSAMPLE_EXPONENT: slower than n (so
# m / n -> 0, required for subsampling/m-out-of-n consistency) but still
# growing without bound as n -> infinity
_DEFAULT_SUBSAMPLE_EXPONENT = 0.6
_DEFAULT_MIN_SUBSAMPLE_SIZE = 2


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


def _subsample_size(n: int, *, exponent: float, min_size: int) -> int:
    return max(min_size, int(n**exponent))


def subsample_bootstrap_ci(
    before: np.ndarray,
    after: np.ndarray,
    statistic_func: TwoSampleStatisticFunc,
    *,
    paired: bool = False,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    subsample_exponent: float = _DEFAULT_SUBSAMPLE_EXPONENT,
    min_subsample_size: int = _DEFAULT_MIN_SUBSAMPLE_SIZE,
) -> Interval:
    """
    m-out-of-n (subsampling) bootstrap CI for a two-sample statistic.

    Unlike `bootstrap_diff_ci`, this doesn't assume `statistic_func` is a
    difference of a per-side summary, or that it's smooth/well-behaved
    everywhere in its parameter space: `statistic_func(before, after)` can
    be any scalar-valued function of the two full samples (e.g. a distance
    metric), which is also why this resamples smaller subsamples of size
    `m < n` from each side rather than full-size (`n`-out-of-`n`)
    resamples. That's what makes this valid at boundary/non-smooth points
    where the ordinary bootstrap (including `bootstrap_diff_ci`'s BCa
    method) is inconsistent -- see Politis & Romano (1994) -- provided `m`
    grows with `n` but at a slower rate (`m / n -> 0`, guaranteed here by
    `subsample_exponent < 1`).

    If `paired`, `before` and `after` must be the same length and
    correspond element-wise: each resample draws one shared index array
    and applies it to both sides (`before[idx]`, `after[idx]`), preserving
    before/after correlation the same way `bootstrap_diff_ci`'s
    `paired=True` does. The resampling unit is then a pair rather than an
    independent before/after observation, so the scaling that turns
    subsample deviations into a CI also switches from the two-independent
    -sample harmonic-mean formula (`n_eff`/`m_eff` combining `n_before`/
    `n_after`) to the single-sample analog (`n_eff = n`, `m_eff = m`,
    where `n` is the shared pair count). Otherwise `before` and `after`
    are resampled independently, valid for unpaired/independent samples.

    Returns `(statistic, ci_low, ci_high)` where `statistic` is the
    observed `statistic_func(before, after)` and the interval is a
    `(1 - 2 * alpha)` one-sided-equivalent interval (mirroring the `alpha`
    convention used everywhere else in this library). This does *not*
    clip the interval to any particular domain (e.g. non-negativity) --
    that's the caller's responsibility, since it depends on what
    `statistic_func` actually measures.
    """
    validate_alpha(alpha)
    validate_n_resamples(n_resamples)

    n_before, n_after = len(before), len(after)
    statistic = float(statistic_func(before, after))

    if paired:
        validate_equal_length(before, after, context="subsampling bootstrap")
        m = _subsample_size(
            n_before, exponent=subsample_exponent, min_size=min_subsample_size
        )
        n_eff = n_before
        m_eff = m
        idx = rng.integers(0, n_before, size=(n_resamples, m))

        subsample_statistics = np.array(
            [statistic_func(before[idx[i]], after[idx[i]]) for i in range(n_resamples)]
        )
    else:
        m_before = _subsample_size(
            n_before, exponent=subsample_exponent, min_size=min_subsample_size
        )
        m_after = _subsample_size(
            n_after, exponent=subsample_exponent, min_size=min_subsample_size
        )
        n_eff = (n_before * n_after) / (n_before + n_after)
        m_eff = (m_before * m_after) / (m_before + m_after)

        before_idx = rng.integers(0, n_before, size=(n_resamples, m_before))
        after_idx = rng.integers(0, n_after, size=(n_resamples, m_after))

        subsample_statistics = np.array(
            [
                statistic_func(before[before_idx[i]], after[after_idx[i]])
                for i in range(n_resamples)
            ]
        )

    deviations = np.sqrt(m_eff) * (subsample_statistics - statistic)
    q_low, q_high = np.percentile(deviations, [100 * alpha, 100 * (1 - alpha)])

    ci_low = statistic - q_high / np.sqrt(n_eff)
    ci_high = statistic - q_low / np.sqrt(n_eff)

    return Interval(statistic, float(ci_low), float(ci_high))
