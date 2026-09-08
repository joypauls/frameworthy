"""Confidence-interval estimators for a difference `after - before`.

Split out from `_stats.py`: this module owns interval *computation*
(Wilson/bootstrap/analytical mean and rate estimators, plus the
`mean_diff_ci`/`rate_diff_ci` dispatchers); `_classify.py` owns turning a
computed interval into a `Decision`.
"""

from collections.abc import Callable

import numpy as np
from scipy import stats

from ._constants import DEFAULT_INFERENCE_METHOD, InferenceMethod, Interval
from ._errors import UsageError
from ._validation import (
    validate_alpha,
    validate_equal_length,
    validate_method,
    validate_min_observations,
    validate_n_resamples,
)


def wilson_interval(count: int, n: int, alpha: float) -> tuple[float, float]:
    """
    Wilson score confidence interval for a single binomial proportion.

    `alpha` is interpreted the same way as elsewhere in this module: the
    returned interval is `(1 - 2 * alpha)` two-sided-equivalent, i.e. it's
    passed to scipy as `confidence_level = 1 - 2 * alpha`.

    Returns `(low, high)`.
    """
    validate_alpha(alpha)
    if n < 1:
        raise UsageError(f"`n` must be positive, got {n}.")
    if not 0 <= count <= n:
        raise UsageError(f"`count` must be in [0, {n}], got {count}.")

    ci = stats.binomtest(count, n).proportion_ci(
        confidence_level=1 - 2 * alpha, method="wilson"
    )
    return float(ci.low), float(ci.high)


def _bootstrap_paired_diffs(
    before: np.ndarray,
    after: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    validate_equal_length(before, after, context="bootstrap")
    validate_min_observations(
        len(before), 2, context="paired observations to bootstrap"
    )

    diffs = after - before
    observed = float(diffs.mean())
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    boot_diffs = diffs[idx].mean(axis=1)

    return observed, boot_diffs


def _bootstrap_unpaired_diffs(
    before: np.ndarray,
    after: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    validate_min_observations(
        min(len(before), len(after)), 2, context="observations per side to bootstrap"
    )

    observed = float(after.mean() - before.mean())
    before_idx = rng.integers(0, len(before), size=(n_resamples, len(before)))
    after_idx = rng.integers(0, len(after), size=(n_resamples, len(after)))
    boot_diffs = after[after_idx].mean(axis=1) - before[before_idx].mean(axis=1)

    return observed, boot_diffs


def bootstrap_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> Interval:
    """
    Bootstrap the difference `after - before` and its confidence interval.

    Valid for any bounded array, including the 0/1 values `.rate()` checks
    resample directly (hence the generic name, rather than `*_mean_*`: this
    is the shared bootstrap fallback for both `mean_diff_ci` and
    `rate_diff_ci`).

    If `paired`, `before` and `after` must be the same length and correspond
    element-wise; the diffs are resampled together. Otherwise, `before` and
    `after` are resampled independently, which is valid for unpaired/
    independent samples.

    Returns `(observed_diff, ci_low, ci_high)` where the interval is the
    `(1 - 2 * alpha)` percentile bootstrap CI.
    """
    validate_alpha(alpha)
    validate_n_resamples(n_resamples)

    if paired:
        observed, boot_diffs = _bootstrap_paired_diffs(
            before, after, n_resamples=n_resamples, rng=rng
        )
    else:
        observed, boot_diffs = _bootstrap_unpaired_diffs(
            before, after, n_resamples=n_resamples, rng=rng
        )

    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha, 100 * (1 - alpha)])
    return Interval(observed, float(ci_low), float(ci_high))


def _t_interval(observed: float, se: float, deg_f: float, alpha: float) -> Interval:
    """Build a `(1 - 2 * alpha)` t-interval around `observed` given its SE and df."""
    if se == 0:
        return Interval(observed, observed, observed)

    margin = float(stats.t.ppf(1 - alpha, deg_f)) * se
    return Interval(observed, observed - margin, observed + margin)


def paired_mean_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> Interval:
    """Paired t-interval for the mean of the within-pair differences
    `after - before`.

    `before` and `after` must be the same length and correspond
    element-wise. Returns `(observed_diff, ci_low, ci_high)` where the
    interval is the `(1 - 2 * alpha)` confidence interval.
    """
    validate_equal_length(before, after, context="comparison")
    validate_min_observations(
        len(before),
        2,
        context="paired observations for an analytical confidence interval",
    )

    diffs = after - before
    n = len(diffs)
    se = float(diffs.std(ddof=1)) / np.sqrt(n)
    return _t_interval(float(diffs.mean()), se, deg_f=n - 1, alpha=alpha)


def independent_mean_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> Interval:
    """Welch's (unequal-variance) t-interval for two independent samples.

    Returns `(observed_diff, ci_low, ci_high)` where `observed_diff` is
    `after.mean() - before.mean()` and the interval is the `(1 - 2 * alpha)`
    confidence interval.
    """
    validate_min_observations(
        min(len(before), len(after)),
        2,
        context="observations per side for an analytical confidence interval",
    )

    n_before, n_after = len(before), len(after)
    se_sq_before = float(before.var(ddof=1)) / n_before
    se_sq_after = float(after.var(ddof=1)) / n_after
    se = float(np.sqrt(se_sq_before + se_sq_after))
    observed = float(after.mean() - before.mean())
    if se == 0:
        return Interval(observed, observed, observed)

    # welch-satterthwaite degrees of freedom
    deg_f = (se_sq_before + se_sq_after) ** 2 / (
        se_sq_before**2 / (n_before - 1) + se_sq_after**2 / (n_after - 1)
    )
    return _t_interval(observed, se, deg_f=deg_f, alpha=alpha)


def analytical_mean_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
) -> Interval:
    """
    Analytically compute the mean difference `after - before` and its CI.

    If `paired`, `before` and `after` must be the same length and correspond
    element-wise; the interval is a t-interval for the mean of the
    within-pair differences. Otherwise, `before` and `after` are treated as
    independent samples and the interval uses Welch's approximation
    (unequal-variance t-interval).

    Returns `(observed_diff, ci_low, ci_high)` where the interval is the
    `(1 - 2 * alpha)` confidence interval.
    """
    validate_alpha(alpha)

    if paired:
        return paired_mean_diff_ci(before, after, alpha=alpha)
    return independent_mean_diff_ci(before, after, alpha=alpha)


def independent_rate_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> Interval:
    """
    Newcombe's hybrid score ("MOVER") CI for the difference of two
    independent binomial proportions.

    `before` and `after` must be 0/1 arrays. Rather than using the sample
    variance of the raw values (which is what treating this as a plain
    mean-difference problem would do), this combines Wilson score
    intervals for the two proportions individually, which avoids the
    zero-width degenerate interval that a naive Wald/t-based approach
    produces when a sample's observed rate is exactly 0 or 1.

    Returns `(observed_diff, ci_low, ci_high)` where `observed_diff` is
    `after.mean() - before.mean()` and the interval is the `(1 - 2 * alpha)`
    confidence interval.
    """
    validate_alpha(alpha)
    validate_min_observations(
        min(len(before), len(after)),
        2,
        context="observations per side for an analytical confidence interval",
    )

    n_before, n_after = len(before), len(after)
    p_before = float(before.mean())
    p_after = float(after.mean())
    diff = p_after - p_before

    l_before, u_before = wilson_interval(int(before.sum()), n_before, alpha)
    l_after, u_after = wilson_interval(int(after.sum()), n_after, alpha)

    low = diff - np.sqrt((p_after - l_after) ** 2 + (u_before - p_before) ** 2)
    high = diff + np.sqrt((u_after - p_after) ** 2 + (p_before - l_before) ** 2)
    return Interval(diff, float(low), float(high))


def _paired_phi(before: np.ndarray, after: np.ndarray) -> float:
    """Newcombe's (1998) adjusted correlation coefficient for a 2x2 paired
    (before, after) table, used to account for the within-pair correlation
    when combining the before/after Wilson intervals.

    Returns `0.0` if any of the table's margins are zero (no information to
    estimate a correlation from).
    """
    n = len(before)
    r = float(np.sum((before == 1) & (after == 1)))
    s = float(np.sum((before == 0) & (after == 1)))
    t = float(np.sum((before == 1) & (after == 0)))
    u = float(np.sum((before == 0) & (after == 0)))

    margins_product = (r + s) * (t + u) * (r + t) * (s + u)
    if margins_product <= 0:
        return 0.0

    b = r * u - s * t
    if b > n / 2:
        c = b - n / 2
    elif b < 0:
        c = b
    else:
        c = 0.0

    return c / np.sqrt(margins_product)


def paired_rate_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> Interval:
    """Newcombe's (1998) score-based CI for the difference between two
    paired/correlated binomial proportions.

    `before` and `after` must be 0/1 arrays of equal length, corresponding
    element-wise. Like `independent_rate_diff_ci`, this combines Wilson
    score intervals for the before/after proportions rather than the
    sample variance of the paired 0/1 differences, and additionally
    corrects for the within-pair correlation via `_paired_phi`. See
    Newcombe, R.G. (1998), "Improved confidence intervals for the
    difference between binomial proportions based on paired data,"
    Statistics in Medicine 17(22).

    Returns `(observed_diff, ci_low, ci_high)` where `observed_diff` is
    `after.mean() - before.mean()` and the interval is the `(1 - 2 * alpha)`
    confidence interval.
    """
    validate_alpha(alpha)
    validate_equal_length(before, after, context="comparison")
    validate_min_observations(
        len(before),
        2,
        context="paired observations for an analytical confidence interval",
    )

    n = len(before)
    p_before = float(before.mean())
    p_after = float(after.mean())
    diff = p_after - p_before

    l_before, u_before = wilson_interval(int(before.sum()), n, alpha)
    l_after, u_after = wilson_interval(int(after.sum()), n, alpha)
    phi = _paired_phi(before, after)

    # each radicand is non-negative by construction given a valid
    # correlation `phi`, but clamp anyway as a floating-point safety net
    # against a borderline case producing a tiny negative value and
    # silently propagating `nan`
    low_radicand = (
        (p_after - l_after) ** 2
        - 2 * phi * (p_after - l_after) * (u_before - p_before)
        + (u_before - p_before) ** 2
    )
    high_radicand = (
        (p_before - l_before) ** 2
        - 2 * phi * (p_before - l_before) * (u_after - p_after)
        + (u_after - p_after) ** 2
    )
    low = diff - np.sqrt(max(low_radicand, 0.0))
    high = diff + np.sqrt(max(high_radicand, 0.0))
    return Interval(diff, float(low), float(high))


def analytical_rate_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
) -> Interval:
    """Analytically compute the rate (proportion) difference `after - before`
    and its CI, dispatching to the paired or independent Newcombe/Wilson
    formula.

    If `paired`, `before` and `after` must be the same length and
    correspond element-wise. Otherwise, `before` and `after` are treated as
    independent samples.

    Returns `(observed_diff, ci_low, ci_high)` where the interval is the
    `(1 - 2 * alpha)` confidence interval.

    `alpha` is validated by whichever of `paired_rate_diff_ci`/
    `independent_rate_diff_ci` this dispatches to, so it isn't re-validated
    here.
    """
    if paired:
        return paired_rate_diff_ci(before, after, alpha=alpha)
    return independent_rate_diff_ci(before, after, alpha=alpha)


AnalyticalDiffFunc = Callable[..., Interval]


def diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    analytical_fn: AnalyticalDiffFunc,
    method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
) -> Interval:
    """
    Select an inference strategy and compute a difference CI.

    This is the single dispatch point shared by every built-in metric: the
    analytical fast path (`analytical_fn`, e.g. `analytical_mean_diff_ci` or
    `analytical_rate_diff_ci`) versus the generic bootstrap fallback
    (`bootstrap_diff_ci`), which works for any metric but doesn't get the
    boundary-case benefits of a metric-specific analytical estimator.

    A new metric only needs its own `analytical_fn`; it can reuse this
    dispatcher directly rather than re-implementing `mean_diff_ci`/
    `rate_diff_ci`'s dispatch logic from scratch.
    """
    validate_method(method)
    if method == "analytical":
        return analytical_fn(before, after, paired=paired, alpha=alpha)
    return bootstrap_diff_ci(
        before,
        after,
        paired=paired,
        alpha=alpha,
        n_resamples=n_resamples,
        rng=rng,
    )


def mean_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
) -> Interval:
    """
    Select an inference strategy and compute the mean difference CI.

    Thin wrapper around `diff_ci` bound to `analytical_mean_diff_ci`: the
    analytical path is a t-interval (paired) or Welch's t-interval
    (independent), used by default for built-in `.mean()` checks.
    """
    return diff_ci(
        before,
        after,
        paired=paired,
        alpha=alpha,
        n_resamples=n_resamples,
        rng=rng,
        analytical_fn=analytical_mean_diff_ci,
        method=method,
    )


def rate_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
) -> Interval:
    """
    Select an inference strategy and compute the rate difference CI.

    Thin wrapper around `diff_ci` bound to `analytical_rate_diff_ci`: the
    analytical path uses the Newcombe/Wilson score-based CIs above, the
    default for built-in `.rate()` checks.
    """
    return diff_ci(
        before,
        after,
        paired=paired,
        alpha=alpha,
        n_resamples=n_resamples,
        rng=rng,
        analytical_fn=analytical_rate_diff_ci,
        method=method,
    )
