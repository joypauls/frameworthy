from typing import Literal

import numpy as np
from scipy import stats

from ._constants import DEFAULT_INFERENCE_METHOD, InferenceMethod
from .decision import Decision

Direction = Literal["greater_than", "less_than"]


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 0.5:
        raise ValueError(f"`alpha` must be in (0, 0.5), got {alpha}.")


def _validate_bootstrap_args(alpha: float, n_resamples: int) -> None:
    _validate_alpha(alpha)
    if n_resamples < 1:
        raise ValueError(f"`n_resamples` must be positive, got {n_resamples}.")


def wilson_interval(count: int, n: int, alpha: float) -> tuple[float, float]:
    """
    Wilson score confidence interval for a single binomial proportion.

    `alpha` is interpreted the same way as elsewhere in this module: the
    returned interval is `(1 - 2 * alpha)` two-sided-equivalent, i.e. it's
    passed to scipy as `confidence_level = 1 - 2 * alpha`.

    Returns `(low, high)`.
    """
    _validate_alpha(alpha)
    if n < 1:
        raise ValueError(f"`n` must be positive, got {n}.")
    if not 0 <= count <= n:
        raise ValueError(f"`count` must be in [0, {n}], got {count}.")

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
    if len(before) != len(after):
        raise ValueError(
            "Paired bootstrap requires `before` and `after` to have the "
            f"same length, got {len(before)} and {len(after)}."
        )
    if len(before) < 2:
        raise ValueError("At least 2 paired observations are required to bootstrap.")

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
    if len(before) < 2 or len(after) < 2:
        raise ValueError("At least 2 observations per side are required to bootstrap.")

    observed = float(after.mean() - before.mean())
    before_idx = rng.integers(0, len(before), size=(n_resamples, len(before)))
    after_idx = rng.integers(0, len(after), size=(n_resamples, len(after)))
    boot_diffs = after[after_idx].mean(axis=1) - before[before_idx].mean(axis=1)

    return observed, boot_diffs


def bootstrap_mean_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Bootstrap the mean difference `after - before` and its confidence interval.

    If `paired`, `before` and `after` must be the same length and correspond
    element-wise; the diffs are resampled together. Otherwise, `before` and
    `after` are resampled independently, which is valid for unpaired/
    independent samples.

    Returns `(observed_diff, ci_low, ci_high)` where the interval is the
    `(1 - 2 * alpha)` percentile bootstrap CI.
    """
    _validate_bootstrap_args(alpha, n_resamples)

    if paired:
        observed, boot_diffs = _bootstrap_paired_diffs(
            before, after, n_resamples=n_resamples, rng=rng
        )
    else:
        observed, boot_diffs = _bootstrap_unpaired_diffs(
            before, after, n_resamples=n_resamples, rng=rng
        )

    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha, 100 * (1 - alpha)])
    return observed, float(ci_low), float(ci_high)


def _t_interval(
    observed: float, se: float, deg_f: float, alpha: float
) -> tuple[float, float, float]:
    """Build a `(1 - 2 * alpha)` t-interval around `observed` given its SE and df."""
    if se == 0:
        return observed, observed, observed

    margin = float(stats.t.ppf(1 - alpha, deg_f)) * se
    return observed, observed - margin, observed + margin


def paired_mean_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> tuple[float, float, float]:
    """Paired t-interval for the mean of the within-pair differences
    `after - before`.

    `before` and `after` must be the same length and correspond
    element-wise. Returns `(observed_diff, ci_low, ci_high)` where the
    interval is the `(1 - 2 * alpha)` confidence interval.
    """
    if len(before) != len(after):
        raise ValueError(
            "Paired comparison requires `before` and `after` to have the "
            f"same length, got {len(before)} and {len(after)}."
        )
    if len(before) < 2:
        raise ValueError(
            "At least 2 paired observations are required for an analytical "
            "confidence interval."
        )

    diffs = after - before
    n = len(diffs)
    se = float(diffs.std(ddof=1)) / np.sqrt(n)
    return _t_interval(float(diffs.mean()), se, deg_f=n - 1, alpha=alpha)


def independent_mean_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> tuple[float, float, float]:
    """Welch's (unequal-variance) t-interval for two independent samples.

    Returns `(observed_diff, ci_low, ci_high)` where `observed_diff` is
    `after.mean() - before.mean()` and the interval is the `(1 - 2 * alpha)`
    confidence interval.
    """
    if len(before) < 2 or len(after) < 2:
        raise ValueError(
            "At least 2 observations per side are required for an analytical "
            "confidence interval."
        )

    n_before, n_after = len(before), len(after)
    se_sq_before = float(before.var(ddof=1)) / n_before
    se_sq_after = float(after.var(ddof=1)) / n_after
    se = float(np.sqrt(se_sq_before + se_sq_after))
    observed = float(after.mean() - before.mean())
    if se == 0:
        return observed, observed, observed

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
) -> tuple[float, float, float]:
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
    _validate_alpha(alpha)

    if paired:
        return paired_mean_diff_ci(before, after, alpha=alpha)
    return independent_mean_diff_ci(before, after, alpha=alpha)


def mean_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
) -> tuple[float, float, float]:
    """
    Select an inference strategy and compute the mean difference CI.

    This is the single dispatch point between the analytical fast path
    (used by default for built-in `.mean()` checks) and the bootstrap
    fallback (kept available for future, arbitrary metrics that don't have
    a closed-form interval).
    """
    if method == "analytical":
        return analytical_mean_diff_ci(before, after, paired=paired, alpha=alpha)
    if method == "bootstrap":
        return bootstrap_mean_diff_ci(
            before,
            after,
            paired=paired,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=rng,
        )
    raise ValueError(f"Unknown inference `method`: {method!r}.")


def independent_rate_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> tuple[float, float, float]:
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
    _validate_alpha(alpha)
    if len(before) < 2 or len(after) < 2:
        raise ValueError(
            "At least 2 observations per side are required for an analytical "
            "confidence interval."
        )

    n_before, n_after = len(before), len(after)
    p_before = float(before.mean())
    p_after = float(after.mean())
    diff = p_after - p_before

    l_before, u_before = wilson_interval(int(before.sum()), n_before, alpha)
    l_after, u_after = wilson_interval(int(after.sum()), n_after, alpha)

    low = diff - np.sqrt((p_after - l_after) ** 2 + (u_before - p_before) ** 2)
    high = diff + np.sqrt((u_after - p_after) ** 2 + (p_before - l_before) ** 2)
    return diff, float(low), float(high)


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
) -> tuple[float, float, float]:
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
    _validate_alpha(alpha)
    if len(before) != len(after):
        raise ValueError(
            "Paired comparison requires `before` and `after` to have the "
            f"same length, got {len(before)} and {len(after)}."
        )
    if len(before) < 2:
        raise ValueError(
            "At least 2 paired observations are required for an analytical "
            "confidence interval."
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
    return diff, float(low), float(high)


def analytical_rate_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
) -> tuple[float, float, float]:
    """Analytically compute the rate (proportion) difference `after - before`
    and its CI, dispatching to the paired or independent Newcombe/Wilson
    formula.

    If `paired`, `before` and `after` must be the same length and
    correspond element-wise. Otherwise, `before` and `after` are treated as
    independent samples.

    Returns `(observed_diff, ci_low, ci_high)` where the interval is the
    `(1 - 2 * alpha)` confidence interval.
    """
    _validate_alpha(alpha)

    if paired:
        return paired_rate_diff_ci(before, after, alpha=alpha)
    return independent_rate_diff_ci(before, after, alpha=alpha)


def rate_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
    method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
) -> tuple[float, float, float]:
    """Select an inference strategy and compute the rate difference CI.

    Mirrors `mean_diff_ci`'s dispatch: `"analytical"` uses the
    Newcombe/Wilson score-based CIs above (the default for built-in
    `.rate()` checks), while `"bootstrap"` falls back to the generic
    percentile bootstrap of the raw (0/1) values, which is valid for any
    bounded array but doesn't get the boundary-case benefits of the Wilson
    interval.
    """
    if method == "analytical":
        return analytical_rate_diff_ci(before, after, paired=paired, alpha=alpha)
    if method == "bootstrap":
        return bootstrap_mean_diff_ci(
            before,
            after,
            paired=paired,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=rng,
        )
    raise ValueError(f"Unknown inference `method`: {method!r}.")


def classify_equivalence(ci_low: float, ci_high: float, within: float) -> Decision:
    """
    Classify a mean-difference CI against an equivalence margin.

    * `equivalent`: the whole CI lies inside `(-within, within)`.
    * `changed`: the whole CI lies outside `(-within, within)`, i.e. it
      doesn't even touch the margin.
    * `inconclusive`: the CI straddles a margin boundary.
    """
    if within <= 0:
        raise ValueError(f"`within` must be positive, got {within}.")

    if -within <= ci_low and ci_high <= within:
        return Decision.EQUIVALENT
    if ci_high < -within or ci_low > within:
        return Decision.CHANGED
    return Decision.INCONCLUSIVE


def classify_change_bound(
    ci_low: float, ci_high: float, threshold: float, *, direction: Direction
) -> Decision:
    """
    Classify a difference CI against a one-sided change threshold.

    `ci_low` and `ci_high` come from the same `(1 - 2 * alpha)` two-sided CI
    used by equivalence checks; each endpoint on its own is also a valid
    `(1 - alpha)` one-sided confidence bound, which is what makes this a
    statistically valid one-sided decision rule without any new interval math.

    * `direction="greater_than"` (ruling out a drop below `threshold`):
      `PASSED` if `ci_low > threshold`, `FAILED` if `ci_high < threshold`,
      `INCONCLUSIVE` otherwise.
    * `direction="less_than"` (ruling out a rise above `threshold`):
      `PASSED` if `ci_high < threshold`, `FAILED` if `ci_low > threshold`,
      `INCONCLUSIVE` otherwise.
    """
    if direction == "greater_than":
        if ci_low > threshold:
            return Decision.PASSED
        if ci_high < threshold:
            return Decision.FAILED
        return Decision.INCONCLUSIVE
    if direction == "less_than":
        if ci_high < threshold:
            return Decision.PASSED
        if ci_low > threshold:
            return Decision.FAILED
        return Decision.INCONCLUSIVE
    raise ValueError(f"Unknown `direction`: {direction!r}.")
