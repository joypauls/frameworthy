import numpy as np
from scipy import stats

from ._constants import DEFAULT_INFERENCE_METHOD, InferenceMethod
from .decision import Decision


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 0.5:
        raise ValueError(f"`alpha` must be in (0, 0.5), got {alpha}.")


def _validate_bootstrap_args(alpha: float, n_resamples: int) -> None:
    _validate_alpha(alpha)
    if n_resamples < 1:
        raise ValueError(f"`n_resamples` must be positive, got {n_resamples}.")


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
    """Bootstrap the mean difference `after - before` and its confidence interval.

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
    observed: float, se: float, df: float, alpha: float
) -> tuple[float, float, float]:
    """Build a `(1 - 2 * alpha)` t-interval around `observed` given its SE and df."""
    if se == 0:
        return observed, observed, observed

    margin = float(stats.t.ppf(1 - alpha, df)) * se
    return observed, observed - margin, observed + margin


def _paired_mean_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> tuple[float, float, float]:
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
    return _t_interval(float(diffs.mean()), se, df=n - 1, alpha=alpha)


def _independent_mean_diff_ci(
    before: np.ndarray, after: np.ndarray, *, alpha: float
) -> tuple[float, float, float]:
    """Welch's (unequal-variance) t-interval for two independent samples."""
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

    # Welch-Satterthwaite degrees of freedom
    df = (se_sq_before + se_sq_after) ** 2 / (
        se_sq_before**2 / (n_before - 1) + se_sq_after**2 / (n_after - 1)
    )
    return _t_interval(observed, se, df=df, alpha=alpha)


def analytical_mean_diff_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
) -> tuple[float, float, float]:
    """Analytically compute the mean difference `after - before` and its CI.

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
        return _paired_mean_diff_ci(before, after, alpha=alpha)
    return _independent_mean_diff_ci(before, after, alpha=alpha)


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
    """Select an inference strategy and compute the mean difference CI.

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


def classify_equivalence(ci_low: float, ci_high: float, within: float) -> Decision:
    """Classify a mean-difference CI against an equivalence margin.

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
