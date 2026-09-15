"""Uncertainty estimation for the Wasserstein-1 (earth mover's) distance
between two independent samples.

Split out from `_intervals.py`: that module owns confidence intervals for a
*difference* statistic (`after - before`); this module owns a confidence
interval for a *distance* statistic, which is non-negative by construction
and needs a different resampling strategy as a result.

Why not the ordinary (percentile) bootstrap used elsewhere in this
library? The plug-in Wasserstein estimator `W(before_hat, after_hat)` is
biased upward in finite samples, and more importantly, when the true
distance is 0 or near it (exactly the "did the distribution stay the
same?" case this check exists for), the statistic sits at a boundary of
its parameter space where it isn't smoothly differentiable. The ordinary
n-out-of-n bootstrap is known to be *inconsistent* at such boundary/
non-smooth points: resampling the full-size dataset doesn't converge to
the right sampling distribution, so a naive percentile bootstrap CI would
be unreliable exactly where correctness matters most.

Subsampling (Politis & Romano, 1994) and the closely related m-out-of-n
bootstrap remain asymptotically valid under much weaker conditions,
including at these boundary/non-smooth points, provided the subsample
size `m` grows with the full sample size `n` but at a slower rate
(`m / n -> 0`). This module implements an m-out-of-n bootstrap: resample
smaller subsamples (with replacement) from each side, rescale their
distribution of the statistic to approximate the full-sample sampling
distribution, and invert that to get a confidence interval for the
full-sample estimate.
"""

import numpy as np
from scipy import stats

from ._constants import Interval
from ._validation import validate_alpha, validate_min_observations, validate_n_resamples

# subsample size grows like n**_SUBSAMPLE_EXPONENT: slower than n (so
# m / n -> 0, required for subsampling/m-out-of-n consistency) but still
# growing without bound as n -> infinity
_SUBSAMPLE_EXPONENT = 0.6
_MIN_OBSERVATIONS = 8
_MIN_SUBSAMPLE_SIZE = 2


def _subsample_size(n: int) -> int:
    return max(_MIN_SUBSAMPLE_SIZE, int(n**_SUBSAMPLE_EXPONENT))


def wasserstein_distance_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> Interval:
    """
    Estimate the Wasserstein-1 distance between `before` and `after` and an
    m-out-of-n bootstrap confidence interval for it.

    `before` and `after` are treated as independent samples (there is no
    paired mode for a distribution check. See this module's docstring
    for why a resampling strategy other than the ordinary percentile
    bootstrap is used).

    Returns `(distance, ci_low, ci_high)` where `distance` is the observed
    `scipy.stats.wasserstein_distance(before, after)` and the interval is
    a `(1 - 2 * alpha)` one-sided-equivalent interval (mirroring the
    `alpha` convention used everywhere else in this library), clipped at
    0 since a distance can't be negative.
    """
    validate_alpha(alpha)
    validate_n_resamples(n_resamples)
    validate_min_observations(
        len(before),
        _MIN_OBSERVATIONS,
        context="before observations for a distribution check",
    )
    validate_min_observations(
        len(after),
        _MIN_OBSERVATIONS,
        context="after observations for a distribution check",
    )

    n_before, n_after = len(before), len(after)
    distance = float(stats.wasserstein_distance(before, after))

    m_before = _subsample_size(n_before)
    m_after = _subsample_size(n_after)
    n_eff = (n_before * n_after) / (n_before + n_after)
    m_eff = (m_before * m_after) / (m_before + m_after)

    before_idx = rng.integers(0, n_before, size=(n_resamples, m_before))
    after_idx = rng.integers(0, n_after, size=(n_resamples, m_after))

    subsample_distances = np.array(
        [
            stats.wasserstein_distance(before[before_idx[i]], after[after_idx[i]])
            for i in range(n_resamples)
        ]
    )

    deviations = np.sqrt(m_eff) * (subsample_distances - distance)
    q_low, q_high = np.percentile(deviations, [100 * alpha, 100 * (1 - alpha)])

    ci_low = max(0.0, distance - q_high / np.sqrt(n_eff))
    ci_high = distance - q_low / np.sqrt(n_eff)

    return Interval(distance, float(ci_low), float(ci_high))
