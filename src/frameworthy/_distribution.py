"""Uncertainty estimation for the Wasserstein-1 (earth mover's) distance
between two samples, independent or paired.

Split out from `_intervals.py`: that module owns confidence intervals for a
*difference* statistic (`after - before`); this module owns a confidence
interval for a *distance* statistic, which is non-negative by construction
and needs a different resampling strategy as a result. The actual bootstrap
mechanics live in `_bootstrap.py` (`subsample_bootstrap_ci`); this module
just supplies the Wasserstein-specific pieces: which statistic to compute,
input validation with distribution-check-specific messages, and clipping
the resulting interval to `[0, inf)` since a distance can't be negative.

Why not the ordinary bootstrap (`_bootstrap.py`'s `bootstrap_diff_ci`) used
elsewhere in this library? The plug-in Wasserstein estimator
`W(before_hat, after_hat)` is biased upward in finite samples, and more
importantly, when the true distance is 0 or near it (exactly the "did the
distribution stay the same?" case this check exists for), the statistic
sits at a boundary of its parameter space where it isn't smoothly
differentiable. The ordinary n-out-of-n bootstrap (and BCa, which still
resamples at full size) is known to be *inconsistent* at such boundary/
non-smooth points: resampling the full-size dataset doesn't converge to
the right sampling distribution, so a naive bootstrap CI would be
unreliable exactly where correctness matters most. `subsample_bootstrap_ci`
implements the m-out-of-n/subsampling alternative instead, which remains
asymptotically valid under much weaker conditions, including at these
boundary/non-smooth points.
"""

import numpy as np
from scipy import stats

from ._bootstrap import subsample_bootstrap_ci
from ._constants import Interval
from ._validation import (
    validate_alpha,
    validate_equal_length,
    validate_min_observations,
    validate_n_resamples,
)

_MIN_OBSERVATIONS = 8


def wasserstein_distance_ci(
    before: np.ndarray,
    after: np.ndarray,
    *,
    paired: bool,
    alpha: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> Interval:
    """
    Estimate the Wasserstein-1 distance between `before` and `after` and an
    m-out-of-n bootstrap confidence interval for it.

    `before` and `after` may be treated as independent samples or, if
    `paired`, as correlated pairs (e.g. the same units measured before and
    after) -- see this module's docstring for why a resampling strategy
    other than the ordinary bootstrap is used, and `subsample_bootstrap_ci`
    for how `paired` changes the resampling itself. The Wasserstein
    distance point estimate doesn't depend on pairing (it's a function of
    the two marginal empirical distributions only), but its sampling
    variability does, so pairing only affects the width of the returned
    interval, not `distance` itself.

    Returns `(distance, ci_low, ci_high)` where `distance` is the observed
    `scipy.stats.wasserstein_distance(before, after)` and the interval is
    a `(1 - 2 * alpha)` one-sided-equivalent interval (mirroring the
    `alpha` convention used everywhere else in this library), clipped at
    0 since a distance can't be negative.
    """
    validate_alpha(alpha)
    validate_n_resamples(n_resamples)
    if paired:
        validate_equal_length(before, after, context="a paired distribution check")
        validate_min_observations(
            len(before),
            _MIN_OBSERVATIONS,
            context="paired observations for a distribution check",
        )
    else:
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

    distance, ci_low, ci_high = subsample_bootstrap_ci(
        before,
        after,
        stats.wasserstein_distance,
        paired=paired,
        alpha=alpha,
        n_resamples=n_resamples,
        rng=rng,
    )

    return Interval(distance, max(0.0, ci_low), ci_high)
