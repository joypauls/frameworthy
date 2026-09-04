from typing import Literal

import numpy as np

Decision = Literal["equivalent", "changed", "inconclusive"]


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
    if not 0 < alpha < 0.5:
        raise ValueError(f"`alpha` must be in (0, 0.5), got {alpha}.")
    if n_resamples < 1:
        raise ValueError(f"`n_resamples` must be positive, got {n_resamples}.")

    if paired:
        if len(before) != len(after):
            raise ValueError(
                "Paired bootstrap requires `before` and `after` to have the "
                f"same length, got {len(before)} and {len(after)}."
            )
        if len(before) < 2:
            raise ValueError(
                "At least 2 paired observations are required to bootstrap."
            )

        diffs = after - before
        observed = float(diffs.mean())
        idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
        boot_diffs = diffs[idx].mean(axis=1)

    else:
        if len(before) < 2 or len(after) < 2:
            raise ValueError(
                "At least 2 observations per side are required to bootstrap."
            )

        observed = float(after.mean() - before.mean())
        before_idx = rng.integers(0, len(before), size=(n_resamples, len(before)))
        after_idx = rng.integers(0, len(after), size=(n_resamples, len(after)))
        boot_diffs = after[after_idx].mean(axis=1) - before[before_idx].mean(axis=1)

    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha, 100 * (1 - alpha)])
    return observed, float(ci_low), float(ci_high)


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
        return "equivalent"
    if ci_high < -within or ci_low > within:
        return "changed"
    return "inconclusive"
