"""Classification of a computed confidence interval into a `Decision`.

Split out from `_stats.py`: `_intervals.py` owns interval *computation*;
this module owns turning a computed interval into a `Decision`.
"""

from ._constants import Direction
from ._errors import InvalidParameterError
from .decision import Decision


def classify_equivalence(ci_low: float, ci_high: float, within: float) -> Decision:
    """
    Classify a mean-difference CI against an equivalence margin.

    * `equivalent`: the whole CI lies inside `(-within, within)`.
    * `changed`: the whole CI lies outside `(-within, within)`, i.e. it
      doesn't even touch the margin.
    * `inconclusive`: the CI straddles a margin boundary.
    """
    if within <= 0:
        raise InvalidParameterError(f"`within` must be positive, got {within}.")

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
    raise InvalidParameterError(f"Unknown `direction`: {direction!r}.")
