import warnings
from dataclasses import dataclass

from ._constants import Direction, Statistic
from ._errors import FrameworthyAssertionError
from .decision import Decision


@dataclass(frozen=True)
class EquivalenceResult:
    """Result of a paired or unpaired equivalence check.

    Instances are returned by `.equivalent(...)` and are not meant to be
    constructed directly.
    """

    decision: Decision
    column: str
    statistic: Statistic
    paired: bool
    before_mean: float
    after_mean: float
    diff: float
    ci_low: float
    ci_high: float
    alpha: float
    within: float
    n_before: int
    n_after: int
    n_resamples: int

    @property
    def passed(self) -> bool:
        """Whether the evidence supports equivalence within the margin."""
        return self.decision == Decision.EQUIVALENT

    def __str__(self) -> str:
        ci_pct = round((1 - 2 * self.alpha) * 100)
        pairing = (
            f"paired, n={self.n_before}"
            if self.paired
            else f"unpaired, n_before={self.n_before}, n_after={self.n_after}"
        )

        if self.statistic == "rate":
            # report rates and their difference/margin in percentage points,
            # which reads more intuitively than raw proportions
            levels = f"before = {self.before_mean:.1%}, after = {self.after_mean:.1%}, "
            diff_str = f"{self.diff * 100:+.4g}pp"
            ci_str = f"[{self.ci_low * 100:+.4g}pp, {self.ci_high * 100:+.4g}pp]"
            margin_str = f"±{self.within * 100:g}pp"
        else:
            levels = ""
            diff_str = f"{self.diff:+.4g}"
            ci_str = f"[{self.ci_low:+.4g}, {self.ci_high:+.4g}]"
            margin_str = f"±{self.within:g}"

        return (
            f"{self.decision.value.upper()}: {self.statistic}({self.column}) "
            f"{levels}"
            f"diff (after - before) = {diff_str}, "
            f"{ci_pct}% CI = {ci_str}, "
            f"margin = {margin_str}, alpha = {self.alpha:g}, {pairing}"
        )

    def raise_for_status(self) -> None:
        """Raise if the evidence supports a change larger than the margin."""
        if self.decision == Decision.CHANGED:
            raise FrameworthyAssertionError(str(self))
        if self.decision == Decision.INCONCLUSIVE:
            warnings.warn(str(self), stacklevel=2)


@dataclass(frozen=True)
class ChangeResult:
    """Result of a one-sided directional change check.

    Instances are returned by `.change_greater_than(...)` and
    `.change_less_than(...)` and are not meant to be constructed directly.
    """

    decision: Decision
    column: str
    statistic: Statistic
    paired: bool
    before_mean: float
    after_mean: float
    diff: float
    ci_low: float
    ci_high: float
    threshold: float
    direction: Direction
    alpha: float
    n_before: int
    n_after: int
    n_resamples: int

    @property
    def passed(self) -> bool:
        """Whether the evidence rules out crossing the threshold."""
        return self.decision == Decision.PASSED

    def __str__(self) -> str:
        # each CI endpoint is individually a (1 - alpha) one-sided bound, so
        # only the endpoint relevant to `direction` is reported
        bound_pct = round((1 - self.alpha) * 100)
        if self.direction == "greater_than":
            bound_label = "lower bound"
            bound = self.ci_low
            method_name = "change_greater_than"
        else:
            bound_label = "upper bound"
            bound = self.ci_high
            method_name = "change_less_than"

        pairing = (
            f"paired, n={self.n_before}"
            if self.paired
            else f"unpaired, n_before={self.n_before}, n_after={self.n_after}"
        )

        if self.statistic == "rate":
            # report rates and their difference/threshold/bound in
            # percentage points, which reads more intuitively than raw
            # proportions
            levels = f"before = {self.before_mean:.1%}, after = {self.after_mean:.1%}, "
            diff_str = f"{self.diff * 100:+.4g}pp"
            bound_str = f"{bound * 100:+.4g}pp"
            threshold_str = f"{self.threshold * 100:+.4g}pp"
        else:
            levels = ""
            diff_str = f"{self.diff:+.4g}"
            bound_str = f"{bound:+.4g}"
            threshold_str = f"{self.threshold:+.4g}"

        return (
            f"{self.decision.value.upper()}: {self.statistic}({self.column}) "
            f"{levels}"
            f"diff (after - before) = {diff_str}, "
            f"{bound_pct}% one-sided {bound_label} = {bound_str}, "
            f"threshold ({method_name}) = {threshold_str}, "
            f"alpha = {self.alpha:g}, {pairing}"
        )

    def raise_for_status(self) -> None:
        """Raise if the evidence confirms crossing the threshold."""
        if self.decision == Decision.FAILED:
            raise FrameworthyAssertionError(str(self))
        if self.decision == Decision.INCONCLUSIVE:
            warnings.warn(str(self), stacklevel=2)
