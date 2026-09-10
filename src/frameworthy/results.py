import warnings
from dataclasses import dataclass

from ._constants import Direction, Metric
from ._errors import FrameworthyAssertionError
from ._format import format_margin, format_point_values, format_value, metric_label
from .decision import Decision


@dataclass(frozen=True)
class ComparisonResult:
    """Shared base for `EquivalenceResult` and `ChangeResult`.

    Not meant to be constructed or subclassed outside this module: it
    holds the fields, `pairing`/`levels` formatting, and `passed`/
    `assert_passed` logic common to both kinds of check, and defers to
    `_claim_summary()` for the part of `__str__` that's specific to each
    (a two-sided CI + margin for equivalence, a one-sided bound + threshold
    for a directional change).
    """

    decision: Decision
    column: str
    metric: Metric | str
    paired: bool
    before_value: float
    after_value: float
    diff: float
    ci_low: float
    ci_high: float
    alpha: float
    n_before: int
    n_after: int
    n_resamples: int

    @property
    def passed(self) -> bool:
        """Whether the evidence supports the claim being tested."""
        return self.decision == Decision.PASSED

    def _claim_summary(self) -> str:
        """The claim-specific middle portion of `__str__` (including its
        own trailing ", "). Implemented by each subclass.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        pairing = (
            f"paired, n={self.n_before}"
            if self.paired
            else f"unpaired, n_before={self.n_before}, n_after={self.n_after}"
        )
        point_values = format_point_values(
            self.before_value, self.after_value, self.metric
        )
        diff_str = format_value(self.diff, self.metric)

        return (
            f"{self.decision.value.upper()}: {metric_label(self.metric)}"
            f"({self.column}) "
            f"{point_values}"
            f"diff (after - before) = {diff_str}, "
            f"{self._claim_summary()}"
            f"alpha = {self.alpha:g}, {pairing}"
        )

    def assert_passed(self) -> None:
        """Raise if the evidence supports the opposite of the claim being
        tested; warn (but don't raise) if the evidence is inconclusive.
        """
        if self.decision == Decision.FAILED:
            raise FrameworthyAssertionError(str(self))
        if self.decision == Decision.INCONCLUSIVE:
            warnings.warn(str(self), stacklevel=2)


@dataclass(frozen=True)
class EquivalenceResult(ComparisonResult):
    """Result of a paired or unpaired equivalence check.

    Instances are returned by `.equivalent(...)` and are not meant to be
    constructed directly.
    """

    within: float

    def _claim_summary(self) -> str:
        ci_pct = round((1 - 2 * self.alpha) * 100)
        ci_str = (
            f"[{format_value(self.ci_low, self.metric)}, "
            f"{format_value(self.ci_high, self.metric)}]"
        )
        margin_str = format_margin(self.within, self.metric)
        return f"{ci_pct}% CI = {ci_str}, margin = {margin_str}, "


@dataclass(frozen=True)
class ChangeResult(ComparisonResult):
    """Result of a one-sided directional change check.

    Instances are returned by `.change_greater_than(...)` and
    `.change_less_than(...)` and are not meant to be constructed directly.
    """

    threshold: float
    direction: Direction

    def _claim_summary(self) -> str:
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

        bound_str = format_value(bound, self.metric)
        threshold_str = format_value(self.threshold, self.metric)
        return (
            f"{bound_pct}% one-sided {bound_label} = {bound_str}, "
            f"threshold ({method_name}) = {threshold_str}, "
        )
