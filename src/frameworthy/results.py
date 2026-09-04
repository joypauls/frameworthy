import warnings
from dataclasses import dataclass

from ._constants import Decision
from ._errors import FrameworthyAssertionError


@dataclass(frozen=True)
class EquivalenceResult:
    """Result of a paired or unpaired equivalence check.

    Instances are returned by `.equivalent(...)` and are not meant to be
    constructed directly.
    """

    verdict: Decision
    column: str
    statistic: str
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
        return self.verdict == Decision.EQUIVALENT

    def __str__(self) -> str:
        ci_pct = round((1 - 2 * self.alpha) * 100)
        pairing = (
            f"paired, n={self.n_before}"
            if self.paired
            else f"unpaired, n_before={self.n_before}, n_after={self.n_after}"
        )
        return (
            f"{self.verdict.value.upper()}: {self.statistic}({self.column}) "
            f"diff (after - before) = {self.diff:+.4g}, "
            f"{ci_pct}% CI = [{self.ci_low:+.4g}, {self.ci_high:+.4g}], "
            f"margin = ±{self.within:g}, alpha = {self.alpha:g}, {pairing}"
        )

    def raise_for_status(self) -> None:
        """Raise if the evidence supports a change larger than the margin."""
        if self.verdict == Decision.CHANGED:
            raise FrameworthyAssertionError(str(self))
        if self.verdict == Decision.INCONCLUSIVE:
            warnings.warn(str(self), stacklevel=2)
