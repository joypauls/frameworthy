import warnings
from dataclasses import dataclass

from ._constants import Direction, Metric
from ._errors import FrameworthyAssertionError
from ._format import format_margin, format_point_rows, format_value, metric_label
from .decision import Decision


@dataclass(frozen=True)
class ComparisonResult:
    """Shared base for `EquivalenceResult` and `ChangeResult`.

    Not meant to be constructed or subclassed outside this module: it
    holds the fields, `pairing`/`levels` formatting, and `passed`/
    `assert_passed` logic common to both kinds of check, and defers to
    `_claim_rows()` for the part of `__str__` that's specific to each
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

    def _claim_rows(self) -> list[tuple[str, str]]:
        """The claim-specific `(label, value)` rows shown between the diff
        and the trailing alpha/pairing line. Implemented by each subclass.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        pairing = (
            f"paired, n={self.n_before}"
            if self.paired
            else f"unpaired, n_before={self.n_before}, n_after={self.n_after}"
        )
        rows = [
            *format_point_rows(self.before_value, self.after_value, self.metric),
            ("diff (after - before)", format_value(self.diff, self.metric)),
            *self._claim_rows(),
        ]
        width = max(len(label) for label, _ in rows)
        body = "\n".join(f"  {label.ljust(width)} = {value}" for label, value in rows)

        header = (
            f"{self.decision.value.upper()}: {metric_label(self.metric)}({self.column})"
        )
        footer = f"  alpha = {self.alpha:g}, {pairing}"
        return f"{header}\n{body}\n{footer}"

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

    def _claim_rows(self) -> list[tuple[str, str]]:
        ci_pct = round((1 - 2 * self.alpha) * 100)
        ci_str = (
            f"[{format_value(self.ci_low, self.metric)}, "
            f"{format_value(self.ci_high, self.metric)}]"
        )
        margin_str = format_margin(self.within, self.metric)
        return [(f"{ci_pct}% CI", ci_str), ("margin", margin_str)]


@dataclass(frozen=True)
class ChangeResult(ComparisonResult):
    """Result of a one-sided directional change check.

    Instances are returned by `.change_greater_than(...)` and
    `.change_less_than(...)` and are not meant to be constructed directly.
    """

    threshold: float
    direction: Direction

    def _claim_rows(self) -> list[tuple[str, str]]:
        # each ci endpoint is individually a (1 - alpha) one sided bound
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
        return [
            (f"{bound_pct}% one-sided {bound_label}", bound_str),
            (f"threshold ({method_name})", threshold_str),
        ]


@dataclass(frozen=True)
class DistributionResult:
    """Result of a distribution-stability check via Wasserstein distance.

    Instances are returned by `.distribution(...).equivalent(...)` and
    `.distribution(...).change_greater_than(...)`, and are not meant to be
    constructed directly.

    Unlike `EquivalenceResult`/`ChangeResult`, there's no `before_value`/
    `after_value`/`diff` here: a Wasserstein distance is a single
    non-negative measure of how far apart two distributions are, not a
    difference of a single-number summary, so it doesn't share
    `ComparisonResult`'s fields or `__str__` layout. Exactly one of
    `within`/`threshold` is set, depending on which claim produced this
    result:

    * `within` (from `.equivalent()`): `within` is treated as an upper
      bound on an acceptable distance -- built the same way as
      `ChangeResult`'s `direction="less_than"` -- `passed` if the whole CI
      is below it, `failed` if the whole CI is above it, `inconclusive`
      otherwise.
    * `threshold` (from `.change_greater_than()`): the opposite claim,
      confirming the distributions have genuinely drifted apart by more
      than `threshold` -- built the same way as `ChangeResult`'s
      `direction="greater_than"` -- `passed` if the whole CI is above it,
      `failed` if the whole CI is below it, `inconclusive` otherwise.

    Either way the verdict is three-state, and `passed`/`assert_passed()`
    work identically regardless of which claim produced it.
    """

    decision: Decision
    column: str
    distance: float
    ci_low: float
    ci_high: float
    alpha: float
    n_before: int
    n_after: int
    n_resamples: int
    method: str = "subsampling"
    within: float | None = None
    threshold: float | None = None

    def __post_init__(self) -> None:
        # internal invariant, not user-facing validation: `DistributionResult`
        # is only ever constructed by `DistributionCheck` itself, from
        # exactly one claim at a time.
        assert (self.within is None) != (self.threshold is None), (
            "DistributionResult requires exactly one of `within`/`threshold`."
        )

    @property
    def passed(self) -> bool:
        """Whether the evidence supports the claim being tested."""
        return self.decision == Decision.PASSED

    def __str__(self) -> str:
        header = f"{self.decision.value.upper()}: wasserstein({self.column})"
        distance_row = ("distance", f"{self.distance:.4g}")

        if self.within is not None:
            ci_pct = round((1 - 2 * self.alpha) * 100)
            rows = [
                distance_row,
                (f"{ci_pct}% CI", f"[{self.ci_low:.4g}, {self.ci_high:.4g}]"),
                ("within", f"{self.within:.4g}"),
            ]
        else:
            bound_pct = round((1 - self.alpha) * 100)
            rows = [
                distance_row,
                (f"{bound_pct}% one-sided lower bound", f"{self.ci_low:.4g}"),
                ("threshold (change_greater_than)", f"{self.threshold:.4g}"),
            ]

        width = max(len(label) for label, _ in rows)
        body = "\n".join(f"  {label.ljust(width)} = {value}" for label, value in rows)

        footer = (
            f"  alpha = {self.alpha:g}, unpaired, n_before={self.n_before}, "
            f"n_after={self.n_after}, method={self.method}"
        )
        return f"{header}\n{body}\n{footer}"

    def assert_passed(self) -> None:
        """Raise if the evidence supports the opposite of the claim being
        tested; warn (but don't raise) if the evidence is inconclusive.
        """
        if self.decision == Decision.FAILED:
            raise FrameworthyAssertionError(str(self))
        if self.decision == Decision.INCONCLUSIVE:
            warnings.warn(str(self), stacklevel=2)
