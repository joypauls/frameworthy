from collections.abc import Sequence
from typing import ClassVar, TypeVar

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrame

from ._arrays import (
    assert_binary_values,
    paired_values_from_columns,
    values_from_two_frames,
)
from ._backend import to_narwhals_frame
from ._classify import classify_change_bound, classify_equivalence
from ._constants import (
    DEFAULT_ALPHA,
    DEFAULT_INFERENCE_METHOD,
    DEFAULT_N_RESAMPLES,
    DiffCIFunc,
    Direction,
    InferenceMethod,
    Statistic,
)
from ._errors import UsageError
from ._intervals import mean_diff_ci, rate_diff_ci
from ._pairing import _normalize_keys, assert_unique_keys
from ._validation import InferenceConfig
from .results import ChangeResult, EquivalenceResult


def _equivalence_result(
    *,
    diff_func: DiffCIFunc,
    statistic: Statistic,
    column: str,
    paired: bool,
    before_values: np.ndarray,
    after_values: np.ndarray,
    within: float,
    inference: InferenceConfig,
) -> EquivalenceResult:
    """Shared implementation behind every metric's `.equivalent()`: run
    `diff_func` (either `mean_diff_ci` or `rate_diff_ci`, which share a
    signature), classify the resulting CI, and package everything into an
    `EquivalenceResult`.
    """
    diff, ci_low, ci_high = diff_func(
        before_values,
        after_values,
        paired=paired,
        alpha=inference.alpha,
        n_resamples=inference.n_resamples,
        rng=inference.rng,
        method=inference.method,
    )
    decision = classify_equivalence(ci_low, ci_high, within)

    return EquivalenceResult(
        decision=decision,
        column=column,
        statistic=statistic,
        paired=paired,
        before_mean=float(before_values.mean()),
        after_mean=float(after_values.mean()),
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=inference.alpha,
        within=within,
        n_before=len(before_values),
        n_after=len(after_values),
        n_resamples=inference.reported_n_resamples,
    )


def _change_result(
    *,
    diff_func: DiffCIFunc,
    statistic: Statistic,
    column: str,
    paired: bool,
    before_values: np.ndarray,
    after_values: np.ndarray,
    threshold: float,
    direction: Direction,
    inference: InferenceConfig,
) -> ChangeResult:
    """Shared implementation behind every metric's `.change_greater_than()`/
    `.change_less_than()`: run `diff_func` (either `mean_diff_ci` or
    `rate_diff_ci`), classify the resulting CI against `threshold` in the
    given `direction`, and package everything into a `ChangeResult`.

    The same `(1 - 2 * alpha)` two-sided CI used by equivalence checks is
    reused here: each of its endpoints is individually a valid `(1 - alpha)`
    one-sided confidence bound, which is exactly what a one-sided directional
    claim needs.
    """
    diff, ci_low, ci_high = diff_func(
        before_values,
        after_values,
        paired=paired,
        alpha=inference.alpha,
        n_resamples=inference.n_resamples,
        rng=inference.rng,
        method=inference.method,
    )
    decision = classify_change_bound(ci_low, ci_high, threshold, direction=direction)

    return ChangeResult(
        decision=decision,
        column=column,
        statistic=statistic,
        paired=paired,
        before_mean=float(before_values.mean()),
        after_mean=float(after_values.mean()),
        diff=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        threshold=threshold,
        direction=direction,
        alpha=inference.alpha,
        n_before=len(before_values),
        n_after=len(after_values),
        n_resamples=inference.reported_n_resamples,
    )


class MetricCheck:
    """Shared base for `MeanCheck`/`RateCheck`, and for any future built-in
    metric: holds the common `before`/`after` construction and defines
    `.equivalent()`, `.change_greater_than()`, and `.change_less_than()`
    exactly once, in terms of two things each subclass declares:

    * `_statistic`: the `Statistic` this metric represents.
    * `_diff_func`: the `DiffCIFunc` (e.g. `mean_diff_ci`/`rate_diff_ci`)
      used to estimate `after - before` and its confidence interval.

    Subclasses may also override `_validate_values()` to add metric-specific
    validation of the extracted `before`/`after` arrays (e.g. `RateCheck`
    requires binary values).

    Not meant to be constructed directly; use `Check.mean(...)`/
    `Check.rate(...)` instead.
    """

    _statistic: ClassVar[Statistic]
    _diff_func: ClassVar[DiffCIFunc]

    def __init__(
        self,
        column: str,
        paired: bool,
        before_values: np.ndarray,
        after_values: np.ndarray,
    ) -> None:
        self._column = column
        self._paired = paired
        self._before_values = np.asarray(before_values, dtype=float)
        self._after_values = np.asarray(after_values, dtype=float)
        self._validate_values()

    def _validate_values(self) -> None:
        """Hook for metric-specific validation of `before`/`after` values.
        No-op by default.
        """

    def equivalent(
        self,
        within: float,
        *,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
        method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
    ) -> EquivalenceResult:
        """Test whether the difference (after - before) is equivalent
        within `within`.

        Builds a `(1 - 2 * alpha)` confidence interval for the difference,
        then classifies it against the `within` margin as `passed`
        (equivalent), `failed` (changed), or `inconclusive`. See
        `frameworthy._classify` for the decision rule, and this class's
        docstring for the metric-specific inference method and what
        `method="bootstrap"` changes.
        """
        inference = InferenceConfig(
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method=method,
        )
        return _equivalence_result(
            diff_func=self._diff_func,
            statistic=self._statistic,
            column=self._column,
            paired=self._paired,
            before_values=self._before_values,
            after_values=self._after_values,
            within=within,
            inference=inference,
        )

    def change_greater_than(
        self,
        threshold: float,
        *,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
        method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
    ) -> ChangeResult:
        """Rule out that the difference (after - before) is `threshold` or
        smaller -- e.g. ruling out an unacceptable drop when `threshold` is
        negative.

        Builds a `(1 - alpha)` one-sided lower confidence bound for the
        difference and checks it against `threshold`: `passed` if the bound
        is above `threshold`, `failed` if the data instead confirms the
        difference is at or below `threshold`, `inconclusive` if the
        available data can't establish either.

        Uses the same inference machinery (and `method`/`n_resamples`/
        `random_state` semantics) as `.equivalent()`; see this class's
        docstring for metric-specific details.
        """
        inference = InferenceConfig(
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method=method,
        )
        return _change_result(
            diff_func=self._diff_func,
            statistic=self._statistic,
            column=self._column,
            paired=self._paired,
            before_values=self._before_values,
            after_values=self._after_values,
            threshold=threshold,
            direction="greater_than",
            inference=inference,
        )

    def change_less_than(
        self,
        threshold: float,
        *,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
        method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
    ) -> ChangeResult:
        """Rule out that the difference (after - before) is `threshold` or
        larger -- e.g. ruling out an unacceptable increase in a metric like
        latency or an error rate.

        Builds a `(1 - alpha)` one-sided upper confidence bound for the
        difference and checks it against `threshold`: `passed` if the bound
        is below `threshold`, `failed` if the data instead confirms the
        difference is at or above `threshold`, `inconclusive` if the
        available data can't establish either.

        Uses the same inference machinery (and `method`/`n_resamples`/
        `random_state` semantics) as `.equivalent()`; see this class's
        docstring for metric-specific details.
        """
        inference = InferenceConfig(
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method=method,
        )
        return _change_result(
            diff_func=self._diff_func,
            statistic=self._statistic,
            column=self._column,
            paired=self._paired,
            before_values=self._before_values,
            after_values=self._after_values,
            threshold=threshold,
            direction="less_than",
            inference=inference,
        )


class MeanCheck(MetricCheck):
    """A check bound to comparing the mean of one column between two datasets.

    Returned by `Check.mean(...)`; not meant to be constructed directly.

    The mean has a closed-form interval, so `method` defaults to
    `"analytical"` (a t-interval for paired differences, or a Welch/
    unequal-variance t-interval for independent samples) instead of
    bootstrap resampling. Pass `method="bootstrap"` on any claim method to
    use percentile bootstrap resampling instead, in which case
    `n_resamples` and `random_state` control the resampling; both are
    unused for the analytical path.
    """

    _statistic = Statistic.MEAN
    _diff_func = staticmethod(mean_diff_ci)


class RateCheck(MetricCheck):
    """A check bound to comparing the rate (proportion) of one binary
    column between two datasets.

    Returned by `Check.rate(...)`; not meant to be constructed directly.

    Unlike `MeanCheck`, the default `method="analytical"` path doesn't fit
    a t-interval to the raw 0/1 values; it combines Wilson score intervals
    for the before/after proportions (Newcombe's method), which stays
    well-behaved even when an observed rate is exactly 0 or 1 -- a case
    where a Wald/t-style interval on the raw values would collapse to a
    single point despite genuine uncertainty.

    Pass `method="bootstrap"` on any claim method to use percentile
    bootstrap resampling instead, in which case `n_resamples` and
    `random_state` control the resampling. Note that the bootstrap path
    resamples the raw 0/1 values directly, so it does *not* get the
    boundary-case protection above: a sample with an observed rate of
    exactly 0 or 1 will still produce a degenerate, zero-width bootstrap
    interval.
    """

    _statistic = Statistic.RATE
    _diff_func = staticmethod(rate_diff_ci)

    def _validate_values(self) -> None:
        assert_binary_values(self._before_values, "before")
        assert_binary_values(self._after_values, "after")


MetricCheckT = TypeVar("MetricCheckT", bound=MetricCheck)


class Check:
    """Entry point for comparing `before` and `after` dataframe-like data.

    Returned by `check(...)`; not meant to be constructed directly.
    """

    def __init__(
        self,
        after: IntoDataFrame,
        before: IntoDataFrame | None,
        paired_by: str | Sequence[str] | None,
    ) -> None:
        self._after = to_narwhals_frame(after)
        self._before = to_narwhals_frame(before) if before is not None else None

        if self._before is None and paired_by is not None:
            raise UsageError(
                "`paired_by` requires a separate `before` dataframe passed to "
                "`check()`. For same-dataframe comparisons, pass "
                "`before=<column name>` to `.mean()` instead."
            )

        self._paired_by = _normalize_keys(paired_by) if paired_by is not None else None
        if self._paired_by is not None:
            assert_unique_keys(self._before, self._paired_by, "before")
            assert_unique_keys(self._after, self._paired_by, "after")

    def _extract_before_after(
        self, column: str, before: str | None, metric: str
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Resolve `before`/`after` values for `column`, shared by `.mean()`
        and `.rate()`. `metric` (e.g. `"mean"`, `"rate"`) is only used to
        name the calling method in error messages.
        """
        if self._before is None:
            if before is None:
                raise UsageError(
                    f"`check()` was given a single dataframe; `.{metric}()` "
                    "requires `before=<column name>` to compare two columns "
                    "in that dataframe."
                )
            if before == column:
                raise UsageError(
                    "`before` must name a different column than the one being "
                    f"compared, got `{column}` for both."
                )

            before_values, after_values = paired_values_from_columns(
                self._after, before, column, "df"
            )
            return before_values, after_values, True

        if before is not None:
            raise UsageError(
                f"`before=` on `.{metric}()` is only used for same-dataframe "
                "comparisons; pass a separate `before` dataframe to `check()` "
                "instead."
            )

        return values_from_two_frames(
            self._before, self._after, column, self._paired_by
        )

    def _metric_check(
        self,
        cls: type[MetricCheckT],
        column: str,
        before: str | None,
        metric: str,
    ) -> MetricCheckT:
        """Shared implementation behind `.mean()` and `.rate()`: extract
        `before`/`after` values for `column` and construct `cls` (either
        `MeanCheck` or `RateCheck`) from them.
        """
        before_values, after_values, paired = self._extract_before_after(
            column, before, metric
        )
        return cls(
            column=column,
            paired=paired,
            before_values=before_values,
            after_values=after_values,
        )

    def mean(self, column: str, before: str | None = None) -> MeanCheck:
        """Select a column and compare its mean between `before` and `after`.

        If `check()` was given a single dataframe, pass `before=<column
        name>` here to compare two columns within that same dataframe as
        paired observations (row-by-row). Otherwise, `before` must be
        omitted and `column` is compared between the two dataframes passed
        to `check()`.
        """
        return self._metric_check(MeanCheck, column, before, "mean")

    def rate(self, column: str, before: str | None = None) -> RateCheck:
        """Select a binary (0/1 or boolean) column and compare its rate
        (proportion) between `before` and `after`.

        If `check()` was given a single dataframe, pass `before=<column
        name>` here to compare two columns within that same dataframe as
        paired observations (row-by-row). Otherwise, `before` must be
        omitted and `column` is compared between the two dataframes passed
        to `check()`.
        """
        return self._metric_check(RateCheck, column, before, "rate")


def check(
    after: IntoDataFrame,
    before: IntoDataFrame | None = None,
    paired_by: str | Sequence[str] | None = None,
) -> Check:
    """Start a statistical check comparing `before` and `after` data.

    `before` may be:

    * a separate dataframe, compared column-by-column with `after` via
      `.mean(column)`. If `paired_by` is given, `before` and `after` are
      aligned on that key (or keys) first; both sides must have at most
      one row per key value. If `paired_by` is omitted, `before` and
      `after` are treated as independent samples.
    * omitted, in which case `after` is the only dataframe and
      `.mean(after_column, before=before_column)` compares two columns
      within it as paired, row-by-row observations. `paired_by` is not
      valid in this mode.
    """
    return Check(after=after, before=before, paired_by=paired_by)
