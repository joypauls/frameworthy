from collections.abc import Sequence
from typing import ClassVar, TypeVar

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrame

from ._arrays import (
    assert_binary_values,
    paired_values_from_columns,
    values_from_two_arrays,
    values_from_two_frames,
)
from ._classify import classify_change_bound, classify_equivalence
from ._constants import (
    DEFAULT_ALPHA,
    DEFAULT_N_RESAMPLES,
    Direction,
    InferenceMethod,
    Metric,
)
from ._dataframes import _normalize_keys, assert_unique_keys, to_narwhals_frame
from ._errors import UsageError
from ._intervals import (
    AnalyticalDiffFunc,
    StatisticFunc,
    analytical_mean_diff_ci,
    analytical_rate_diff_ci,
    diff_ci,
)
from ._validation import (
    InferenceConfig,
    validate_custom_metric,
    validate_statistic_func,
)
from .results import ChangeResult, EquivalenceResult


def _equivalence_result(
    *,
    analytical_func: AnalyticalDiffFunc | None,
    statistic_func: StatisticFunc,
    metric: Metric | str,
    column: str,
    paired: bool,
    before_values: np.ndarray,
    after_values: np.ndarray,
    within: float,
    inference: InferenceConfig,
) -> EquivalenceResult:
    """Shared implementation behind every metric's `.equivalent()`: run
    `diff_ci` (dispatching on `analytical_func`/`statistic_func`), classify
    the resulting CI, and package everything into an `EquivalenceResult`.

    `statistic_func` is reused for both the bootstrap statistic and the
    reported `before_value`/`after_value` -- the same function that
    estimates the difference is also what "before"/"after" mean for this
    metric, so there's only one function to keep in sync rather than two.
    """
    diff, ci_low, ci_high = diff_ci(
        before_values,
        after_values,
        paired=paired,
        alpha=inference.alpha,
        n_resamples=inference.n_resamples,
        rng=inference.rng,
        method=inference.method,
        analytical_func=analytical_func,
        statistic_func=statistic_func,
    )
    decision = classify_equivalence(ci_low, ci_high, within)

    return EquivalenceResult(
        decision=decision,
        column=column,
        metric=metric,
        paired=paired,
        before_value=float(statistic_func(before_values)),
        after_value=float(statistic_func(after_values)),
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
    analytical_func: AnalyticalDiffFunc | None,
    statistic_func: StatisticFunc,
    metric: Metric | str,
    column: str,
    paired: bool,
    before_values: np.ndarray,
    after_values: np.ndarray,
    threshold: float,
    direction: Direction,
    inference: InferenceConfig,
) -> ChangeResult:
    """Shared implementation behind every metric's `.change_greater_than()`/
    `.change_less_than()`: run `diff_ci` (dispatching on `analytical_func`/
    `statistic_func`), classify the resulting CI against `threshold` in the
    given `direction`, and package everything into a `ChangeResult`.

    The same `(1 - 2 * alpha)` two-sided CI used by equivalence checks is
    reused here: each of its endpoints is individually a valid `(1 - alpha)`
    one-sided confidence bound, which is exactly what a one-sided directional
    claim needs.
    """
    diff, ci_low, ci_high = diff_ci(
        before_values,
        after_values,
        paired=paired,
        alpha=inference.alpha,
        n_resamples=inference.n_resamples,
        rng=inference.rng,
        method=inference.method,
        analytical_func=analytical_func,
        statistic_func=statistic_func,
    )
    decision = classify_change_bound(ci_low, ci_high, threshold, direction=direction)

    return ChangeResult(
        decision=decision,
        column=column,
        metric=metric,
        paired=paired,
        before_value=float(statistic_func(before_values)),
        after_value=float(statistic_func(after_values)),
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
    """Shared base for `MeanCheck`/`RateCheck`/`MedianCheck`, and for any
    future built-in metric: holds the common `before`/`after` construction
    and defines `.equivalent()`, `.change_greater_than()`, and
    `.change_less_than()` exactly once, in terms of what each subclass
    declares:

    * `_metric`: the `Metric` this check represents, or a plain `str`
      name for a `.custom()` check (see `CustomCheck`).
    * `_statistic_func`: the point-summary function (e.g. `np.mean`/
      `np.median`) used both to report `before_value`/`after_value` on the
      result *and*, on the bootstrap path, as the statistic whose
      difference is resampled -- one function serves both roles, so
      "before"/"after" always mean the same thing that the CI estimates a
      difference of.
    * `_analytical_func`: the closed-form CI estimator (e.g.
      `analytical_mean_diff_ci`/`analytical_rate_diff_ci`) used on the
      `method="analytical"` path, or `None` if this metric has no
      closed-form estimator (in which case `method="analytical"` raises a
      `UsageError` from `diff_ci`, and `_default_method` below resolves to
      `"bootstrap"`).
    * `_default_method`: the `InferenceMethod` used when a claim method's
      `method` argument is omitted -- derived automatically from whether
      `_analytical_func` is set, so metrics with no analytical estimator
      (e.g. `MedianCheck`) don't need to declare it separately.

    Subclasses may also override `_validate_values()` to add metric-specific
    validation of the extracted `before`/`after` arrays (e.g. `RateCheck`
    requires binary values).

    Not meant to be constructed directly; use `Check.mean(...)`/
    `Check.rate(...)`/`Check.median(...)` instead.
    """

    _metric: ClassVar[Metric | str]
    _statistic_func: ClassVar[StatisticFunc]
    _analytical_func: ClassVar[AnalyticalDiffFunc | None] = None

    @property
    def _default_method(self) -> InferenceMethod:
        return "analytical" if self._analytical_func is not None else "bootstrap"

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
        method: InferenceMethod | None = None,
    ) -> EquivalenceResult:
        """Test whether the difference (after - before) is equivalent
        within `within`.

        Builds a `(1 - 2 * alpha)` confidence interval for the difference,
        then classifies it against the `within` margin as `passed`
        (equivalent), `failed` (changed), or `inconclusive`. See
        `frameworthy._classify` for the decision rule, and this class's
        docstring for the metric-specific inference method and what
        `method="bootstrap"` changes. `method` defaults to
        `self._default_method` when omitted.
        """
        inference = InferenceConfig(
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method=self._default_method if method is None else method,
        )
        return _equivalence_result(
            analytical_func=self._analytical_func,
            statistic_func=self._statistic_func,
            metric=self._metric,
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
        method: InferenceMethod | None = None,
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
        docstring for metric-specific details. `method` defaults to
        `self._default_method` when omitted.
        """
        inference = InferenceConfig(
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method=self._default_method if method is None else method,
        )
        return _change_result(
            analytical_func=self._analytical_func,
            statistic_func=self._statistic_func,
            metric=self._metric,
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
        method: InferenceMethod | None = None,
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
        docstring for metric-specific details. `method` defaults to
        `self._default_method` when omitted.
        """
        inference = InferenceConfig(
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method=self._default_method if method is None else method,
        )
        return _change_result(
            analytical_func=self._analytical_func,
            statistic_func=self._statistic_func,
            metric=self._metric,
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

    _metric = Metric.MEAN
    _statistic_func = staticmethod(np.mean)
    _analytical_func = staticmethod(analytical_mean_diff_ci)


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

    _metric = Metric.RATE
    _statistic_func = staticmethod(np.mean)
    _analytical_func = staticmethod(analytical_rate_diff_ci)

    def _validate_values(self) -> None:
        assert_binary_values(self._before_values, "before")
        assert_binary_values(self._after_values, "after")


class MedianCheck(MetricCheck):
    """A check bound to comparing the median of one column between two
    datasets.

    Returned by `Check.median(...)`; not meant to be constructed directly.

    Unlike the mean (CLT/t-interval) or rate (Wilson/Newcombe), there's no
    simple closed-form confidence interval for a difference of medians, so
    `method` defaults to `"bootstrap"` here instead of `"analytical"`;
    passing `method="analytical"` explicitly raises a `UsageError`.

    The target quantity is always `median(after) - median(before)`. If
    `paired`, bootstrap resamples draw the same indices for both sides
    (preserving before/after correlation); otherwise `before` and `after`
    are resampled independently.
    """

    _metric = Metric.MEDIAN
    _statistic_func = staticmethod(np.median)


class CustomCheck(MetricCheck):
    """A check bound to comparing a user-supplied `statistic_func` of one
    column between two datasets.

    Returned by `Check.custom(...)`/`ArrayCheck.custom(...)`; not meant to
    be constructed directly.

    There's no general closed-form confidence interval for an arbitrary
    statistic, so `CustomCheck` only supports bootstrap resampling: unlike
    `MeanCheck`/`RateCheck`/`MedianCheck`, its `.equivalent()`/
    `.change_greater_than()`/`.change_less_than()` don't even accept a
    `method` argument -- there's nothing to switch to.

    `statistic_func` is used both to report `before_value`/`after_value`
    and, on the bootstrap path, as the statistic whose difference is
    resampled (the same "one function, two roles" design as every
    built-in metric). It must:

    * Return a single scalar when called on a 1-D array, e.g.
      `statistic_func(before_values)`.
    * Accept an `axis=` keyword and apply row-wise when called on a 2-D
      array, like `np.mean`/`np.median` do -- this is what lets bootstrap
      resampling apply it to every resample at once instead of looping in
      Python. Bind extra arguments with `functools.partial` (e.g.
      `functools.partial(np.percentile, q=95)`) rather than a plain
      `lambda x: np.percentile(x, 95)`, which won't accept `axis=`.

    Both are checked eagerly at construction time (see
    `frameworthy._validation.validate_statistic_func`), so a `statistic_func`
    that doesn't support this fails fast with a clear `UsageError` here
    rather than surfacing an opaque numpy error deep inside a
    multi-thousand-iteration bootstrap loop.

    Bootstrap resampling can also be statistically unstable for exotic
    statistics (e.g. a high percentile) on small samples -- that's a
    property of the data/statistic, not something validated here.
    """

    _analytical_func = None

    def __init__(
        self,
        column: str,
        paired: bool,
        before_values: np.ndarray,
        after_values: np.ndarray,
        *,
        name: str,
        statistic_func: StatisticFunc,
    ) -> None:
        validate_custom_metric(name, statistic_func)
        self._metric = name
        self._statistic_func = statistic_func
        super().__init__(column, paired, before_values, after_values)

    def _validate_values(self) -> None:
        validate_statistic_func(self._statistic_func, self._before_values)

    def equivalent(
        self,
        within: float,
        *,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
    ) -> EquivalenceResult:
        """Test whether the difference (after - before) is equivalent
        within `within`. See `MetricCheck.equivalent()`; `method` isn't
        exposed here since `CustomCheck` only supports bootstrap.
        """
        return super().equivalent(
            within,
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method="bootstrap",
        )

    def change_greater_than(
        self,
        threshold: float,
        *,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
    ) -> ChangeResult:
        """Rule out that the difference (after - before) is `threshold` or
        smaller. See `MetricCheck.change_greater_than()`; `method` isn't
        exposed here since `CustomCheck` only supports bootstrap.
        """
        return super().change_greater_than(
            threshold,
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method="bootstrap",
        )

    def change_less_than(
        self,
        threshold: float,
        *,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
    ) -> ChangeResult:
        """Rule out that the difference (after - before) is `threshold` or
        larger. See `MetricCheck.change_less_than()`; `method` isn't
        exposed here since `CustomCheck` only supports bootstrap.
        """
        return super().change_less_than(
            threshold,
            alpha=alpha,
            n_resamples=n_resamples,
            random_state=random_state,
            method="bootstrap",
        )


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
        **extra_kwargs: object,
    ) -> MetricCheckT:
        """Shared implementation behind `.mean()`, `.rate()`, `.median()`,
        and `.custom()`: extract `before`/`after` values for `column` and
        construct `cls` (`MeanCheck`/`RateCheck`/`MedianCheck`/
        `CustomCheck`) from them. `extra_kwargs` passes through to `cls`,
        for `CustomCheck`'s `name`/`statistic_func`.
        """
        before_values, after_values, paired = self._extract_before_after(
            column, before, metric
        )
        return cls(
            column=column,
            paired=paired,
            before_values=before_values,
            after_values=after_values,
            **extra_kwargs,
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

    def median(self, column: str, before: str | None = None) -> MedianCheck:
        """Select a column and compare its median between `before` and
        `after`.

        If `check()` was given a single dataframe, pass `before=<column
        name>` here to compare two columns within that same dataframe as
        paired observations (row-by-row). Otherwise, `before` must be
        omitted and `column` is compared between the two dataframes passed
        to `check()`.
        """
        return self._metric_check(MedianCheck, column, before, "median")

    def custom(
        self,
        column: str,
        statistic_func: StatisticFunc,
        name: str,
        before: str | None = None,
    ) -> CustomCheck:
        """Select a column and compare a user-supplied `statistic_func` of
        it between `before` and `after`.

        Unlike `.mean()`/`.rate()`/`.median()`, there's no closed-form CI
        for an arbitrary statistic, so the returned `CustomCheck` only
        supports bootstrap resampling; see `CustomCheck` for
        `statistic_func`'s requirements (checked eagerly) and `name`'s
        role in result output.

        If `check()` was given a single dataframe, pass `before=<column
        name>` here to compare two columns within that same dataframe as
        paired observations (row-by-row). Otherwise, `before` must be
        omitted and `column` is compared between the two dataframes passed
        to `check()`.
        """
        return self._metric_check(
            CustomCheck,
            column,
            before,
            "custom",
            name=name,
            statistic_func=statistic_func,
        )


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

    For two 1-D numpy arrays of already-extracted metric values, use
    `check_arrays(...)` instead.
    """
    return Check(after=after, before=before, paired_by=paired_by)


class ArrayCheck:
    """Entry point for comparing two 1-D numpy arrays of already-extracted
    metric values, as independent (unpaired) samples.

    Returned by `check_arrays(...)`; not meant to be constructed directly.
    Unlike `Check`, there's no dataframe/column concept here, so there's
    also no paired mode -- arrays have no key to align pairs by, which is
    why `check_arrays()` (unlike `check()`) has no `paired_by` parameter.
    """

    def __init__(self, after: np.ndarray, before: np.ndarray) -> None:
        self._before_values, self._after_values, self._paired = values_from_two_arrays(
            before, after
        )

    def mean(self, column: str = "value") -> MeanCheck:
        """Compare the mean of the two arrays.

        `column` is used purely as a display label (e.g. shows up as
        "mean(column)" in results/messages) -- there's nothing to select,
        since both arrays were already given in full to `check_arrays()`.
        """
        return MeanCheck(
            column=column,
            paired=self._paired,
            before_values=self._before_values,
            after_values=self._after_values,
        )

    def rate(self, column: str = "value") -> RateCheck:
        """Compare the rate (proportion) of the two binary arrays.

        `column` is used purely as a display label; see `.mean()`.
        """
        return RateCheck(
            column=column,
            paired=self._paired,
            before_values=self._before_values,
            after_values=self._after_values,
        )

    def median(self, column: str = "value") -> MedianCheck:
        """Compare the median of the two arrays.

        `column` is used purely as a display label; see `.mean()`.
        """
        return MedianCheck(
            column=column,
            paired=self._paired,
            before_values=self._before_values,
            after_values=self._after_values,
        )

    def custom(
        self, statistic_func: StatisticFunc, name: str, column: str = "value"
    ) -> CustomCheck:
        """Compare a user-supplied `statistic_func` of the two arrays.

        `column` is used purely as a display label; see `.mean()`. See
        `CustomCheck` for `statistic_func`'s requirements (checked
        eagerly) and `name`'s role in result output.
        """
        return CustomCheck(
            column=column,
            paired=self._paired,
            before_values=self._before_values,
            after_values=self._after_values,
            name=name,
            statistic_func=statistic_func,
        )


def check_arrays(after: np.ndarray, before: np.ndarray) -> ArrayCheck:
    """Start a statistical check comparing two 1-D numpy arrays of
    already-extracted metric values.

    Always treated as independent samples -- there's no equivalent of
    `paired_by` since arrays have no key column to align pairs by. Use
    `check(...)` instead for dataframe input, which supports both paired
    and unpaired comparisons.
    """
    return ArrayCheck(after=after, before=before)
