from collections.abc import Sequence

import numpy as np
from narwhals.stable.v2.typing import IntoDataFrame

from ._arrays import (
    assert_binary_values,
    paired_values_from_columns,
    values_from_two_frames,
)
from ._backend import to_narwhals_frame
from ._constants import (
    DEFAULT_ALPHA,
    DEFAULT_INFERENCE_METHOD,
    DEFAULT_N_RESAMPLES,
    InferenceMethod,
)
from ._pairing import _normalize_keys, assert_unique_keys
from ._stats import classify_equivalence, mean_diff_ci, rate_diff_ci
from .results import EquivalenceResult


class MeanCheck:
    """A check bound to comparing the mean of one column between two datasets.

    Returned by `Check.mean(...)`; not meant to be constructed directly.
    """

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

    def equivalent(
        self,
        within: float,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
        method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
    ) -> EquivalenceResult:
        """Test whether the mean difference is equivalent within `within`.

        Builds a `(1 - 2 * alpha)` confidence interval for the mean
        difference (after - before), then classifies it against the
        `within` margin as `equivalent`, `changed`, or `inconclusive`. See
        `frameworthy._stats` for the decision rule.

        The mean has a closed-form interval, so `method` defaults to
        `"analytical"` (a t-interval for paired differences, or a Welch/
        unequal-variance t-interval for independent samples) instead of
        bootstrap resampling. Pass `method="bootstrap"` to use percentile
        bootstrap resampling instead, in which case `n_resamples` and
        `random_state` control the resampling; both are unused for the
        analytical path.
        """
        rng = np.random.default_rng(random_state)
        diff, ci_low, ci_high = mean_diff_ci(
            self._before_values,
            self._after_values,
            paired=self._paired,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=rng,
            method=method,
        )
        decision = classify_equivalence(ci_low, ci_high, within)

        return EquivalenceResult(
            decision=decision,
            column=self._column,
            statistic="mean",
            paired=self._paired,
            before_mean=float(self._before_values.mean()),
            after_mean=float(self._after_values.mean()),
            diff=diff,
            ci_low=ci_low,
            ci_high=ci_high,
            alpha=alpha,
            within=within,
            n_before=len(self._before_values),
            n_after=len(self._after_values),
            n_resamples=n_resamples if method == "bootstrap" else 0,
        )


class RateCheck:
    """A check bound to comparing the rate (proportion) of one binary
    column between two datasets.

    Returned by `Check.rate(...)`; not meant to be constructed directly.
    """

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
        assert_binary_values(self._before_values, "before")
        assert_binary_values(self._after_values, "after")

    def equivalent(
        self,
        within: float,
        alpha: float = DEFAULT_ALPHA,
        n_resamples: int = DEFAULT_N_RESAMPLES,
        random_state: int | np.random.Generator | None = None,
        method: InferenceMethod = DEFAULT_INFERENCE_METHOD,
    ) -> EquivalenceResult:
        """Test whether the rate (proportion) difference is equivalent
        within `within`.

        Builds a `(1 - 2 * alpha)` confidence interval for the difference
        in rates (after - before), then classifies it against the `within`
        margin as `equivalent`, `changed`, or `inconclusive`. `within` is a
        plain proportion, e.g. `within=0.005` means half a percentage
        point. See `frameworthy._stats` for the decision rule.

        Unlike `MeanCheck`, the default `method="analytical"` path doesn't
        fit a t-interval to the raw 0/1 values; it combines Wilson score
        intervals for the before/after proportions (Newcombe's method),
        which stays well-behaved even when an observed rate is exactly 0
        or 1 -- a case where a Wald/t-style interval on the raw values
        would collapse to a single point despite genuine uncertainty.

        Pass `method="bootstrap"` to use percentile bootstrap resampling
        instead, in which case `n_resamples` and `random_state` control the
        resampling. Note that the bootstrap path resamples the raw 0/1
        values directly, so it does *not* get the boundary-case protection
        above: a sample with an observed rate of exactly 0 or 1 will still
        produce a degenerate, zero-width bootstrap interval.
        """
        rng = np.random.default_rng(random_state)
        diff, ci_low, ci_high = rate_diff_ci(
            self._before_values,
            self._after_values,
            paired=self._paired,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=rng,
            method=method,
        )
        decision = classify_equivalence(ci_low, ci_high, within)

        return EquivalenceResult(
            decision=decision,
            column=self._column,
            statistic="rate",
            paired=self._paired,
            before_mean=float(self._before_values.mean()),
            after_mean=float(self._after_values.mean()),
            diff=diff,
            ci_low=ci_low,
            ci_high=ci_high,
            alpha=alpha,
            within=within,
            n_before=len(self._before_values),
            n_after=len(self._after_values),
            n_resamples=n_resamples if method == "bootstrap" else 0,
        )


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
            raise ValueError(
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
                raise ValueError(
                    f"`check()` was given a single dataframe; `.{metric}()` "
                    "requires `before=<column name>` to compare two columns "
                    "in that dataframe."
                )
            if before == column:
                raise ValueError(
                    "`before` must name a different column than the one being "
                    f"compared, got `{column}` for both."
                )

            before_values, after_values = paired_values_from_columns(
                self._after, before, column, "df"
            )
            return before_values, after_values, True

        if before is not None:
            raise ValueError(
                f"`before=` on `.{metric}()` is only used for same-dataframe "
                "comparisons; pass a separate `before` dataframe to `check()` "
                "instead."
            )

        return values_from_two_frames(
            self._before, self._after, column, self._paired_by
        )

    def mean(self, column: str, before: str | None = None) -> MeanCheck:
        """Select a column and compare its mean between `before` and `after`.

        If `check()` was given a single dataframe, pass `before=<column
        name>` here to compare two columns within that same dataframe as
        paired observations (row-by-row). Otherwise, `before` must be
        omitted and `column` is compared between the two dataframes passed
        to `check()`.
        """
        before_values, after_values, paired = self._extract_before_after(
            column, before, "mean"
        )
        return MeanCheck(
            column=column,
            paired=paired,
            before_values=before_values,
            after_values=after_values,
        )

    def rate(self, column: str, before: str | None = None) -> RateCheck:
        """Select a binary (0/1 or boolean) column and compare its rate
        (proportion) between `before` and `after`.

        If `check()` was given a single dataframe, pass `before=<column
        name>` here to compare two columns within that same dataframe as
        paired observations (row-by-row). Otherwise, `before` must be
        omitted and `column` is compared between the two dataframes passed
        to `check()`.
        """
        before_values, after_values, paired = self._extract_before_after(
            column, before, "rate"
        )
        return RateCheck(
            column=column,
            paired=paired,
            before_values=before_values,
            after_values=after_values,
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
    """
    return Check(after=after, before=before, paired_by=paired_by)
