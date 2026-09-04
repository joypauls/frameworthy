from collections.abc import Sequence

import narwhals.stable.v2 as nw
import numpy as np
from narwhals.stable.v2.typing import IntoDataFrame

from ._backend import to_narwhals_frame
from ._pairing import _normalize_keys, assert_unique_keys, join_paired
from ._stats import bootstrap_mean_diff_ci, classify_equivalence
from .results import EquivalenceResult


def _assert_column_exists(columns: Sequence[str], column: str, label: str) -> None:
    if column not in columns:
        raise KeyError(f"Column `{column}` not found in `{label}`.")


def _assert_min_count(n: int, min_count: int, label: str) -> None:
    if n == 0:
        raise ValueError(f"No usable (non-null) values found for {label}.")
    if n < min_count:
        raise ValueError(
            f"At least {min_count} usable (non-null) values are required for "
            f"{label}, got {n}."
        )


def _assert_equal_pairs(
    before_values: np.ndarray, after_values: np.ndarray, label: str
) -> None:
    if len(before_values) != len(after_values):
        raise ValueError(
            f"Paired comparison for {label} produced unequal numbers of usable "
            f"before ({len(before_values)}) and after ({len(after_values)}) "
            "values; before/after values must stay aligned pair-by-pair."
        )


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
        alpha: float = 0.05,
        n_resamples: int = 10_000,
        random_state: int | np.random.Generator | None = None,
    ) -> EquivalenceResult:
        """Test whether the mean difference is equivalent within `within`.

        Uses percentile bootstrap resampling to build a `(1 - 2 * alpha)`
        confidence interval for the mean difference (after - before), then
        classifies it against the `within` margin as `equivalent`,
        `changed`, or `inconclusive`. See `frameworthy._stats` for the
        decision rule.
        """
        rng = np.random.default_rng(random_state)
        diff, ci_low, ci_high = bootstrap_mean_diff_ci(
            self._before_values,
            self._after_values,
            paired=self._paired,
            alpha=alpha,
            n_resamples=n_resamples,
            rng=rng,
        )
        verdict = classify_equivalence(ci_low, ci_high, within)

        return EquivalenceResult(
            verdict=verdict,
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
            n_resamples=n_resamples,
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

    def mean(self, column: str, before: str | None = None) -> MeanCheck:
        """Select a column and compare its mean between `before` and `after`.

        If `check()` was given a single dataframe, pass `before=<column
        name>` here to compare two columns within that same dataframe as
        paired observations (row-by-row). Otherwise, `before` must be
        omitted and `column` is compared between the two dataframes passed
        to `check()`.
        """
        if self._before is None:
            if before is None:
                raise ValueError(
                    "`check()` was given a single dataframe; `.mean()` requires "
                    "`before=<column name>` to compare two columns in that "
                    "dataframe."
                )
            return self._mean_same_frame(after_column=column, before_column=before)

        if before is not None:
            raise ValueError(
                "`before=` on `.mean()` is only used for same-dataframe "
                "comparisons; pass a separate `before` dataframe to `check()` "
                "instead."
            )

        return self._mean_two_frames(column)

    def _mean_same_frame(self, after_column: str, before_column: str) -> MeanCheck:
        if after_column == before_column:
            raise ValueError(
                "`before` must name a different column than the one being "
                f"compared, got `{after_column}` for both."
            )
        _assert_column_exists(self._after.columns, after_column, "df")
        _assert_column_exists(self._after.columns, before_column, "df")

        paired = self._after.select(before_column, after_column).drop_nulls(
            subset=[before_column, after_column]
        )
        before_values = paired[before_column].to_numpy()
        after_values = paired[after_column].to_numpy()

        _assert_equal_pairs(before_values, after_values, "df")
        _assert_min_count(len(before_values), 2, "df")

        return MeanCheck(
            column=after_column,
            paired=True,
            before_values=before_values,
            after_values=after_values,
        )

    def _mean_two_frames(self, column: str) -> MeanCheck:
        assert self._before is not None
        _assert_column_exists(self._before.columns, column, "before")
        _assert_column_exists(self._after.columns, column, "after")

        if self._paired_by is not None:
            after_column = f"{column}_after"
            joined: nw.DataFrame = join_paired(
                self._before, self._after, self._paired_by, [column]
            ).drop_nulls(subset=[column, after_column])

            before_values = joined[column].to_numpy()
            after_values = joined[after_column].to_numpy()

            _assert_equal_pairs(before_values, after_values, "before/after")
            _assert_min_count(len(before_values), 2, "before/after")
            paired = True
        else:
            before_values = self._before.select(column).drop_nulls()[column].to_numpy()
            after_values = self._after.select(column).drop_nulls()[column].to_numpy()

            _assert_min_count(len(before_values), 2, "before")
            _assert_min_count(len(after_values), 2, "after")
            paired = False

        return MeanCheck(
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
