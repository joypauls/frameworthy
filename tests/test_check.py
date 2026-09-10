import functools

import numpy as np
import pytest
from conftest import paired_mean_arrays, rate_array, unpaired_mean_arrays

import frameworthy as fw
from frameworthy._backend import to_narwhals_frame

# --- Deterministic decision fixtures ------------------------------------
#
# Each entry is a `(before, after)` pair of numpy arrays, tuned (via
# `linspace`/exact counts, not RNG) so that the resulting confidence
# interval lands unambiguously on the target decision for the margin/
# threshold used alongside it below. See `conftest.py` for the builders.

MARGIN = {"mean": 2.0, "rate": 0.05, "median": 2.0}
GREATER_THAN_THRESHOLD = {"mean": -2.0, "rate": -0.005, "median": -2.0}
LESS_THAN_THRESHOLD = {"mean": 20.0, "rate": 0.05, "median": 20.0}

# `median`'s bootstrap CI is (necessarily) non-deterministic without a
# fixed `random_state`, but for these same mean-tuned arrays/margins the
# resulting median-difference CI lands comfortably clear of the decision
# boundary regardless of seed (spot-checked across many seeds), so no
# `random_state` is needed here to avoid flakiness.
EQUIVALENT_ARRAYS = {
    ("mean", True): paired_mean_arrays(n=20, diff=0.3, spread=1.0),
    ("mean", False): unpaired_mean_arrays(20, 22, 100.0, 100.3, spread=1.0),
    ("rate", True): (rate_array(500, 0.30), rate_array(500, 0.31)),
    ("rate", False): (rate_array(2000, 0.30), rate_array(2000, 0.31)),
    ("median", True): paired_mean_arrays(n=20, diff=0.3, spread=1.0),
    ("median", False): unpaired_mean_arrays(20, 22, 100.0, 100.3, spread=1.0),
}
CHANGED_ARRAYS = {
    ("mean", True): paired_mean_arrays(n=20, diff=10.0, spread=1.0),
    ("mean", False): unpaired_mean_arrays(20, 22, 100.0, 110.0, spread=1.0),
    ("rate", True): (rate_array(500, 0.30), rate_array(500, 0.50)),
    ("rate", False): (rate_array(400, 0.30), rate_array(450, 0.50)),
    ("median", True): paired_mean_arrays(n=20, diff=10.0, spread=1.0),
    ("median", False): unpaired_mean_arrays(20, 22, 100.0, 110.0, spread=1.0),
}
EQUIVALENCE_INCONCLUSIVE_ARRAYS = {
    ("mean", True): paired_mean_arrays(n=6, diff=1.0, spread=4.0),
    ("mean", False): unpaired_mean_arrays(4, 4, 100.0, 101.0, spread=2.5),
    ("rate", True): (rate_array(30, 0.30), rate_array(30, 0.40)),
    ("rate", False): (rate_array(30, 0.30), rate_array(30, 0.45)),
    ("median", True): paired_mean_arrays(n=6, diff=1.0, spread=4.0),
    ("median", False): unpaired_mean_arrays(4, 4, 100.0, 101.0, spread=2.5),
}

# `change_greater_than(GREATER_THAN_THRESHOLD)`: rules out a drop.
GREATER_THAN_PASSED_ARRAYS = {
    "mean": paired_mean_arrays(n=20, diff=0.3, spread=1.0),
    "rate": (rate_array(500, 0.30), rate_array(500, 0.30)),
    "median": paired_mean_arrays(n=20, diff=0.3, spread=1.0),
}
GREATER_THAN_FAILED_ARRAYS = {
    "mean": paired_mean_arrays(n=20, diff=-10.0, spread=1.0),
    "rate": (rate_array(500, 0.40), rate_array(500, 0.10)),
    "median": paired_mean_arrays(n=20, diff=-10.0, spread=1.0),
}
GREATER_THAN_INCONCLUSIVE_ARRAYS = {
    "mean": paired_mean_arrays(n=6, diff=-1.5, spread=4.0),
    "rate": (rate_array(40, 0.30), rate_array(40, 0.28)),
    "median": paired_mean_arrays(n=6, diff=-1.5, spread=4.0),
}

# `change_less_than(LESS_THAN_THRESHOLD)`: rules out a rise.
LESS_THAN_PASSED_ARRAYS = {
    "mean": unpaired_mean_arrays(300, 300, 100.0, 102.0, spread=5.0),
    "rate": (rate_array(500, 0.02), rate_array(500, 0.025)),
    "median": unpaired_mean_arrays(300, 300, 100.0, 102.0, spread=5.0),
}
LESS_THAN_FAILED_ARRAYS = {
    "mean": unpaired_mean_arrays(300, 300, 100.0, 140.0, spread=5.0),
    "rate": (rate_array(500, 0.02), rate_array(500, 0.15)),
    "median": unpaired_mean_arrays(300, 300, 100.0, 140.0, spread=5.0),
}
LESS_THAN_INCONCLUSIVE_ARRAYS = {
    "mean": unpaired_mean_arrays(6, 6, 100.0, 118.0, spread=8.0),
    "rate": (rate_array(60, 0.02), rate_array(60, 0.08)),
    "median": unpaired_mean_arrays(6, 6, 100.0, 118.0, spread=8.0),
}


def _column_name(metric: str) -> str:
    return "revenue" if metric in ("mean", "median") else "converted"


def _build_check(frame_factory, metric: str, paired: bool, before, after):
    """Build a `MeanCheck`/`RateCheck`/`MedianCheck` from raw `before`/
    `after` arrays, wiring up `paired_by` (paired) or two independent
    dataframes (unpaired).
    """
    column = _column_name(metric)
    if paired:
        ids = list(range(len(before)))
        before_df = frame_factory({"id": ids, column: before})
        after_df = frame_factory({"id": ids, column: after})
        check_ = fw.check(after_df, before=before_df, paired_by="id")
    else:
        before_df = frame_factory({column: before})
        after_df = frame_factory({column: after})
        check_ = fw.check(after_df, before=before_df)

    return getattr(check_, metric)(column)


def _build_custom_check(
    frame_factory,
    paired: bool,
    before,
    after,
    *,
    statistic_func=np.mean,
    name: str = "custom_metric",
):
    """Build a `CustomCheck` from raw `before`/`after` arrays, mirroring
    `_build_check` above but going through `.custom()` instead of a
    metric-named method.
    """
    column = "revenue"
    if paired:
        ids = list(range(len(before)))
        before_df = frame_factory({"id": ids, column: before})
        after_df = frame_factory({"id": ids, column: after})
        check_ = fw.check(after_df, before=before_df, paired_by="id")
    else:
        before_df = frame_factory({column: before})
        after_df = frame_factory({column: after})
        check_ = fw.check(after_df, before=before_df)

    return check_.custom(column, statistic_func, name)


class TestCheckConstruction:
    def test_rejects_paired_by_without_separate_before_dataframe(self, frame_factory):
        df = frame_factory({"customer_id": [1, 2], "score": [1.0, 2.0]})

        with pytest.raises(fw.UsageError, match="paired_by"):
            fw.check(df, paired_by="customer_id")

    def test_paired_raises_on_duplicate_keys(self, frame_factory):
        before = frame_factory({"customer_id": [1, 1], "revenue": [10.0, 11.0]})
        after = frame_factory({"customer_id": [1], "revenue": [12.0]})

        with pytest.raises(fw.InvalidDataError, match="before"):
            fw.check(after, before=before, paired_by="customer_id")

    def test_accepts_narwhals_frame_via_backend_helper(self, frame_factory):
        # sanity check that check() works when given already-native frames,
        # matching how to_narwhals_frame is used elsewhere in the codebase
        before = to_narwhals_frame(frame_factory({"revenue": [1.0, 2.0, 3.0, 4.0]}))
        after = to_narwhals_frame(frame_factory({"revenue": [2.0, 3.0, 4.0, 5.0]}))

        result = (
            fw.check(after.to_native(), before=before.to_native())
            .mean("revenue")
            .equivalent(within=5.0, random_state=0)
        )

        assert result.diff == pytest.approx(1.0)


class TestColumnResolution:
    def test_paired_aligns_on_key_and_drops_unmatched(self, frame_factory):
        before = frame_factory(
            {"customer_id": [1, 2, 3], "revenue": [10.0, 20.0, 30.0]}
        )
        after = frame_factory({"customer_id": [2, 3, 4], "revenue": [21.0, 31.0, 41.0]})

        mean_check = fw.check(after, before=before, paired_by="customer_id").mean(
            "revenue"
        )

        assert mean_check._before_values.tolist() == [20.0, 30.0]
        assert mean_check._after_values.tolist() == [21.0, 31.0]

    def test_missing_column_raises_key_error(self, frame_factory):
        before = frame_factory({"customer_id": [1, 2], "revenue": [10.0, 20.0]})
        after = frame_factory({"customer_id": [1, 2], "revenue": [11.0, 19.0]})

        with pytest.raises(fw.ColumnNotFoundError):
            fw.check(after, before=before, paired_by="customer_id").mean("missing_col")

    def test_two_dataframe_unpaired_drops_nulls_independently(self, frame_factory):
        before = frame_factory({"revenue": [10.0, None, 30.0, 40.0]})
        after = frame_factory({"revenue": [None, 21.0, 31.0, 41.0]})

        mean_check = fw.check(after, before=before).mean("revenue")

        assert mean_check._before_values.tolist() == [10.0, 30.0, 40.0]
        assert mean_check._after_values.tolist() == [21.0, 31.0, 41.0]

    def test_two_dataframe_paired_drops_rows_with_null_in_either_side(
        self, frame_factory
    ):
        before = frame_factory(
            {"customer_id": [1, 2, 3, 4], "revenue": [10.0, None, 30.0, 40.0]}
        )
        after = frame_factory(
            {"customer_id": [1, 2, 3, 4], "revenue": [11.0, 21.0, None, 41.0]}
        )

        mean_check = fw.check(after, before=before, paired_by="customer_id").mean(
            "revenue"
        )

        assert mean_check._before_values.tolist() == [10.0, 40.0]
        assert mean_check._after_values.tolist() == [11.0, 41.0]


class TestSameDataframeComparison:
    def test_rows_stay_paired_row_by_row(self, frame_factory):
        # a constant per-row shift should be recovered exactly regardless of
        # row order, proving before/after values are compared row-by-row
        # rather than as independent samples
        df = frame_factory(
            {
                "score_before": [10.0, 50.0, 5.0, 100.0],
                "score_after": [11.0, 51.0, 6.0, 101.0],
            }
        )

        mean_check = fw.check(df).mean("score_after", before="score_before")

        assert mean_check._paired is True
        assert (mean_check._after_values - mean_check._before_values == 1.0).all()

    def test_requires_two_distinct_columns(self, frame_factory):
        df = frame_factory({"score": [1.0, 2.0, 3.0]})

        with pytest.raises(fw.UsageError, match="different column"):
            fw.check(df).mean("score", before="score")

    def test_requires_before_kwarg_for_single_dataframe(self, frame_factory):
        df = frame_factory({"score_before": [1.0, 2.0], "score_after": [2.0, 3.0]})

        with pytest.raises(fw.UsageError, match="before="):
            fw.check(df).mean("score_after")

    def test_rejects_before_kwarg_for_two_dataframe_mode(self, frame_factory):
        before = frame_factory({"revenue": [1.0, 2.0]})
        after = frame_factory({"revenue": [2.0, 3.0]})

        with pytest.raises(fw.UsageError, match="same-dataframe"):
            fw.check(after, before=before).mean("revenue", before="revenue")

    def test_missing_column_raises_key_error(self, frame_factory):
        df = frame_factory({"score_before": [1.0, 2.0], "score_after": [2.0, 3.0]})

        with pytest.raises(fw.ColumnNotFoundError):
            fw.check(df).mean("missing_col", before="score_before")

        with pytest.raises(fw.ColumnNotFoundError):
            fw.check(df).mean("score_after", before="missing_col")

    def test_null_only_column_raises_value_error(self, frame_factory):
        df = frame_factory(
            {"score_before": [None, None, None], "score_after": [1.0, 2.0, 3.0]}
        )

        with pytest.raises(fw.InvalidDataError, match="No usable"):
            fw.check(df).mean("score_after", before="score_before")

    def test_drops_rows_with_nulls_in_either_column(self, frame_factory):
        df = frame_factory(
            {
                "score_before": [10.0, None, 30.0, 40.0],
                "score_after": [11.0, 21.0, None, 41.0],
            }
        )

        mean_check = fw.check(df).mean("score_after", before="score_before")

        assert mean_check._before_values.tolist() == [10.0, 40.0]
        assert mean_check._after_values.tolist() == [11.0, 41.0]

    def test_too_few_usable_pairs_raises_value_error(self, frame_factory):
        df = frame_factory({"score_before": [10.0, None], "score_after": [11.0, None]})

        with pytest.raises(fw.InvalidDataError, match="At least 2"):
            fw.check(df).mean("score_after", before="score_before")


class TestRateSpecific:
    def test_rejects_non_binary_column(self, frame_factory):
        before = frame_factory({"count": [0.0, 1.0, 2.0]})
        after = frame_factory({"count": [0.0, 1.0, 1.0]})

        with pytest.raises(fw.InvalidDataError, match="binary"):
            fw.check(after, before=before).rate("count")

    def test_boundary_all_zero_does_not_collapse_to_a_point(self, frame_factory):
        before = frame_factory({"converted": [0.0] * 50})
        after = frame_factory({"converted": [0.0] * 5 + [1.0] * 45})

        result = (
            fw.check(after, before=before).rate("converted").equivalent(within=0.05)
        )

        # a boundary observed rate of 0 shouldn't produce a zero-width CI
        assert result.ci_low != result.ci_high
        assert result.decision == "failed"


class TestMedianSpecific:
    """Behavior unique to `MedianCheck`: unlike `MeanCheck`/`RateCheck`,
    there's no analytical estimator for a median difference, so it
    defaults to (and is limited to) `method="bootstrap"`.
    """

    def test_default_method_is_bootstrap(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["median", True]
        result = _build_check(frame_factory, "median", True, before, after).equivalent(
            within=MARGIN["median"]
        )

        assert result.n_resamples > 0

    @pytest.mark.parametrize(
        "claim",
        [
            lambda check_: check_.equivalent(within=5.0, method="analytical"),
            lambda check_: check_.change_greater_than(-5.0, method="analytical"),
            lambda check_: check_.change_less_than(5.0, method="analytical"),
        ],
        ids=["equivalent", "change_greater_than", "change_less_than"],
    )
    def test_rejects_analytical_method(self, frame_factory, claim):
        before, after = EQUIVALENT_ARRAYS["median", True]
        check_ = _build_check(frame_factory, "median", True, before, after)

        with pytest.raises(fw.UsageError, match="no closed-form"):
            claim(check_)


class TestCustomSpecific:
    """Behavior unique to `CustomCheck`: unlike every built-in metric,
    there's no `method` kwarg at all (bootstrap is the only option), and
    `name`/`statistic_func` are validated eagerly at construction time.

    Decision coverage reuses the existing `EQUIVALENT_ARRAYS`/
    `CHANGED_ARRAYS["mean", ...]` fixtures with `statistic_func=np.mean`,
    since a mean-difference bootstrap CI on the same tuned arrays lands on
    the same decisions as `MeanCheck`'s own bootstrap path -- no need to
    invent new arrays just to prove the plumbing works.
    """

    def test_is_always_bootstrap(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["mean", True]
        result = _build_custom_check(frame_factory, True, before, after).equivalent(
            within=MARGIN["mean"]
        )

        assert result.n_resamples > 0

    @pytest.mark.parametrize(
        "claim",
        [
            lambda check_: check_.equivalent(within=5.0, method="bootstrap"),
            lambda check_: check_.change_greater_than(-5.0, method="bootstrap"),
            lambda check_: check_.change_less_than(5.0, method="bootstrap"),
        ],
        ids=["equivalent", "change_greater_than", "change_less_than"],
    )
    def test_claim_methods_do_not_accept_a_method_kwarg(self, frame_factory, claim):
        before, after = EQUIVALENT_ARRAYS["mean", True]
        check_ = _build_custom_check(frame_factory, True, before, after)

        with pytest.raises(TypeError, match="method"):
            claim(check_)

    def test_equivalent_within_margin(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["mean", True]
        result = _build_custom_check(frame_factory, True, before, after).equivalent(
            within=MARGIN["mean"], random_state=0
        )

        assert result.decision == "passed"

    def test_changed_beyond_margin(self, frame_factory):
        before, after = CHANGED_ARRAYS["mean", True]
        result = _build_custom_check(frame_factory, True, before, after).equivalent(
            within=MARGIN["mean"], random_state=0
        )

        assert result.decision == "failed"

    def test_supports_a_genuinely_custom_statistic(self, frame_factory):
        # not just `np.mean` wearing a `.custom()` costume: a statistic no
        # built-in check offers, bound via `functools.partial` since a
        # plain `lambda x: np.percentile(x, 95)` wouldn't accept `axis=`.
        rng = np.random.default_rng(0)
        before = rng.normal(100.0, 5.0, size=300)
        after = before + 3.0  # constant shift, comfortably inside margin

        p95 = functools.partial(np.percentile, q=95)
        check_ = _build_custom_check(
            frame_factory,
            False,
            before,
            after,
            statistic_func=p95,
            name="p95_latency",
        )
        result = check_.equivalent(within=10.0, n_resamples=2000, random_state=0)

        assert result.metric == "p95_latency"
        assert result.before_value == pytest.approx(np.percentile(before, 95))
        assert result.after_value == pytest.approx(np.percentile(after, 95))
        assert result.decision == "passed"

    def test_custom_name_shows_up_in_str(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["mean", True]
        check_ = _build_custom_check(
            frame_factory, True, before, after, name="p95_latency"
        )

        result = check_.equivalent(within=MARGIN["mean"], random_state=0)

        assert "p95_latency(revenue)" in str(result)
        assert "pp" not in str(result)  # a plain str metric is never a proportion

    def test_rejects_empty_name(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["mean", True]

        with pytest.raises(fw.UsageError, match="`name`"):
            _build_custom_check(frame_factory, True, before, after, name="")

    def test_rejects_non_callable_statistic_func(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["mean", True]

        with pytest.raises(fw.UsageError, match="`statistic_func`"):
            _build_custom_check(
                frame_factory, True, before, after, statistic_func="not callable"
            )

    def test_rejects_statistic_func_without_axis_support(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["mean", True]

        def p95(values):
            return np.percentile(values, 95)

        with pytest.raises(fw.UsageError, match="axis"):
            _build_custom_check(frame_factory, True, before, after, statistic_func=p95)

    def test_rejects_statistic_func_that_does_not_return_a_scalar(self, frame_factory):
        before, after = EQUIVALENT_ARRAYS["mean", True]

        with pytest.raises(fw.UsageError, match="single scalar"):
            _build_custom_check(
                frame_factory,
                True,
                before,
                after,
                statistic_func=lambda x, axis=None: x,
            )


class TestEquivalent:
    """Decision coverage for `.equivalent()`, across both built-in metrics
    and both pairing modes. Each case's `(before, after)` arrays are
    constructed deterministically (see `conftest.py`) to land on the target
    decision -- no seed-fishing required.
    """

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    @pytest.mark.parametrize("paired", [True, False])
    def test_equivalent_within_margin(self, frame_factory, metric, paired):
        before, after = EQUIVALENT_ARRAYS[metric, paired]
        result = _build_check(frame_factory, metric, paired, before, after).equivalent(
            within=MARGIN[metric]
        )

        assert result.decision == "passed"
        assert result.passed is True
        assert result.paired is paired
        result.assert_passed()  # should not raise

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    @pytest.mark.parametrize("paired", [True, False])
    def test_changed_beyond_margin(self, frame_factory, metric, paired):
        before, after = CHANGED_ARRAYS[metric, paired]
        result = _build_check(frame_factory, metric, paired, before, after).equivalent(
            within=MARGIN[metric]
        )

        assert result.decision == "failed"
        assert result.passed is False
        with pytest.raises(fw.FrameworthyAssertionError):
            result.assert_passed()

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    @pytest.mark.parametrize("paired", [True, False])
    def test_inconclusive_with_small_noisy_sample(self, frame_factory, metric, paired):
        before, after = EQUIVALENCE_INCONCLUSIVE_ARRAYS[metric, paired]
        result = _build_check(frame_factory, metric, paired, before, after).equivalent(
            within=MARGIN[metric]
        )

        assert result.decision == "inconclusive"
        with pytest.warns(UserWarning):
            result.assert_passed()  # should not raise, only warn


class TestChangeGreaterThan:
    """Decision coverage for `.change_greater_than()` (ruling out a drop),
    across both built-in metrics. Uses paired data, mirroring the
    "same customers, before vs. after" scenario this claim targets.
    """

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_passed_when_lower_bound_above_threshold(self, frame_factory, metric):
        before, after = GREATER_THAN_PASSED_ARRAYS[metric]
        result = _build_check(
            frame_factory, metric, True, before, after
        ).change_greater_than(GREATER_THAN_THRESHOLD[metric])

        assert result.decision == "passed"
        assert result.passed is True
        assert result.direction == "greater_than"
        assert result.threshold == GREATER_THAN_THRESHOLD[metric]
        result.assert_passed()  # should not raise

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_failed_when_upper_bound_below_threshold(self, frame_factory, metric):
        before, after = GREATER_THAN_FAILED_ARRAYS[metric]
        result = _build_check(
            frame_factory, metric, True, before, after
        ).change_greater_than(GREATER_THAN_THRESHOLD[metric])

        assert result.decision == "failed"
        assert result.passed is False
        with pytest.raises(fw.FrameworthyAssertionError):
            result.assert_passed()

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_inconclusive_with_small_noisy_sample(self, frame_factory, metric):
        before, after = GREATER_THAN_INCONCLUSIVE_ARRAYS[metric]
        result = _build_check(
            frame_factory, metric, True, before, after
        ).change_greater_than(GREATER_THAN_THRESHOLD[metric])

        assert result.decision == "inconclusive"
        with pytest.warns(UserWarning):
            result.assert_passed()  # should not raise, only warn


class TestChangeLessThan:
    """Decision coverage for `.change_less_than()` (ruling out a rise),
    across both built-in metrics. Uses unpaired data, mirroring the
    "independent before/after samples" scenario this claim targets.
    """

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_passed_when_upper_bound_below_threshold(self, frame_factory, metric):
        before, after = LESS_THAN_PASSED_ARRAYS[metric]
        result = _build_check(
            frame_factory, metric, False, before, after
        ).change_less_than(LESS_THAN_THRESHOLD[metric])

        assert result.decision == "passed"
        assert result.passed is True
        assert result.direction == "less_than"
        assert result.threshold == LESS_THAN_THRESHOLD[metric]
        result.assert_passed()  # should not raise

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_failed_when_lower_bound_above_threshold(self, frame_factory, metric):
        before, after = LESS_THAN_FAILED_ARRAYS[metric]
        result = _build_check(
            frame_factory, metric, False, before, after
        ).change_less_than(LESS_THAN_THRESHOLD[metric])

        assert result.decision == "failed"
        with pytest.raises(fw.FrameworthyAssertionError):
            result.assert_passed()

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_inconclusive_with_small_noisy_sample(self, frame_factory, metric):
        before, after = LESS_THAN_INCONCLUSIVE_ARRAYS[metric]
        result = _build_check(
            frame_factory, metric, False, before, after
        ).change_less_than(LESS_THAN_THRESHOLD[metric])

        assert result.decision == "inconclusive"
        with pytest.warns(UserWarning):
            result.assert_passed()  # should not raise, only warn


class TestInferenceOptions:
    """`alpha`/`n_resamples`/`random_state`/`method` behavior, which is
    shared plumbing (`InferenceConfig`) rather than metric-specific, so one
    parametrized pass over `mean`/`rate` is enough -- no need to repeat
    these for every claim method too.
    """

    @pytest.mark.parametrize("metric", ["mean", "rate"])
    def test_analytical_method_uses_zero_resamples(self, frame_factory, metric):
        before, after = EQUIVALENT_ARRAYS[metric, True]
        result = _build_check(frame_factory, metric, True, before, after).equivalent(
            within=MARGIN[metric]
        )

        assert result.n_resamples == 0

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_bootstrap_method_is_supported(self, frame_factory, metric):
        before, after = EQUIVALENT_ARRAYS[metric, True]
        result = _build_check(frame_factory, metric, True, before, after).equivalent(
            within=MARGIN[metric], n_resamples=1000, random_state=0, method="bootstrap"
        )

        assert result.n_resamples == 1000

    @pytest.mark.parametrize("metric", ["mean", "rate", "median"])
    def test_random_state_makes_bootstrap_result_reproducible(
        self, frame_factory, metric
    ):
        before, after = EQUIVALENT_ARRAYS[metric, True]
        check_a = _build_check(frame_factory, metric, True, before, after)
        check_b = _build_check(frame_factory, metric, True, before, after)

        result_a = check_a.equivalent(
            within=MARGIN[metric], random_state=42, method="bootstrap"
        )
        result_b = check_b.equivalent(
            within=MARGIN[metric], random_state=42, method="bootstrap"
        )

        assert result_a.ci_low == result_b.ci_low
        assert result_a.ci_high == result_b.ci_high

    @pytest.mark.parametrize("metric", ["mean", "rate"])
    def test_random_state_has_no_effect_on_analytical_path(self, frame_factory, metric):
        before, after = EQUIVALENT_ARRAYS[metric, True]
        check_a = _build_check(frame_factory, metric, True, before, after)
        check_b = _build_check(frame_factory, metric, True, before, after)

        result_a = check_a.equivalent(within=MARGIN[metric], random_state=1)
        result_b = check_b.equivalent(within=MARGIN[metric], random_state=999)

        assert result_a.diff == result_b.diff
        assert result_a.ci_low == result_b.ci_low
        assert result_a.ci_high == result_b.ci_high

    @pytest.mark.parametrize(
        "claim",
        [
            lambda check_: check_.equivalent(within=5.0, method="magic"),
            lambda check_: check_.change_greater_than(-5.0, method="magic"),
            lambda check_: check_.change_less_than(5.0, method="magic"),
        ],
        ids=["equivalent", "change_greater_than", "change_less_than"],
    )
    def test_rejects_unknown_method(self, frame_factory, claim):
        # method validation happens in `InferenceConfig`, which doesn't
        # care which metric is asking, so one representative metric (mean)
        # is enough here -- unlike the tests above, parametrizing this over
        # `metric` too would double the case count for zero extra coverage.
        before, after = EQUIVALENT_ARRAYS["mean", True]
        check_ = _build_check(frame_factory, "mean", True, before, after)

        with pytest.raises(fw.UsageError, match="Unknown inference"):
            claim(check_)

    def test_rejects_invalid_alpha_before_computing_ci(self, frame_factory):
        # alpha is validated eagerly by `InferenceConfig` as soon as
        # `.equivalent()` is called, rather than only failing deep inside CI
        # computation.
        before, after = EQUIVALENT_ARRAYS["mean", True]
        check_ = _build_check(frame_factory, "mean", True, before, after)

        with pytest.raises(fw.UsageError, match="alpha"):
            check_.equivalent(within=5.0, alpha=0.6)


class TestArrayCheck:
    """`ArrayCheck`/`check_arrays()`: the array-input counterpart to
    `Check`/`check()`, always unpaired. Only `.custom()` is covered here in
    detail (mirroring `TestCustomSpecific` above); `.mean()`/`.rate()`/
    `.median()` just construct the same `MeanCheck`/`RateCheck`/
    `MedianCheck` classes already covered via `Check`.
    """

    def test_mean_matches_check_result(self):
        before, after = EQUIVALENT_ARRAYS["mean", False]

        result = fw.check_arrays(after, before).mean().equivalent(within=MARGIN["mean"])

        assert result.paired is False
        assert result.decision == "passed"

    def test_custom_compares_a_user_supplied_statistic(self):
        before, after = EQUIVALENT_ARRAYS["mean", False]

        result = (
            fw.check_arrays(after, before)
            .custom(np.mean, name="array_mean")
            .equivalent(within=MARGIN["mean"], random_state=0)
        )

        assert result.metric == "array_mean"
        assert result.paired is False
        assert result.decision == "passed"
        assert result.n_resamples > 0

    def test_custom_rejects_empty_name(self):
        before, after = EQUIVALENT_ARRAYS["mean", False]

        with pytest.raises(fw.UsageError, match="`name`"):
            fw.check_arrays(after, before).custom(np.mean, name="")
