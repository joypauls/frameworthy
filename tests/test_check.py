import numpy as np
import pytest

import frameworthy as fw
from frameworthy._backend import to_narwhals_frame


def test_paired_equivalent_within_margin(frame_factory):
    rng = np.random.default_rng(1)
    ids = list(range(200))
    before_revenue = rng.normal(loc=100.0, scale=5.0, size=200)
    after_revenue = before_revenue + rng.normal(loc=0.2, scale=0.5, size=200)

    before = frame_factory({"customer_id": ids, "revenue": before_revenue})
    after = frame_factory({"customer_id": ids, "revenue": after_revenue})

    result = (
        fw.check(after, before=before, paired_by="customer_id")
        .mean("revenue")
        .equivalent(within=2.0, alpha=0.05, random_state=0)
    )

    assert result.decision == "equivalent"
    assert result.passed is True
    assert result.paired is True
    assert result.n_before == result.n_after == 200
    result.raise_for_status()  # should not raise


def test_paired_changed_beyond_margin(frame_factory):
    rng = np.random.default_rng(2)
    ids = list(range(200))
    before_revenue = rng.normal(loc=100.0, scale=5.0, size=200)
    after_revenue = before_revenue + rng.normal(loc=10.0, scale=0.5, size=200)

    before = frame_factory({"customer_id": ids, "revenue": before_revenue})
    after = frame_factory({"customer_id": ids, "revenue": after_revenue})

    result = (
        fw.check(after, before=before, paired_by="customer_id")
        .mean("revenue")
        .equivalent(within=2.0, alpha=0.05, random_state=0)
    )

    assert result.decision == "changed"
    assert result.passed is False
    with pytest.raises(fw.FrameworthyAssertionError):
        result.raise_for_status()


def test_paired_inconclusive_with_small_noisy_sample(frame_factory):
    rng = np.random.default_rng(3)
    ids = list(range(6))
    before_revenue = rng.normal(loc=100.0, scale=5.0, size=6)
    after_revenue = before_revenue + rng.normal(loc=1.5, scale=3.0, size=6)

    before = frame_factory({"customer_id": ids, "revenue": before_revenue})
    after = frame_factory({"customer_id": ids, "revenue": after_revenue})

    result = (
        fw.check(after, before=before, paired_by="customer_id")
        .mean("revenue")
        .equivalent(within=2.0, alpha=0.05, random_state=0)
    )

    assert result.decision == "inconclusive"
    with pytest.warns(UserWarning):
        result.raise_for_status()  # should not raise, only warn


def test_paired_aligns_on_key_and_drops_unmatched(frame_factory):
    before = frame_factory({"customer_id": [1, 2, 3], "revenue": [10.0, 20.0, 30.0]})
    after = frame_factory({"customer_id": [2, 3, 4], "revenue": [21.0, 31.0, 41.0]})

    mean_check = fw.check(after, before=before, paired_by="customer_id").mean("revenue")

    assert mean_check._before_values.tolist() == [20.0, 30.0]
    assert mean_check._after_values.tolist() == [21.0, 31.0]


def test_paired_raises_on_duplicate_keys(frame_factory):
    before = frame_factory({"customer_id": [1, 1], "revenue": [10.0, 11.0]})
    after = frame_factory({"customer_id": [1], "revenue": [12.0]})

    with pytest.raises(ValueError, match="before"):
        fw.check(after, before=before, paired_by="customer_id")


def test_unpaired_when_no_paired_by_given(frame_factory):
    rng = np.random.default_rng(4)
    before = frame_factory({"revenue": rng.normal(100.0, 5.0, size=300)})
    after = frame_factory({"revenue": rng.normal(100.5, 5.0, size=350)})

    result = (
        fw.check(after, before=before)
        .mean("revenue")
        .equivalent(within=2.0, alpha=0.05, random_state=0)
    )

    assert result.paired is False
    assert result.n_before == 300
    assert result.n_after == 350
    assert result.decision == "equivalent"


def test_missing_column_raises_key_error(frame_factory):
    before = frame_factory({"customer_id": [1, 2], "revenue": [10.0, 20.0]})
    after = frame_factory({"customer_id": [1, 2], "revenue": [11.0, 19.0]})

    with pytest.raises(KeyError):
        fw.check(after, before=before, paired_by="customer_id").mean("missing_col")


def test_random_state_makes_result_reproducible(frame_factory):
    rng = np.random.default_rng(5)
    ids = list(range(50))
    before_revenue = rng.normal(100.0, 5.0, size=50)
    after_revenue = before_revenue + rng.normal(0.5, 1.0, size=50)

    before = frame_factory({"customer_id": ids, "revenue": before_revenue})
    after = frame_factory({"customer_id": ids, "revenue": after_revenue})

    check_a = fw.check(after, before=before, paired_by="customer_id").mean("revenue")
    check_b = fw.check(after, before=before, paired_by="customer_id").mean("revenue")

    result_a = check_a.equivalent(within=2.0, random_state=42)
    result_b = check_b.equivalent(within=2.0, random_state=42)

    assert result_a.ci_low == result_b.ci_low
    assert result_a.ci_high == result_b.ci_high


def test_mean_check_uses_analytical_path_with_zero_resamples(frame_factory):
    before = frame_factory({"revenue": [1.0, 2.0, 3.0, 4.0]})
    after = frame_factory({"revenue": [2.0, 3.0, 4.0, 5.0]})

    result = fw.check(after, before=before).mean("revenue").equivalent(within=5.0)

    assert result.n_resamples == 0


def test_mean_check_can_opt_into_bootstrap(frame_factory):
    before = frame_factory({"revenue": [1.0, 2.0, 3.0, 4.0, 5.0]})
    after = frame_factory({"revenue": [2.0, 3.0, 4.0, 5.0, 6.0]})

    result = (
        fw.check(after, before=before)
        .mean("revenue")
        .equivalent(within=5.0, n_resamples=1000, random_state=0, method="bootstrap")
    )

    assert result.n_resamples == 1000
    assert result.diff == pytest.approx(1.0)


def test_mean_check_rejects_unknown_method(frame_factory):
    before = frame_factory({"revenue": [1.0, 2.0, 3.0]})
    after = frame_factory({"revenue": [2.0, 3.0, 4.0]})

    with pytest.raises(ValueError, match="Unknown inference"):
        fw.check(after, before=before).mean("revenue").equivalent(
            within=5.0, method="magic"
        )


def test_mean_check_is_deterministic_regardless_of_random_state(frame_factory):
    rng = np.random.default_rng(6)
    ids = list(range(50))
    before_revenue = rng.normal(100.0, 5.0, size=50)
    after_revenue = before_revenue + rng.normal(0.5, 1.0, size=50)

    before = frame_factory({"customer_id": ids, "revenue": before_revenue})
    after = frame_factory({"customer_id": ids, "revenue": after_revenue})

    check_a = fw.check(after, before=before, paired_by="customer_id").mean("revenue")
    check_b = fw.check(after, before=before, paired_by="customer_id").mean("revenue")

    result_a = check_a.equivalent(within=2.0, random_state=1)
    result_b = check_b.equivalent(within=2.0, random_state=999)

    assert result_a.diff == result_b.diff
    assert result_a.ci_low == result_b.ci_low
    assert result_a.ci_high == result_b.ci_high


def test_unpaired_equivalent_within_margin(frame_factory):
    rng = np.random.default_rng(20)
    before = frame_factory({"revenue": rng.normal(100.0, 5.0, size=200)})
    after = frame_factory({"revenue": rng.normal(100.2, 5.0, size=220)})

    result = fw.check(after, before=before).mean("revenue").equivalent(within=2.0)

    assert result.decision == "equivalent"
    assert result.paired is False


def test_unpaired_changed_beyond_margin(frame_factory):
    rng = np.random.default_rng(21)
    before = frame_factory({"revenue": rng.normal(100.0, 5.0, size=200)})
    after = frame_factory({"revenue": rng.normal(110.0, 5.0, size=220)})

    result = fw.check(after, before=before).mean("revenue").equivalent(within=2.0)

    assert result.decision == "changed"


def test_unpaired_inconclusive_with_small_noisy_sample(frame_factory):
    rng = np.random.default_rng(0)
    before = frame_factory({"revenue": rng.normal(100.0, 5.0, size=6)})
    after = frame_factory({"revenue": rng.normal(101.0, 5.0, size=6)})

    result = fw.check(after, before=before).mean("revenue").equivalent(within=2.0)

    assert result.decision == "inconclusive"


def test_accepts_narwhals_frame_via_backend_helper(frame_factory):
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


class TestSameDataframeComparison:
    def test_equivalent_within_margin(self, frame_factory):
        rng = np.random.default_rng(10)
        score_before = rng.normal(loc=80.0, scale=5.0, size=200)
        score_after = score_before + rng.normal(loc=0.2, scale=0.5, size=200)

        df = frame_factory({"score_before": score_before, "score_after": score_after})

        result = (
            fw.check(df)
            .mean("score_after", before="score_before")
            .equivalent(within=1.0, alpha=0.05, random_state=0)
        )

        assert result.decision == "equivalent"
        assert result.paired is True
        assert result.column == "score_after"
        assert result.n_before == result.n_after == 200
        result.raise_for_status()  # should not raise

    def test_changed_beyond_margin(self, frame_factory):
        rng = np.random.default_rng(11)
        score_before = rng.normal(loc=80.0, scale=5.0, size=200)
        score_after = score_before + rng.normal(loc=10.0, scale=0.5, size=200)

        df = frame_factory({"score_before": score_before, "score_after": score_after})

        result = (
            fw.check(df)
            .mean("score_after", before="score_before")
            .equivalent(within=1.0, alpha=0.05, random_state=0)
        )

        assert result.decision == "changed"
        with pytest.raises(fw.FrameworthyAssertionError):
            result.raise_for_status()

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

        with pytest.raises(ValueError, match="different column"):
            fw.check(df).mean("score", before="score")

    def test_requires_before_kwarg_for_single_dataframe(self, frame_factory):
        df = frame_factory({"score_before": [1.0, 2.0], "score_after": [2.0, 3.0]})

        with pytest.raises(ValueError, match="before="):
            fw.check(df).mean("score_after")

    def test_rejects_before_kwarg_for_two_dataframe_mode(self, frame_factory):
        before = frame_factory({"revenue": [1.0, 2.0]})
        after = frame_factory({"revenue": [2.0, 3.0]})

        with pytest.raises(ValueError, match="same-dataframe"):
            fw.check(after, before=before).mean("revenue", before="revenue")

    def test_rejects_paired_by_without_separate_before_dataframe(self, frame_factory):
        df = frame_factory({"customer_id": [1, 2], "score": [1.0, 2.0]})

        with pytest.raises(ValueError, match="paired_by"):
            fw.check(df, paired_by="customer_id")

    def test_missing_column_raises_key_error(self, frame_factory):
        df = frame_factory({"score_before": [1.0, 2.0], "score_after": [2.0, 3.0]})

        with pytest.raises(KeyError):
            fw.check(df).mean("missing_col", before="score_before")

        with pytest.raises(KeyError):
            fw.check(df).mean("score_after", before="missing_col")

    def test_null_only_column_raises_value_error(self, frame_factory):
        df = frame_factory(
            {"score_before": [None, None, None], "score_after": [1.0, 2.0, 3.0]}
        )

        with pytest.raises(ValueError, match="No usable"):
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

        with pytest.raises(ValueError, match="At least 2"):
            fw.check(df).mean("score_after", before="score_before")


class TestRateCheck:
    def test_paired_by_equivalent_within_margin(self, frame_factory):
        rng = np.random.default_rng(30)
        ids = list(range(500))
        before_converted = (rng.random(500) < 0.30).astype(float)
        # after mostly agrees with before, with a small amount of noise
        flip = rng.random(500) < 0.02
        after_converted = np.where(flip, 1 - before_converted, before_converted)

        before = frame_factory({"customer_id": ids, "converted": before_converted})
        after = frame_factory({"customer_id": ids, "converted": after_converted})

        result = (
            fw.check(after, before=before, paired_by="customer_id")
            .rate("converted")
            .equivalent(within=0.05, alpha=0.05)
        )

        assert result.decision == "equivalent"
        assert result.paired is True
        assert result.statistic == "rate"
        assert result.n_before == result.n_after == 500
        result.raise_for_status()  # should not raise

    def test_paired_by_changed_beyond_margin(self, frame_factory):
        rng = np.random.default_rng(31)
        ids = list(range(500))
        before_converted = (rng.random(500) < 0.20).astype(float)
        flip = rng.random(500) < 0.35
        after_converted = np.where(flip, 1 - before_converted, before_converted)

        before = frame_factory({"customer_id": ids, "converted": before_converted})
        after = frame_factory({"customer_id": ids, "converted": after_converted})

        result = (
            fw.check(after, before=before, paired_by="customer_id")
            .rate("converted")
            .equivalent(within=0.005, alpha=0.05)
        )

        assert result.decision == "changed"
        with pytest.raises(fw.FrameworthyAssertionError):
            result.raise_for_status()

    def test_unpaired_when_no_paired_by_given(self, frame_factory):
        rng = np.random.default_rng(32)
        before = frame_factory({"converted": (rng.random(400) < 0.30).astype(float)})
        after = frame_factory({"converted": (rng.random(450) < 0.31).astype(float)})

        result = fw.check(after, before=before).rate("converted").equivalent(within=0.1)

        assert result.paired is False
        assert result.n_before == 400
        assert result.n_after == 450

    def test_boundary_all_zero_does_not_collapse_to_a_point(self, frame_factory):
        before = frame_factory({"converted": [0.0] * 50})
        after = frame_factory({"converted": [0.0] * 5 + [1.0] * 45})

        result = (
            fw.check(after, before=before).rate("converted").equivalent(within=0.05)
        )

        # a boundary observed rate of 0 shouldn't produce a zero-width CI
        assert result.ci_low != result.ci_high
        assert result.decision == "changed"

    def test_same_dataframe_paired_columns(self, frame_factory):
        rng = np.random.default_rng(33)
        converted_before = (rng.random(300) < 0.25).astype(float)
        flip = rng.random(300) < 0.02
        converted_after = np.where(flip, 1 - converted_before, converted_before)

        df = frame_factory(
            {"converted_before": converted_before, "converted_after": converted_after}
        )

        result = (
            fw.check(df)
            .rate("converted_after", before="converted_before")
            .equivalent(within=0.05)
        )

        assert result.paired is True
        assert result.column == "converted_after"

    def test_rejects_non_binary_column(self, frame_factory):
        before = frame_factory({"count": [0.0, 1.0, 2.0]})
        after = frame_factory({"count": [0.0, 1.0, 1.0]})

        with pytest.raises(ValueError, match="binary"):
            fw.check(after, before=before).rate("count")

    def test_bootstrap_method_is_supported(self, frame_factory):
        rng = np.random.default_rng(34)
        before = frame_factory({"converted": (rng.random(200) < 0.3).astype(float)})
        after = frame_factory({"converted": (rng.random(200) < 0.32).astype(float)})

        result = (
            fw.check(after, before=before)
            .rate("converted")
            .equivalent(
                within=0.1, n_resamples=1000, random_state=0, method="bootstrap"
            )
        )

        assert result.n_resamples == 1000

    def test_analytical_method_uses_zero_resamples(self, frame_factory):
        before = frame_factory({"converted": [0.0, 1.0, 1.0, 0.0]})
        after = frame_factory({"converted": [1.0, 1.0, 1.0, 0.0]})

        result = fw.check(after, before=before).rate("converted").equivalent(within=0.5)

        assert result.n_resamples == 0

    def test_random_state_makes_bootstrap_result_reproducible(self, frame_factory):
        rng = np.random.default_rng(35)
        before = frame_factory({"converted": (rng.random(100) < 0.3).astype(float)})
        after = frame_factory({"converted": (rng.random(100) < 0.32).astype(float)})

        check_a = fw.check(after, before=before).rate("converted")
        check_b = fw.check(after, before=before).rate("converted")

        result_a = check_a.equivalent(within=0.1, random_state=42, method="bootstrap")
        result_b = check_b.equivalent(within=0.1, random_state=42, method="bootstrap")

        assert result_a.ci_low == result_b.ci_low
        assert result_a.ci_high == result_b.ci_high

    def test_missing_column_raises_key_error(self, frame_factory):
        before = frame_factory({"converted": [0.0, 1.0]})
        after = frame_factory({"converted": [0.0, 1.0]})

        with pytest.raises(KeyError):
            fw.check(after, before=before).rate("missing_col")

    def test_rejects_unknown_method(self, frame_factory):
        before = frame_factory({"converted": [0.0, 1.0, 1.0]})
        after = frame_factory({"converted": [1.0, 1.0, 0.0]})

        with pytest.raises(ValueError, match="Unknown inference"):
            fw.check(after, before=before).rate("converted").equivalent(
                within=0.5, method="magic"
            )


class TestChangeGreaterThan:
    def test_mean_passed_when_lower_bound_above_threshold(self, frame_factory):
        rng = np.random.default_rng(40)
        ids = list(range(200))
        before_revenue = rng.normal(loc=100.0, scale=5.0, size=200)
        after_revenue = before_revenue + rng.normal(loc=0.2, scale=0.5, size=200)

        before = frame_factory({"customer_id": ids, "revenue": before_revenue})
        after = frame_factory({"customer_id": ids, "revenue": after_revenue})

        result = (
            fw.check(after, before=before, paired_by="customer_id")
            .mean("revenue")
            .change_greater_than(-2.0, alpha=0.05, random_state=0)
        )

        assert result.decision == "passed"
        assert result.passed is True
        assert result.direction == "greater_than"
        assert result.threshold == -2.0
        result.raise_for_status()  # should not raise

    def test_mean_failed_when_upper_bound_below_threshold(self, frame_factory):
        rng = np.random.default_rng(41)
        ids = list(range(200))
        before_revenue = rng.normal(loc=100.0, scale=5.0, size=200)
        after_revenue = before_revenue - rng.normal(loc=10.0, scale=0.5, size=200)

        before = frame_factory({"customer_id": ids, "revenue": before_revenue})
        after = frame_factory({"customer_id": ids, "revenue": after_revenue})

        result = (
            fw.check(after, before=before, paired_by="customer_id")
            .mean("revenue")
            .change_greater_than(-2.0, alpha=0.05, random_state=0)
        )

        assert result.decision == "failed"
        assert result.passed is False
        with pytest.raises(fw.FrameworthyAssertionError):
            result.raise_for_status()

    def test_mean_inconclusive_with_small_noisy_sample(self, frame_factory):
        rng = np.random.default_rng(0)
        ids = list(range(6))
        before_revenue = rng.normal(loc=100.0, scale=5.0, size=6)
        after_revenue = before_revenue + rng.normal(loc=0.0, scale=3.0, size=6)

        before = frame_factory({"customer_id": ids, "revenue": before_revenue})
        after = frame_factory({"customer_id": ids, "revenue": after_revenue})

        result = (
            fw.check(after, before=before, paired_by="customer_id")
            .mean("revenue")
            .change_greater_than(-2.0, alpha=0.05, random_state=0)
        )

        assert result.decision == "inconclusive"
        with pytest.warns(UserWarning):
            result.raise_for_status()  # should not raise, only warn

    def test_rate_passed_rules_out_conversion_drop(self, frame_factory):
        rng = np.random.default_rng(43)
        ids = list(range(500))
        before_converted = (rng.random(500) < 0.30).astype(float)
        flip = rng.random(500) < 0.01
        after_converted = np.where(flip, 1 - before_converted, before_converted)

        before = frame_factory({"customer_id": ids, "converted": before_converted})
        after = frame_factory({"customer_id": ids, "converted": after_converted})

        result = (
            fw.check(after, before=before, paired_by="customer_id")
            .rate("converted")
            .change_greater_than(-0.05, alpha=0.05)
        )

        assert result.decision == "passed"
        assert result.paired is True
        assert result.statistic == "rate"
        result.raise_for_status()  # should not raise

    def test_rate_failed_confirms_conversion_drop(self, frame_factory):
        rng = np.random.default_rng(44)
        ids = list(range(500))
        before_converted = (rng.random(500) < 0.40).astype(float)
        after_converted = (rng.random(500) < 0.10).astype(float)

        before = frame_factory({"customer_id": ids, "converted": before_converted})
        after = frame_factory({"customer_id": ids, "converted": after_converted})

        result = (
            fw.check(after, before=before, paired_by="customer_id")
            .rate("converted")
            .change_greater_than(-0.005, alpha=0.05)
        )

        assert result.decision == "failed"
        with pytest.raises(fw.FrameworthyAssertionError):
            result.raise_for_status()


class TestChangeLessThan:
    def test_mean_passed_when_upper_bound_below_threshold(self, frame_factory):
        rng = np.random.default_rng(50)
        before = frame_factory({"latency_ms": rng.normal(100.0, 5.0, size=300)})
        after = frame_factory({"latency_ms": rng.normal(102.0, 5.0, size=300)})

        result = (
            fw.check(after, before=before)
            .mean("latency_ms")
            .change_less_than(20.0, alpha=0.05)
        )

        assert result.decision == "passed"
        assert result.passed is True
        assert result.direction == "less_than"
        assert result.threshold == 20.0
        result.raise_for_status()  # should not raise

    def test_mean_failed_when_lower_bound_above_threshold(self, frame_factory):
        rng = np.random.default_rng(51)
        before = frame_factory({"latency_ms": rng.normal(100.0, 5.0, size=300)})
        after = frame_factory({"latency_ms": rng.normal(140.0, 5.0, size=300)})

        result = (
            fw.check(after, before=before)
            .mean("latency_ms")
            .change_less_than(20.0, alpha=0.05)
        )

        assert result.decision == "failed"
        with pytest.raises(fw.FrameworthyAssertionError):
            result.raise_for_status()

    def test_mean_inconclusive_with_small_noisy_sample(self, frame_factory):
        rng = np.random.default_rng(52)
        before = frame_factory({"latency_ms": rng.normal(100.0, 5.0, size=6)})
        after = frame_factory({"latency_ms": rng.normal(115.0, 8.0, size=6)})

        result = (
            fw.check(after, before=before)
            .mean("latency_ms")
            .change_less_than(20.0, alpha=0.05)
        )

        assert result.decision == "inconclusive"
        with pytest.warns(UserWarning):
            result.raise_for_status()  # should not raise, only warn

    def test_rate_passed_rules_out_error_rate_increase(self, frame_factory):
        rng = np.random.default_rng(53)
        before = frame_factory({"errored": (rng.random(500) < 0.02).astype(float)})
        after = frame_factory({"errored": (rng.random(500) < 0.025).astype(float)})

        result = fw.check(after, before=before).rate("errored").change_less_than(0.05)

        assert result.decision == "passed"
        assert result.statistic == "rate"
        result.raise_for_status()  # should not raise

    def test_bootstrap_method_is_supported(self, frame_factory):
        before = frame_factory({"latency_ms": [100.0, 101.0, 99.0, 102.0, 98.0]})
        after = frame_factory({"latency_ms": [101.0, 102.0, 100.0, 103.0, 99.0]})

        result = (
            fw.check(after, before=before)
            .mean("latency_ms")
            .change_less_than(
                20.0, n_resamples=1000, random_state=0, method="bootstrap"
            )
        )

        assert result.n_resamples == 1000

    def test_rejects_unknown_method(self, frame_factory):
        before = frame_factory({"latency_ms": [100.0, 101.0, 99.0]})
        after = frame_factory({"latency_ms": [101.0, 102.0, 100.0]})

        with pytest.raises(ValueError, match="Unknown inference"):
            fw.check(after, before=before).mean("latency_ms").change_less_than(
                20.0, method="magic"
            )


class TestTwoDataframeNullHandling:
    def test_unpaired_drops_nulls_independently(self, frame_factory):
        before = frame_factory({"revenue": [10.0, None, 30.0, 40.0]})
        after = frame_factory({"revenue": [None, 21.0, 31.0, 41.0]})

        mean_check = fw.check(after, before=before).mean("revenue")

        assert mean_check._before_values.tolist() == [10.0, 30.0, 40.0]
        assert mean_check._after_values.tolist() == [21.0, 31.0, 41.0]

    def test_paired_drops_rows_with_null_in_either_side(self, frame_factory):
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
