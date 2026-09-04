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

    assert result.verdict == "equivalent"
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

    assert result.verdict == "changed"
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

    assert result.verdict == "inconclusive"
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
    assert result.verdict == "equivalent"


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

        assert result.verdict == "equivalent"
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

        assert result.verdict == "changed"
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
