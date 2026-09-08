from pathlib import Path

import pandas as pd
import pytest

import frameworthy as fw

# generated from scripts/generate_e2e_datasets.py
BEFORE_CSV = Path(__file__).parent / "data" / "before.csv"
AFTER_CHANGED_CSV = Path(__file__).parent / "data" / "after_changed.csv"
AFTER_UNCHANGED_CSV = Path(__file__).parent / "data" / "after_unchanged.csv"


@pytest.fixture
def before_df() -> pd.DataFrame:
    return pd.read_csv(BEFORE_CSV)


@pytest.fixture
def after_changed_df() -> pd.DataFrame:
    return pd.read_csv(AFTER_CHANGED_CSV)


@pytest.fixture
def after_unchanged_df() -> pd.DataFrame:
    return pd.read_csv(AFTER_UNCHANGED_CSV)


def test_dataframe_check_equivalent_for_unchanged(before_df, after_unchanged_df):
    result = (
        fw.check(after_unchanged_df, before=before_df)
        .mean("normal")
        .equivalent(within=0.1)
    )
    assert result.decision == "passed"
    assert result.n_before == result.n_after == len(before_df)
    result.assert_passed()


def test_dataframe_check_equivalent_for_changed_with_buffer(
    before_df, after_changed_df
):
    result = (
        fw.check(after_changed_df, before=before_df)
        .mean("normal")
        .equivalent(within=2.0)
    )
    assert result.decision == "passed"
    assert result.n_before == result.n_after == len(before_df)
    result.assert_passed()


# def test_two_dataframe_unpaired_check_detects_species_difference(test_df):
#     # Setosa and Virginica petal_length distributions are well-known to
#     # differ substantially, so an unpaired (no paired_by) comparison should
#     # find the difference exceeds a reasonably tight margin.
#     setosa = test_df[test_df["variety"] == "Setosa"]
#     virginica = test_df[test_df["variety"] == "Virginica"]

#     result = (
#         fw.check(virginica, before=setosa)
#         .mean("petal_length")
#         .equivalent(within=1.0, alpha=0.05, random_state=0)
#     )

#     assert result.decision == "failed"
#     assert result.paired is False
#     with pytest.raises(fw.FrameworthyAssertionError):
#         result.assert_passed()


# def test_same_df_paired_rate_check_is_equivalent_for_small_change(test_df):
#     # Derive a binary "large petal" flag from a real measurement, then
#     # simulate a small amount of relabeling noise (a few percent of rows
#     # flip) between "before" and "after" -- the kind of thing you'd expect
#     # from a metric/labeling pipeline change that shouldn't move the rate
#     # meaningfully.
#     rng = np.random.default_rng(0)
#     df = test_df.assign(large_petal_before=(test_df["petal_length"] > 4.0))
#     flip = rng.random(len(df)) < 0.02
#     df = df.assign(
#         large_petal_after=np.where(
#             flip, ~df["large_petal_before"], df["large_petal_before"]
#         )
#     )

#     result = (
#         fw.check(df)
#         .rate("large_petal_after", before="large_petal_before")
#         .equivalent(within=0.05, alpha=0.05, random_state=0)
#     )

#     assert result.decision == "passed"
#     assert result.paired is True
#     assert result.statistic == "rate"
#     assert result.n_before == result.n_after == len(test_df)
#     result.assert_passed()


# def test_two_dataframe_unpaired_rate_check_detects_species_difference(test_df):
#     # Setosa never has a petal_length above 3, Virginica always does, so
#     # the rate of "large petal" (>3) is 0% vs 100% -- a boundary case for
#     # each side individually, and an obvious rate difference overall.
#     setosa = test_df[test_df["variety"] == "Setosa"].assign(
#         large_petal=lambda d: d["petal_length"] > 3.0
#     )
#     virginica = test_df[test_df["variety"] == "Virginica"].assign(
#         large_petal=lambda d: d["petal_length"] > 3.0
#     )

#     result = (
#         fw.check(virginica, before=setosa)
#         .rate("large_petal")
#         .equivalent(within=0.1, alpha=0.05, random_state=0)
#     )

#     assert result.decision == "failed"
#     assert result.paired is False
#     assert result.before_mean == pytest.approx(0.0)
#     assert result.after_mean == pytest.approx(1.0)
#     # the boundary rates (0% and 100%) shouldn't collapse the CI to a point
#     assert result.ci_low != result.ci_high
#     with pytest.raises(fw.FrameworthyAssertionError):
#         result.assert_passed()
