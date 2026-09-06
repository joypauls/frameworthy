"""End-to-end tests exercising `fw.check` against a real dataset (iris.csv)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import frameworthy as fw

IRIS_CSV = Path(__file__).parent / "data" / "iris.csv"


@pytest.fixture
def test_df() -> pd.DataFrame:
    return pd.read_csv(IRIS_CSV)


def test_same_df_paired_check_is_equivalent_for_small_shift(test_df):
    # Simulate a "before/after" scenario from real measurements: petal_length
    # perturbed by noise far smaller than the equivalence margin should be
    # judged equivalent.
    rng = np.random.default_rng(0)
    df = test_df.assign(
        petal_length_after=test_df["petal_length"] + rng.normal(0, 0.02, len(test_df))
    )

    result = (
        fw.check(df)
        .mean("petal_length_after", before="petal_length")
        .equivalent(within=0.3, alpha=0.05, random_state=0)
    )

    assert result.decision == "equivalent"
    assert result.n_before == result.n_after == len(test_df)
    result.raise_for_status()  # should not raise


def test_two_dataframe_unpaired_check_detects_species_difference(test_df):
    # Setosa and Virginica petal_length distributions are well-known to
    # differ substantially, so an unpaired (no paired_by) comparison should
    # find the difference exceeds a reasonably tight margin.
    setosa = test_df[test_df["variety"] == "Setosa"]
    virginica = test_df[test_df["variety"] == "Virginica"]

    result = (
        fw.check(virginica, before=setosa)
        .mean("petal_length")
        .equivalent(within=1.0, alpha=0.05, random_state=0)
    )

    assert result.decision == "changed"
    assert result.paired is False
    with pytest.raises(fw.FrameworthyAssertionError):
        result.raise_for_status()


def test_same_df_paired_rate_check_is_equivalent_for_small_change(test_df):
    # Derive a binary "large petal" flag from a real measurement, then
    # simulate a small amount of relabeling noise (a few percent of rows
    # flip) between "before" and "after" -- the kind of thing you'd expect
    # from a metric/labeling pipeline change that shouldn't move the rate
    # meaningfully.
    rng = np.random.default_rng(0)
    df = test_df.assign(large_petal_before=(test_df["petal_length"] > 4.0))
    flip = rng.random(len(df)) < 0.02
    df = df.assign(
        large_petal_after=np.where(
            flip, ~df["large_petal_before"], df["large_petal_before"]
        )
    )

    result = (
        fw.check(df)
        .rate("large_petal_after", before="large_petal_before")
        .equivalent(within=0.05, alpha=0.05, random_state=0)
    )

    assert result.decision == "equivalent"
    assert result.paired is True
    assert result.statistic == "rate"
    assert result.n_before == result.n_after == len(test_df)
    result.raise_for_status()  # should not raise


def test_two_dataframe_unpaired_rate_check_detects_species_difference(test_df):
    # Setosa never has a petal_length above 3, Virginica always does, so
    # the rate of "large petal" (>3) is 0% vs 100% -- a boundary case for
    # each side individually, and an obvious rate difference overall.
    setosa = test_df[test_df["variety"] == "Setosa"].assign(
        large_petal=lambda d: d["petal_length"] > 3.0
    )
    virginica = test_df[test_df["variety"] == "Virginica"].assign(
        large_petal=lambda d: d["petal_length"] > 3.0
    )

    result = (
        fw.check(virginica, before=setosa)
        .rate("large_petal")
        .equivalent(within=0.1, alpha=0.05, random_state=0)
    )

    assert result.decision == "changed"
    assert result.paired is False
    assert result.before_mean == pytest.approx(0.0)
    assert result.after_mean == pytest.approx(1.0)
    # the boundary rates (0% and 100%) shouldn't collapse the CI to a point
    assert result.ci_low != result.ci_high
    with pytest.raises(fw.FrameworthyAssertionError):
        result.raise_for_status()
