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

    assert result.verdict == "equivalent"
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

    assert result.verdict == "changed"
    assert result.paired is False
    with pytest.raises(fw.FrameworthyAssertionError):
        result.raise_for_status()
