from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest


# this fixture will be used to create dataframes in tests
# parameterized to test both pandas and polars
@pytest.fixture(params=["pandas", "polars"])
def frame_factory(request: pytest.FixtureRequest) -> Callable[[dict[str, Any]], Any]:
    if request.param == "pandas":
        return pd.DataFrame
    if request.param == "polars":
        return pl.DataFrame

    raise AssertionError(f"Unsupported backend: {request.param}")


# --- Deterministic data builders for tests that need a specific decision
# outcome (equivalent/changed/inconclusive, passed/failed/inconclusive).
#
# These build exact arrays via `linspace`/counting rather than picking an
# RNG seed and checking whether it happens to land on the desired verdict:
# the resulting CI is a fully deterministic function of the arguments, so
# there's no seed to go stale if the underlying interval math changes.


def paired_mean_arrays(
    n: int, diff: float, spread: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """`n` paired before/after values with mean difference exactly `diff`.

    `spread` linearly varies the per-pair difference by `+/-spread` around
    `diff`, which controls the width of the resulting confidence interval
    (0 for a zero-variance/zero-width CI, larger for a wider one).
    """
    before = np.linspace(90.0, 110.0, n)
    perturbation = np.linspace(-spread, spread, n)
    after = before + diff + perturbation
    return before, after


def unpaired_mean_arrays(
    n_before: int,
    n_after: int,
    before_level: float,
    after_level: float,
    spread: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent before/after samples centered on `before_level`/
    `after_level`, each linearly spread by `+/-spread`.
    """
    before = np.linspace(before_level - spread, before_level + spread, n_before)
    after = np.linspace(after_level - spread, after_level + spread, n_after)
    return before, after


def rate_array(n: int, rate: float) -> np.ndarray:
    """An exact-`rate` 0/1 array of length `n` (`round(rate * n)` ones)."""
    n_ones = round(rate * n)
    return np.array([1.0] * n_ones + [0.0] * (n - n_ones))
