from collections.abc import Callable
from typing import Any

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
