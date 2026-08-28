from pathlib import Path

import pandas as pd
import polars as pl

import frameworthy as fw

DATA_PATH = Path(__file__).parent / "data" / "iris.csv"


def test_pandas_dataset_preserves_rows():
    before = pd.read_csv(DATA_PATH)

    after = before.copy()
    after["sepal_area"] = after["sepal_length"] * after["sepal_width"]

    fw.expect(after, before=before).preserves_rows()


def test_polars_dataset_preserves_rows():
    before = pl.read_csv(DATA_PATH)

    after = before.with_columns(
        (pl.col("sepal_length") * pl.col("sepal_width")).alias("sepal_area")
    )

    fw.expect(after, before=before).preserves_rows()
