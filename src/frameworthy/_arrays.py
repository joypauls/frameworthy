from collections.abc import Sequence

import narwhals.stable.v2 as nw
import numpy as np

from ._pairing import join_paired


def assert_column_exists(columns: Sequence[str], column: str, label: str) -> None:
    if column not in columns:
        raise KeyError(f"Column `{column}` not found in `{label}`.")


def assert_min_count(n: int, min_count: int, label: str) -> None:
    if n == 0:
        raise ValueError(f"No usable (non-null) values found for {label}.")
    if n < min_count:
        raise ValueError(
            f"At least {min_count} usable (non-null) values are required for "
            f"{label}, got {n}."
        )


def assert_equal_pairs(
    before_values: np.ndarray, after_values: np.ndarray, label: str
) -> None:
    if len(before_values) != len(after_values):
        raise ValueError(
            f"Paired comparison for {label} produced unequal numbers of usable "
            f"before ({len(before_values)}) and after ({len(after_values)}) "
            "values; before/after values must stay aligned pair-by-pair."
        )


def paired_values_from_columns(
    frame: nw.DataFrame,
    before_column: str,
    after_column: str,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract two same-frame columns as row-aligned paired arrays.

    Rows where either column is null are dropped together, so the pairing
    is preserved. Validates that at least 2 usable pairs remain.
    """
    assert_column_exists(frame.columns, before_column, label)
    assert_column_exists(frame.columns, after_column, label)

    paired = frame.select(before_column, after_column).drop_nulls(
        subset=[before_column, after_column]
    )
    before_values = paired[before_column].to_numpy()
    after_values = paired[after_column].to_numpy()

    assert_equal_pairs(before_values, after_values, label)
    assert_min_count(len(before_values), 2, label)

    return before_values, after_values


def values_from_two_frames(
    before_frame: nw.DataFrame,
    after_frame: nw.DataFrame,
    column: str,
    paired_by: list[str] | None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract before/after values for `column` from two separate frames.

    If `paired_by` is given, `before_frame` and `after_frame` are aligned
    on that key first and null pairs are dropped together. Otherwise the
    two sides are treated as independent samples, with nulls dropped
    independently on each side.

    Returns `(before_values, after_values, paired)`.
    """
    assert_column_exists(before_frame.columns, column, "before")
    assert_column_exists(after_frame.columns, column, "after")

    if paired_by is not None:
        after_column = f"{column}_after"
        joined = join_paired(before_frame, after_frame, paired_by, [column]).drop_nulls(
            subset=[column, after_column]
        )

        before_values = joined[column].to_numpy()
        after_values = joined[after_column].to_numpy()

        assert_equal_pairs(before_values, after_values, "before/after")
        assert_min_count(len(before_values), 2, "before/after")
        return before_values, after_values, True

    before_values = before_frame.select(column).drop_nulls()[column].to_numpy()
    after_values = after_frame.select(column).drop_nulls()[column].to_numpy()

    assert_min_count(len(before_values), 2, "before")
    assert_min_count(len(after_values), 2, "after")
    return before_values, after_values, False
