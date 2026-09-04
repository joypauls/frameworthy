import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import narwhals.stable.v2 as nw

from ._constants import NULL_KEY


def _normalize_keys(key: str | Sequence[str]) -> list[str]:
    if isinstance(key, str):
        return [key]

    keys = list(key)
    if not keys:
        raise ValueError("At least one key column is required.")

    return keys


def _normalize_key_value(value: Any) -> Any:
    if value is None:
        return NULL_KEY
    if isinstance(value, float) and math.isnan(value):
        return NULL_KEY
    return value


def key_counts(
    df: nw.DataFrame,
    key: str | Sequence[str],
) -> Counter[tuple[Any, ...]]:
    """Count occurrences of each distinct key value combination in a dataframe."""
    keys = _normalize_keys(key)
    grouped = df.group_by(*keys, drop_null_keys=False).agg(
        nw.len().alias("__frameworthy_count")
    )
    counts: Counter[tuple[Any, ...]] = Counter()

    for row in grouped.iter_rows():
        *key_values, count = row
        normalized_key = tuple(_normalize_key_value(value) for value in key_values)
        counts[normalized_key] = int(count)

    return counts


def duplicate_keys(
    df: nw.DataFrame,
    key: str | Sequence[str],
) -> list[tuple[Any, ...]]:
    """Return key value combinations that occur more than once in a frame."""
    counts = key_counts(df, key)
    return [values for values, count in counts.items() if count > 1]


def assert_unique_keys(df: nw.DataFrame, keys: list[str], label: str) -> None:
    dupes = duplicate_keys(df, keys)
    if dupes:
        raise ValueError(
            f"`{label}` has duplicate values for key {keys}: {dupes[:5]}. "
            "Paired comparisons require exactly one row per key on both sides."
        )


def join_paired(
    before: nw.DataFrame,
    after: nw.DataFrame,
    key: str | Sequence[str],
    columns: Sequence[str],
) -> nw.DataFrame:
    """Align `before` and `after` on `key`, keeping only matched pairs."""
    keys = _normalize_keys(key)
    assert_unique_keys(before, keys, "before")
    assert_unique_keys(after, keys, "after")

    before_selected = before.select(*keys, *columns)
    after_selected = after.select(*keys, *columns)

    return before_selected.join(after_selected, on=keys, how="inner", suffix="_after")
