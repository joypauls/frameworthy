from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrame

from frameworthy._constants import NULL_KEY
from frameworthy._errors import FrameworthyAssertionError
from frameworthy._formatting import (
    format_key_failure,
    format_key_names,
    format_key_value,
    format_row_count_failure,
    format_value_mismatch_failure,
    format_value_population_failure,
)

logger = logging.getLogger(__name__)


def _normalize_keys(
    key: str | Sequence[str],
) -> list[str]:
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


def _key_counts(
    frame: nw.DataFrame,
    keys: list[str],
) -> Counter[tuple[Any, ...]]:
    grouped = frame.group_by(*keys, drop_null_keys=False).agg(
        nw.len().alias("__frameworthy_count")
    )
    counts = Counter()

    for row in grouped.iter_rows():
        *key_values, count = row
        normalized_key = tuple(_normalize_key_value(value) for value in key_values)
        counts[normalized_key] = int(count)

    return counts


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _values_equal(before: Any, after: Any) -> bool:
    if _is_missing(before) or _is_missing(after):
        return _is_missing(before) and _is_missing(after)
    return bool(before == after)


class TransformationExpectation:
    """Assertions about the relationship between two DataFrame states"""

    def __init__(
        self,
        after: IntoDataFrame,
        *,
        before: IntoDataFrame,
    ) -> None:
        self._before = nw.from_native(before, eager_only=True)
        self._after = nw.from_native(after, eager_only=True)

        # should we store the native frames too?
        # self._before_native = before
        # self._after_native = after

    def preserves_rows(self) -> None:
        """Assert that the transformation preserves row count"""
        before_rows = self._before.shape[0]
        after_rows = self._after.shape[0]
        if before_rows == after_rows:
            return
        difference = after_rows - before_rows

        formatted_message = format_row_count_failure(
            before_rows=before_rows,
            after_rows=after_rows,
            difference=difference,
        )
        raise FrameworthyAssertionError(formatted_message)

    def preserves_key(
        self,
        key: str | Sequence[str],
    ) -> None:
        """Assert that key values and their multiplicities are preserved."""
        keys = _normalize_keys(key)
        missing_before = [
            column for column in keys if column not in self._before.columns
        ]

        if missing_before:
            raise ValueError(
                f"Key columns not found in reference frame: {', '.join(missing_before)}"
            )
        missing_after = [column for column in keys if column not in self._after.columns]

        if missing_after:
            raise FrameworthyAssertionError(
                "Expected transformation to preserve key "
                f"{format_key_names(keys)}, but the following "
                "key columns are missing from the result: "
                f"{', '.join(missing_after)}"
            )

        before_counts = _key_counts(self._before, keys)
        after_counts = _key_counts(self._after, keys)

        if before_counts == after_counts:
            return

        missing = before_counts - after_counts
        added = after_counts - before_counts

        raise FrameworthyAssertionError(
            format_key_failure(
                keys=keys,
                missing=missing,
                added=added,
            )
        )

    def preserves_values(
        self,
        *columns: str,
        on: str | Sequence[str],
    ) -> None:
        """Assert that value columns are unchanged for rows aligned by key(s)."""
        if not columns:
            raise ValueError("At least one value column is required.")

        keys = _normalize_keys(on)
        value_columns = list(columns)

        overlap = sorted(set(keys) & set(value_columns))
        if overlap:
            raise ValueError(
                "Columns cannot be used as both alignment key and value "
                f"columns: {', '.join(overlap)}"
            )

        required = keys + value_columns
        missing_before = [
            column for column in required if column not in self._before.columns
        ]
        if missing_before:
            raise ValueError(
                f"Columns not found in reference frame: {', '.join(missing_before)}"
            )

        missing_after = [
            column for column in required if column not in self._after.columns
        ]
        if missing_after:
            raise FrameworthyAssertionError(
                f"Expected values in {format_key_names(value_columns)} to be "
                f"preserved for rows aligned on {format_key_names(keys)}, but "
                "the following columns are missing from the result: "
                f"{', '.join(missing_after)}"
            )

        before_counts = _key_counts(self._before, keys)
        duplicated_before = [key for key, count in before_counts.items() if count > 1]
        if duplicated_before:
            sample = ", ".join(
                format_key_value(keys, key) for key in duplicated_before[:5]
            )
            raise ValueError(
                "Alignment key must uniquely identify rows in the reference "
                f"frame, but found duplicate keys: {sample}"
            )

        after_counts = _key_counts(self._after, keys)
        duplicated_after = [key for key, count in after_counts.items() if count > 1]
        if duplicated_after:
            sample = ", ".join(
                format_key_value(keys, key) for key in duplicated_after[:5]
            )
            raise ValueError(
                "Alignment key must uniquely identify rows in the result "
                f"frame, but found duplicate keys: {sample}"
            )

        before_key_set = set(before_counts)
        after_key_set = set(after_counts)
        if before_key_set != after_key_set:
            raise FrameworthyAssertionError(
                format_value_population_failure(
                    keys=keys,
                    columns=value_columns,
                    missing=before_key_set - after_key_set,
                    added=after_key_set - before_key_set,
                )
            )

        before_rows = self._before.select(*keys, *value_columns)
        after_rows = self._after.select(*keys, *value_columns)
        n_keys = len(keys)

        after_values = {}
        for row in after_rows.iter_rows():
            key_tuple = tuple(_normalize_key_value(value) for value in row[:n_keys])
            after_values[key_tuple] = row[n_keys:]

        mismatch_count = 0
        sample: list[tuple[tuple[Any, ...], list[tuple[str, Any, Any]]]] = []

        for row in before_rows.iter_rows():
            key_tuple = tuple(_normalize_key_value(value) for value in row[:n_keys])
            before_vals = row[n_keys:]
            after_vals = after_values[key_tuple]

            diffs = [
                (column, before_val, after_val)
                for column, before_val, after_val in zip(
                    value_columns, before_vals, after_vals, strict=True
                )
                if not _values_equal(before_val, after_val)
            ]

            if diffs:
                mismatch_count += 1
                if len(sample) < 5:
                    sample.append((key_tuple, diffs))

        if mismatch_count == 0:
            return

        raise FrameworthyAssertionError(
            format_value_mismatch_failure(
                keys=keys,
                columns=value_columns,
                mismatch_count=mismatch_count,
                sample=sample,
            )
        )


# main interface for end users
def expect(
    after: IntoDataFrame,
    *,
    before: IntoDataFrame,
) -> TransformationExpectation:
    """Create expectations about a DataFrame relative to an earlier state"""
    return TransformationExpectation(
        after,
        before=before,
    )
