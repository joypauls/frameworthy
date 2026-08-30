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
    format_row_count_failure,
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

    def preserves_rows(self, throw: bool = True) -> None:
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
        if throw:
            raise FrameworthyAssertionError(formatted_message)
        else:
            logger.warning(formatted_message)

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
