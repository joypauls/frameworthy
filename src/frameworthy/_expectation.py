from __future__ import annotations

import logging

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrame

from frameworthy._errors import FrameworthyAssertionError

logger = logging.getLogger(__name__)


class TransformationExpectation:
    """Assertions about the relationship between two DataFrame states"""

    def __init__(
        self,
        after: IntoDataFrame,
        *,
        relative_to: IntoDataFrame,
    ) -> None:
        self._before = nw.from_native(relative_to, eager_only=True)
        self._after = nw.from_native(after, eager_only=True)

        # should we store the native frames too?
        # self._before_native = relative_to
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


def expect(
    after: IntoDataFrame,
    *,
    relative_to: IntoDataFrame,
) -> TransformationExpectation:
    """Create expectations about a DataFrame relative to an earlier state"""
    return TransformationExpectation(
        after,
        relative_to=relative_to,
    )


def format_row_count_failure(
    *,
    before_rows: int,
    after_rows: int,
    difference: int,
) -> str:
    absolute_difference = abs(difference)
    row_wording = "row" if absolute_difference == 1 else "rows"
    direction = "introduced" if difference > 0 else "removed"

    if before_rows == 0:
        change_display = f"{difference:+,} {row_wording}"
    else:
        percentage = difference / before_rows
        change_display = f"{difference:+,} {row_wording} ({percentage:+.2%})"

    return (
        "Expected transformation to preserve row count.\n"
        "\n"
        f"before    {before_rows:,} rows\n"
        f"after     {after_rows:,} rows\n"
        f"change    {change_display}\n"
        "\n"
        f"The transformation {direction} {absolute_difference:,} {row_wording}."
    )
