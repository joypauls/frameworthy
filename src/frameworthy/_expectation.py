from __future__ import annotations

import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrame

from frameworthy._errors import FrameworthyAssertionError


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

    def preserves_rows(self) -> None:
        """Assert that the transformation preserves row count"""
        before_rows = self._before.shape[0]
        after_rows = self._after.shape[0]

        if before_rows == after_rows:
            return

        difference = after_rows - before_rows

        raise FrameworthyAssertionError(
            format_row_count_failure(
                before_rows=before_rows,
                after_rows=after_rows,
                difference=difference,
            )
        )


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
    if before_rows == 0:
        percentage = None
    else:
        percentage = difference / before_rows

    direction = "introduced" if difference > 0 else "removed"
    absolute_difference = abs(difference)

    if percentage is None:
        change_display = f"{difference:+,} rows"
    else:
        change_display = f"{difference:+,} rows ({percentage:+.2%})"

    return (
        "Expected transformation to preserve row count.\n"
        "\n"
        f"before    {before_rows:,} rows\n"
        f"after     {after_rows:,} rows\n"
        f"change    {change_display}\n"
        "\n"
        f"The transformation {direction} "
        f"{absolute_difference:,} "
        f"{'row' if absolute_difference == 1 else 'rows'}."
    )
