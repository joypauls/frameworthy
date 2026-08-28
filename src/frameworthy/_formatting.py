from collections import Counter
from typing import Any

from frameworthy._constants import NULL_KEY


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


def format_key_names(keys: list[str]) -> str:
    return ", ".join(repr(key) for key in keys)


def format_key_component(value: Any) -> str:
    if value is NULL_KEY:
        return "<null>"

    return repr(value)


def format_key_value(
    keys: list[str],
    values: tuple[Any, ...],
) -> str:
    if len(keys) == 1:
        return format_key_component(values[0])

    return ", ".join(
        f"{key}={format_key_component(value)}"
        for key, value in zip(keys, values, strict=True)
    )


def format_key_failure(
    *,
    keys: list[str],
    missing: Counter[tuple[Any, ...]],
    added: Counter[tuple[Any, ...]],
) -> str:
    lines = [
        f"Expected transformation to preserve key {format_key_names(keys)}.",
        "",
        "Key population changed.",
    ]

    if missing:
        missing_count = sum(missing.values())
        noun = "occurrence" if missing_count == 1 else "occurrences"

        lines.extend(
            [
                "",
                f"missing    {missing_count:,} key {noun}",
            ]
        )
        for values, count in missing.most_common(5):
            lines.append(f"  {format_key_value(keys, values)} × {count}")

    if added:
        added_count = sum(added.values())
        noun = "occurrence" if added_count == 1 else "occurrences"

        lines.extend(
            [
                "",
                f"added      {added_count:,} key {noun}",
            ]
        )
        for values, count in added.most_common(5):
            lines.append(f"  {format_key_value(keys, values)} × {count}")

    return "\n".join(lines)
