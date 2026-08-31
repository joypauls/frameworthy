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


def format_value_population_failure(
    *,
    keys: list[str],
    columns: list[str],
    missing: set[tuple[Any, ...]],
    added: set[tuple[Any, ...]],
) -> str:
    lines = [
        (
            f"Expected transformation to preserve values for "
            f"{format_key_names(columns)} aligned on {format_key_names(keys)}."
        ),
        "",
        "Alignment key population changed.",
    ]

    if missing:
        noun = "key" if len(missing) == 1 else "keys"
        lines.extend(
            [
                "",
                f"missing    {len(missing):,} {noun}",
            ]
        )
        for values in sorted(missing, key=repr)[:5]:
            lines.append(f"  {format_key_value(keys, values)}")

    if added:
        noun = "key" if len(added) == 1 else "keys"
        lines.extend(
            [
                "",
                f"added      {len(added):,} {noun}",
            ]
        )
        for values in sorted(added, key=repr)[:5]:
            lines.append(f"  {format_key_value(keys, values)}")

    return "\n".join(lines)


def format_value_mismatch_failure(
    *,
    keys: list[str],
    columns: list[str],
    mismatch_count: int,
    sample: list[tuple[tuple[Any, ...], list[tuple[str, Any, Any]]]],
) -> str:
    noun = "row" if mismatch_count == 1 else "rows"
    lines = [
        (
            f"Expected values in {format_key_names(columns)} to be preserved "
            f"for rows aligned on {format_key_names(keys)}."
        ),
        "",
        f"mismatched    {mismatch_count:,} {noun}",
        "",
    ]

    for key_values, diffs in sample:
        key_display = format_key_value(keys, key_values)
        diff_display = ", ".join(
            f"{column}: {before!r} \u2192 {after!r}" for column, before, after in diffs
        )
        lines.append(f"  {key_display}: {diff_display}")

    return "\n".join(lines)


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
