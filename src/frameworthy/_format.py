"""Unit-aware formatting for `ComparisonResult.__str__`.

Centralizes the pp-vs-native-units decision (driven by `Statistic.is_proportion`)
in one place, rather than each result class re-implementing its own
`if statistic == "rate"` branch.
"""

from ._constants import Statistic


def format_value(value: float, statistic: Statistic) -> str:
    """Format a signed value (a diff, bound, or threshold) in the
    statistic's natural unit: percentage points for proportions, raw units
    otherwise.
    """
    if statistic.is_proportion:
        return f"{value * 100:+.4g}pp"
    return f"{value:+.4g}"


def format_margin(value: float, statistic: Statistic) -> str:
    """Format an unsigned `±` margin (e.g. an equivalence `within`) in the
    statistic's natural unit.
    """
    if statistic.is_proportion:
        return f"±{value * 100:g}pp"
    return f"±{value:g}"


def format_level(value: float, statistic: Statistic) -> str:
    """Format an absolute level (e.g. `before_mean`/`after_mean`) in the
    statistic's natural unit.
    """
    if statistic.is_proportion:
        return f"{value:.1%}"
    return f"{value:.4g}"


def format_levels(before: float, after: float, statistic: Statistic) -> str:
    """Format the `before = ..., after = ..., ` prefix shown for
    proportions (where the absolute level is useful context alongside the
    diff), or an empty string for statistics that don't need it.
    """
    if not statistic.is_proportion:
        return ""
    return (
        f"before = {format_level(before, statistic)}, "
        f"after = {format_level(after, statistic)}, "
    )
