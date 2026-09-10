"""Unit-aware formatting for `ComparisonResult.__str__`.

Centralizes the pp-vs-native-units decision
in one place, rather than each result class re-implementing its own
`if metric == "rate"` branch.
"""

from ._constants import Metric


def _is_proportion(metric: Metric | str) -> bool:
    """Whether `metric` should be rendered in percentage points.

    Only built-in `Metric` members opt into this; a plain `str` metric
    (a `.custom()` check's `name`) always renders in raw units.
    """
    return isinstance(metric, Metric) and metric == Metric.RATE


def metric_label(metric: Metric | str) -> str:
    """The display name for `metric`: a `Metric` member's `.value` (e.g.
    `"mean"`), or the string itself for a `.custom()` check's `name`.
    """
    return metric.value if isinstance(metric, Metric) else metric


def format_value(value: float, metric: Metric | str) -> str:
    """Format a signed value (a diff, bound, or threshold) in the
    metric's natural unit: percentage points for proportions, raw units
    otherwise.
    """
    if metric == Metric.RATE:
        return f"{value * 100:+.4g}pp"
    return f"{value:+.4g}"


def format_margin(value: float, metric: Metric | str) -> str:
    """Format an unsigned `±` margin (e.g. an equivalence `within`) in the
    metric's natural unit.
    """
    if metric == Metric.RATE:
        return f"±{value * 100:g}pp"
    return f"±{value:g}"


def format_point_value(value: float, metric: Metric | str) -> str:
    """Format an absolute point value (e.g. `before_value`/`after_value`)
    in the metric's natural unit.
    """
    if metric == Metric.RATE:
        return f"{value:.1%}"
    return f"{value:.4g}"


def format_point_values(before: float, after: float, metric: Metric | str) -> str:
    """Format the `before = ..., after = ..., ` prefix shown for
    proportions (where the absolute point value is useful context alongside
    the diff), or an empty string for metrics that don't need it.
    """
    if metric != Metric.RATE:
        return ""
    return (
        f"before = {format_point_value(before, metric)}, "
        f"after = {format_point_value(after, metric)}, "
    )
