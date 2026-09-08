from enum import Enum
from typing import Literal, NamedTuple, Protocol

import numpy as np

NULL_KEY = object()

DEFAULT_ALPHA = 0.05
DEFAULT_N_RESAMPLES = 10_000

# inference strategy options
InferenceMethod = Literal["analytical", "bootstrap"]
DEFAULT_INFERENCE_METHOD = "analytical"

# one-sided claim direction for `.change_greater_than()`/`.change_less_than()`
Direction = Literal["greater_than", "less_than"]


class Statistic(str, Enum):
    """Which built-in statistic a check compares between `before` and `after`.

    Being `str`-based keeps `result.statistic == "rate"`-style comparisons
    working, mirroring `Decision` in `decision.py`.
    """

    MEAN = "mean"
    RATE = "rate"
    MEDIAN = "median"

    @property
    def is_proportion(self) -> bool:
        """Whether this statistic's natural unit is a proportion, and
        should therefore be rendered in percentage points (rather than raw
        units) in result output.
        """
        return self is Statistic.RATE


class Interval(NamedTuple):
    """A point estimate and its confidence interval: `(diff, low, high)`.

    Returned by every `*_diff_ci` function in `_intervals.py`. Being a
    `NamedTuple`, it still unpacks positionally like the plain 3-tuple it
    replaces (`diff, low, high = interval`), so existing call sites and
    tuple-equality comparisons keep working unchanged.
    """

    diff: float
    low: float
    high: float


class DiffCIFunc(Protocol):
    """Shared call signature for `<metric>_diff_ci`: given
    paired or independent `before`/`after` arrays, estimate the difference
    `after - before` and its confidence interval under the given inference
    `method`.
    """

    def __call__(
        self,
        before: np.ndarray,
        after: np.ndarray,
        *,
        paired: bool,
        alpha: float,
        n_resamples: int,
        rng: np.random.Generator,
        method: InferenceMethod = ...,
    ) -> Interval: ...
