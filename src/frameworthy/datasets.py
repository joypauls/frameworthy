"""Tiny numpy-only sample-data generators for self-contained examples.

These exist so a `frameworthy` example can run without loading a real
dataset (or pandas/polars) at all.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampleData:
    """A `before`/`after` pair of 1-D numpy arrays for testing."""

    before: np.ndarray
    after: np.ndarray


def sample_normal(
    n: int = 1000,
    shift: float = 0.0,
    seed: int | None = None,
) -> SampleData:
    """Generate a `before`/`after` pair of normally-distributed samples.

    Pass the result straight to `check()`, e.g.:
        data = sample_normal(shift=0.5)
        fw.check(data.after, before=data.before).mean().equivalent(within=0.1)
    """
    rng = np.random.default_rng(seed)
    before = rng.normal(0.0, 1.0, n)
    after = rng.normal(shift, 1.0, n)
    return SampleData(before=before, after=after)


def sample_rate(
    n: int = 1000,
    p_before: float = 0.5,
    p_after: float = 0.5,
    seed: int | None = None,
) -> SampleData:
    """Generate a `before`/`after` pair of binary (0/1) samples, for
    `.rate()` examples.

    Pass the result straight to `check()`, e.g.:
        data = sample_rate(p_before=0.5, p_after=0.7)
        fw.check(data.after, before=data.before).rate().equivalent(within=0.1)
    """
    rng = np.random.default_rng(seed)
    before = rng.binomial(1, p_before, n)
    after = rng.binomial(1, p_after, n)
    return SampleData(before=before, after=after)
