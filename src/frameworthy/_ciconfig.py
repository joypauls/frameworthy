"""Stateful configuration built on top of `_validation.py`'s stateless
guard functions: bundles the validated `alpha`/`n_resamples`/
`random_state`/`method` parameters used to compute a confidence interval
together with the derived values (an RNG, a reporting rule) every check
needs.

Shared by `check.py`'s `MetricCheck` (which exposes `method=` as a public
analytical-vs-bootstrap switch) and `DistributionCheck` (which always
subsample-bootstraps and so always passes `method="bootstrap"` explicitly
rather than relying on the dataclass default).
"""

from dataclasses import dataclass

import numpy as np

from ._constants import (
    DEFAULT_ALPHA,
    DEFAULT_INFERENCE_METHOD,
    DEFAULT_N_RESAMPLES,
    InferenceMethod,
)
from ._validation import validate_alpha, validate_method, validate_n_resamples


@dataclass(frozen=True)
class CIConfig:
    """Validated bundle of the `alpha`/`n_resamples`/`random_state`/
    `method` parameters used to compute a confidence interval, shared by
    every check.

    Validating in `__post_init__` means an invalid `alpha` or unknown
    `method` fails as soon as a claim method (e.g. `.equivalent()`) is
    called, rather than after `before`/`after` data has already been
    extracted and validated, partway through CI computation.
    """

    alpha: float = DEFAULT_ALPHA
    n_resamples: int = DEFAULT_N_RESAMPLES
    random_state: int | np.random.Generator | None = None
    method: InferenceMethod = DEFAULT_INFERENCE_METHOD

    def __post_init__(self) -> None:
        validate_alpha(self.alpha)
        validate_n_resamples(self.n_resamples)
        validate_method(self.method)

    @property
    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    @property
    def reported_n_resamples(self) -> int:
        """`n_resamples` if it was actually used (bootstrap), else `0` --
        the rule both `EquivalenceResult` and `ChangeResult` apply to their
        own `n_resamples` field. `DistributionResult` doesn't use this: it
        always passes `method="bootstrap"`, so its own `n_resamples` field
        is always `n_resamples` as-is.
        """
        return self.n_resamples if self.method == "bootstrap" else 0
