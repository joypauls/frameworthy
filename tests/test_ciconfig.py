import pytest

from frameworthy._ciconfig import CIConfig
from frameworthy._errors import UsageError


class TestCIConfig:
    # `__post_init__` eagerly calling `validate_alpha`/`validate_n_resamples`/
    # `validate_method` (each already unit-tested in `test_validation.py`)
    # is exercised end-to-end by `test_check.py`'s
    # `test_rejects_invalid_alpha_before_computing_ci`/
    # `test_rejects_unknown_method`, so it isn't re-tested here.

    def test_rejects_invalid_alpha(self):
        with pytest.raises(UsageError, match="alpha"):
            CIConfig(alpha=0.6)

    def test_rejects_non_positive_n_resamples(self):
        with pytest.raises(UsageError, match="n_resamples"):
            CIConfig(n_resamples=0)

    def test_rejects_unknown_method(self):
        with pytest.raises(UsageError, match="Unknown inference"):
            CIConfig(method="magic")

    def test_reported_n_resamples_is_zero_for_analytical(self):
        config = CIConfig(n_resamples=1000, method="analytical")

        assert config.reported_n_resamples == 0

    def test_reported_n_resamples_is_n_resamples_for_bootstrap(self):
        config = CIConfig(n_resamples=1000, method="bootstrap")

        assert config.reported_n_resamples == 1000

    def test_rng_is_reproducible_with_seeded_random_state(self):
        config_a = CIConfig(random_state=42)
        config_b = CIConfig(random_state=42)

        assert config_a.rng.integers(0, 1000) == config_b.rng.integers(0, 1000)
