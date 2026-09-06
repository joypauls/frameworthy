import numpy as np
import pytest

from frameworthy._errors import (
    InsufficientDataError,
    InvalidColumnDataError,
    InvalidParameterError,
)
from frameworthy._validation import (
    InferenceConfig,
    validate_alpha,
    validate_equal_length,
    validate_method,
    validate_min_observations,
    validate_n_resamples,
)


class TestValidateAlpha:
    def test_accepts_values_in_open_range(self):
        validate_alpha(0.05)  # should not raise

    def test_rejects_invalid_alpha(self):
        with pytest.raises(InvalidParameterError, match="alpha"):
            validate_alpha(0.6)


class TestValidateNResamples:
    def test_accepts_positive_values(self):
        validate_n_resamples(100)  # should not raise

    def test_rejects_non_positive_values(self):
        with pytest.raises(InvalidParameterError, match="n_resamples"):
            validate_n_resamples(0)


class TestValidateMethod:
    def test_accepts_analytical_and_bootstrap(self):
        validate_method("analytical")  # should not raise
        validate_method("bootstrap")  # should not raise

    def test_rejects_unknown_method(self):
        with pytest.raises(InvalidParameterError, match="Unknown inference"):
            validate_method("magic")


class TestValidateMinObservations:
    def test_accepts_at_or_above_minimum(self):
        validate_min_observations(
            2, 2, context="paired observations"
        )  # should not raise

    def test_rejects_below_minimum(self):
        with pytest.raises(InsufficientDataError, match="At least 2"):
            validate_min_observations(1, 2, context="paired observations")


class TestValidateEqualLength:
    def test_accepts_equal_length_arrays(self):
        validate_equal_length(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]), context="test"
        )

    def test_rejects_unequal_length_arrays(self):
        with pytest.raises(InvalidColumnDataError, match="same length"):
            validate_equal_length(
                np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0]), context="test"
            )


class TestInferenceConfig:
    # `__post_init__` eagerly calling `validate_alpha`/`validate_n_resamples`/
    # `validate_method` (each already unit-tested above) is exercised
    # end-to-end by `test_check.py`'s `test_rejects_invalid_alpha_before_
    # computing_ci`/`test_rejects_unknown_method`, so it isn't re-tested here.

    def test_reported_n_resamples_is_zero_for_analytical(self):
        config = InferenceConfig(n_resamples=1000, method="analytical")

        assert config.reported_n_resamples == 0

    def test_reported_n_resamples_is_n_resamples_for_bootstrap(self):
        config = InferenceConfig(n_resamples=1000, method="bootstrap")

        assert config.reported_n_resamples == 1000

    def test_rng_is_reproducible_with_seeded_random_state(self):
        config_a = InferenceConfig(random_state=42)
        config_b = InferenceConfig(random_state=42)

        assert config_a.rng.integers(0, 1000) == config_b.rng.integers(0, 1000)
