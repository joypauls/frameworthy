import functools

import numpy as np
import pytest

from frameworthy._errors import InvalidDataError, UsageError
from frameworthy._validation import (
    InferenceConfig,
    validate_alpha,
    validate_custom_metric,
    validate_equal_length,
    validate_method,
    validate_min_observations,
    validate_n_resamples,
    validate_statistic_func,
)


class TestValidateAlpha:
    def test_accepts_values_in_open_range(self):
        validate_alpha(0.05)  # should not raise

    def test_rejects_invalid_alpha(self):
        with pytest.raises(UsageError, match="alpha"):
            validate_alpha(0.6)


class TestValidateNResamples:
    def test_accepts_positive_values(self):
        validate_n_resamples(100)  # should not raise

    def test_rejects_non_positive_values(self):
        with pytest.raises(UsageError, match="n_resamples"):
            validate_n_resamples(0)


class TestValidateMethod:
    def test_accepts_analytical_and_bootstrap(self):
        validate_method("analytical")  # should not raise
        validate_method("bootstrap")  # should not raise

    def test_rejects_unknown_method(self):
        with pytest.raises(UsageError, match="Unknown inference"):
            validate_method("magic")


class TestValidateMinObservations:
    def test_accepts_at_or_above_minimum(self):
        validate_min_observations(
            2, 2, context="paired observations"
        )  # should not raise

    def test_rejects_below_minimum(self):
        with pytest.raises(InvalidDataError, match="At least 2"):
            validate_min_observations(1, 2, context="paired observations")


class TestValidateEqualLength:
    def test_accepts_equal_length_arrays(self):
        validate_equal_length(
            np.array([1.0, 2.0]), np.array([3.0, 4.0]), context="test"
        )

    def test_rejects_unequal_length_arrays(self):
        with pytest.raises(InvalidDataError, match="same length"):
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


class TestValidateCustomMetric:
    def test_accepts_a_valid_name_and_callable(self):
        validate_custom_metric("p95_latency", np.mean)  # should not raise

    def test_rejects_empty_name(self):
        with pytest.raises(UsageError, match="`name`"):
            validate_custom_metric("", np.mean)

    def test_rejects_non_string_name(self):
        with pytest.raises(UsageError, match="`name`"):
            validate_custom_metric(123, np.mean)

    def test_rejects_non_callable_statistic_func(self):
        with pytest.raises(UsageError, match="`statistic_func`"):
            validate_custom_metric("p95_latency", "not callable")


class TestValidateStatisticFunc:
    """`validate_statistic_func` is `.custom()`'s fail-fast smoke test:
    it should accept anything that follows the `np.mean`/`np.median`
    convention, and reject (with a clear `UsageError`, not a raw numpy
    error) anything that doesn't return a scalar or doesn't support
    `axis=`.
    """

    def test_accepts_numpy_reduction_functions(self):
        sample = np.array([1.0, 2.0, 3.0, 4.0])

        validate_statistic_func(np.mean, sample)  # should not raise
        validate_statistic_func(np.median, sample)  # should not raise

    def test_accepts_functools_partial_bound_percentile(self):
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p95 = functools.partial(np.percentile, q=95)

        validate_statistic_func(p95, sample)  # should not raise

    def test_rejects_function_that_does_not_support_axis(self):
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        def p95(values):
            return np.percentile(values, 95)

        with pytest.raises(UsageError, match="axis"):
            validate_statistic_func(p95, sample)

    def test_rejects_function_that_does_not_return_a_scalar(self):
        sample = np.array([1.0, 2.0, 3.0])

        with pytest.raises(UsageError, match="single scalar"):
            validate_statistic_func(lambda x, axis=None: x, sample)

    def test_rejects_function_that_raises(self):
        sample = np.array([1.0, 2.0, 3.0])

        def broken(_values):
            raise RuntimeError("boom")

        with pytest.raises(UsageError, match="single scalar"):
            validate_statistic_func(broken, sample)
