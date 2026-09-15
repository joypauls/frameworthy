import numpy as np
import pytest

from frameworthy._distribution import wasserstein_distance_ci
from frameworthy._errors import InvalidDataError


class TestWassersteinDistanceCi:
    def test_identical_distribution_gives_near_zero_distance(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=10.0, scale=2.0, size=500)
        after = rng.normal(loc=10.0, scale=2.0, size=500)

        distance, ci_low, ci_high = wasserstein_distance_ci(
            before, after, alpha=0.05, n_resamples=500, rng=rng
        )

        assert distance == pytest.approx(0.0, abs=0.5)
        assert ci_low == pytest.approx(0.0, abs=0.5)
        assert ci_low <= ci_high

    def test_ci_low_never_negative(self):
        rng = np.random.default_rng(1)
        before = rng.normal(loc=0.0, scale=1.0, size=50)
        after = rng.normal(loc=0.0, scale=1.0, size=50)

        _, ci_low, ci_high = wasserstein_distance_ci(
            before, after, alpha=0.05, n_resamples=200, rng=rng
        )

        assert ci_low >= 0.0
        assert ci_low <= ci_high

    def test_ci_bounds_stay_ordered_for_a_clear_shift(self):
        rng = np.random.default_rng(2)
        before = rng.normal(loc=0.0, scale=1.0, size=300)
        after = rng.normal(loc=20.0, scale=1.0, size=300)

        distance, ci_low, ci_high = wasserstein_distance_ci(
            before, after, alpha=0.05, n_resamples=500, rng=rng
        )

        assert distance == pytest.approx(20.0, abs=1.0)
        assert 0.0 < ci_low <= distance <= ci_high

    def test_is_reproducible_with_seeded_rng(self):
        before = np.linspace(0.0, 10.0, 100)
        after = np.linspace(1.0, 11.0, 100)

        result_a = wasserstein_distance_ci(
            before, after, alpha=0.05, n_resamples=300, rng=np.random.default_rng(42)
        )
        result_b = wasserstein_distance_ci(
            before, after, alpha=0.05, n_resamples=300, rng=np.random.default_rng(42)
        )

        assert result_a == result_b

    def test_raises_on_too_few_observations(self):
        rng = np.random.default_rng(0)
        before = np.array([1.0, 2.0, 3.0])
        after = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        with pytest.raises(InvalidDataError, match="before observations"):
            wasserstein_distance_ci(before, after, alpha=0.05, n_resamples=100, rng=rng)
