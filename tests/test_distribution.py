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
            before, after, paired=False, alpha=0.05, n_resamples=500, rng=rng
        )

        assert distance == pytest.approx(0.0, abs=0.5)
        assert ci_low == pytest.approx(0.0, abs=0.5)
        assert ci_low <= ci_high

    def test_ci_low_never_negative(self):
        rng = np.random.default_rng(1)
        before = rng.normal(loc=0.0, scale=1.0, size=50)
        after = rng.normal(loc=0.0, scale=1.0, size=50)

        _, ci_low, ci_high = wasserstein_distance_ci(
            before, after, paired=False, alpha=0.05, n_resamples=200, rng=rng
        )

        assert ci_low >= 0.0
        assert ci_low <= ci_high

    def test_ci_bounds_stay_ordered_for_a_clear_shift(self):
        rng = np.random.default_rng(2)
        before = rng.normal(loc=0.0, scale=1.0, size=300)
        after = rng.normal(loc=20.0, scale=1.0, size=300)

        distance, ci_low, ci_high = wasserstein_distance_ci(
            before, after, paired=False, alpha=0.05, n_resamples=500, rng=rng
        )

        assert distance == pytest.approx(20.0, abs=1.0)
        assert 0.0 < ci_low <= distance <= ci_high

    def test_is_reproducible_with_seeded_rng(self):
        before = np.linspace(0.0, 10.0, 100)
        after = np.linspace(1.0, 11.0, 100)

        result_a = wasserstein_distance_ci(
            before,
            after,
            paired=False,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )
        result_b = wasserstein_distance_ci(
            before,
            after,
            paired=False,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )

        assert result_a == result_b

    def test_raises_on_too_few_observations(self):
        rng = np.random.default_rng(0)
        before = np.array([1.0, 2.0, 3.0])
        after = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        with pytest.raises(InvalidDataError, match="before observations"):
            wasserstein_distance_ci(
                before, after, paired=False, alpha=0.05, n_resamples=100, rng=rng
            )


class TestWassersteinDistanceCiPaired:
    def test_identical_distribution_gives_near_zero_distance(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=10.0, scale=2.0, size=500)
        after = before + rng.normal(loc=0.0, scale=0.1, size=500)

        distance, ci_low, ci_high = wasserstein_distance_ci(
            before, after, paired=True, alpha=0.05, n_resamples=500, rng=rng
        )

        assert distance == pytest.approx(0.0, abs=0.5)
        assert ci_low == pytest.approx(0.0, abs=0.5)
        assert ci_low <= ci_high

    def test_ci_low_never_negative(self):
        rng = np.random.default_rng(1)
        before = rng.normal(loc=0.0, scale=1.0, size=50)
        after = before + rng.normal(loc=0.0, scale=0.2, size=50)

        _, ci_low, ci_high = wasserstein_distance_ci(
            before, after, paired=True, alpha=0.05, n_resamples=200, rng=rng
        )

        assert ci_low >= 0.0
        assert ci_low <= ci_high

    def test_is_reproducible_with_seeded_rng(self):
        before = np.linspace(0.0, 10.0, 100)
        after = before + 1.0

        result_a = wasserstein_distance_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )
        result_b = wasserstein_distance_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )

        assert result_a == result_b

    def test_raises_on_mismatched_lengths(self):
        rng = np.random.default_rng(0)
        before = rng.normal(size=20)
        after = rng.normal(size=25)

        with pytest.raises(InvalidDataError, match="same length"):
            wasserstein_distance_ci(
                before, after, paired=True, alpha=0.05, n_resamples=100, rng=rng
            )

    def test_raises_on_too_few_observations(self):
        rng = np.random.default_rng(0)
        before = np.array([1.0, 2.0, 3.0])
        after = np.array([1.5, 2.5, 3.5])

        with pytest.raises(InvalidDataError, match="paired observations"):
            wasserstein_distance_ci(
                before, after, paired=True, alpha=0.05, n_resamples=100, rng=rng
            )

    def test_narrower_ci_than_unpaired_for_strongly_correlated_data(self):
        # a strong per-unit correlation between before/after should let
        # the paired bootstrap (which resamples matched pairs together)
        # produce a tighter CI than treating the same data as independent
        # samples, which throws away that correlation.
        rng = np.random.default_rng(3)
        before = rng.normal(loc=0.0, scale=5.0, size=300)
        after = before + 10.0 + rng.normal(loc=0.0, scale=0.1, size=300)

        _, paired_low, paired_high = wasserstein_distance_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(0),
        )
        _, unpaired_low, unpaired_high = wasserstein_distance_ci(
            before,
            after,
            paired=False,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(0),
        )

        assert (paired_high - paired_low) < (unpaired_high - unpaired_low)
