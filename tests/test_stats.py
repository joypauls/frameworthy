import numpy as np
import pytest

from frameworthy._stats import bootstrap_mean_diff_ci, classify_equivalence


class TestBootstrapMeanDiffCi:
    def test_paired_recovers_known_constant_shift(self):
        rng = np.random.default_rng(0)
        before = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        after = before + 1.0  # constant shift, zero variance in diffs

        observed, ci_low, ci_high = bootstrap_mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=1000,
            rng=rng,
        )

        assert observed == pytest.approx(1.0)
        assert ci_low == pytest.approx(1.0)
        assert ci_high == pytest.approx(1.0)

    def test_paired_requires_equal_length(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same length"):
            bootstrap_mean_diff_ci(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                paired=True,
                alpha=0.05,
                n_resamples=100,
                rng=rng,
            )

    def test_paired_requires_at_least_two_observations(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="At least 2"):
            bootstrap_mean_diff_ci(
                np.array([1.0]),
                np.array([2.0]),
                paired=True,
                alpha=0.05,
                n_resamples=100,
                rng=rng,
            )

    def test_unpaired_recovers_approximate_shift(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=10.0, scale=0.01, size=500)
        after = rng.normal(loc=11.0, scale=0.01, size=600)

        observed, ci_low, ci_high = bootstrap_mean_diff_ci(
            before,
            after,
            paired=False,
            alpha=0.05,
            n_resamples=2000,
            rng=rng,
        )

        assert observed == pytest.approx(1.0, abs=0.05)
        assert ci_low < observed < ci_high

    def test_unpaired_allows_different_lengths(self):
        rng = np.random.default_rng(0)
        before = np.array([1.0, 2.0, 3.0])
        after = np.array([4.0, 5.0])

        observed, ci_low, ci_high = bootstrap_mean_diff_ci(
            before,
            after,
            paired=False,
            alpha=0.05,
            n_resamples=100,
            rng=rng,
        )

        assert observed == pytest.approx(4.5 - 2.0)
        assert ci_low <= ci_high

    def test_unpaired_requires_at_least_two_observations_per_side(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="At least 2"):
            bootstrap_mean_diff_ci(
                np.array([1.0]),
                np.array([2.0, 3.0]),
                paired=False,
                alpha=0.05,
                n_resamples=100,
                rng=rng,
            )

    def test_rejects_invalid_alpha(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="alpha"):
            bootstrap_mean_diff_ci(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                paired=True,
                alpha=0.6,
                n_resamples=100,
                rng=rng,
            )

    def test_is_reproducible_with_seeded_rng(self):
        before = np.array([1.0, 2.0, 3.0, 4.0])
        after = np.array([2.0, 2.0, 5.0, 3.0])

        result_a = bootstrap_mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(42),
        )
        result_b = bootstrap_mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(42),
        )

        assert result_a == result_b


class TestClassifyEquivalence:
    def test_equivalent_when_ci_fully_inside_margin(self):
        assert classify_equivalence(-1.0, 1.0, within=2.0) == "equivalent"

    def test_changed_when_ci_fully_outside_margin_above(self):
        assert classify_equivalence(3.0, 5.0, within=2.0) == "changed"

    def test_changed_when_ci_fully_outside_margin_below(self):
        assert classify_equivalence(-5.0, -3.0, within=2.0) == "changed"

    def test_inconclusive_when_ci_straddles_upper_margin(self):
        assert classify_equivalence(1.0, 3.0, within=2.0) == "inconclusive"

    def test_inconclusive_when_ci_straddles_lower_margin(self):
        assert classify_equivalence(-3.0, -1.0, within=2.0) == "inconclusive"

    def test_inconclusive_when_ci_spans_both_margins(self):
        assert classify_equivalence(-5.0, 5.0, within=2.0) == "inconclusive"

    def test_rejects_non_positive_within(self):
        with pytest.raises(ValueError, match="within"):
            classify_equivalence(-1.0, 1.0, within=0.0)
