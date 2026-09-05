import numpy as np
import pytest
from scipy import stats as scipy_stats

from frameworthy._stats import (
    analytical_mean_diff_ci,
    bootstrap_mean_diff_ci,
    classify_equivalence,
    mean_diff_ci,
)


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


class TestAnalyticalMeanDiffCi:
    def test_paired_recovers_known_constant_shift(self):
        before = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        after = before + 1.0  # constant shift, zero variance in diffs

        observed, ci_low, ci_high = analytical_mean_diff_ci(
            before, after, paired=True, alpha=0.05
        )

        assert observed == pytest.approx(1.0)
        assert ci_low == pytest.approx(1.0)
        assert ci_high == pytest.approx(1.0)

    def test_paired_matches_scipy_ttest(self):
        rng = np.random.default_rng(1)
        before = rng.normal(50.0, 5.0, size=20)
        after = before + rng.normal(1.0, 2.0, size=20)

        observed, ci_low, ci_high = analytical_mean_diff_ci(
            before, after, paired=True, alpha=0.05
        )

        ref_ci = scipy_stats.ttest_1samp(after - before, popmean=0).confidence_interval(
            confidence_level=0.90
        )

        assert observed == pytest.approx((after - before).mean())
        assert ci_low == pytest.approx(ref_ci.low)
        assert ci_high == pytest.approx(ref_ci.high)

    def test_paired_requires_equal_length(self):
        with pytest.raises(ValueError, match="same length"):
            analytical_mean_diff_ci(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                paired=True,
                alpha=0.05,
            )

    def test_paired_requires_at_least_two_observations(self):
        with pytest.raises(ValueError, match="At least 2"):
            analytical_mean_diff_ci(
                np.array([1.0]), np.array([2.0]), paired=True, alpha=0.05
            )

    def test_unpaired_recovers_approximate_shift(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=10.0, scale=0.01, size=500)
        after = rng.normal(loc=11.0, scale=0.01, size=600)

        observed, ci_low, ci_high = analytical_mean_diff_ci(
            before, after, paired=False, alpha=0.05
        )

        assert observed == pytest.approx(1.0, abs=0.05)
        assert ci_low < observed < ci_high

    def test_unpaired_matches_scipy_welch_ttest(self):
        rng = np.random.default_rng(2)
        before = rng.normal(50.0, 5.0, size=30)
        after = rng.normal(52.0, 12.0, size=45)  # deliberately unequal variance

        observed, ci_low, ci_high = analytical_mean_diff_ci(
            before, after, paired=False, alpha=0.05
        )

        ref_ci = scipy_stats.ttest_ind(
            after, before, equal_var=False
        ).confidence_interval(confidence_level=0.90)

        assert observed == pytest.approx(after.mean() - before.mean())
        assert ci_low == pytest.approx(ref_ci.low)
        assert ci_high == pytest.approx(ref_ci.high)

    def test_unpaired_allows_different_lengths(self):
        before = np.array([1.0, 2.0, 3.0])
        after = np.array([4.0, 5.0])

        observed, ci_low, ci_high = analytical_mean_diff_ci(
            before, after, paired=False, alpha=0.05
        )

        assert observed == pytest.approx(4.5 - 2.0)
        assert ci_low <= ci_high

    def test_unpaired_requires_at_least_two_observations_per_side(self):
        with pytest.raises(ValueError, match="At least 2"):
            analytical_mean_diff_ci(
                np.array([1.0]), np.array([2.0, 3.0]), paired=False, alpha=0.05
            )

    def test_unpaired_handles_zero_variance_on_both_sides(self):
        before = np.array([5.0, 5.0, 5.0])
        after = np.array([6.0, 6.0, 6.0, 6.0])

        observed, ci_low, ci_high = analytical_mean_diff_ci(
            before, after, paired=False, alpha=0.05
        )

        assert observed == pytest.approx(1.0)
        assert ci_low == pytest.approx(1.0)
        assert ci_high == pytest.approx(1.0)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            analytical_mean_diff_ci(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                paired=True,
                alpha=0.6,
            )

    def test_is_deterministic(self):
        before = np.array([1.0, 2.0, 3.0, 4.0])
        after = np.array([2.0, 2.0, 5.0, 3.0])

        result_a = analytical_mean_diff_ci(before, after, paired=True, alpha=0.05)
        result_b = analytical_mean_diff_ci(before, after, paired=True, alpha=0.05)

        assert result_a == result_b

    def test_paired_and_bootstrap_are_reasonably_consistent(self):
        rng = np.random.default_rng(7)
        before = rng.normal(100.0, 5.0, size=200)
        after = before + rng.normal(1.0, 2.0, size=200)

        analytical = analytical_mean_diff_ci(before, after, paired=True, alpha=0.05)
        bootstrap = bootstrap_mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=5000,
            rng=np.random.default_rng(8),
        )

        assert analytical[0] == pytest.approx(bootstrap[0], abs=1e-9)
        assert analytical[1] == pytest.approx(bootstrap[1], abs=0.3)
        assert analytical[2] == pytest.approx(bootstrap[2], abs=0.3)

    def test_unpaired_and_bootstrap_are_reasonably_consistent(self):
        rng = np.random.default_rng(9)
        before = rng.normal(100.0, 5.0, size=300)
        after = rng.normal(101.0, 6.0, size=350)

        analytical = analytical_mean_diff_ci(before, after, paired=False, alpha=0.05)
        bootstrap = bootstrap_mean_diff_ci(
            before,
            after,
            paired=False,
            alpha=0.05,
            n_resamples=5000,
            rng=np.random.default_rng(10),
        )

        assert analytical[0] == pytest.approx(bootstrap[0], abs=1e-9)
        assert analytical[1] == pytest.approx(bootstrap[1], abs=0.3)
        assert analytical[2] == pytest.approx(bootstrap[2], abs=0.3)


class TestMeanDiffCiDispatcher:
    def test_default_method_is_analytical(self):
        before = np.array([10.0, 20.0, 30.0])
        after = before + 1.0

        result = mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=100,
            rng=np.random.default_rng(0),
        )

        assert result == analytical_mean_diff_ci(before, after, paired=True, alpha=0.05)

    def test_method_analytical_matches_direct_call(self):
        before = np.array([10.0, 20.0, 30.0])
        after = before + 1.0

        result = mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=100,
            rng=np.random.default_rng(0),
            method="analytical",
        )

        assert result == analytical_mean_diff_ci(before, after, paired=True, alpha=0.05)

    def test_method_bootstrap_matches_direct_call(self):
        before = np.array([10.0, 20.0, 30.0, 40.0])
        after = np.array([11.0, 19.0, 33.0, 42.0])

        result = mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(42),
            method="bootstrap",
        )
        expected = bootstrap_mean_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(42),
        )

        assert result == expected

    def test_rejects_unknown_method(self):
        before = np.array([10.0, 20.0, 30.0])
        after = before + 1.0

        with pytest.raises(ValueError, match="Unknown inference"):
            mean_diff_ci(
                before,
                after,
                paired=True,
                alpha=0.05,
                n_resamples=100,
                rng=np.random.default_rng(0),
                method="magic",
            )


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
