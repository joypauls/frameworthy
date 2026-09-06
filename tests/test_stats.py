import numpy as np
import pytest
from scipy import stats as scipy_stats

from frameworthy._errors import (
    InsufficientDataError,
    InvalidColumnDataError,
    InvalidParameterError,
)
from frameworthy._stats import (
    _validate_alpha,
    analytical_mean_diff_ci,
    analytical_rate_diff_ci,
    bootstrap_mean_diff_ci,
    classify_change_bound,
    classify_equivalence,
    independent_rate_diff_ci,
    mean_diff_ci,
    paired_rate_diff_ci,
    rate_diff_ci,
    wilson_interval,
)


class TestValidateAlpha:
    """`_validate_alpha` is the single guard shared by every function below
    that takes an `alpha`; each of those call sites is covered once here
    rather than re-testing the same one-line check at every call site.
    """

    def test_accepts_values_in_open_range(self):
        _validate_alpha(0.05)  # should not raise

    def test_rejects_invalid_alpha(self):
        with pytest.raises(InvalidParameterError, match="alpha"):
            _validate_alpha(0.6)


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
        with pytest.raises(InvalidColumnDataError, match="same length"):
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
        with pytest.raises(InsufficientDataError, match="At least 2"):
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
        with pytest.raises(InsufficientDataError, match="At least 2"):
            bootstrap_mean_diff_ci(
                np.array([1.0]),
                np.array([2.0, 3.0]),
                paired=False,
                alpha=0.05,
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
        with pytest.raises(InvalidColumnDataError, match="same length"):
            analytical_mean_diff_ci(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                paired=True,
                alpha=0.05,
            )

    def test_paired_requires_at_least_two_observations(self):
        with pytest.raises(InsufficientDataError, match="At least 2"):
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
        with pytest.raises(InsufficientDataError, match="At least 2"):
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

        with pytest.raises(InvalidParameterError, match="Unknown inference"):
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
        with pytest.raises(InvalidParameterError, match="within"):
            classify_equivalence(-1.0, 1.0, within=0.0)


class TestClassifyChangeBound:
    def test_greater_than_passed_when_lower_bound_above_threshold(self):
        result = classify_change_bound(-0.001, 0.01, -0.005, direction="greater_than")
        assert result == "passed"

    def test_greater_than_failed_when_upper_bound_below_threshold(self):
        result = classify_change_bound(-0.02, -0.01, -0.005, direction="greater_than")
        assert result == "failed"

    def test_greater_than_inconclusive_when_threshold_inside_ci(self):
        result = classify_change_bound(-0.02, 0.01, -0.005, direction="greater_than")
        assert result == "inconclusive"

    def test_less_than_passed_when_upper_bound_below_threshold(self):
        result = classify_change_bound(-5.0, 15.0, 20.0, direction="less_than")
        assert result == "passed"

    def test_less_than_failed_when_lower_bound_above_threshold(self):
        result = classify_change_bound(25.0, 35.0, 20.0, direction="less_than")
        assert result == "failed"

    def test_less_than_inconclusive_when_threshold_inside_ci(self):
        result = classify_change_bound(15.0, 25.0, 20.0, direction="less_than")
        assert result == "inconclusive"

    def test_rejects_unknown_direction(self):
        with pytest.raises(InvalidParameterError, match="direction"):
            classify_change_bound(-1.0, 1.0, 0.0, direction="sideways")


class TestWilsonInterval:
    def test_matches_scipy_binomtest_wilson_ci(self):
        # alpha=0.05 here is one-sided, matching this module's convention of
        # a `(1 - 2 * alpha)` two-sided-equivalent interval, i.e. a 90% CI.
        low, high = wilson_interval(8, 20, alpha=0.05)

        ref = scipy_stats.binomtest(8, 20).proportion_ci(
            confidence_level=0.90, method="wilson"
        )

        assert low == pytest.approx(ref.low)
        assert high == pytest.approx(ref.high)

    def test_rejects_non_positive_n(self):
        with pytest.raises(InvalidParameterError, match="`n`"):
            wilson_interval(0, 0, alpha=0.05)

    def test_rejects_count_out_of_range(self):
        with pytest.raises(InvalidParameterError, match="`count`"):
            wilson_interval(11, 10, alpha=0.05)


class TestIndependentRateDiffCi:
    def test_matches_published_newcombe_reference(self):
        # Reference values from Fagerland et al. 2015, as reproduced in
        # statsmodels' test suite for confint_proportions_2indep(...,
        # method="newcomb"): count1=7, nobs1=34, count2=1, nobs2=34 gives a
        # 95% CI of [0.019, 0.340] for diff = p1 - p2.
        before = np.zeros(34)
        before[:1] = 1.0
        after = np.zeros(34)
        after[:7] = 1.0

        diff, ci_low, ci_high = independent_rate_diff_ci(before, after, alpha=0.025)

        assert diff == pytest.approx(7 / 34 - 1 / 34)
        assert ci_low == pytest.approx(0.019, abs=0.005)
        assert ci_high == pytest.approx(0.340, abs=0.005)

    def test_recovers_known_difference_for_large_samples(self):
        rng = np.random.default_rng(0)
        before = (rng.random(2000) < 0.30).astype(float)
        after = (rng.random(2000) < 0.35).astype(float)

        diff, ci_low, ci_high = independent_rate_diff_ci(before, after, alpha=0.05)

        assert diff == pytest.approx(0.05, abs=0.03)
        assert ci_low < diff < ci_high

    def test_does_not_collapse_when_one_side_is_all_zero(self):
        before = np.zeros(50)
        after = np.zeros(50)
        after[:5] = 1.0

        diff, ci_low, ci_high = independent_rate_diff_ci(before, after, alpha=0.05)

        assert diff == pytest.approx(0.1)
        assert ci_low < diff < ci_high
        assert ci_low > -1.0  # not degenerate

    def test_zero_diff_when_both_sides_identical(self):
        before = np.array([0.0, 1.0, 0.0, 1.0, 1.0])
        after = before.copy()

        diff, ci_low, ci_high = independent_rate_diff_ci(before, after, alpha=0.05)

        assert diff == pytest.approx(0.0)
        assert ci_low < 0.0 < ci_high

    def test_requires_at_least_two_observations_per_side(self):
        with pytest.raises(InsufficientDataError, match="At least 2"):
            independent_rate_diff_ci(np.array([1.0]), np.array([1.0, 0.0]), alpha=0.05)


class TestPairedRateDiffCi:
    def test_matches_hand_computed_newcombe_paired_ci(self):
        # 2x2 paired table: a=40 (both 1), b=20 (before=1,after=0),
        # c=8 (before=0,after=1), d=32 (both 0); n=100.
        # Hand-computed (and cross-checked against a from-scratch
        # reimplementation of Newcombe's 1998 paired formula) reference:
        # phi = 0.43718, diff = after - before = -0.12,
        # ci = [-0.21873, -0.01664] for a 95% CI (alpha=0.025 here).
        before = np.array([1.0] * 40 + [1.0] * 20 + [0.0] * 8 + [0.0] * 32)
        after = np.array([1.0] * 40 + [0.0] * 20 + [1.0] * 8 + [0.0] * 32)

        diff, ci_low, ci_high = paired_rate_diff_ci(before, after, alpha=0.025)

        assert diff == pytest.approx(-0.12)
        assert ci_low == pytest.approx(-0.21873, abs=1e-4)
        assert ci_high == pytest.approx(-0.01664, abs=1e-4)

    def test_recovers_known_difference_for_large_samples(self):
        rng = np.random.default_rng(0)
        before = (rng.random(2000) < 0.30).astype(float)
        # correlated after: mostly agrees with before, shifted up slightly
        flip = rng.random(2000) < 0.1
        after = np.where(flip, 1 - before, before)

        diff, ci_low, ci_high = paired_rate_diff_ci(before, after, alpha=0.05)

        assert ci_low < diff < ci_high
        assert abs(diff) < 0.1

    def test_zero_diff_when_both_sides_identical(self):
        before = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        after = before.copy()

        diff, ci_low, ci_high = paired_rate_diff_ci(before, after, alpha=0.05)

        assert diff == pytest.approx(0.0)
        assert ci_low < 0.0 < ci_high

    def test_does_not_collapse_when_both_sides_are_all_zero(self):
        before = np.zeros(50)
        after = np.zeros(50)

        diff, ci_low, ci_high = paired_rate_diff_ci(before, after, alpha=0.05)

        assert diff == pytest.approx(0.0)
        # degenerate table (all margins zero) => phi=0, but Wilson intervals
        # around 0/1 still aren't zero-width, so the CI isn't a single point
        assert ci_low < 0.0
        assert ci_high > 0.0

    def test_requires_equal_length(self):
        with pytest.raises(InvalidColumnDataError, match="same length"):
            paired_rate_diff_ci(
                np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0]), alpha=0.05
            )

    def test_requires_at_least_two_pairs(self):
        with pytest.raises(InsufficientDataError, match="At least 2"):
            paired_rate_diff_ci(np.array([1.0]), np.array([0.0]), alpha=0.05)


class TestAnalyticalRateDiffCi:
    def test_paired_dispatches_to_paired_rate_diff_ci(self):
        before = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        after = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

        result = analytical_rate_diff_ci(before, after, paired=True, alpha=0.05)

        assert result == paired_rate_diff_ci(before, after, alpha=0.05)

    def test_independent_dispatches_to_independent_rate_diff_ci(self):
        before = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        after = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])

        result = analytical_rate_diff_ci(before, after, paired=False, alpha=0.05)

        assert result == independent_rate_diff_ci(before, after, alpha=0.05)


class TestRateDiffCiDispatcher:
    def test_default_method_is_analytical(self):
        before = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        after = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

        result = rate_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=100,
            rng=np.random.default_rng(0),
        )

        assert result == analytical_rate_diff_ci(before, after, paired=True, alpha=0.05)

    def test_method_bootstrap_matches_direct_call(self):
        before = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
        after = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])

        result = rate_diff_ci(
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
        before = np.array([1.0, 0.0, 1.0])
        after = np.array([1.0, 1.0, 0.0])

        with pytest.raises(InvalidParameterError, match="Unknown inference"):
            rate_diff_ci(
                before,
                after,
                paired=True,
                alpha=0.05,
                n_resamples=100,
                rng=np.random.default_rng(0),
                method="magic",
            )
