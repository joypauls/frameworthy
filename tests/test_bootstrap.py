import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats

from frameworthy._bootstrap import bootstrap_diff_ci, subsample_bootstrap_ci
from frameworthy._errors import InvalidDataError, UsageError


class TestBootstrapDiffCiValidation:
    def test_paired_requires_equal_length(self):
        with pytest.raises(InvalidDataError, match="same length"):
            bootstrap_diff_ci(
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
                paired=True,
                alpha=0.05,
                n_resamples=100,
                rng=np.random.default_rng(0),
            )

    def test_paired_requires_at_least_two_observations(self):
        with pytest.raises(InvalidDataError, match="At least 2"):
            bootstrap_diff_ci(
                np.array([1.0]),
                np.array([2.0]),
                paired=True,
                alpha=0.05,
                n_resamples=100,
                rng=np.random.default_rng(0),
            )

    def test_unpaired_requires_at_least_two_observations_per_side(self):
        with pytest.raises(InvalidDataError, match="At least 2"):
            bootstrap_diff_ci(
                np.array([1.0]),
                np.array([2.0, 3.0]),
                paired=False,
                alpha=0.05,
                n_resamples=100,
                rng=np.random.default_rng(0),
            )


class TestBootstrapDiffCiContract:
    """Behavior that's ours to guarantee, not scipy's: the reported
    `observed` value, that unpaired samples of different lengths are
    allowed, reproducibility given a seeded rng, and that an arbitrary
    `statistic_func` is plumbed through correctly. We deliberately don't
    re-test `scipy.stats.bootstrap`'s own statistical correctness (e.g.
    that a BCa interval actually achieves its nominal coverage) -- that's
    scipy's job, already covered by scipy's own test suite.
    """

    def test_observed_is_the_plain_difference_of_statistics_not_bias_corrected(self):
        """There is no "bias-corrected point estimate" reported anywhere:
        `observed` must always be the plain difference of statistics, not
        derived from the (BCa) bootstrap distribution in any way.
        """
        rng = np.random.default_rng(3)
        before = rng.normal(loc=10.0, scale=2.0, size=40)
        after = rng.normal(loc=12.0, scale=2.0, size=45)

        observed, _, _ = bootstrap_diff_ci(
            before,
            after,
            paired=False,
            alpha=0.05,
            n_resamples=500,
            rng=rng,
        )

        assert observed == pytest.approx(np.mean(after) - np.mean(before))

    def test_unpaired_allows_different_lengths(self):
        observed, ci_low, ci_high = bootstrap_diff_ci(
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0]),
            paired=False,
            alpha=0.05,
            n_resamples=100,
            rng=np.random.default_rng(0),
        )

        assert observed == pytest.approx(4.5 - 2.0)
        assert ci_low <= ci_high

    def test_is_reproducible_with_seeded_rng(self):
        before = np.array([1.0, 2.0, 3.0, 4.0, 7.0, 9.0])
        after = np.array([2.0, 2.0, 5.0, 3.0, 6.0, 8.0])

        result_a = bootstrap_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(42),
        )
        result_b = bootstrap_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(42),
        )

        assert result_a == result_b

    def test_arbitrary_statistic_func_is_used_for_observed_and_resampling(self):
        rng = np.random.default_rng(0)
        before = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        after = before + 1.0  # constant shift under any subset -> also
        # exercises the degenerate fallback below, but the point here is
        # that `statistic_func=np.median` (not the default `np.mean`) is
        # actually what's used.

        observed, ci_low, ci_high = bootstrap_diff_ci(
            before,
            after,
            paired=True,
            alpha=0.05,
            n_resamples=1000,
            rng=rng,
            statistic_func=np.median,
        )

        assert observed == pytest.approx(1.0)
        assert ci_low == pytest.approx(1.0)
        assert ci_high == pytest.approx(1.0)


class TestBootstrapDiffCiDegenerateFallback:
    """`scipy.stats.bootstrap`'s BCa method can't compute a valid interval
    when the bootstrap distribution has no variability at all (e.g.
    `before`/`after` are identical, or a paired shift is exactly
    constant): it emits a `DegenerateDataWarning` and returns a `nan`
    interval. Catching that and falling back to something sensible is
    genuinely our own logic (not scipy's), so it's worth testing directly:
    unlike scipy's default behavior, we never leak the warning or return
    `nan` to the caller.
    """

    def test_both_sides_identical_does_not_raise_or_warn(self):
        before = np.zeros(50)
        after = np.zeros(50)

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            observed, ci_low, ci_high = bootstrap_diff_ci(
                before,
                after,
                paired=False,
                alpha=0.05,
                n_resamples=500,
                rng=np.random.default_rng(0),
            )

        assert observed == pytest.approx(0.0)
        assert ci_low == pytest.approx(0.0)
        assert ci_high == pytest.approx(0.0)
        assert not np.isnan(ci_low)
        assert not np.isnan(ci_high)

    def test_paired_constant_shift_falls_back_to_point_interval(self):
        before = np.array([1.0, 2.0])
        after = before + 1.0

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            observed, ci_low, ci_high = bootstrap_diff_ci(
                before,
                after,
                paired=True,
                alpha=0.05,
                n_resamples=200,
                rng=np.random.default_rng(0),
            )

        assert observed == pytest.approx(1.0)
        assert ci_low == pytest.approx(1.0)
        assert ci_high == pytest.approx(1.0)


class TestBootstrapDiffCiScipyWiring:
    """We don't need to re-verify `scipy.stats.bootstrap`'s statistical
    behavior -- just that we're calling it with the arguments we intend
    (BCa by default, `paired=` forwarded, `confidence_level` derived from
    `alpha` the same way as everywhere else in this library).
    """

    def _spy_on_bootstrap(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        captured: dict = {}
        original = scipy_stats.bootstrap

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(scipy_stats, "bootstrap", spy)
        return captured

    def test_uses_bca_method(self, monkeypatch):
        captured = self._spy_on_bootstrap(monkeypatch)

        bootstrap_diff_ci(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 4.0, 6.0]),
            paired=True,
            alpha=0.05,
            n_resamples=50,
            rng=np.random.default_rng(0),
        )

        assert captured["method"] == "BCa"

    @pytest.mark.parametrize("paired", [True, False])
    def test_forwards_paired_flag(self, monkeypatch, paired):
        captured = self._spy_on_bootstrap(monkeypatch)

        before = np.array([1.0, 2.0, 3.0])
        after = np.array([2.0, 4.0, 6.0]) if paired else np.array([2.0, 4.0])
        bootstrap_diff_ci(
            before,
            after,
            paired=paired,
            alpha=0.05,
            n_resamples=50,
            rng=np.random.default_rng(0),
        )

        assert captured["paired"] is paired

    def test_confidence_level_is_one_minus_two_alpha(self, monkeypatch):
        captured = self._spy_on_bootstrap(monkeypatch)

        bootstrap_diff_ci(
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            paired=True,
            alpha=0.025,
            n_resamples=50,
            rng=np.random.default_rng(0),
        )

        assert captured["confidence_level"] == pytest.approx(0.95)


class TestSubsampleBootstrapCi:
    """Unlike `bootstrap_diff_ci`, this bootstrap is entirely our own
    mechanics (no `scipy.stats.bootstrap` underneath), so it's worth
    testing more directly rather than just checking wiring. `_distribution.py`
    covers the Wasserstein-specific end-to-end behavior (clipping,
    validation messages); these tests use a simple, cheap `statistic_func`
    to isolate the generic subsampling/rescaling mechanics themselves.
    """

    @staticmethod
    def _mean_diff(before: np.ndarray, after: np.ndarray) -> float:
        return float(np.mean(after) - np.mean(before))

    def test_statistic_is_the_plain_observed_value(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=10.0, scale=1.0, size=100)
        after = rng.normal(loc=12.0, scale=1.0, size=100)

        statistic, _, _ = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            alpha=0.05,
            n_resamples=200,
            rng=rng,
        )

        assert statistic == pytest.approx(self._mean_diff(before, after))

    def test_is_reproducible_with_seeded_rng(self):
        before = np.linspace(0.0, 10.0, 50)
        after = np.linspace(1.0, 11.0, 50)

        result_a = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )
        result_b = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )

        assert result_a == result_b

    def test_resamples_before_and_after_independently(self):
        """By default (`paired=False`), resampling `before`/`after` at
        different sizes must work, which is only possible if they're drawn
        independently rather than sharing indices.
        """
        rng = np.random.default_rng(0)
        before = rng.normal(loc=0.0, scale=1.0, size=40)
        after = rng.normal(loc=5.0, scale=1.0, size=70)

        statistic, ci_low, ci_high = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            alpha=0.05,
            n_resamples=300,
            rng=rng,
        )

        assert statistic == pytest.approx(self._mean_diff(before, after), abs=0.5)
        assert ci_low < statistic < ci_high

    def test_does_not_clip_to_any_domain(self):
        """Domain-specific clipping (e.g. a distance can't be negative) is
        the caller's responsibility, not this function's -- with a
        statistic that can go negative and a clear negative shift, the
        interval should be allowed to sit entirely below 0.
        """
        rng = np.random.default_rng(1)
        before = rng.normal(loc=10.0, scale=0.5, size=200)
        after = rng.normal(loc=5.0, scale=0.5, size=200)

        _, ci_low, ci_high = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            alpha=0.05,
            n_resamples=500,
            rng=rng,
        )

        assert ci_low < 0.0
        assert ci_high < 0.0

    def test_rejects_invalid_alpha(self):
        rng = np.random.default_rng(0)
        before = rng.normal(size=20)
        after = rng.normal(size=20)

        with pytest.raises(UsageError, match="alpha"):
            subsample_bootstrap_ci(
                before, after, self._mean_diff, alpha=0.6, n_resamples=100, rng=rng
            )

    def test_rejects_non_positive_n_resamples(self):
        rng = np.random.default_rng(0)
        before = rng.normal(size=20)
        after = rng.normal(size=20)

        with pytest.raises(UsageError, match="n_resamples"):
            subsample_bootstrap_ci(
                before, after, self._mean_diff, alpha=0.05, n_resamples=0, rng=rng
            )

    def test_paired_rejects_mismatched_lengths(self):
        rng = np.random.default_rng(0)
        before = rng.normal(size=20)
        after = rng.normal(size=25)

        with pytest.raises(InvalidDataError, match="same length"):
            subsample_bootstrap_ci(
                before,
                after,
                self._mean_diff,
                paired=True,
                alpha=0.05,
                n_resamples=100,
                rng=rng,
            )

    def test_paired_is_reproducible_with_seeded_rng(self):
        before = np.linspace(0.0, 10.0, 50)
        after = before + 1.0

        result_a = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            paired=True,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )
        result_b = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            paired=True,
            alpha=0.05,
            n_resamples=300,
            rng=np.random.default_rng(42),
        )

        assert result_a == result_b

    def test_paired_gives_narrower_ci_for_strongly_correlated_pairs(self):
        # a constant-plus-tiny-noise paired shift is highly correlated;
        # resampling shared indices (paired=True) should preserve that
        # correlation and produce a tighter CI than resampling before/after
        # independently, which discards it.
        rng = np.random.default_rng(1)
        before = rng.normal(loc=0.0, scale=5.0, size=200)
        after = before + 3.0 + rng.normal(loc=0.0, scale=0.05, size=200)

        _, paired_low, paired_high = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            paired=True,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(0),
        )
        _, unpaired_low, unpaired_high = subsample_bootstrap_ci(
            before,
            after,
            self._mean_diff,
            paired=False,
            alpha=0.05,
            n_resamples=500,
            rng=np.random.default_rng(0),
        )

        assert (paired_high - paired_low) < (unpaired_high - unpaired_low)
