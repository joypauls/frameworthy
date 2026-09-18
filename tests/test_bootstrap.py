import warnings

import numpy as np
import pytest
from scipy import stats as scipy_stats

from frameworthy._bootstrap import bootstrap_diff_ci
from frameworthy._errors import InvalidDataError


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
