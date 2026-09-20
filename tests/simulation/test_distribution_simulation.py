"""
Simulation-based sanity checks for DistributionCheck.equivalent() and
DistributionCheck.change_greater_than()
"""

import numpy as np
import pandas as pd
import pytest

import frameworthy as fw


def _distribution_check(before: np.ndarray, after: np.ndarray):
    return fw.check(after, before).distribution()


class TestIdenticalDistributions:
    def test_does_not_falsely_fail(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            before = rng.normal(loc=50.0, scale=5.0, size=400)
            after = rng.normal(loc=50.0, scale=5.0, size=400)

            result = _distribution_check(before, after).equivalent(
                within=2.0, n_resamples=500, random_state=seed
            )

            assert result.decision != "failed"


class TestShiftedDistributions:
    def test_fails(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=0.0, scale=1.0, size=400)
        after = rng.normal(loc=15.0, scale=1.0, size=400)

        result = _distribution_check(before, after).equivalent(
            within=1.0, n_resamples=500, random_state=0
        )

        assert result.decision == "failed"
        assert result.distance == pytest.approx(15.0, abs=1.0)


class TestBorderlineDistributions:
    def test_is_not_confidently_one_sided(self):
        # the true wasserstein distance between same-scale normals shifted
        # by delta is exactly delta so setting within equal to the
        # shift puts the true distance right at the decision boundary
        decisions = set()
        for seed in range(10):
            rng = np.random.default_rng(seed)
            before = rng.normal(loc=0.0, scale=5.0, size=150)
            after = rng.normal(loc=2.0, scale=5.0, size=150)

            result = _distribution_check(before, after).equivalent(
                within=2.0, n_resamples=500, random_state=seed
            )
            decisions.add(result.decision)

        # neither decision should be unanimous but ok if inconclusive
        assert decisions != {"passed"}
        assert decisions != {"failed"}


class TestChangeGreaterThan:
    """`change_greater_than` makes the opposite claim from `equivalent`:
    it should confirm drift for a clearly shifted distribution, never
    falsely claim drift for an identical distribution, and stay uncertain
    right at the decision boundary.
    """

    def test_does_not_falsely_confirm_drift_for_identical_distributions(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            before = rng.normal(loc=50.0, scale=5.0, size=400)
            after = rng.normal(loc=50.0, scale=5.0, size=400)

            result = _distribution_check(before, after).change_greater_than(
                threshold=2.0, n_resamples=500, random_state=seed
            )

            assert result.decision != "passed"

    def test_confirms_drift_for_a_clear_shift(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=0.0, scale=1.0, size=400)
        after = rng.normal(loc=15.0, scale=1.0, size=400)

        result = _distribution_check(before, after).change_greater_than(
            threshold=1.0, n_resamples=500, random_state=0
        )

        assert result.decision == "passed"
        assert result.distance == pytest.approx(15.0, abs=1.0)

    def test_is_not_confidently_one_sided_at_the_boundary(self):
        decisions = set()
        for seed in range(10):
            rng = np.random.default_rng(seed)
            before = rng.normal(loc=0.0, scale=5.0, size=150)
            after = rng.normal(loc=2.0, scale=5.0, size=150)

            result = _distribution_check(before, after).change_greater_than(
                threshold=2.0, n_resamples=500, random_state=seed
            )
            decisions.add(result.decision)

        assert decisions != {"passed"}
        assert decisions != {"failed"}


class TestPairedDistributions:
    """Sanity checks for the `paired=True` path, built via `paired_by` so
    `before`/`after` are correlated pairs rather than independent samples.
    """

    def _paired_distribution_check(self, before: np.ndarray, after: np.ndarray):
        ids = list(range(len(before)))
        before_df = pd.DataFrame({"id": ids, "value": before})
        after_df = pd.DataFrame({"id": ids, "value": after})
        return fw.check(after_df, before=before_df, paired_by="id").distribution(
            "value"
        )

    def test_does_not_falsely_fail_for_matched_pairs(self):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            before = rng.normal(loc=50.0, scale=5.0, size=400)
            after = before + rng.normal(loc=0.0, scale=0.2, size=400)

            result = self._paired_distribution_check(before, after).equivalent(
                within=2.0, n_resamples=500, random_state=seed
            )

            assert result.paired is True
            assert result.decision != "failed"

    def test_fails_for_a_clear_paired_shift(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=0.0, scale=1.0, size=400)
        after = before + 15.0 + rng.normal(loc=0.0, scale=0.1, size=400)

        result = self._paired_distribution_check(before, after).equivalent(
            within=1.0, n_resamples=500, random_state=0
        )

        assert result.decision == "failed"
        assert result.distance == pytest.approx(15.0, abs=1.0)

    def test_change_greater_than_confirms_drift_for_matched_pairs(self):
        rng = np.random.default_rng(0)
        before = rng.normal(loc=0.0, scale=1.0, size=400)
        after = before + 15.0 + rng.normal(loc=0.0, scale=0.1, size=400)

        result = self._paired_distribution_check(before, after).change_greater_than(
            threshold=1.0, n_resamples=500, random_state=0
        )

        assert result.decision == "passed"
        assert result.paired is True
