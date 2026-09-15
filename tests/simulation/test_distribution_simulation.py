"""
Simulation-based sanity checks for DistributionCheck.equivalent()
"""

import numpy as np
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
