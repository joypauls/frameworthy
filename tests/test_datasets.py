import numpy as np
import pytest

import frameworthy as fw
from frameworthy.datasets import SampleData, sample_normal, sample_rate


class TestSampleNormal:
    def test_returns_named_arrays_of_requested_size(self):
        data = sample_normal(n=50, seed=0)

        assert isinstance(data, SampleData)
        assert data.before.shape[0] == 50
        assert data.after.shape[0] == 50

    def test_shift_moves_the_after_mean(self):
        data = sample_normal(n=5000, shift=3.0, seed=0)

        assert data.after.mean() - data.before.mean() > 2.0

    def test_default_shift_has_no_real_effect(self):
        data = sample_normal(n=5000, seed=0)

        assert abs(data.after.mean() - data.before.mean()) < 0.5

    def test_chains_directly_into_check(self):
        data = sample_normal(n=1000, shift=0.0, seed=0)
        result = fw.check(data.after, before=data.before).mean().equivalent(within=0.5)

        assert result.decision == "passed"


class TestSampleRate:
    def test_returns_named_binary_arrays_of_requested_size(self):
        data = sample_rate(n=50, seed=0)

        assert isinstance(data, SampleData)
        assert set(np.unique(data.before)) <= {0, 1}
        assert set(np.unique(data.after)) <= {0, 1}

    def test_p_after_moves_the_rate(self):
        data = sample_rate(n=5000, p_before=0.3, p_after=0.7, seed=0)

        assert data.after.mean() - data.before.mean() > 0.2

    def test_chains_directly_into_check(self):
        data = sample_rate(n=1000, p_before=0.5, p_after=0.5, seed=0)
        result = fw.check(data.after, before=data.before).rate().equivalent(within=0.1)

        assert result.decision == "passed"

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_extreme_probabilities_still_binary(self, p):
        data = sample_rate(n=20, p_before=p, p_after=p, seed=0)

        assert set(np.unique(data.before)) <= {0, 1}
