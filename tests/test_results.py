import pytest

from frameworthy._errors import FrameworthyAssertionError
from frameworthy.decision import Decision
from frameworthy.results import EquivalenceResult


def _make_result(decision: Decision, **overrides) -> EquivalenceResult:
    defaults = {
        "decision": decision,
        "column": "revenue",
        "statistic": "mean",
        "paired": True,
        "before_mean": 100.0,
        "after_mean": 101.0,
        "diff": 1.0,
        "ci_low": -1.0,
        "ci_high": 3.0,
        "alpha": 0.05,
        "within": 2.0,
        "n_before": 50,
        "n_after": 50,
        "n_resamples": 10_000,
    }
    defaults.update(overrides)
    return EquivalenceResult(**defaults)


class TestPassed:
    def test_true_for_equivalent(self):
        assert _make_result(Decision.EQUIVALENT).passed is True

    def test_false_for_changed(self):
        assert _make_result(Decision.CHANGED).passed is False

    def test_false_for_inconclusive(self):
        assert _make_result(Decision.INCONCLUSIVE).passed is False


class TestStr:
    def test_includes_key_details(self):
        result = _make_result(
            Decision.EQUIVALENT,
            ci_low=-1.0,
            ci_high=3.0,
            diff=1.0,
            within=2.0,
            alpha=0.05,
        )
        text = str(result)

        assert "EQUIVALENT" in text
        assert "mean(revenue)" in text
        assert "+1" in text
        assert "90% CI" in text
        assert "-1" in text and "+3" in text
        assert "\u00b12" in text  # ±2
        assert "paired, n=50" in text

    def test_reports_unpaired_sample_sizes(self):
        result = _make_result(Decision.CHANGED, paired=False, n_before=30, n_after=45)
        text = str(result)

        assert "unpaired" in text
        assert "n_before=30" in text
        assert "n_after=45" in text


class TestRaiseForStatus:
    def test_equivalent_does_not_raise_or_warn(self, recwarn):
        _make_result(Decision.EQUIVALENT).raise_for_status()
        assert len(recwarn) == 0

    def test_changed_raises_frameworthy_assertion_error(self):
        result = _make_result(Decision.CHANGED)
        with pytest.raises(FrameworthyAssertionError, match="CHANGED"):
            result.raise_for_status()

    def test_inconclusive_warns_but_does_not_raise(self):
        result = _make_result(Decision.INCONCLUSIVE)
        with pytest.warns(UserWarning, match="INCONCLUSIVE"):
            result.raise_for_status()
