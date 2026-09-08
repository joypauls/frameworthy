import pytest

from frameworthy._constants import Statistic
from frameworthy._errors import FrameworthyAssertionError
from frameworthy.decision import Decision
from frameworthy.results import ChangeResult, EquivalenceResult


def _make_result(decision: Decision, **overrides) -> EquivalenceResult:
    defaults = {
        "decision": decision,
        "column": "revenue",
        "statistic": Statistic.MEAN,
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


def _make_change_result(decision: Decision, **overrides) -> ChangeResult:
    defaults = {
        "decision": decision,
        "column": "latency_ms",
        "statistic": Statistic.MEAN,
        "paired": True,
        "before_mean": 100.0,
        "after_mean": 105.0,
        "diff": 5.0,
        "ci_low": 1.0,
        "ci_high": 9.0,
        "threshold": 20.0,
        "direction": "less_than",
        "alpha": 0.05,
        "n_before": 50,
        "n_after": 50,
        "n_resamples": 10_000,
    }
    defaults.update(overrides)
    return ChangeResult(**defaults)


class TestPassed:
    @pytest.mark.parametrize(
        "decision, expected",
        [
            (Decision.PASSED, True),
            (Decision.FAILED, False),
            (Decision.INCONCLUSIVE, False),
        ],
    )
    def test_passed(self, decision, expected):
        assert _make_result(decision).passed is expected


class TestStr:
    def test_includes_key_details(self):
        result = _make_result(
            Decision.PASSED,
            ci_low=-1.0,
            ci_high=3.0,
            diff=1.0,
            within=2.0,
            alpha=0.05,
        )
        text = str(result)

        assert "PASSED" in text
        assert "mean(revenue)" in text
        assert "+1" in text
        assert "90% CI" in text
        assert "-1" in text and "+3" in text
        assert "\u00b12" in text  # ±2
        assert "paired, n=50" in text

    def test_reports_unpaired_sample_sizes(self):
        result = _make_result(Decision.FAILED, paired=False, n_before=30, n_after=45)
        text = str(result)

        assert "unpaired" in text
        assert "n_before=30" in text
        assert "n_after=45" in text

    def test_rate_formats_diff_and_margin_in_percentage_points(self):
        result = _make_result(
            Decision.PASSED,
            statistic=Statistic.RATE,
            before_mean=0.40,
            after_mean=0.45,
            diff=0.05,
            ci_low=-0.01,
            ci_high=0.03,
            within=0.005,
        )
        text = str(result)

        assert "rate(revenue)" in text
        assert "before = 40.0%" in text
        assert "after = 45.0%" in text
        assert "+5pp" in text
        assert "[-1pp, +3pp]" in text
        assert "\u00b10.5pp" in text  # ±0.5pp
        # a raw proportion like 0.005 shouldn't leak through unformatted
        assert "0.005" not in text

    def test_mean_does_not_show_percentage_point_formatting(self):
        result = _make_result(Decision.PASSED, statistic=Statistic.MEAN)
        text = str(result)

        assert "pp" not in text
        assert "before =" not in text
        assert "after =" not in text


class TestAssertPassed:
    def test_passed_does_not_raise_or_warn(self, recwarn):
        _make_result(Decision.PASSED).assert_passed()
        assert len(recwarn) == 0

    def test_failed_raises_frameworthy_assertion_error(self):
        result = _make_result(Decision.FAILED)
        with pytest.raises(FrameworthyAssertionError, match="FAILED"):
            result.assert_passed()

    def test_inconclusive_warns_but_does_not_raise(self):
        result = _make_result(Decision.INCONCLUSIVE)
        with pytest.warns(UserWarning, match="INCONCLUSIVE"):
            result.assert_passed()


class TestChangeResultPassed:
    @pytest.mark.parametrize(
        "decision, expected",
        [
            (Decision.PASSED, True),
            (Decision.FAILED, False),
            (Decision.INCONCLUSIVE, False),
        ],
    )
    def test_passed(self, decision, expected):
        assert _make_change_result(decision).passed is expected


class TestChangeResultStr:
    def test_less_than_includes_upper_bound_and_threshold(self):
        result = _make_change_result(
            Decision.PASSED,
            direction="less_than",
            diff=5.0,
            ci_low=1.0,
            ci_high=9.0,
            threshold=20.0,
            alpha=0.05,
        )
        text = str(result)

        assert "PASSED" in text
        assert "mean(latency_ms)" in text
        assert "+5" in text
        assert "95% one-sided upper bound" in text
        assert "+9" in text
        assert "change_less_than" in text
        assert "+20" in text
        assert "paired, n=50" in text
        # the lower bound isn't the relevant one for this direction
        assert "lower bound" not in text

    def test_greater_than_rate_includes_lower_bound_and_pp_formatting(self):
        # also covers pp-formatting for ChangeResult; EquivalenceResult's
        # copy of the same underlying formatting is proven in `TestStr`
        result = _make_change_result(
            Decision.PASSED,
            direction="greater_than",
            diff=-0.002,
            ci_low=-0.004,
            ci_high=0.001,
            threshold=-0.005,
            statistic=Statistic.RATE,
            before_mean=0.40,
            after_mean=0.398,
        )
        text = str(result)

        assert "rate(latency_ms)" in text
        assert "before = 40.0%" in text
        assert "after = 39.8%" in text
        assert "95% one-sided lower bound" in text
        assert "change_greater_than" in text
        assert "upper bound" not in text
        assert "-0.2pp" in text
        assert "-0.4pp" in text
        assert "-0.5pp" in text
        # a raw proportion like 0.005 shouldn't leak through unformatted
        assert "0.005" not in text


class TestChangeResultAssertPassed:
    def test_passed_does_not_raise_or_warn(self, recwarn):
        _make_change_result(Decision.PASSED).assert_passed()
        assert len(recwarn) == 0

    def test_failed_raises_frameworthy_assertion_error(self):
        result = _make_change_result(Decision.FAILED)
        with pytest.raises(FrameworthyAssertionError, match="FAILED"):
            result.assert_passed()

    def test_inconclusive_warns_but_does_not_raise(self):
        result = _make_change_result(Decision.INCONCLUSIVE)
        with pytest.warns(UserWarning, match="INCONCLUSIVE"):
            result.assert_passed()
