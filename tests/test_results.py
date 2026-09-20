import pytest

from frameworthy._constants import Metric
from frameworthy._errors import FrameworthyAssertionError
from frameworthy.decision import Decision
from frameworthy.results import ChangeResult, DistributionResult, EquivalenceResult


def _normalize(text: str) -> str:
    """Collapse all whitespace so assertions don't depend on the exact
    column-alignment padding used by `__str__`.
    """
    return " ".join(text.split())


def _make_result(decision: Decision, **overrides) -> EquivalenceResult:
    defaults = {
        "decision": decision,
        "column": "revenue",
        "metric": Metric.MEAN,
        "paired": True,
        "before_value": 100.0,
        "after_value": 101.0,
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


def _make_distribution_result(decision: Decision, **overrides) -> DistributionResult:
    defaults = {
        "decision": decision,
        "column": "revenue",
        "paired": False,
        "distance": 0.8,
        "ci_low": 0.4,
        "ci_high": 1.2,
        "within": 5.0,
        "alpha": 0.05,
        "n_before": 500,
        "n_after": 500,
        "n_resamples": 2000,
    }
    defaults.update(overrides)
    return DistributionResult(**defaults)


def _make_distribution_change_result(
    decision: Decision, **overrides
) -> DistributionResult:
    defaults = {
        "decision": decision,
        "column": "revenue",
        "paired": False,
        "distance": 8.0,
        "ci_low": 6.0,
        "ci_high": 10.0,
        "threshold": 5.0,
        "alpha": 0.05,
        "n_before": 500,
        "n_after": 500,
        "n_resamples": 2000,
    }
    defaults.update(overrides)
    return DistributionResult(**defaults)


def _make_change_result(decision: Decision, **overrides) -> ChangeResult:
    defaults = {
        "decision": decision,
        "column": "latency_ms",
        "metric": Metric.MEAN,
        "paired": True,
        "before_value": 100.0,
        "after_value": 105.0,
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
    """String representation tests for EquivalenceResult and ChangeResult."""

    def test_includes_key_details(self):
        result = _make_result(
            Decision.PASSED,
            ci_low=-1.0,
            ci_high=3.0,
            diff=1.0,
            within=2.0,
            alpha=0.05,
        )
        text = _normalize(str(result))

        assert "PASSED" in text
        assert "mean(revenue)" in text
        assert "+1" in text
        assert "90% CI" in text
        assert "-1" in text and "+3" in text
        assert "\u00b12" in text  # ±2
        assert "paired, n=50" in text

    def test_reports_unpaired_sample_sizes(self):
        result = _make_result(Decision.FAILED, paired=False, n_before=30, n_after=45)
        text = _normalize(str(result))

        assert "unpaired" in text
        assert "n_before=30" in text
        assert "n_after=45" in text

    def test_rate_formats_diff_and_margin_in_percentage_points(self):
        result = _make_result(
            Decision.PASSED,
            metric=Metric.RATE,
            before_value=0.40,
            after_value=0.45,
            diff=0.05,
            ci_low=-0.01,
            ci_high=0.03,
            within=0.005,
        )
        text = _normalize(str(result))

        assert "rate(revenue)" in text
        assert "before = 40.0%" in text
        assert "after = 45.0%" in text
        assert "+5pp" in text
        assert "[-1pp, +3pp]" in text
        assert "\u00b10.5pp" in text  # ±0.5pp
        # a raw proportion like 0.005 shouldn't leak through unformatted
        assert "0.005" not in text

    def test_mean_does_not_show_percentage_point_formatting(self):
        result = _make_result(Decision.PASSED, metric=Metric.MEAN)
        text = _normalize(str(result))

        assert "pp" not in text
        assert "before =" not in text
        assert "after =" not in text

    def test_custom_str_metric_renders_in_raw_units(self):
        # `.custom()` checks use a plain `str` for `name` not `Metric`
        result = _make_result(
            Decision.PASSED,
            metric="p95_latency",
            before_value=100.0,
            after_value=101.0,
        )
        text = _normalize(str(result))

        assert "p95_latency(revenue)" in text
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
        text = _normalize(str(result))

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
            metric=Metric.RATE,
            before_value=0.40,
            after_value=0.398,
        )
        text = _normalize(str(result))

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


class TestDistributionResultPassed:
    @pytest.mark.parametrize(
        "decision, expected",
        [
            (Decision.PASSED, True),
            (Decision.FAILED, False),
            (Decision.INCONCLUSIVE, False),
        ],
    )
    def test_passed(self, decision, expected):
        assert _make_distribution_result(decision).passed is expected


class TestDistributionResultStr:
    def test_includes_key_details(self):
        result = _make_distribution_result(
            Decision.PASSED,
            distance=0.8,
            ci_low=0.4,
            ci_high=1.2,
            within=5.0,
            alpha=0.05,
        )
        text = _normalize(str(result))

        assert "PASSED" in text
        assert "wasserstein(revenue)" in text
        assert "distance = 0.8" in text
        assert "90% CI" in text
        assert "[0.4, 1.2]" in text
        assert "within = 5" in text
        assert "alpha = 0.05" in text
        assert "unpaired" in text
        assert "n_before=500" in text
        assert "n_after=500" in text
        assert "method=subsampling" in text

    def test_reports_paired_sample_size(self):
        result = _make_distribution_result(
            Decision.PASSED, paired=True, n_before=500, n_after=500
        )
        text = _normalize(str(result))

        assert "paired, n=500" in text
        assert "unpaired" not in text


class TestDistributionResultAssertPassed:
    def test_passed_does_not_raise_or_warn(self, recwarn):
        _make_distribution_result(Decision.PASSED).assert_passed()
        assert len(recwarn) == 0

    def test_failed_raises_frameworthy_assertion_error(self):
        result = _make_distribution_result(Decision.FAILED)
        with pytest.raises(FrameworthyAssertionError, match="FAILED"):
            result.assert_passed()

    def test_inconclusive_warns_but_does_not_raise(self):
        result = _make_distribution_result(Decision.INCONCLUSIVE)
        with pytest.warns(UserWarning, match="INCONCLUSIVE"):
            result.assert_passed()


class TestDistributionResultRequiresExactlyOneClaim:
    def test_rejects_neither_within_nor_threshold(self):
        with pytest.raises(AssertionError):
            _make_distribution_result(Decision.PASSED, within=None)

    def test_rejects_both_within_and_threshold(self):
        with pytest.raises(AssertionError):
            _make_distribution_result(Decision.PASSED, within=5.0, threshold=5.0)


class TestDistributionChangeGreaterThanResultPassed:
    @pytest.mark.parametrize(
        "decision, expected",
        [
            (Decision.PASSED, True),
            (Decision.FAILED, False),
            (Decision.INCONCLUSIVE, False),
        ],
    )
    def test_passed(self, decision, expected):
        assert _make_distribution_change_result(decision).passed is expected


class TestDistributionChangeGreaterThanResultStr:
    def test_includes_key_details(self):
        result = _make_distribution_change_result(
            Decision.PASSED,
            distance=8.0,
            ci_low=6.0,
            ci_high=10.0,
            threshold=5.0,
            alpha=0.05,
        )
        text = _normalize(str(result))

        assert "PASSED" in text
        assert "wasserstein(revenue)" in text
        assert "distance = 8" in text


class TestDistributionChangeGreaterThanResultAssertPassed:
    def test_passed_does_not_raise_or_warn(self, recwarn):
        _make_distribution_change_result(Decision.PASSED).assert_passed()
        assert len(recwarn) == 0

    def test_failed_raises_frameworthy_assertion_error(self):
        result = _make_distribution_change_result(Decision.FAILED)
        with pytest.raises(FrameworthyAssertionError, match="FAILED"):
            result.assert_passed()

    def test_inconclusive_warns_but_does_not_raise(self):
        result = _make_distribution_change_result(Decision.INCONCLUSIVE)
        with pytest.warns(UserWarning, match="INCONCLUSIVE"):
            result.assert_passed()
