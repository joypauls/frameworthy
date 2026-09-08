import pytest

from frameworthy._classify import classify_change_bound, classify_equivalence
from frameworthy._errors import UsageError


class TestClassifyEquivalence:
    def test_equivalent_when_ci_fully_inside_margin(self):
        assert classify_equivalence(-1.0, 1.0, within=2.0) == "passed"

    def test_changed_when_ci_fully_outside_margin_above(self):
        assert classify_equivalence(3.0, 5.0, within=2.0) == "failed"

    def test_changed_when_ci_fully_outside_margin_below(self):
        assert classify_equivalence(-5.0, -3.0, within=2.0) == "failed"

    def test_inconclusive_when_ci_straddles_upper_margin(self):
        assert classify_equivalence(1.0, 3.0, within=2.0) == "inconclusive"

    def test_inconclusive_when_ci_straddles_lower_margin(self):
        assert classify_equivalence(-3.0, -1.0, within=2.0) == "inconclusive"

    def test_inconclusive_when_ci_spans_both_margins(self):
        assert classify_equivalence(-5.0, 5.0, within=2.0) == "inconclusive"

    def test_rejects_non_positive_within(self):
        with pytest.raises(UsageError, match="within"):
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
        with pytest.raises(UsageError, match="direction"):
            classify_change_bound(-1.0, 1.0, 0.0, direction="sideways")
