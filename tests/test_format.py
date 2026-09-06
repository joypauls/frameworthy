from frameworthy._constants import Statistic
from frameworthy._format import format_level, format_levels, format_margin, format_value


class TestFormatValue:
    def test_mean_renders_raw_units(self):
        assert format_value(1.2345, Statistic.MEAN) == "+1.234"

    def test_rate_renders_percentage_points(self):
        assert format_value(0.05, Statistic.RATE) == "+5pp"

    def test_negative_values_keep_sign(self):
        assert format_value(-0.005, Statistic.RATE) == "-0.5pp"


class TestFormatMargin:
    def test_mean_renders_raw_units(self):
        assert format_margin(2.0, Statistic.MEAN) == "\u00b12"

    def test_rate_renders_percentage_points(self):
        assert format_margin(0.005, Statistic.RATE) == "\u00b10.5pp"


class TestFormatLevel:
    def test_mean_renders_raw_units(self):
        assert format_level(100.0, Statistic.MEAN) == "100"

    def test_rate_renders_percent(self):
        assert format_level(0.4, Statistic.RATE) == "40.0%"


class TestFormatLevels:
    def test_mean_returns_empty_string(self):
        assert format_levels(100.0, 101.0, Statistic.MEAN) == ""

    def test_rate_returns_before_after_percentages(self):
        assert (
            format_levels(0.40, 0.45, Statistic.RATE)
            == "before = 40.0%, after = 45.0%, "
        )
