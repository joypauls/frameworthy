import pytest

from frameworthy._arrays import (
    assert_column_exists,
    assert_equal_pairs,
    assert_min_count,
    paired_values_from_columns,
    values_from_two_frames,
)
from frameworthy._backend import to_narwhals_frame


def test_assert_column_exists_raises_when_missing():
    with pytest.raises(KeyError, match="missing"):
        assert_column_exists(["a", "b"], "missing", "df")


def test_assert_min_count_raises_below_minimum():
    with pytest.raises(ValueError, match="No usable"):
        assert_min_count(0, 2, "df")
    with pytest.raises(ValueError, match="At least 2"):
        assert_min_count(1, 2, "df")


def test_assert_equal_pairs_raises_when_unequal_length():
    with pytest.raises(ValueError, match="unequal numbers"):
        assert_equal_pairs([1.0, 2.0], [3.0], "df")


def test_paired_values_from_columns_drops_rows_with_null_in_either_column(
    frame_factory,
):
    frame = to_narwhals_frame(
        frame_factory({"before": [1.0, None, 3.0, 4.0], "after": [2.0, 3.0, None, 5.0]})
    )

    before_values, after_values = paired_values_from_columns(
        frame, "before", "after", "df"
    )

    assert before_values.tolist() == [1.0, 4.0]
    assert after_values.tolist() == [2.0, 5.0]


def test_paired_values_from_columns_raises_when_too_few_pairs_remain(frame_factory):
    frame = to_narwhals_frame(
        frame_factory({"before": [1.0, None], "after": [2.0, None]})
    )

    with pytest.raises(ValueError, match="At least 2"):
        paired_values_from_columns(frame, "before", "after", "df")


def test_values_from_two_frames_paired_joins_and_drops_null_pairs(frame_factory):
    before = to_narwhals_frame(
        frame_factory({"id": [1, 2, 3, 4], "value": [1.0, None, 3.0, 4.0]})
    )
    after = to_narwhals_frame(
        frame_factory({"id": [1, 2, 3, 4], "value": [2.0, 3.0, None, 5.0]})
    )

    before_values, after_values, paired = values_from_two_frames(
        before, after, "value", ["id"]
    )

    assert paired is True
    assert before_values.tolist() == [1.0, 4.0]
    assert after_values.tolist() == [2.0, 5.0]


def test_values_from_two_frames_unpaired_drops_nulls_independently(frame_factory):
    before = to_narwhals_frame(frame_factory({"value": [1.0, None, 3.0]}))
    after = to_narwhals_frame(frame_factory({"value": [None, 4.0, 5.0]}))

    before_values, after_values, paired = values_from_two_frames(
        before, after, "value", None
    )

    assert paired is False
    assert before_values.tolist() == [1.0, 3.0]
    assert after_values.tolist() == [4.0, 5.0]


def test_values_from_two_frames_raises_on_missing_column(frame_factory):
    before = to_narwhals_frame(frame_factory({"other": [1.0]}))
    after = to_narwhals_frame(frame_factory({"value": [1.0]}))

    with pytest.raises(KeyError, match="before"):
        values_from_two_frames(before, after, "value", None)


def test_values_from_two_frames_raises_when_too_few_usable_values(frame_factory):
    before = to_narwhals_frame(frame_factory({"value": [1.0, None]}))
    after = to_narwhals_frame(frame_factory({"value": [1.0, 2.0]}))

    with pytest.raises(ValueError, match="At least 2"):
        values_from_two_frames(before, after, "value", None)
