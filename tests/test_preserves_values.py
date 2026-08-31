import pytest

import frameworthy as fw


def test_passes_when_values_unchanged(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "score": [30, 40, 50],
            "segment": ["A", "B", "C"],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 3],
            "score": [30, 40, 50],
            "segment": ["A", "B", "C"],
            "extra": [1, 2, 3],
        }
    )
    fw.expect(after, before=before).preserves_values(
        "score",
        "segment",
        on="id",
    )


def test_ignores_row_order(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "score": [30, 40, 50],
        }
    )
    after = frame_factory(
        {
            "id": [3, 1, 2],
            "score": [50, 30, 40],
        }
    )
    fw.expect(after, before=before).preserves_values("score", on="id")


def test_detects_changed_value(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [123, 456],
            "segment": ["A", "B"],
        }
    )
    after = frame_factory(
        {
            "id": [123, 456],
            "segment": ["B", "B"],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_values("segment", on="id")


def test_supports_composite_key(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 1, 2],
            "secondary_id": [10, 11, 20],
            "metric": [5.0, 6.0, 7.0],
        }
    )
    after = frame_factory(
        {
            "id": [2, 1, 1],
            "secondary_id": [20, 11, 10],
            "metric": [7.0, 6.0, 5.0],
        }
    )
    fw.expect(after, before=before).preserves_values(
        "metric",
        on=["id", "secondary_id"],
    )


def test_detects_composite_key_value_change(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 1, 2],
            "secondary_id": [10, 11, 20],
            "metric": [5.0, 6.0, 7.0],
        }
    )
    after = frame_factory(
        {
            "id": [1, 1, 2],
            "secondary_id": [10, 11, 20],
            "metric": [5.0, 9.0, 7.0],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_values(
            "metric",
            on=["id", "secondary_id"],
        )


def test_matching_nulls_are_equal(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2],
            "value": [None, 2.0],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2],
            "value": [None, 2.0],
        }
    )
    fw.expect(after, before=before).preserves_values("value", on="id")


def test_matching_nans_are_equal(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2],
            "value": [float("nan"), 2.0],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2],
            "value": [float("nan"), 2.0],
        }
    )
    fw.expect(after, before=before).preserves_values("value", on="id")


def test_detects_key_population_change(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "value": [1, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 4],
            "value": [1, 2, 3],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_values("value", on="id")


def test_raises_value_error_when_no_columns_supplied(
    frame_factory,
):
    before = frame_factory({"id": [1, 2, 3]})
    after = frame_factory({"id": [1, 2, 3]})
    with pytest.raises(ValueError):
        fw.expect(after, before=before).preserves_values(on="id")


def test_raises_value_error_when_missing_reference_value_column(
    frame_factory,
):
    before = frame_factory({"id": [1, 2, 3]})
    after = frame_factory({"id": [1, 2, 3], "value": [1, 2, 3]})
    with pytest.raises(ValueError):
        fw.expect(after, before=before).preserves_values("value", on="id")


def test_raises_value_error_when_missing_reference_key_column(
    frame_factory,
):
    before = frame_factory({"value": [1, 2, 3]})
    after = frame_factory({"id": [1, 2, 3], "value": [1, 2, 3]})
    with pytest.raises(ValueError):
        fw.expect(after, before=before).preserves_values("value", on="id")


def test_raises_value_error_on_duplicate_reference_keys(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 1, 2],
            "value": [1, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2],
            "value": [1, 3],
        }
    )
    with pytest.raises(ValueError):
        fw.expect(after, before=before).preserves_values("value", on="id")


def test_raises_value_error_on_duplicate_result_keys(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2],
            "value": [1, 3],
        }
    )
    after = frame_factory(
        {
            "id": [1, 1, 2],
            "value": [1, 1, 3],
        }
    )
    with pytest.raises(ValueError):
        fw.expect(after, before=before).preserves_values("value", on="id")


def test_raises_assertion_error_when_result_missing_value_column(
    frame_factory,
):
    before = frame_factory({"id": [1, 2, 3], "value": [1, 2, 3]})
    after = frame_factory({"id": [1, 2, 3]})
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_values("value", on="id")


def test_dtype_change_does_not_fail(
    frame_factory,
):
    before = frame_factory({"id": [1, 2, 3], "value": [1, 2, 3]})
    after = frame_factory({"id": [1, 2, 3], "value": [1.0, 2.0, 3.0]})
    fw.expect(after, before=before).preserves_values("value", on="id")
