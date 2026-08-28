import pytest

import frameworthy as fw


@pytest.mark.parametrize(
    "before_keys,after_keys",
    [
        pytest.param([1, 2, 2, 3], [1, 2, 3, 3], id="int"),
        pytest.param(["a", "b", "b", "c"], ["a", "b", "c", "c"], id="str"),
        pytest.param([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 3.0], id="float"),
    ],
)
def test_detects_changed_multiplicity(
    frame_factory,
    before_keys,
    after_keys,
):
    before = frame_factory({"id": before_keys})
    after = frame_factory({"id": after_keys})
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_key("id")


def test_passes_when_key_population_same(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 3],
            "metric": [0.1, 0.2, 0.3],
        }
    )
    fw.expect(after, before=before).preserves_key("id")


def test_ignores_row_order(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [3, 2, 1],
        }
    )
    fw.expect(after, before=before).preserves_key("id")


def test_detects_key_value_change(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 4],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_key("id")


def test_handles_null_values(
    frame_factory,
) -> None:
    before = frame_factory(
        {
            "id": [1.0, None, 3.0],
        }
    )
    after = frame_factory(
        {
            "id": [None, 3.0, 1.0],
        }
    )
    fw.expect(after, before=before).preserves_key("id")


def test_detects_null_multiplicity_change(
    frame_factory,
) -> None:
    before = frame_factory(
        {
            "id": [1.0, None, 3.0],
        }
    )
    after = frame_factory(
        {
            "id": [None, None, 3.0],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_key("id")


###
# COMPOSITE KEYS
###


def test_passes_with_composite_key(
    frame_factory,
) -> None:
    before = frame_factory(
        {
            "customer_id": [1, 1, 2],
            "order_id": [10, 11, 20],
        }
    )
    after = frame_factory(
        {
            "customer_id": [2, 1, 1],
            "order_id": [20, 11, 10],
        }
    )
    fw.expect(after, before=before).preserves_key(["customer_id", "order_id"])


def test_detects_composite_key_value_change(
    frame_factory,
) -> None:
    before = frame_factory(
        {
            "id1": [1, 1, 2],
            "id2": [10, 11, 20],
        }
    )
    after = frame_factory(
        {
            "id1": [1, 1, 2],
            "id2": [10, 12, 20],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_key(["id1", "id2"])
