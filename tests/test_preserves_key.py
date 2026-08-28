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
        fw.expect(
            after,
            relative_to=before,
        ).preserves_key("id")


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
    fw.expect(after, relative_to=before).preserves_key("id")


def test_preserves_key_ignores_row_order(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [3, 2, 1, 2],
        }
    )
    fw.expect(after, relative_to=before).preserves_key("id")


def test_preserves_key_fails_when_key_value_changes(
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
    with pytest.raises(
        fw.FrameworthyAssertionError,
        match="Expected transformation to preserve key",
    ):
        fw.expect(after, relative_to=before).preserves_key("id")
