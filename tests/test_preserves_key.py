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
