import pytest

import frameworthy as fw


def test_detects_changed_multiplicity(
    frame_factory,
):
    before = frame_factory({"id": [1, 2, 2, 3]})
    after = frame_factory({"id": [1, 2, 3, 3]})

    # fw.expect(after, relative_to=before).preserves_rows()

    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(
            after,
            relative_to=before,
        ).preserves_key("id")
