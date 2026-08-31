import pytest

import frameworthy as fw


def test_passes_when_row_count_equal(
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
    fw.expect(after, before=before).preserves_rows()


def test_detects_dropped_rows(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_rows()


def test_detects_added_rows(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 3],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError):
        fw.expect(after, before=before).preserves_rows()


def test_passes_with_empty_frames(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [],
        }
    )
    after = frame_factory(
        {
            "id": [],
        }
    )
    fw.expect(after, before=before).preserves_rows()
