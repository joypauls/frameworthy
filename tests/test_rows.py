# import pytest

import frameworthy as fw


def test_preserves_rows_passes_when_row_count_equal(
    frame_factory,
) -> None:
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

    fw.expect(after, relative_to=before).preserves_rows()
