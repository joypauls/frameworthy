import pytest

import frameworthy as fw


def test_passes_when_columns_identical(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "email": ["a", "b", "c"],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 3],
            "email": ["a", "b", "c"],
        }
    )
    fw.expect(after, before=before).preserves_columns()


def test_passes_when_columns_reordered(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "email": ["a", "b", "c"],
        }
    )
    after = frame_factory(
        {
            "email": ["a", "b", "c"],
            "id": [1, 2, 3],
        }
    )
    fw.expect(after, before=before).preserves_columns()


def test_detects_added_column(
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
            "propensity_score": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError) as exc_info:
        fw.expect(after, before=before).preserves_columns()

    message = str(exc_info.value)
    assert "propensity_score" in message
    assert "Added columns" in message
    assert "Removed columns" not in message


def test_detects_removed_column(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "email": ["a", "b", "c"],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 3],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError) as exc_info:
        fw.expect(after, before=before).preserves_columns()

    message = str(exc_info.value)
    assert "email" in message
    assert "Removed columns" in message
    assert "Added columns" not in message


def test_detects_added_and_removed_columns(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "email": ["a", "b", "c"],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 3],
            "propensity_score": [0.1, 0.2, 0.3],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError) as exc_info:
        fw.expect(after, before=before).preserves_columns()

    message = str(exc_info.value)
    assert "Added columns" in message
    assert "propensity_score" in message
    assert "Removed columns" in message
    assert "email" in message


def test_reports_deterministic_output_for_multiple_changes(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [1, 2, 3],
            "zeta": [1, 2, 3],
            "alpha": [1, 2, 3],
        }
    )
    after = frame_factory(
        {
            "id": [1, 2, 3],
            "yankee": [1, 2, 3],
            "bravo": [1, 2, 3],
        }
    )
    with pytest.raises(fw.FrameworthyAssertionError) as exc_info:
        fw.expect(after, before=before).preserves_columns()

    expected_message = (
        "Columns were not preserved.\n"
        "\n"
        "Added columns:\n"
        "  + bravo\n"
        "  + yankee\n"
        "\n"
        "Removed columns:\n"
        "  - alpha\n"
        "  - zeta"
    )

    assert str(exc_info.value) == expected_message


def test_passes_with_empty_frames_and_matching_columns(
    frame_factory,
):
    before = frame_factory(
        {
            "id": [],
            "email": [],
        }
    )
    after = frame_factory(
        {
            "id": [],
            "email": [],
        }
    )
    fw.expect(after, before=before).preserves_columns()
