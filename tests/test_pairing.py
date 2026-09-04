import pytest

from frameworthy._backend import to_narwhals_frame
from frameworthy._pairing import duplicate_keys, join_paired, key_counts


def test_key_counts_single_column(frame_factory):
    frame = to_narwhals_frame(frame_factory({"id": [1, 2, 2, 3]}))
    counts = key_counts(frame, "id")

    assert counts[(1,)] == 1
    assert counts[(2,)] == 2
    assert counts[(3,)] == 1


def test_key_counts_multiple_columns(frame_factory):
    frame = to_narwhals_frame(frame_factory({"a": [1, 1, 2], "b": ["x", "x", "y"]}))
    counts = key_counts(frame, ["a", "b"])

    assert counts[(1, "x")] == 2
    assert counts[(2, "y")] == 1


def test_key_counts_normalizes_null_values(frame_factory):
    frame = to_narwhals_frame(frame_factory({"id": [1, None, None]}))
    counts = key_counts(frame, "id")

    assert sum(counts.values()) == 3

    null_keys = [key for key in counts if key != (1,)]

    assert len(null_keys) == 1
    assert counts[null_keys[0]] == 2


def test_duplicate_keys_returns_only_repeated_values(frame_factory):
    frame = to_narwhals_frame(frame_factory({"id": [1, 2, 2, 3, 3, 3]}))
    duplicates = duplicate_keys(frame, "id")

    assert set(duplicates) == {(2,), (3,)}


def test_duplicate_keys_empty_when_all_unique(frame_factory):
    frame = to_narwhals_frame(frame_factory({"id": [1, 2, 3]}))

    assert duplicate_keys(frame, "id") == []


def test_join_paired_aligns_matching_keys(frame_factory):
    before = to_narwhals_frame(
        frame_factory({"id": [1, 2, 3], "metric": [10.0, 20.0, 30.0]})
    )
    after = to_narwhals_frame(
        frame_factory({"id": [1, 2, 3], "metric": [11.0, 19.0, 33.0]})
    )

    joined = join_paired(before, after, "id", ["metric"])
    rows = {row[0]: (row[1], row[2]) for row in joined.iter_rows()}

    assert rows == {1: (10.0, 11.0), 2: (20.0, 19.0), 3: (30.0, 33.0)}


def test_join_paired_drops_unmatched_keys(frame_factory):
    before = to_narwhals_frame(frame_factory({"id": [1, 2], "metric": [10.0, 20.0]}))
    after = to_narwhals_frame(frame_factory({"id": [2, 3], "metric": [21.0, 30.0]}))

    joined = join_paired(before, after, "id", ["metric"])

    assert joined.shape[0] == 1
    assert joined["id"].to_list() == [2]


def test_join_paired_supports_multiple_key_columns(frame_factory):
    before = to_narwhals_frame(
        frame_factory({"a": [1, 1], "b": ["x", "y"], "value": [1.0, 2.0]})
    )
    after = to_narwhals_frame(
        frame_factory({"a": [1, 1], "b": ["x", "y"], "value": [1.5, 2.5]})
    )

    joined = join_paired(before, after, ["a", "b"], ["value"])

    assert joined.shape[0] == 2
    assert "value_after" in joined.columns


def test_join_paired_raises_on_duplicate_keys_in_before(frame_factory):
    before = to_narwhals_frame(frame_factory({"id": [1, 1], "metric": [10.0, 11.0]}))
    after = to_narwhals_frame(frame_factory({"id": [1], "metric": [12.0]}))

    with pytest.raises(ValueError, match="before"):
        join_paired(before, after, "id", ["metric"])


def test_join_paired_raises_on_duplicate_keys_in_after(frame_factory):
    before = to_narwhals_frame(frame_factory({"id": [1], "metric": [10.0]}))
    after = to_narwhals_frame(frame_factory({"id": [1, 1], "metric": [12.0, 13.0]}))

    with pytest.raises(ValueError, match="after"):
        join_paired(before, after, "id", ["metric"])
