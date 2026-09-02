from frameworthy._backend import to_narwhals_frame
from frameworthy._pairing import duplicate_keys, key_counts


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
