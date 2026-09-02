import narwhals.stable.v2 as nw

from frameworthy._backend import to_narwhals_frame


def test_wraps_native_frame_in_narwhals(frame_factory):
    native = frame_factory({"id": [1, 2, 3]})

    frame = to_narwhals_frame(native)

    assert isinstance(frame, nw.DataFrame)
    assert frame.columns == ["id"]
    assert frame.shape[0] == 3
