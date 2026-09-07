import narwhals.stable.v2 as nw
from narwhals.stable.v2.typing import IntoDataFrame


def to_narwhals_frame(data: IntoDataFrame) -> nw.DataFrame:
    """Wrap a native dataframe (pandas/polars) in a narwhals frame."""
    return nw.from_native(data, eager_only=True)
