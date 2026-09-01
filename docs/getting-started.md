# Getting Started

## Installation

Available on [PyPI](https://pypi.org/project/frameworthy/), use pip or your preferred package manager.

```bash
pip install frameworthy
# or
uv add frameworthy
```

## Example Usage

```python
import frameworthy as fw
import pandas as pd # or: import polars as pl

def transformation(df):
    return df.assign(col=df.col + 1)

input_df = pd.DataFrame({"col": [1, 2, 3]})
output_df = transformation(input_df)

# test that the transformation preserves row count
fw.expect(output_df, before=input_df).preserves_rows()

# test that the transformation preserves a key column
fw.expect(output_df, before=input_df).preserves_key("col")

# test that the transformation doesn't add or remove columns
fw.expect(output_df, before=input_df).preserves_columns()
```

You can also write multiple assertions like this:
```python
exp = fw.expect(output_df, before=input_df)
exp.preserves_rows()
exp.preserves_key("col")
exp.preserves_columns()
```
