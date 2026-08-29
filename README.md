# frameworthy

![PyPI Version](https://img.shields.io/pypi/v/frameworthy)

Frameworthy is a lightweight Python testing library for dataframes and analytical transformations, supporting both Pandas and Polars. Instead of requiring exact expected outputs, it lets you express the properties a transformation should preserve or change: rows, keys, columns, values, aggregates, and more.

Many dataframe tests never get written because verifying a transformation means carefully constructing a second expected dataset. The goal of frameworthy is to make it easy to test what you actually care about, and extremely easy to understand why a test fails.

## Getting Started

### Installation

Available on [PyPI](https://pypi.org/project/frameworthy/).

```bash
pip install frameworthy
# or
uv add frameworthy
# or
poetry add frameworthy
```

### Example

```python
import frameworthy as fw
import pandas as pd

def transformation(df):
    return df.assign(col=df.col + 1)

input_df = pd.DataFrame({"col": [1, 2, 3]})
output_df = transformation(input_df)

# test that the transformation preserves row count
fw.expect(output_df, before=input_df).preserves_rows()

# test that the transformation preserves a key column
fw.expect(output_df, before=input_df).preserves_key("col")
```


## Features

- **Compatible**: Works with both Pandas and Polars dataframes through [Narwhals](https://narwhals-dev.github.io/narwhals/). Support for more dataframe libraries is being explored for future releases.
- **User Friendly**: Clear error messages that explain what failed and why.
- **Flexible**: Intended to be equally useful with testing frameworks like pytest or in scripts for real-world validation.

## Comparison with Similar Tools

| | What it checks | Best for |
|---|---|---|
| [Pandera](https://pandera.readthedocs.io/) / [Great Expectations](https://greatexpectations.io/) | Does this dataframe conform to a schema? | Validating pipeline inputs/outputs in critical production settings |
| **Pandas/Polars** `assert_frame_equal` | Does this dataframe exactly equal that one? | Regression tests once you already have a golden output reference |
| **Frameworthy** | Did this *transformation* preserve invariants or change what it should? | Testing transformation logic without building a full expected fixture |

Frameworthy is built around asserting relational invariants between a transformation's input and output. Schema validators check one dataframe against a fixed spec; equality checks compare two dataframes exactly. In either case, you're expected to provide the expected schema output. Given the nature of real-world data transformation code, this means tests go unwritten. Moreover, in some cases schema definitions can be brittle and change frequently. This library aims to meet you where you are and help you write tests that are easy to understand and maintain.

We do recommend using Pandera for critical schema validation, it is very powerful. It can also validate a function's input and output via `check_io`, but each is still checked independently against its own schema.
