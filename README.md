# frameworthy [WIP]

![PyPI Version](https://img.shields.io/pypi/v/frameworthy)

Frameworthy is a lightweight Python testing library for dataframes and analytical transformations, supporting both Pandas and Polars. Instead of requiring exact expected outputs, it lets you express the properties a transformation should preserve or change: rows, keys, columns, values, aggregates, and more.

Many dataframe tests never get written because verifying a transformation means carefully constructing a second expected dataset. The goal of frameworthy is to make it easy to test what you actually care about, and extremely easy to understand why a test fails.

## Features

- **Compatible**: Works with both Pandas and Polars dataframes through [Narwhals](https://narwhals-dev.github.io/narwhals/). Support for more dataframe libraries is being explored for future releases.
- **User Friendly**: Clear error messages that explain what failed and why.
- **Flexible**: Intended to be equally useful with testing frameworks like pytest or in scripts for real-world validation.



