# frameworthy [WIP]

![PyPI Version](https://img.shields.io/pypi/v/frameworthy)

Frameworthy is a lightweight Python testing library for dataframes and analytical transformations, supporting both pandas and polars. Instead of requiring exact expected outputs, it lets you express the properties a transformation should preserve or change: rows, keys, columns, values, aggregates, and more.

Many dataframe tests never get written because verifying a transformation means carefully constructing a second expected dataset. The goal of frameworthy is to make it easy to test what you actually care about, and extremely easy to understand why a test fails.

## Compatibility

Supports both **pandas** and **polars** dataframes through [Narwhals](https://narwhals-dev.github.io/narwhals/).

Intended to be equally useful with testing frameworks like **pytest** or in scripts for real-world data validation.



