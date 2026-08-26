# frameworthy [WIP]

Frameworthy is a lightweight Python testing library for dataframes and analytical transformations, supporting both pandas and polars. Instead of requiring exact expected outputs, it lets you express the properties a transformation should preserve or change: rows, keys, columns, values, aggregates, and more.

Many dataframe tests never get written because verifying a transformation means constructing a second expected dataset. The goal of Frameworthy is to make it easy to test the guarantees you actually care about.





