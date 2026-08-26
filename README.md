# frameworthy

Expressive testing for dataframes.

Frameworthy is a lightweight Python testing library for DataFrames and analytical transformations, supporting both pandas and polars. Instead of requiring exact expected outputs, it lets you express the properties a transformation should preserve or change: rows, keys, columns, values, aggregates, and more.

Many DataFrame tests never get written because verifying a transformation means constructing a second expected dataset. The goal of Frameworthy is to make it easy to test the guarantees you actually care about.





