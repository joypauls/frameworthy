> ⚠️ WIP: All 0.1.x releases are unstable. 0.2.0 will be the first stable release.

> Releases >0.1.3 are functional and ready for use, but the API is subject tochange.

# frameworthy

![PyPI Version](https://img.shields.io/pypi/v/frameworthy) 
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/frameworthy.svg?x=1)](https://pypi.org/project/frameworthy/)
[![codecov](https://codecov.io/gh/joypauls/frameworthy/branch/main/graph/badge.svg?token=npu0JtY8hc)](https://codecov.io/gh/joypauls/frameworthy)

Lightweight statistical validation library for data changes.

<div align="center"><img src="docs/public/banner.png" width="600"></div>

## Getting Started

### Installation

Available on [PyPI](https://pypi.org/project/frameworthy/), use `pip` or your preferred package manager.

```bash
pip install frameworthy
# or
uv add frameworthy
```


### Examples

See `scripts/examples.py` for a few quick examples.

```python
import frameworthy as fw
import polars as pl # or pandas

before_df = pl.read_csv("before.csv")
after_df = pl.read_csv("after.csv")

result = (
    fw.check(after_df, before=before_df)
    .mean("column_name")
    .equivalent(within=0.1)
)

# inspect the results
print(result)
# or raise on failure
result.assert_passed()
```


### Usage

Bring your data: two pandas/polars DataFrames (before and after / pre and post).

A standard `frameworthy` check looks like this:

```python
fw.check(after_df, before_df).mean("column_name").equivalent(within=0.1)
```

1. Start a **check** with `fw.check(after_df, before_df)`
2. Specify the **metric** to check, with `.mean("column_name")`
3. Make a **claim** about the change, using `.equivalent()`, `.change_greater_than()`, or `.change_less_than()`
4. Examine the **results** or assert that the check passed with `.assert_passed()` for testing


## Methodology

These are the most important notes to be aware of for correct usage. For further details, see the more extensive [docs](https://joypauls.github.io/frameworthy/methodology).

### Confidence Intervals

The parameter `alpha` is a one-sided significance level everywhere; `.equivalent()`'s displayed CI is (1-2α), not (1-α). This is to maintain consistency with the TOST (Two One-Sided Tests) equivalence testing method.

Examples:
- `.equivalent()`
    - `alpha=0.05` → 90% CI 
    - `alpha=0.025` → 95% CI
- `.change_greater_than()` and `.change_less_than()`
    - `alpha=0.1` → 90% CI 
    - `alpha=0.05` → 95% CI


## Development

To run the tests:
```bash
make test
```

To run the examples:
```bash
uv run scripts/examples.py
```

