---
layout: home

hero:
  name: frameworthy
#   text: Uncertainty-aware regression testing for data and metrics.
  tagline: Lightweight statistical validation library for data changes.
  image:
    src: /logo.png
    alt: Frameworthy
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/joypauls/frameworthy

features:
  - title: Simple API
    details: Build reliable validation pipelines with a clean, readable chain

  - title: Library Compatibility
    details: Works just as well with either Pandas or Polars dataframes

  - title: Statistical Rigor
    details: Built on well-established statistical methods and best practices
---

## Example

```python
import frameworthy as fw
import polars as pl # or pandas

# load your data if necessary
before_df = pl.read_csv("before.csv")
after_df = pl.read_csv("after.csv")

# run a check
result = (
    fw.check(after_df, before=before_df)
    .mean("column_name")
    .equivalent(within=0.1)
)

# inspect the results
print(result)
# or raise an exception on failure
result.assert_passed()
```

