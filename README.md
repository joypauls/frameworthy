> ⚠️ WIP: All 0.1.x releases are unstable. 0.2.0 will be the first stable release.

# frameworthy

![PyPI Version](https://img.shields.io/pypi/v/frameworthy) 
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/frameworthy.svg?x=1)](https://pypi.org/project/frameworthy/)
[![codecov](https://codecov.io/gh/joypauls/frameworthy/branch/main/graph/badge.svg?token=npu0JtY8hc)](https://codecov.io/gh/joypauls/frameworthy)

Lightweight statistical validation library for data changes.

<div align="center"><img src="docs/public/banner.png" width="600"></div>

## Getting Started

### Installation

Available on [PyPI](https://pypi.org/project/frameworthy/), use pip or your preferred package manager.

```bash
pip install frameworthy
# or
uv add frameworthy
```

## Methodology

These are the most important notes to be aware of. For further details, see the more extensive [docs]().

### Confidence Intervals

The parameter `alpha` is a one-sided significance level everywhere; `.equivalent()`'s displayed CI is (1-2α), not (1-α). This is to maintain consistency with the TOST (Two One-Sided Tests) equivalence testing method.

Examples:
- `.equivalent()`
    - `alpha=0.05` → 90% CI 
    - `alpha=0.025` → 95% CI
- `.change_greater_than()` and `.change_less_than()`
    - `alpha=0.1` → 90% CI 
    - `alpha=0.05` → 95% CI


