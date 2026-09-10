# Methodology

## Confidence Intervals

The parameter `alpha` is a one-sided significance level everywhere; `.equivalent()`'s displayed CI is (1-2α), not (1-α). This is to maintain consistency with the TOST (Two One-Sided Tests) equivalence testing method.

Examples:
- `.equivalent()`
    - `alpha=0.05` → 90% CI 
    - `alpha=0.025` → 95% CI
- `.change_greater_than()` and `.change_less_than()`
    - `alpha=0.1` → 90% CI 
    - `alpha=0.05` → 95% CI