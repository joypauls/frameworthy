---
layout: home

hero:
  name: frameworthy
#   text: A lightweight testing library for dataframes and analytical transformations.
  tagline: Expressive assertions for the structural and statistical behavior of data transformations.
  image:
    src: /logo.png
    alt: Frameworthy
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/YOUR_USERNAME/frameworthy

features:
  - title: Transformation-aware
    details: Assert that transformations preserve rows, keys, grain, and other properties that matter.

  - title: Built for real data work
    details: Test relationships between DataFrames instead of only comparing exact outputs.

  - title: Expressive
    details: Write tests that describe the contract your transformation is supposed to uphold.
---

## Example

```python
expect(after).preserves_rows(before)
```

