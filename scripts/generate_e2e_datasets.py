"""
Generate synthetic datasets for end-to-end tests.
"""

import numpy as np
import pandas as pd

np.random.seed(1729)

ROWS = 1000


def generate_normal(mean: float, std: float, size: int) -> np.ndarray:
    return np.random.normal(mean, std, size)


def generate_bernoulli(size: int, p: float) -> np.ndarray:
    return np.random.binomial(1, p, size)


def main():
    normal_before = generate_normal(0, 1, ROWS)
    binomial_before = generate_bernoulli(ROWS, p=0.5)

    normal_after_unchanged = generate_normal(0, 1, ROWS)
    binomial_after_unchanged = generate_bernoulli(ROWS, p=0.5)

    normal_after_changed = generate_normal(0.5, 1, ROWS)
    binomial_after_changed = generate_bernoulli(ROWS, p=0.7)

    df_before = pd.DataFrame(
        {
            "normal": normal_before,
            "binomial": binomial_before,
        }
    )
    df_after_changed = pd.DataFrame(
        {
            "normal": normal_after_changed,
            "binomial": binomial_after_changed,
        }
    )
    df_after_unchanged = pd.DataFrame(
        {
            "normal": normal_after_unchanged,
            "binomial": binomial_after_unchanged,
        }
    )

    df_before.to_csv("tests/integration/data/before.csv", index=False)
    df_after_changed.to_csv("tests/integration/data/after_changed.csv", index=False)
    df_after_unchanged.to_csv("tests/integration/data/after_unchanged.csv", index=False)


if __name__ == "__main__":
    main()
