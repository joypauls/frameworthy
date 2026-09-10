from pathlib import Path

import numpy as np
import pandas as pd

import frameworthy as fw

np.random.seed(1729)

# generated from scripts/generate_e2e_datasets.py
BEFORE_CSV = Path("tests/integration/data/before.csv")
AFTER_CHANGED_CSV = Path("tests/integration/data/after_changed.csv")
AFTER_UNCHANGED_CSV = Path("tests/integration/data/after_unchanged.csv")

before_df = pd.read_csv(BEFORE_CSV)
after_changed_df = pd.read_csv(AFTER_CHANGED_CSV)
after_unchanged_df = pd.read_csv(AFTER_UNCHANGED_CSV)


def main():
    # passing example: mean of unchanged distribution should be equivalent
    result = (
        fw.check(after_unchanged_df, before=before_df)
        .mean("normal")
        .equivalent(within=0.1)
    )
    print(result)

    # failing example: mean of changed distribution should not be equivalent
    result = (
        fw.check(after_changed_df, before=before_df)
        .mean("normal")
        .equivalent(within=0.1)
    )
    print(result)
    result = result.assert_passed()


if __name__ == "__main__":
    main()
