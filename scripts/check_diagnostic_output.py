import pandas as pd

import frameworthy as fw

before = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Joy", "Ada", "Grace"],
    }
)
after = pd.DataFrame(
    {
        "id": [1, 2, 2, 3],
        "name": ["Joy", "Ada", "Ada", "Grace"],
    }
)

fw.expect(
    after,
    before=before,
).preserves_rows()
