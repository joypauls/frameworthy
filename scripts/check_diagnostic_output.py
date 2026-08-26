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
    relative_to=before,
).preserves_rows()
