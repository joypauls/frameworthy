from enum import Enum


class Decision(str, Enum):
    """Classification decision for equivalence and directional change checks.

    `EQUIVALENT`/`CHANGED` are produced by equivalence checks (`.equivalent()`);
    `PASSED`/`FAILED` are produced by directional change checks
    (`.change_greater_than()`/`.change_less_than()`). `INCONCLUSIVE` is shared
    by both: the available data can't establish the requested claim either way.
    """

    EQUIVALENT = "equivalent"
    CHANGED = "changed"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
