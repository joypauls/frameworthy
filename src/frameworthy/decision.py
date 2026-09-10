from enum import Enum


class Decision(str, Enum):
    """Classification decision for equivalence and directional change checks.

    Shared by both equivalence checks (`.equivalent()`) and directional
    change checks (`.change_greater_than()`/`.change_less_than()`):
    `PASSED` if the evidence supports the claim being tested, `FAILED` if it
    supports the opposite, and `INCONCLUSIVE` if the available data can't
    establish the claim either way.
    """

    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
