from enum import Enum


class Decision(str, Enum):
    """Classification decision for equivalence testing."""

    EQUIVALENT = "equivalent"
    CHANGED = "changed"
    INCONCLUSIVE = "inconclusive"
