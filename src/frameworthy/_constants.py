from enum import Enum

NULL_KEY = object()


class Decision(str, Enum):
    """Classification decision for equivalence testing."""

    EQUIVALENT = "equivalent"
    CHANGED = "changed"
    INCONCLUSIVE = "inconclusive"
