class FrameworthyError(Exception):
    """Base class for all exceptions raised by frameworthy.

    Never raised directly; catch one of the subclasses below (or this base
    class to catch any of them at once). Each subclass also inherits from
    the built-in exception type it replaces (e.g. `ValueError`), so existing
    `except ValueError` handling keeps working unchanged.
    """


class UsageError(FrameworthyError, ValueError):
    """Raised when `check()`/`.mean()`/`.rate()` are called in a way that's
    inconsistent with the single- vs. two-dataframe comparison mode, e.g.
    passing `paired_by` without a separate `before` dataframe.
    """


class InvalidParameterError(FrameworthyError, ValueError):
    """Raised when a check parameter (e.g. `alpha`, `within`, `threshold`,
    `method`, `direction`, `n_resamples`) has an invalid value.
    """


class InsufficientDataError(FrameworthyError, ValueError):
    """Raised when too few usable (non-null) observations are available to
    compute a confidence interval.
    """


class InvalidColumnDataError(FrameworthyError, ValueError):
    """Raised when column data doesn't satisfy a required shape or
    constraint, e.g. a `.rate()` column containing non-binary values, or
    before/after data that can't be aligned pair-by-pair.
    """


class ColumnNotFoundError(FrameworthyError, KeyError):
    """Raised when a named column is missing from a dataframe."""


# needs to subclass AssertionError to work with pytest
class FrameworthyAssertionError(FrameworthyError, AssertionError):
    """Raised when a Frameworthy expectation is not satisfied"""
