# needs to subclass AssertionError to work with pytest
class FrameworthyAssertionError(AssertionError):
    """Raised when a Frameworthy expectation is not satisfied"""
