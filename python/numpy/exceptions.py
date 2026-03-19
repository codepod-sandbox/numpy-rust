"""numpy.exceptions - exception classes."""

from numpy._helpers import AxisError


class ComplexWarning(UserWarning):
    pass


class VisibleDeprecationWarning(UserWarning):
    pass


class RankWarning(UserWarning):
    pass


class DTypePromotionError(TypeError):
    pass


class TooHardError(RuntimeError):
    pass
