"""numpy.exceptions - exception classes."""


class AxisError(Exception):
    def __init__(self, axis=None, ndim=None, msg_prefix=None):
        if axis is not None and ndim is not None:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
            if msg_prefix:
                msg = f"{msg_prefix}: {msg}"
        elif axis is not None:
            msg = str(axis)
        else:
            msg = ""
        super().__init__(msg)
        self.axis = axis
        self.ndim = ndim


class ComplexWarning(UserWarning):
    pass


class VisibleDeprecationWarning(UserWarning):
    pass


class RankWarning(UserWarning):
    pass
