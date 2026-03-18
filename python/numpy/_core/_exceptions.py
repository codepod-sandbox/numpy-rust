"""numpy._core._exceptions - internal exception classes."""


class _UFuncNoLoopError(TypeError):
    """Exception for when a ufunc has no matching loop for the given types."""

    def __init__(self, ufunc, dtypes):
        self.ufunc = ufunc
        self.dtypes = dtypes
        super().__init__(
            f"ufunc '{ufunc}' did not contain a loop with signature "
            f"matching types {dtypes}"
        )


class UFuncTypeError(TypeError):
    """Exception for ufunc type errors."""

    def __init__(self, ufunc, *args, **kwargs):
        self.ufunc = ufunc
        super().__init__(*args, **kwargs)


class _ArrayMemoryError(MemoryError):
    """Exception for array memory errors with shape/dtype info."""

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype
        size = 1
        for s in shape:
            size *= s
        super().__init__(
            f"Unable to allocate array with shape {shape} and dtype {dtype}"
        )


class AxisError(ValueError, IndexError):
    """Exception for invalid axis."""

    def __init__(self, axis, ndim=None, msg_prefix=None):
        if ndim is not None:
            msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
            if msg_prefix:
                msg = f"{msg_prefix}: {msg}"
        else:
            msg = str(axis)
        self.axis = axis
        self.ndim = ndim
        super().__init__(msg)
