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
        super().__init__(
            f"Unable to allocate {self._size_to_string(self._total_size)} "
            f"for an array with shape {shape} and data type {dtype}"
        )

    def __reduce__(self):
        return (type(self), (self.shape, self.dtype))

    @property
    def _total_size(self):
        size = 1
        for s in self.shape:
            size *= s
        itemsize = getattr(self.dtype, 'itemsize', 8)
        return size * itemsize

    @staticmethod
    def _size_to_string(num_bytes):
        """Convert a number of bytes to a human-readable string."""
        if num_bytes < 1024:
            return '{} bytes'.format(num_bytes)

        units = ['KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']
        last_unit = 'EiB'
        unit_val = 1024.0
        for unit in units:
            new_val = num_bytes / unit_val
            # Check if displaying in this unit would round to >= 1024
            # If so, move to the next larger unit
            if new_val < 1024.0 - 0.5 or unit == last_unit:
                if abs(new_val) < 10.0:
                    return '{:.2f} {}'.format(new_val, unit)
                elif abs(new_val) < 100.0:
                    return '{:.1f} {}'.format(new_val, unit)
                else:
                    return '{:.0f}. {}'.format(new_val, unit)
            unit_val *= 1024.0
        return '{:.0f}. {}'.format(new_val, last_unit)


from numpy._helpers import AxisError
