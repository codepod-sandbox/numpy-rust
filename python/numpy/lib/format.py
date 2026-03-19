"""numpy.lib.format - .npy/.npz file format."""

# Re-export from numpy io if available
try:
    from numpy._io import open_memmap
except (ImportError, AttributeError):
    def open_memmap(*args, **kwargs):
        raise NotImplementedError("open_memmap not available")

try:
    from numpy._io import read_array, write_array
except (ImportError, AttributeError):
    def read_array(*args, **kwargs):
        raise NotImplementedError("read_array not available")

    def write_array(*args, **kwargs):
        raise NotImplementedError("write_array not available")

MAGIC_PREFIX = b'\x93NUMPY'
MAGIC_LEN = 6


def descr_to_dtype(descr):
    """Convert a dtype description to a dtype object."""
    import numpy as np
    if isinstance(descr, str):
        return np.dtype(descr)
    if isinstance(descr, (list, tuple)):
        # Structured dtype description: [(name, type, shape), ...]
        try:
            return np.dtype(descr)
        except Exception:
            return np.dtype('float64')
    return np.dtype(descr)


def dtype_to_descr(dtype):
    """Convert a dtype to a description string."""
    import numpy as np
    dt = np.dtype(dtype)
    return dt.str
