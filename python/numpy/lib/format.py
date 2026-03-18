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
