"""numpy._core._multiarray_tests - stub for C extension test helpers."""
import numpy


def array_indexing(arr, idx_type, idx):
    """Stub for array indexing test helper."""
    raise NotImplementedError("C extension test helper not available")


def create_custom_field_dtype(*args, **kwargs):
    """Stub for custom field dtype creation."""
    raise NotImplementedError("C extension test helper not available")


def internal_overlap(arr):
    """Stub for internal overlap check."""
    return False


def fromstring_null_term_c_api(*args, **kwargs):
    """Stub for C API fromstring test."""
    raise NotImplementedError("C extension test helper not available")


def npy_char_deprecation():
    """Stub."""
    pass


def get_buffer_info(arr, *args):
    """Stub for buffer info."""
    return (arr.shape, arr.strides)


def run_scalar_intp_converter(*args, **kwargs):
    """Stub for C extension scalar intp converter test."""
    raise NotImplementedError("C extension test helper not available")


def run_scalar_intp_from_sequence(*args, **kwargs):
    """Stub for C extension scalar intp from sequence test."""
    raise NotImplementedError("C extension test helper not available")


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
