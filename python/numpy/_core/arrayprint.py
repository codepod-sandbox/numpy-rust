"""numpy._core.arrayprint - array printing utilities."""
import numpy

# Set of types that should be printed without type prefix
_typelessdata = {
    numpy.bool_,
    numpy.int64,
    numpy.float64,
    numpy.complex128,
}


def set_printoptions(**kwargs):
    """Stub for set_printoptions."""
    pass


def get_printoptions():
    """Stub for get_printoptions."""
    return {}


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
