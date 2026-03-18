"""numpy._core._internal - internal utilities."""
import numpy


def _dtype_from_pep3118(spec):
    """Parse a PEP 3118 buffer format string into a dtype (stub)."""
    return numpy.dtype(spec)


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
