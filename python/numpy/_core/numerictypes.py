"""numpy._core.numerictypes - type-related utilities."""
import numpy


def obj2sctype(x, default=None):
    """Return the scalar dtype or NumPy equivalent of type of an object."""
    try:
        return numpy.dtype(x).type
    except (TypeError, KeyError):
        return default


def issctype(rep):
    """Determines whether the given object represents a scalar data-type."""
    try:
        obj2sctype(rep)
        return True
    except Exception:
        return False


def maximum_sctype(t):
    """Return the scalar type of highest precision of the same kind as the input."""
    import numpy as np
    dt = np.dtype(t)
    kind = dt.kind
    if kind == 'f':
        return np.float64
    elif kind == 'i':
        return np.int64
    elif kind == 'u':
        return np.uint64
    elif kind == 'c':
        return np.complex128
    elif kind == 'b':
        return np.bool_
    return dt.type


# Type aliases
bool_ = numpy.bool_
int8 = numpy.int8
int16 = numpy.int16
int32 = numpy.int32
int64 = numpy.int64
uint8 = numpy.uint8
uint16 = numpy.uint16
uint32 = numpy.uint32
uint64 = numpy.uint64
float16 = numpy.float16
float32 = numpy.float32
float64 = numpy.float64
complex64 = numpy.complex64
complex128 = numpy.complex128

typecodes = {
    'All': '?bhilqBHILQefdgFDGSUVOMm',
    'AllFloat': 'efdgFDG',
    'AllInteger': 'bBhHiIlLqQ',
    'Character': 'c',
    'Complex': 'FDG',
    'Float': 'efdg',
    'Integer': 'bhilq',
    'UnsignedInteger': 'BHILQ',
    'Datetime': 'Mm',
}


def sctype2char(sctype):
    """Return the string representation of a scalar dtype."""
    dt = numpy.dtype(sctype)
    return dt.char


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
