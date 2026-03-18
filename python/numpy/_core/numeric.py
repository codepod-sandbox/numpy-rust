"""numpy._core.numeric - numeric utilities and re-exports."""
import numpy


def normalize_axis_index(axis, ndim, msg_prefix=None):
    """Normalizes an axis index to [0, ndim)."""
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        if msg_prefix:
            msg = f"{msg_prefix}: {msg}"
        raise numpy.AxisError(axis, ndim, msg_prefix)
    return axis % ndim


def normalize_axis_tuple(axis, ndim, argname=None, allow_duplicate=False):
    """Normalizes an axis argument into a tuple of non-negative integer axes."""
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(normalize_axis_index(ax, ndim, argname) for ax in axis)
    if not allow_duplicate and len(set(axis)) != len(axis):
        if argname:
            raise ValueError(f"repeated axis in `{argname}` argument")
        else:
            raise ValueError("repeated axis")
    return axis


# Re-export commonly used functions
array = numpy.array
asarray = numpy.asarray
zeros = numpy.zeros
ones = numpy.ones
empty = numpy.empty
full = numpy.full
arange = numpy.arange
where = numpy.where
concatenate = numpy.concatenate
dot = numpy.dot
ndarray = numpy.ndarray
dtype = numpy.dtype
isnan = numpy.isnan
isinf = numpy.isinf
isfinite = numpy.isfinite
count_nonzero = numpy.count_nonzero
newaxis = None
bool_ = numpy.bool_
int_ = numpy.int64
float_ = numpy.float64
complex_ = numpy.complex128
inf = float("inf")
nan = float("nan")


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
