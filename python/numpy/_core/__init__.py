"""numpy._core - internal core package, re-exports from numpy."""
import sys
import numpy

# Re-export core functions
from numpy import (
    array, asarray, zeros, ones, empty, full, arange,
    linspace, logspace, eye, identity,
    dot, concatenate, stack, vstack, hstack,
    reshape, transpose, swapaxes, ravel,
    expand_dims, squeeze,
    sort, argsort, argmax, argmin, searchsorted,
    where, nonzero, count_nonzero,
    sum, prod, mean, std, var, min, max, all, any,
    cumsum, cumprod,
    clip, around,
    abs, sign, sqrt, log, exp,
    sin, cos, tan, arcsin, arccos, arctan,
    isnan, isinf, isfinite,
    floor, ceil, trunc,
    finfo, iinfo, dtype,
    bool_, int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64, complex64, complex128,
    ndarray,
)

# sctypes: mapping of type categories
sctypes = {
    "float": [float32, float64],
    "int": [int8, int16, int32, int64],
    "uint": [uint8, uint16, uint32, uint64],
    "complex": [complex64, complex128],
    "others": [bool_],
}

# Import submodules
from numpy._core import numeric, multiarray, fromnumeric, umath, numerictypes
from numpy._core import shape_base, function_base, arrayprint
from numpy._core import _exceptions, overrides

# Register in sys.modules
sys.modules['numpy._core.numeric'] = numeric
sys.modules['numpy._core.multiarray'] = multiarray
sys.modules['numpy._core.fromnumeric'] = fromnumeric
sys.modules['numpy._core.umath'] = umath
sys.modules['numpy._core.numerictypes'] = numerictypes
sys.modules['numpy._core.shape_base'] = shape_base
sys.modules['numpy._core.function_base'] = function_base
sys.modules['numpy._core.arrayprint'] = arrayprint
sys.modules['numpy._core._exceptions'] = _exceptions
sys.modules['numpy._core.overrides'] = overrides


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
