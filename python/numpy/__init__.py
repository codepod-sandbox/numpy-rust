"""NumPy-compatible Python package wrapping the Rust native module."""
import sys as _sys
import math as _stdlib_math
import json as _json
from functools import reduce as _reduce

__version__ = "1.26.0"

# Import from native Rust module
import _numpy_native as _native
from _numpy_native import ndarray
from _numpy_native import dot
from _numpy_native import concatenate as _native_concatenate
from ._helpers import *
from ._core_types import *
from ._datetime import *
from ._creation import *
from ._math import *
from ._reductions import *
from ._manipulation import *
from ._bitwise import *
from ._ufunc import *
from ._poly import *
from ._indexing import *
from ._window import *
from ._io import *
from ._stubs import *
from ._linalg_ext import *
from ._fft_ext import *
from ._random_ext import *


# Import submodules so they're accessible as numpy.linalg etc.
from _numpy_native import linalg, fft, random

# Register Rust submodules in sys.modules so `from numpy.random import ...` works
_sys.modules["numpy.linalg"] = linalg
_sys.modules["numpy.fft"] = fft
_sys.modules["numpy.random"] = random


# --- Constants --------------------------------------------------------------
nan = float("nan")
inf = float("inf")
pi = _stdlib_math.pi
e = _stdlib_math.e
newaxis = None
PINF = float("inf")
NINF = float("-inf")
PZERO = 0.0
NZERO = -0.0
Inf = inf
Infinity = inf
NaN = nan
NAN = nan
euler_gamma = 0.5772156649015329
ALLOW_THREADS = 1
little_endian = True

# numpy 1.x compat: np.bool (deprecated, can't shadow builtin 'bool' in module
# scope since isinstance checks recurse). We set it via __getattr__ below.


# ---------------------------------------------------------------------------
# _parse_dtype_json, void scalar, StructuredArray wrapper
# ---------------------------------------------------------------------------

def _parse_dtype_json(json_str):
    """Convert dtype_json string back to StructuredDtype.
    dtype_json format: [["x","float64"],["y","int32"]]
    Returns a StructuredDtype instance.
    """
    from numpy._core_types import StructuredDtype
    pairs = _json.loads(json_str)   # list of [name, dtype_str]
    return StructuredDtype([(name, dt_str) for name, dt_str in pairs])


class void:
    """Scalar returned by arr[i] on a structured array."""

    def __init__(self, data, dtype):
        # Use object.__setattr__ throughout to avoid any __setattr__ override issues
        object.__setattr__(self, '_data', data)   # dict {fieldname: scalar_value}
        object.__setattr__(self, 'dtype', dtype)

    def __getitem__(self, key):
        return object.__getattribute__(self, '_data')[key]

    def __getattr__(self, name):
        # __getattr__ is only called when normal lookup fails
        data = object.__getattribute__(self, '_data')
        if name in data:
            return data[name]
        raise AttributeError(name)

    def __repr__(self):
        data = object.__getattribute__(self, '_data')
        dt = object.__getattribute__(self, 'dtype')
        vals = tuple(data[n] for n in dt.names)
        return repr(vals)

    def __iter__(self):
        data = object.__getattribute__(self, '_data')
        dt = object.__getattribute__(self, 'dtype')
        return iter(data[n] for n in dt.names)


class StructuredArray:
    """Python wrapper for _native.StructuredArray (columnar Rust-backed structured array)."""

    def __init__(self, native_arr):
        object.__setattr__(self, '_native_arr', native_arr)
        dt = _parse_dtype_json(native_arr.dtype)
        object.__setattr__(self, 'dtype', dt)

    def __getitem__(self, key):
        native = object.__getattribute__(self, '_native_arr')
        result = native[key]
        dt = object.__getattribute__(self, 'dtype')
        # Integer key → Rust returns a list of scalars in field order
        if isinstance(result, list):
            return void({n: v for n, v in zip(dt.names, result)}, dt)
        # List of strings → Rust returns _native.StructuredArray → wrap
        if hasattr(result, 'field_names'):
            return StructuredArray(result)
        # String key → Rust returns PyNdArray column directly
        return result

    def __setitem__(self, key, val):
        native = object.__getattribute__(self, '_native_arr')
        # Unwrap StructuredArray wrappers for Rust
        if isinstance(val, StructuredArray):
            val = object.__getattribute__(val, '_native_arr')
        native[key] = val

    def __len__(self):
        return len(object.__getattribute__(self, '_native_arr'))

    def __iter__(self):
        dt = object.__getattribute__(self, 'dtype')
        native = object.__getattribute__(self, '_native_arr')
        for i in range(len(self)):
            row_list = native[i]   # list of scalars from Rust __getitem__(int)
            yield void({n: v for n, v in zip(dt.names, row_list)}, dt)

    @property
    def shape(self):
        return tuple(object.__getattribute__(self, '_native_arr').shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def field_names(self):
        return object.__getattribute__(self, '_native_arr').field_names()

    def __repr__(self):
        dt = object.__getattribute__(self, 'dtype')
        rows = list(self)
        return f"StructuredArray({rows}, dtype={dt})"


# --- Aliases ----------------------------------------------------------------

absolute = abs
conjugate = conj

radians = deg2rad
degrees = rad2deg

round_ = around
round = around

mod = remainder
divmod = divmod_

special = type('special', (), {
    'gamma': staticmethod(gamma),
    'erf': staticmethod(erf),
    'erfc': staticmethod(erfc),
    'lgamma': staticmethod(lgamma),
    'j0': staticmethod(j0),
    'j1': staticmethod(j1),
    'y0': staticmethod(y0),
    'y1': staticmethod(y1),
})()

# Link top-level functions into linalg module
linalg.trace = trace
linalg.cross = cross        # delegate to top-level cross()
linalg.diagonal = diagonal  # delegate to top-level diagonal()
linalg.outer = outer        # delegate to top-level outer()

# --- Import submodules so np.ma and np.polynomial are accessible ------------
import numpy.ma as ma
import numpy.polynomial as polynomial

# --- Module-level __getattr__ for deprecated aliases like np.bool -----------
def __getattr__(name):
    _bi = __import__("builtins")
    _deprecated_aliases = {
        'bool': _bi.bool,
        'int': _bi.int,
        'float': _bi.float,
        'complex': _bi.complex,
        'str': _bi.str,
        'object': _bi.object,
    }
    if name in _deprecated_aliases:
        return _deprecated_aliases[name]
    raise AttributeError(f"module 'numpy' has no attribute '{name}'")
