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
from ._ufunc import _ufunc_reconstruct, gcd, lcm
# RustPython treats builtin names like 'divmod' specially in module scope;
# setattr on the module object is the only reliable way to override it.
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
    """Scalar returned by arr[i] on a structured array, or a placeholder void of N bytes."""

    def __init__(self, data, dtype=None):
        if isinstance(data, int) and dtype is None:
            # np.void(n) creates a placeholder void of n bytes
            object.__setattr__(self, '_data', {})
            object.__setattr__(self, 'dtype', None)
            return
        # Use object.__setattr__ throughout to avoid any __setattr__ override issues
        object.__setattr__(self, '_data', data)   # dict {fieldname: scalar_value}
        object.__setattr__(self, 'dtype', dtype)

    def __bool__(self):
        return False  # void scalars are always considered zero

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
        if dt is None or not data:
            return 'void()'
        vals = tuple(data[n] for n in dt.names)
        return repr(vals)

    def __iter__(self):
        data = object.__getattribute__(self, '_data')
        dt = object.__getattribute__(self, 'dtype')
        if dt is None or not data:
            return iter([])
        return iter(data[n] for n in dt.names)


class StructuredArray:
    """Python wrapper for _native.StructuredArray (columnar Rust-backed structured array)."""

    def __init__(self, native_arr, py_shape=None):
        object.__setattr__(self, '_native_arr', native_arr)
        dt = _parse_dtype_json(native_arr.dtype)
        object.__setattr__(self, 'dtype', dt)
        # _py_shape: None = use Rust 1D shape; tuple = Python-tracked multi-D shape
        object.__setattr__(self, '_py_shape', tuple(py_shape) if py_shape is not None else None)

    def __getitem__(self, key):
        native = object.__getattribute__(self, '_native_arr')
        dt = object.__getattribute__(self, 'dtype')
        shape = self.shape

        # Tuple key: (i, j) for 2D element access
        if isinstance(key, tuple) and len(shape) > 1:
            if len(key) != len(shape):
                raise IndexError(
                    "too many indices for array: array is {}-dimensional, "
                    "but {} were indexed".format(len(shape), len(key))
                )
            flat = 0
            stride = 1
            for k in _builtin_range(len(shape) - 1, -1, -1):
                flat += key[k] * stride
                stride *= shape[k]
            result = native[flat]
            if isinstance(result, list):
                return void({n: v for n, v in zip(dt.names, result)}, dt)
            return result

        # Integer key on multi-D: return a row (1D StructuredArray)
        if isinstance(key, int) and len(shape) > 1:
            cols = shape[1]  # number of elements per row
            row_native = native[key * cols : (key + 1) * cols]
            if hasattr(row_native, 'field_names'):
                return StructuredArray(row_native)
            # fallback: shouldn't happen
            return row_native

        # All other cases: delegate to existing behavior
        result = native[key]

        # String key: field access — reshape if multi-D, mark as unaligned
        if isinstance(key, str):
            if len(shape) > 1:
                result = result.reshape(shape)
            # Structured field views are unaligned (interleaved memory semantics)
            if hasattr(result, '_set_unaligned'):
                result._set_unaligned()
            return result

        # Integer key (1D) → void scalar
        if isinstance(result, list):
            return void({n: v for n, v in zip(dt.names, result)}, dt)

        # Slice or list-of-strings → StructuredArray
        if hasattr(result, 'field_names'):
            return StructuredArray(result)

        return result

    def __setitem__(self, key, val):
        native = object.__getattribute__(self, '_native_arr')
        # Unwrap StructuredArray wrappers for Rust
        if isinstance(val, StructuredArray):
            val = object.__getattribute__(val, '_native_arr')
        native[key] = val

    def __len__(self):
        return self.shape[0]

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (list, tuple)):
            new_shape = tuple(new_shape[0])
        else:
            new_shape = tuple(new_shape)
        total = 1
        for s in new_shape:
            total *= s
        native = object.__getattribute__(self, '_native_arr')
        if total != len(native):
            raise ValueError(
                "cannot reshape structured array of size {} into shape {}".format(
                    len(native), new_shape
                )
            )
        # If result is 1D, clear _py_shape so Rust shape is used
        if len(new_shape) == 1:
            return StructuredArray(native)
        return StructuredArray(native, py_shape=new_shape)

    def __iter__(self):
        shape = self.shape
        if len(shape) > 1:
            for i in _builtin_range(shape[0]):
                yield self[i]
            return
        dt = object.__getattribute__(self, 'dtype')
        native = object.__getattribute__(self, '_native_arr')
        for i in _builtin_range(len(native)):
            row_list = native[i]
            yield void({n: v for n, v in zip(dt.names, row_list)}, dt)

    @property
    def shape(self):
        py_shape = object.__getattribute__(self, '_py_shape')
        if py_shape is not None:
            return py_shape
        return tuple(object.__getattribute__(self, '_native_arr').shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def field_names(self):
        return object.__getattribute__(self, '_native_arr').field_names()

    @property
    def flags(self):
        from ._helpers import _ArrayFlags
        return _ArrayFlags(c_contiguous=True, f_contiguous=False)

    def tolist(self):
        """Return the array as a nested Python list of tuples."""
        dt = object.__getattribute__(self, 'dtype')
        return [tuple(row[n] for n in dt.names) for row in self]

    def __repr__(self):
        dt = object.__getattribute__(self, 'dtype')
        rows = list(self)
        return f"StructuredArray({rows}, dtype={dt})"


class recarray:
    """Structured array with attribute-style field access (np.recarray)."""

    def __init__(self, shape, dtype):
        """Create an empty (zero-filled) structured array."""
        if isinstance(shape, int):
            shape = (shape,)
        if len(shape) != 1:
            raise ValueError(
                "recarray only supports 1D arrays; got shape {}".format(shape)
            )
        arr = zeros(shape[0], dtype=dtype)   # returns StructuredArray
        object.__setattr__(self, '_arr', arr)
        object.__setattr__(self, 'dtype', arr.dtype)

    @classmethod
    def _from_structured(cls, structured_arr):
        """Wrap an existing StructuredArray as a recarray (no data copy)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, '_arr', structured_arr)
        object.__setattr__(obj, 'dtype', structured_arr.dtype)
        return obj

    def __getattr__(self, name):
        try:
            arr = object.__getattribute__(self, '_arr')
            dt = object.__getattribute__(self, 'dtype')
        except AttributeError:
            raise AttributeError(name)
        if name in dt.names:
            return arr[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        try:
            arr = object.__getattribute__(self, '_arr')
            dt = object.__getattribute__(self, 'dtype')
        except AttributeError:
            object.__setattr__(self, name, value)
            return
        if name in dt.names:
            arr[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getitem__(self, key):
        return object.__getattribute__(self, '_arr')[key]

    def __setitem__(self, key, val):
        object.__getattribute__(self, '_arr')[key] = val

    def __len__(self):
        return len(object.__getattribute__(self, '_arr'))

    def __iter__(self):
        return iter(object.__getattribute__(self, '_arr'))

    @property
    def shape(self):
        return object.__getattribute__(self, '_arr').shape

    @property
    def ndim(self):
        return object.__getattribute__(self, '_arr').ndim


class _RecModule:
    """Stub for numpy.rec submodule."""

    def array(self, data, dtype=None, **kwargs):
        """Create a recarray from data (list of tuples)."""
        arr = array(data, dtype=dtype)   # → StructuredArray or ndarray
        if isinstance(arr, StructuredArray):
            return recarray._from_structured(arr)
        return arr

    def fromarrays(self, arrayList, dtype=None, names=None, **kwargs):
        """Create a recarray from a list of arrays, one per field."""
        if names is not None and dtype is None:
            # Build a structured dtype from the names and inferred column types
            fields = []
            for name, col in zip(names, arrayList):
                col_arr = array(col)
                fields.append((name, str(col_arr.dtype)))
            dtype = globals()['dtype'](fields)
        arr = zeros(len(arrayList[0]), dtype=dtype)
        for name, col in zip(dtype.names, arrayList):
            arr[name] = array(col)
        return recarray._from_structured(arr)


rec = _RecModule()


# --- Aliases ----------------------------------------------------------------

absolute = abs
conjugate = conj

radians = deg2rad
degrees = rad2deg

round_ = around
round = around

mod = remainder
# divmod is exposed as a ufunc (see _ufunc.py); divmod_ is the plain function alias
# divmod = divmod_  # (replaced by ufunc version imported from _ufunc)

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
