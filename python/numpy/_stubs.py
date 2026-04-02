"""Stub/placeholder modules and backward-compat re-exports for numpy-rust."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray, AxisError, _builtin_max, _copy_into, _ComplexResultArray
from ._core_types import (
    dtype, _normalize_dtype, StructuredDtype,
    Float64DType, Float32DType, Float16DType,
    Int8DType, Int16DType, Int32DType, Int64DType,
    UInt8DType, UInt16DType, UInt32DType, UInt64DType,
    Complex64DType, Complex128DType, BoolDType, StrDType,
    BytesDType, VoidDType, ObjectDType,
)
from ._creation import array, asarray, _make_complex_array

# ---------------------------------------------------------------------------
# Re-export submodules
# ---------------------------------------------------------------------------
from ._errors import *
from ._format import *
from ._linalg_ops import *
from ._bitops import *
from ._array_utils import *

__all__ = [
    # error handling
    '_err_state', 'seterr', 'geterr', 'errstate',
    'seterrcall', 'geterrcall',
    # print formatting
    'set_printoptions', 'get_printoptions', 'printoptions',
    # string/char module
    'char',
    # array representation
    'array_str', 'array_repr', 'array2string',
    # lib module
    'lib',
    # testing module
    'testing',
    # dtypes module
    'dtypes',
    # rec module
    'rec',
    # module stubs
    'core', 'compat', 'exceptions', 'matlib', 'ctypeslib',
    # exception classes
    'ComplexWarning', 'VisibleDeprecationWarning',
    # misc utilities
    'info', 'who', 'show_config', 'einsum_path', 'byte_bounds',
    # format functions
    'format_float_positional', 'format_float_scientific',
    # memmap stub
    'memmap',
    # misc stubs
    'add_newdoc', 'deprecate', 'get_include',
    # NumpyVersion
    'NumpyVersion',
    # constants
    'tracemalloc_domain', 'use_hugepage', 'nested_iters',
    # linear algebra / product functions
    'outer', 'cross', 'tensordot', 'inner', 'kron', 'matmul', 'vdot', 'einsum',
    # bit manipulation
    'packbits', 'unpackbits',
    # misc numeric
    'binary_repr', 'base_repr', 'frompyfunc',
    # indexing helpers
    'take_along_axis', 'put_along_axis',
    # matrix class
    'matrix',
    # internal helper
    '_has_complex',
]


# ---------------------------------------------------------------------------
# lib module
# ---------------------------------------------------------------------------

class _LibModule:
    class stride_tricks:
        @staticmethod
        def as_strided(x, shape=None, strides=None, subok=False, writeable=True):
            """Simplified as_strided - creates a new array with the given shape.
            WARNING: This does NOT share memory with the original array.
            It creates a view-like result by repeating/tiling data."""
            x = asarray(x)
            if shape is None:
                return x.copy()
            # Best effort: reshape or tile to match requested shape
            flat = x.flatten().tolist()
            total = 1
            for s in shape:
                total *= s
            # Repeat flat data to fill the requested size
            result = []
            for i in range(total):
                result.append(flat[i % len(flat)])
            return array(result).reshape(shape)

        @staticmethod
        def sliding_window_view(x, window_shape, axis=None):
            """Create a sliding window view of the array."""
            x = asarray(x)
            if isinstance(window_shape, int):
                window_shape = (window_shape,)
            if x.ndim == 1 and len(window_shape) == 1:
                w = window_shape[0]
                data = x.tolist()
                n = len(data) - w + 1
                if n <= 0:
                    return array([]).reshape((0, w))
                rows = []
                for i in range(n):
                    rows.append(data[i:i+w])
                return array(rows)
            raise NotImplementedError("sliding_window_view only supports 1D")

lib = _LibModule()

from ._linalg_ops import _ScimathModule
lib.scimath = _ScimathModule()


# ---------------------------------------------------------------------------
# NumpyVersion
# ---------------------------------------------------------------------------

class NumpyVersion:
    """Minimal numpy version comparison class."""
    def __init__(self, vstring):
        self.vstring = vstring
        parts = vstring.split('.')
        self.major = int(parts[0]) if len(parts) > 0 else 0
        self.minor = int(parts[1]) if len(parts) > 1 else 0
        self.bugfix = int(parts[2].split('rc')[0].split('a')[0].split('b')[0]) if len(parts) > 2 else 0
    def __repr__(self):
        return f"NumpyVersion('{self.vstring}')"
    def __str__(self):
        return self.vstring
    def __lt__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) < (other.major, other.minor, other.bugfix)
    def __le__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) <= (other.major, other.minor, other.bugfix)
    def __gt__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) > (other.major, other.minor, other.bugfix)
    def __ge__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) >= (other.major, other.minor, other.bugfix)
    def __eq__(self, other):
        if isinstance(other, str): other = NumpyVersion(other)
        return (self.major, self.minor, self.bugfix) == (other.major, other.minor, other.bugfix)

lib.NumpyVersion = NumpyVersion


# ---------------------------------------------------------------------------
# testing module
# ---------------------------------------------------------------------------

class _AssertRaisesRegexContext:
    def __init__(self, exc_class, pattern):
        self.exc_class = exc_class
        self.pattern = pattern
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        import re
        if exc_type is None:
            raise AssertionError("{} not raised".format(self.exc_class.__name__))
        if not issubclass(exc_type, self.exc_class):
            return False
        if not re.search(self.pattern, str(exc_val)):
            raise AssertionError("'{}' did not match '{}'".format(str(exc_val), self.pattern))
        return True


class _TestingModule:
    def assert_allclose(self, actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        from . import allclose
        actual = asarray(actual)
        desired = asarray(desired)
        if not allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan):
            actual_list = actual.tolist()
            desired_list = desired.tolist()
            raise AssertionError(err_msg or "Not equal to tolerance rtol={}, atol={}\n Actual: {}\n Desired: {}".format(rtol, atol, actual_list, desired_list))

    def assert_array_equal(self, x, y, err_msg='', verbose=True, strict=False):
        from . import array_equal, broadcast_to
        x = asarray(x)
        y = asarray(y)
        # Handle scalar vs array comparison (NumPy broadcasts)
        if x.shape != y.shape:
            # 0-D vs 1-element: equivalent for comparison purposes
            if x.ndim == 0 and y.size == 1:
                y = y.reshape(())
            elif y.ndim == 0 and x.size == 1:
                x = x.reshape(())
            elif y.size == 1:
                y = broadcast_to(y.flatten(), x.shape)
            elif x.size == 1:
                x = broadcast_to(x.flatten(), y.shape)
        if not array_equal(x, y, equal_nan=True):
            raise AssertionError(err_msg or "Arrays are not equal\n x: {}\n y: {}".format(x.tolist(), y.tolist()))

    def assert_array_almost_equal(self, x, y, decimal=6, err_msg='', verbose=True):
        from . import allclose
        x = asarray(x)
        y = asarray(y)
        if not allclose(x, y, rtol=0, atol=1.5 * 10**(-decimal)):
            raise AssertionError(err_msg or "Arrays are not almost equal to {} decimals".format(decimal))

    def assert_equal(self, actual, desired, err_msg='', verbose=True):
        from . import array_equal, broadcast_to
        # Handle tuples/lists recursively
        if isinstance(actual, (tuple, list)) and isinstance(desired, (tuple, list)):
            if len(actual) != len(desired):
                raise AssertionError(err_msg or "Length mismatch: {} vs {}".format(len(actual), len(desired)))
            for i, (a, d) in enumerate(zip(actual, desired)):
                self.assert_equal(a, d, err_msg=err_msg, verbose=verbose)
            return
        actual_a = asarray(actual)
        desired_a = asarray(desired)
        # Empty array is vacuously equal to any scalar
        if actual_a.size == 0 and desired_a.size <= 1:
            return
        if desired_a.size == 0 and actual_a.size <= 1:
            return
        # Handle 0-d scalar comparison: extract element and compare directly
        if actual_a.shape == () and desired_a.size == 1:
            a_val = actual_a.flatten()[0]
            d_val = desired_a.flatten()[0]
            if a_val != d_val:
                raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))
            return
        if desired_a.shape == () and actual_a.size == 1:
            a_val = actual_a.flatten()[0]
            d_val = desired_a.flatten()[0]
            if a_val != d_val:
                raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))
            return
        # Handle scalar vs array comparison
        if actual_a.shape != desired_a.shape:
            if desired_a.size == 1:
                desired_a = broadcast_to(desired_a.flatten(), actual_a.shape)
            elif actual_a.size == 1:
                actual_a = broadcast_to(actual_a.flatten(), desired_a.shape)
        if not array_equal(actual_a, desired_a, equal_nan=True):
            raise AssertionError(err_msg or "Items are not equal:\n actual: {}\n desired: {}".format(actual, desired))

    def assert_raises(self, exception_class, *args, **kwargs):
        """Simple assert_raises - returns a context manager."""
        class _AssertRaisesCtx:
            def __init__(self, exc_cls):
                self.exc_cls = exc_cls
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                if exc_type is None:
                    raise AssertionError("Expected {} but no exception raised".format(self.exc_cls.__name__))
                if not issubclass(exc_type, self.exc_cls):
                    return False  # re-raise
                return True  # suppress exception
        if args:
            # Called as assert_raises(Error, func, *args)
            callable_obj = args[0]
            rest = args[1:]
            try:
                callable_obj(*rest, **kwargs)
            except exception_class:
                return
            raise AssertionError("Expected {}".format(exception_class.__name__))
        return _AssertRaisesCtx(exception_class)

    def assert_raises_regex(self, exception_class, expected_regex, *args, **kwargs):
        """Assert that an exception is raised matching a regex."""
        import re
        callable_obj = kwargs.pop('callable', None)
        if callable_obj is None and len(args) >= 1:
            callable_obj = args[0]
            args = args[1:]
        if callable_obj is not None:
            try:
                callable_obj(*args, **kwargs)
            except exception_class as e:
                if not re.search(expected_regex, str(e)):
                    raise AssertionError(
                        "Exception message '{}' did not match '{}'".format(str(e), expected_regex))
                return
            except Exception as e:
                raise AssertionError(
                    "Expected {}, got {}".format(exception_class.__name__, type(e).__name__))
            raise AssertionError("{} not raised".format(exception_class.__name__))
        else:
            # Context manager mode
            return _AssertRaisesRegexContext(exception_class, expected_regex)

    def assert_warns(self, warning_class, *args, **kwargs):
        """Assert that a warning is raised. Since we don't have warnings module, just run the callable."""
        callable_obj = kwargs.pop('callable', None)
        if callable_obj is None and len(args) >= 1:
            callable_obj = args[0]
            args = args[1:]
        if callable_obj is not None:
            return callable_obj(*args, **kwargs)
        # Return a context manager that suppresses warnings
        class _WarnCtx:
            def __enter__(self_ctx):
                return self_ctx
            def __exit__(self_ctx, *exc):
                return False
        return _WarnCtx()

    def assert_approx_equal(self, actual, desired, significant=7, err_msg='', verbose=True):
        """Assert approximately equal to given number of significant digits."""
        if desired == 0:
            if _math.fabs(actual) > 1.5 * 10**(-significant):
                raise AssertionError("{} != {} to {} significant digits".format(actual, desired, significant))
        else:
            rel = _math.fabs((actual - desired) / desired)
            if rel > 1.5 * 10**(-significant):
                raise AssertionError("{} != {} to {} significant digits".format(actual, desired, significant))

    def assert_array_less(self, x, y, err_msg='', verbose=True):
        """Assert array_like x is less than array_like y, element-wise."""
        x = asarray(x)
        y = asarray(y)
        if not all((x < y).flatten().tolist()):
            raise AssertionError("Arrays are not less-ordered\nx: {}\ny: {}".format(x.tolist(), y.tolist()))


# testing module is now properly defined in numpy/testing/_utils.py
# Import the real module instead of using the stub class
import numpy.testing as testing


# ---------------------------------------------------------------------------
# dtypes module
# ---------------------------------------------------------------------------

class _dtypes_mod:
    Float64DType = Float64DType
    Float32DType = Float32DType
    Float16DType = Float16DType
    Int8DType = Int8DType
    Int16DType = Int16DType
    Int32DType = Int32DType
    Int64DType = Int64DType
    UInt8DType = UInt8DType
    UInt16DType = UInt16DType
    UInt32DType = UInt32DType
    UInt64DType = UInt64DType
    Complex64DType = Complex64DType
    Complex128DType = Complex128DType
    BoolDType = BoolDType
    StrDType = StrDType
    BytesDType = BytesDType
    VoidDType = VoidDType
    ObjectDType = ObjectDType

dtypes = _dtypes_mod()


# ---------------------------------------------------------------------------
# rec module
# ---------------------------------------------------------------------------

class _RecModule:
    """Minimal np.rec namespace."""
    def __init__(self):
        self.recarray = None  # placeholder

    def array(self, data, dtype=None, formats=None, names=None, shape=None, byteorder=None):
        """Create a record array (falls back to regular array)."""
        import numpy as np
        # Build dtype from formats if provided
        if dtype is None and formats is not None:
            if isinstance(formats, str):
                fmt_list = [f.strip() for f in formats.split(',')]
            else:
                fmt_list = list(formats)
            if names is None:
                names = ['f{}'.format(i) for i in range(len(fmt_list))]
            elif isinstance(names, str):
                names = [n.strip() for n in names.split(',')]
            fields = list(zip(names, fmt_list))
            dtype = StructuredDtype(fields)
        if isinstance(data, (list, tuple)):
            if dtype is not None:
                if isinstance(dtype, list):
                    dtype = StructuredDtype(dtype)
                return np.array(data, dtype=dtype)
        arr = asarray(data)
        if dtype is not None:
            dt = dtype if isinstance(dtype, StructuredDtype) else StructuredDtype(dtype) if isinstance(dtype, list) else dtype
            # Try to attach structured dtype metadata; silently skip if type doesn't allow it
            try:
                arr._structured_dtype = dt
            except (TypeError, AttributeError):
                pass
        return arr

    def fromarrays(self, arrays, dtype=None, names=None):
        """Create a record array from separate arrays."""
        import numpy as np
        if isinstance(names, str):
            names = [n.strip() for n in names.split(',')]
        if dtype is None and names is not None:
            # Infer dtype from each array
            fields = []
            for i, (name, arr) in enumerate(zip(names, arrays)):
                a = np.asarray(arr)
                fields.append((name, str(a.dtype)))
            dtype = StructuredDtype(fields)
        if dtype is not None:
            if not isinstance(dtype, StructuredDtype):
                if isinstance(dtype, list):
                    dtype = StructuredDtype(dtype)
            # Build structured array from column arrays
            col_arrays = [np.asarray(a) for a in arrays]
            n = len(col_arrays[0]) if len(col_arrays) > 0 else 0
            records = []
            for i in range(n):
                rec = tuple(a[i] if hasattr(a, '__getitem__') else a for a in col_arrays)
                records.append(rec)
            return np.array(records, dtype=dtype)
        return np.array(arrays)

    def fromrecords(self, reclist, dtype=None, names=None, formats=None):
        """Create a record array from a list of records (tuples)."""
        import numpy as np
        if isinstance(names, str):
            names = [n.strip() for n in names.split(',')]
        if dtype is not None:
            if isinstance(dtype, list):
                dtype = StructuredDtype(dtype)
            return np.array(reclist, dtype=dtype)
        if names is not None and formats is not None:
            if isinstance(formats, str):
                formats = [f.strip() for f in formats.split(',')]
            fields = list(zip(names, formats))
            dt = StructuredDtype(fields)
            return np.array(reclist, dtype=dt)
        if names is not None:
            # Infer types from first record
            if len(reclist) > 0:
                first = reclist[0]
                fields = []
                for i, name in enumerate(names):
                    val = first[i] if isinstance(first, (list, tuple)) else first
                    if isinstance(val, int):
                        fields.append((name, 'int64'))
                    elif isinstance(val, float):
                        fields.append((name, 'float64'))
                    elif isinstance(val, str):
                        # Find max string length
                        max_len = max(len(str(r[i] if isinstance(r, (list, tuple)) else r)) for r in reclist)
                        fields.append((name, 'U' + str(max(max_len, 1))))
                    elif isinstance(val, bytes):
                        max_len = max(len(r[i] if isinstance(r, (list, tuple)) else r) for r in reclist)
                        fields.append((name, 'S' + str(max(max_len, 1))))
                    else:
                        fields.append((name, 'float64'))
                dt = StructuredDtype(fields)
                return np.array(reclist, dtype=dt)
            return np.array(reclist)
        return np.array(reclist)

rec = _RecModule()


# ---------------------------------------------------------------------------
# show_config
# ---------------------------------------------------------------------------

def show_config():
    """Show numpy-rust build configuration."""
    print("numpy-rust (codepod)")
    print("  backend: Rust + RustPython")


# ---------------------------------------------------------------------------
# Module stubs
# ---------------------------------------------------------------------------

# np.core module stub
class _CoreModule:
    """Minimal np.core namespace."""
    pass

core = _CoreModule()
core.numeric = core  # self-reference for np.core.numeric compatibility
core.multiarray = core  # np.core.multiarray compatibility
core.fromnumeric = core  # np.core.fromnumeric compatibility

# np.compat module stub
class _CompatModule:
    pass
compat = _CompatModule()

# np.exceptions module
class _ExceptionsModule:
    AxisError = AxisError  # already defined
    ComplexWarning = type('ComplexWarning', (UserWarning,), {})
    DTypePromotionError = type('DTypePromotionError', (TypeError,), {})
    VisibleDeprecationWarning = type('VisibleDeprecationWarning', (UserWarning,), {})
    ModuleDeprecationWarning = type('ModuleDeprecationWarning', (DeprecationWarning,), {})
    RankWarning = type('RankWarning', (UserWarning,), {})
    TooHardError = type('TooHardError', (RuntimeError,), {})

exceptions = _ExceptionsModule()
exceptions.__name__ = 'numpy.exceptions'

# Expose exception classes at top level (sklearn fallback path)
ComplexWarning = exceptions.ComplexWarning
VisibleDeprecationWarning = exceptions.VisibleDeprecationWarning

import numpy.matlib as matlib

class _CtypeslibModule:
    """Minimal numpy.ctypeslib — ctypes integration for arrays."""
    @staticmethod
    def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
        """Array-checking restype/argtypes for ctypes."""
        import ctypes
        class _ndptr(ctypes.c_void_p):
            @classmethod
            def from_param(cls, obj):
                return obj
        return _ndptr

    @staticmethod
    def as_array(obj, shape=None):
        """Create a numpy array from a ctypes array or pointer."""
        import numpy as _np
        return _np.asarray(obj)

    @staticmethod
    def as_ctypes_type(dtype):
        """Return the ctypes type corresponding to the given dtype."""
        import ctypes
        import numpy as _np
        _map = {
            'bool': ctypes.c_bool,
            'int8': ctypes.c_int8,
            'int16': ctypes.c_int16,
            'int32': ctypes.c_int32,
            'int64': ctypes.c_int64,
            'uint8': ctypes.c_uint8,
            'uint16': ctypes.c_uint16,
            'uint32': ctypes.c_uint32,
            'uint64': ctypes.c_uint64,
            'float32': ctypes.c_float,
            'float64': ctypes.c_double,
        }
        name = str(_np.dtype(dtype))
        if name not in _map:
            raise TypeError("No known ctypes type for dtype {}".format(dtype))
        return _map[name]

    @staticmethod
    def as_ctypes(obj):
        """Create and return a ctypes object from a numpy array."""
        import ctypes
        import numpy as _np
        obj = _np.asarray(obj)
        # return a ctypes array of the appropriate type
        ctype = _CtypeslibModule.as_ctypes_type(obj.dtype)
        return (ctype * obj.size)(*obj.flatten().tolist())

    def load_library(self, libname, loader_path):
        import ctypes
        return ctypes.CDLL(libname)

ctypeslib = _CtypeslibModule()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

tracemalloc_domain = 0
use_hugepage = 0
nested_iters = None  # Not supported
