"""NumPy-compatible Python package wrapping the Rust native module."""
import sys as _sys
import math as _math

# Import from native Rust module
import _numpy_native as _native
from _numpy_native import ndarray
from _numpy_native import dot, concatenate

# Wrap creation functions to accept (and currently ignore) dtype keyword
class _ObjectArray:
    """Lightweight fallback for arrays with non-numeric dtypes (strings, structured, etc.)."""
    def __init__(self, data, dt=None):
        if isinstance(data, (list, tuple)):
            self._data = list(data)
        else:
            self._data = [data]
        self._dtype = dt or "object"
        if isinstance(self._data, list) and len(self._data) > 0 and isinstance(self._data[0], (list, tuple)):
            self._shape = (len(self._data), len(self._data[0]))
            self._ndim = 2
        else:
            self._shape = (len(self._data),)
            self._ndim = 1

    @property
    def shape(self): return self._shape
    @property
    def ndim(self): return self._ndim
    @property
    def dtype(self): return self._dtype
    @property
    def size(self): return len(self._data)
    @property
    def T(self): return self

    def copy(self): return _ObjectArray(list(self._data), self._dtype)
    def astype(self, dtype): return _ObjectArray(list(self._data), str(dtype))
    def flatten(self): return self
    def ravel(self): return self
    def all(self): return all(self._data)
    def any(self): return any(self._data)
    def __len__(self): return len(self._data)
    def __getitem__(self, key): return self._data[key]
    def __eq__(self, other):
        if isinstance(other, _ObjectArray):
            return _ObjectArray([a == b for a, b in zip(self._data, other._data)], "bool")
        return NotImplemented
    def __repr__(self): return f"array({self._data!r}, dtype='{self._dtype}')"


def array(data, dtype=None, copy=None, order=None, subok=False, ndmin=0, like=None):
    # Check if dtype forces a non-numeric path
    if dtype is not None:
        dt = str(dtype)
        # String dtypes: route to Rust native (S-prefixed, U-prefixed, "str")
        if dt.startswith("S") or dt.startswith("U") or dt == "str":
            if isinstance(data, str):
                data = [data]
            if isinstance(data, (list, tuple)):
                return _native.array([str(x) for x in data])
            return _native.array(data)
        if dt == "object" or "," in dt:
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data], dt)
    if isinstance(data, _ObjectArray):
        return data.copy() if copy else data
    if isinstance(data, ndarray):
        return data.copy() if copy else data
    if isinstance(data, (int, float)):
        return _native.array([float(data)])
    if isinstance(data, str):
        # Single string → string array
        return _native.array([data])
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        # List of strings → string array
        return _native.array(data)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
        return _native.array([float(x) for x in data])
    # Try the native array constructor
    try:
        return _native.array(data)
    except (TypeError, ValueError):
        try:
            return _native.array(_to_float_list(data))
        except (TypeError, ValueError):
            # Final fallback for non-numeric data
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data])


def _to_float_list(data):
    """Recursively convert nested lists to flat float lists for the Rust array constructor."""
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, (int, float)):
                result.append(float(item))
            elif isinstance(item, list):
                result.extend(_to_float_list(item))
            else:
                result.append(float(item))
        return result
    return [float(data)]

def zeros(shape, dtype=None, order="C", like=None):
    return _native.zeros(shape)

def ones(shape, dtype=None, order="C", like=None):
    return _native.ones(shape)

def arange(*args, dtype=None, like=None, **kwargs):
    float_args = [float(a) for a in args]
    return _native.arange(*float_args, **kwargs)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start = float(start)
    stop = float(stop)
    num = int(num)
    result = _native.linspace(start, stop, num)
    if retstep:
        step = (stop - start) / max(num - 1, 1) if num > 1 else 0.0
        return result, step
    return result

def eye(N, M=None, k=0, dtype=None, order="C", like=None):
    if M is not None:
        return _native.eye(N, M, k)
    if k != 0:
        return _native.eye(N, N, k)
    return _native.eye(N)

def where(condition, x=None, y=None):
    if x is None and y is None:
        return nonzero(condition)
    return _native.where_(condition, x, y)

# Import submodules so they're accessible as numpy.linalg etc.
from _numpy_native import linalg, fft, random

# Register Rust submodules in sys.modules so `from numpy.random import ...` works
_sys.modules["numpy.linalg"] = linalg
_sys.modules["numpy.fft"] = fft
_sys.modules["numpy.random"] = random

# --- Dtype aliases ----------------------------------------------------------
# NumPy tests reference these as np.float64(value), np.int32(value), etc.
# They must be callable (constructing scalars) AND usable as dtype identifiers.

class _ScalarType:
    """A callable dtype alias that can construct scalars and be used as a dtype string."""
    def __init__(self, name, python_type=float):
        self._name = name
        self._type = python_type

    def __call__(self, value=0, *args, **kwargs):
        try:
            return self._type(value)
        except (ValueError, TypeError):
            # Return the value as-is for unsupported conversions (e.g. NaT)
            return value

    def __repr__(self):
        return f"<class 'numpy.{self._name}'>"

    def __str__(self):
        return self._name

    def __eq__(self, other):
        if isinstance(other, _ScalarType):
            return self._name == other._name
        if isinstance(other, str):
            return self._name == other
        return NotImplemented

    def __hash__(self):
        return hash(self._name)


float64 = _ScalarType("float64", float)
float32 = _ScalarType("float32", float)
float16 = _ScalarType("float16", float)
float128 = _ScalarType("float128", float)
int64 = _ScalarType("int64", int)
int32 = _ScalarType("int32", int)
int16 = _ScalarType("int16", int)
int8 = _ScalarType("int8", int)
uint64 = _ScalarType("uint64", int)
uint32 = _ScalarType("uint32", int)
uint16 = _ScalarType("uint16", int)
uint8 = _ScalarType("uint8", int)
complex64 = _ScalarType("complex64", complex)
complex128 = _ScalarType("complex128", complex)
bool_ = _ScalarType("bool", bool)
intp = _ScalarType("int64", int)
intc = _ScalarType("int32", int)
uintp = _ScalarType("uint64", int)
byte = _ScalarType("int8", int)
ubyte = _ScalarType("uint8", int)
short = _ScalarType("int16", int)
ushort = _ScalarType("uint16", int)
longlong = _ScalarType("int64", int)
ulonglong = _ScalarType("uint64", int)
single = _ScalarType("float32", float)
double = _ScalarType("float64", float)
longdouble = _ScalarType("float64", float)
csingle = _ScalarType("complex64", complex)
cdouble = _ScalarType("complex128", complex)
clongdouble = _ScalarType("complex128", complex)
object_ = _ScalarType("object", object)
str_ = _ScalarType("str", str)
bytes_ = _ScalarType("bytes", bytes)
void = _ScalarType("void")
timedelta64 = _ScalarType("timedelta64", int)
datetime64 = _ScalarType("datetime64", int)
string_ = _ScalarType("str", str)
unicode_ = _ScalarType("str", str)
half = _ScalarType("float16", float)
int_ = int64

# --- Constants --------------------------------------------------------------
nan = float("nan")
inf = float("inf")
pi = _math.pi
e = _math.e
newaxis = None
PINF = float("inf")
NINF = float("-inf")
PZERO = 0.0
NZERO = -0.0

# --- typecodes (mapping of type character codes) ----------------------------
typecodes = {
    "All": "?bhilqBHILQefdgFDGSUVO",
    "AllFloat": "efdgFDG",
    "AllInteger": "bhilqBHILQ",
    "Character": "c",
    "Complex": "FDG",
    "Float": "efdg",
    "Integer": "bhilq",
    "UnsignedInteger": "BHILQ",
}

# --- Scalar type checks (stubs) --------------------------------------------
integer = int
floating = float
complexfloating = complex
number = (int, float, complex)
signedinteger = int
unsignedinteger = int
inexact = float
flexible = (str, bytes)
character = (str, bytes)
generic = object

class iinfo:
    """Stub for integer type info."""
    def __init__(self, dtype):
        self.dtype = dtype
        if dtype in ("int8", int8):
            self.min, self.max, self.bits = -128, 127, 8
        elif dtype in ("int16", int16):
            self.min, self.max, self.bits = -32768, 32767, 16
        elif dtype in ("int32", int32):
            self.min, self.max, self.bits = -2147483648, 2147483647, 32
        elif dtype in ("int64", int64, intp):
            self.min, self.max, self.bits = -9223372036854775808, 9223372036854775807, 64
        else:
            self.min, self.max, self.bits = -9223372036854775808, 9223372036854775807, 64

class finfo:
    """Stub for float type info."""
    def __init__(self, dtype=None):
        self.dtype = dtype or float64
        self.eps = 2.220446049250313e-16
        self.max = 1.7976931348623157e+308
        self.min = -1.7976931348623157e+308
        self.tiny = 2.2250738585072014e-308
        self.resolution = 1e-15

# --- Missing functions (stubs) ----------------------------------------------
def empty(shape, dtype=None, order="C"):
    """Stub: returns zeros instead of uninitialized."""
    return zeros(shape)

def empty_like(a, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    return zeros(s)

def full(shape, fill_value, dtype=None, order="C"):
    a = ones(shape)
    fill_arr = ones(shape) * array([float(fill_value)]) if not isinstance(fill_value, ndarray) else fill_value
    # element-wise multiply: ones * scalar_array
    return a * fill_arr if isinstance(fill_arr, ndarray) else a

def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    return ones(s) * fill_value

def zeros_like(a, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    return zeros(s)

def ones_like(a, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    return ones(s)

def isnan(x):
    """Check for NaN element-wise."""
    if isinstance(x, ndarray):
        # Compare: NaN != NaN
        return array([1.0 if _math.isnan(float(x.flatten()[i])) else 0.0
                       for i in range(x.size)]).reshape(x.shape)
    return _math.isnan(x)

def isfinite(x):
    if isinstance(x, ndarray):
        return array([1.0 if _math.isfinite(float(x.flatten()[i])) else 0.0
                       for i in range(x.size)]).reshape(x.shape)
    return _math.isfinite(x)

def isinf(x):
    if isinstance(x, ndarray):
        return array([1.0 if _math.isinf(float(x.flatten()[i])) else 0.0
                       for i in range(x.size)]).reshape(x.shape)
    return _math.isinf(x)

def isscalar(x):
    return isinstance(x, (int, float, complex, bool))

def asarray(a, dtype=None, order=None):
    if isinstance(a, ndarray):
        return a
    return array(a)

def ascontiguousarray(a, dtype=None):
    return asarray(a)

def copy(a, order="K"):
    if isinstance(a, ndarray):
        return a.copy()
    return array(a)

def clip(a, a_min, a_max, out=None):
    """Clip array values to [a_min, a_max]."""
    if not isinstance(a, ndarray):
        a = array([a])
    result = where(a < a_min, full(a.shape, a_min), a) if a_min is not None else a
    result = where(result > a_max, full(result.shape, a_max), result) if a_max is not None else result
    return result

def abs(x):
    if isinstance(x, ndarray):
        return x.abs()
    return __builtins__["abs"](x) if isinstance(__builtins__, dict) else _math.fabs(x)

absolute = abs

def sqrt(x):
    if isinstance(x, ndarray):
        return x.sqrt()
    return _math.sqrt(x)

def exp(x):
    if isinstance(x, ndarray):
        return x.exp()
    return _math.exp(x)

def log(x):
    if isinstance(x, ndarray):
        return x.log()
    return _math.log(x)

def sin(x):
    if isinstance(x, ndarray):
        return x.sin()
    return _math.sin(x)

def cos(x):
    if isinstance(x, ndarray):
        return x.cos()
    return _math.cos(x)

def tan(x):
    if isinstance(x, ndarray):
        return x.tan()
    return _math.tan(x)

def floor(x):
    if isinstance(x, ndarray):
        return x.floor()
    return _math.floor(x)

def ceil(x):
    if isinstance(x, ndarray):
        return x.ceil()
    return _math.ceil(x)

def around(a, decimals=0, out=None):
    factor = 10 ** decimals
    if isinstance(a, ndarray):
        flat = a.flatten()
        result = array([round(float(flat[i]) * factor) / factor for i in range(flat.size)])
        return result.reshape(a.shape)
    return round(a * factor) / factor

round_ = around

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    try:
        diff = a - b
        abs_diff = diff.abs() if hasattr(diff, 'abs') else array([_math.fabs(float(diff))])
        abs_b = b.abs() if hasattr(b, 'abs') else array([_math.fabs(float(b))])
        threshold = full(abs_diff.shape, atol) + full(abs_b.shape, rtol) * abs_b
        return (abs_diff < threshold).all() if hasattr((abs_diff < threshold), 'all') else abs_diff < threshold
    except Exception:
        return False

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    diff = (a - b).abs() if hasattr(a - b, 'abs') else array([_math.fabs(float(a) - float(b))])
    abs_b = b.abs() if hasattr(b, 'abs') else array([_math.fabs(float(b))])
    threshold = full(diff.shape, atol) + full(abs_b.shape, rtol) * abs_b
    return diff < threshold

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.sum(axis, keepdims)
        return a.sum(None, keepdims)
    return __builtins__["sum"](a) if isinstance(__builtins__, dict) else a

def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        flat = a.flatten()
        result = 1.0
        for i in range(flat.size):
            result *= float(flat[i])
        return result
    return a

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.mean(axis, keepdims)
        return a.mean(None, keepdims)
    return a

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.std(axis, ddof, keepdims)
        return a.std(None, ddof, keepdims)
    return 0.0

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.var(axis, ddof, keepdims)
        return a.var(None, ddof, keepdims)
    return 0.0

def max(a, axis=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.max(axis, keepdims)
        return a.max(None, keepdims)
    return a

amax = max

def min(a, axis=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            return a.min(axis, keepdims)
        return a.min(None, keepdims)
    return a

amin = min

def argmax(a, axis=None, out=None):
    if isinstance(a, ndarray):
        return a.argmax(axis)
    return 0

def argmin(a, axis=None, out=None):
    if isinstance(a, ndarray):
        return a.argmin(axis)
    return 0

def reshape(a, newshape, order="C"):
    return a.reshape(newshape)

def transpose(a, axes=None):
    return a.T

def flatten(a, order="C"):
    return a.flatten()

def ravel(a, order="C"):
    if isinstance(a, ndarray):
        return a.ravel()
    return array(a).ravel()

def squeeze(a, axis=None):
    if isinstance(a, ndarray):
        return a.squeeze(axis)
    return a

def expand_dims(a, axis):
    if isinstance(a, ndarray):
        return a.expand_dims(axis)
    return a

def stack(arrays, axis=0, out=None):
    return concatenate(arrays, axis=axis)

def vstack(tup):
    return concatenate(tup, axis=0)

def hstack(tup):
    return concatenate(tup, axis=1) if tup[0].ndim > 1 else concatenate(tup, axis=0)

row_stack = vstack

def can_cast(from_, to, casting="safe"):
    return True  # stub

def result_type(*arrays_and_dtypes):
    return float64  # stub

def promote_types(type1, type2):
    return float64  # stub

def seterr(**kwargs):
    """Stub for floating point error handling."""
    old = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
    return old

def geterr():
    return {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}

class errstate:
    """Context manager for floating point error handling."""
    def __init__(self, **kwargs):
        self._old = None
    def __enter__(self):
        self._old = geterr()
        return self
    def __exit__(self, *args):
        pass

def set_printoptions(**kwargs):
    pass

def get_printoptions():
    return {}

# --- dtype class (stub) -----------------------------------------------------
class dtype:
    """Stub for numpy dtype objects."""
    def __init__(self, tp=None):
        if isinstance(tp, dtype):
            self.name = tp.name
            self.kind = tp.kind
            self.itemsize = tp.itemsize
            self.char = tp.char
        elif isinstance(tp, str):
            self.name = tp
            self._init_from_name(tp)
        elif tp is float or tp is float64:
            self.name = "float64"
            self.kind = "f"
            self.itemsize = 8
            self.char = "d"
        elif tp is int or tp is int64:
            self.name = "int64"
            self.kind = "i"
            self.itemsize = 8
            self.char = "q"
        elif tp is bool or tp is bool_:
            self.name = "bool"
            self.kind = "b"
            self.itemsize = 1
            self.char = "?"
        else:
            self.name = str(tp) if tp else "float64"
            self.kind = "f"
            self.itemsize = 8
            self.char = "d"
        self.type = type(tp) if tp else float
        self.str = self.name

    def _init_from_name(self, name):
        _info = {
            "float64": ("f", 8, "d"), "float32": ("f", 4, "f"),
            "float16": ("f", 2, "e"), "int64": ("i", 8, "q"),
            "int32": ("i", 4, "l"), "int16": ("i", 2, "h"),
            "int8": ("i", 1, "b"), "uint64": ("u", 8, "Q"),
            "uint32": ("u", 4, "L"), "uint16": ("u", 2, "H"),
            "uint8": ("u", 1, "B"), "bool": ("b", 1, "?"),
            "complex128": ("c", 16, "D"), "complex64": ("c", 8, "F"),
        }
        if name in _info:
            self.kind, self.itemsize, self.char = _info[name]
        else:
            self.kind, self.itemsize, self.char = "f", 8, "d"

    def __repr__(self):
        return f"dtype('{self.name}')"

    def __eq__(self, other):
        if isinstance(other, dtype):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def newbyteorder(self, new_order="S"):
        return dtype(self.name)

# --- More missing stubs for test_numeric.py ---------------------------------
True_ = True
False_ = False
int_ = int64

class broadcast:
    """Stub for np.broadcast."""
    def __init__(self, *args):
        self.shape = args[0].shape if hasattr(args[0], 'shape') else (1,)
        self.nd = len(self.shape)
        self.ndim = self.nd
        self.size = 1
        for s in self.shape:
            self.size *= s

def nonzero(a):
    if isinstance(a, ndarray):
        flat = a.flatten()
        indices = [i for i in range(flat.size) if float(flat[i]) != 0.0]
        return (array([float(i) for i in indices]),)
    return (array([]),)

def count_nonzero(a, axis=None):
    if isinstance(a, ndarray):
        flat = a.flatten()
        return _builtin_sum(1 for i in range(flat.size) if float(flat[i]) != 0.0)
    return 0

# Keep builtin sum reference
_builtin_sum = __builtins__["sum"] if isinstance(__builtins__, dict) else __import__("builtins").sum

def diagonal(a, offset=0, axis1=0, axis2=1):
    """Extract diagonal from 2D array."""
    if a.ndim != 2:
        raise ValueError("diagonal requires 2-d array")
    rows, cols = a.shape
    n = _builtin_min(rows, cols - offset) if offset >= 0 else _builtin_min(rows + offset, cols)
    if n <= 0:
        return array([])
    result = []
    for i in range(n):
        r = i - offset if offset < 0 else i
        c = i + offset if offset >= 0 else i
        result.append(float(a[r, c]))
    return array(result)

_builtin_min = __builtins__["min"] if isinstance(__builtins__, dict) else __import__("builtins").min
_builtin_max = __builtins__["max"] if isinstance(__builtins__, dict) else __import__("builtins").max

def trace(a, offset=0, axis1=0, axis2=1):
    d = diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    return d.sum()

def ptp(a, axis=None):
    if isinstance(a, ndarray):
        return float(a.max()) - float(a.min())
    return 0

def repeat(a, repeats, axis=None):
    flat = a.flatten()
    result = []
    for i in range(flat.size):
        for _ in range(repeats):
            result.append(float(flat[i]))
    return array(result)

def tile(a, reps):
    if isinstance(reps, int):
        reps = (reps,)
    flat = list(float(a.flatten()[i]) for i in range(a.size))
    result = flat * reps[-1]
    return array(result)

def resize(a, new_shape):
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    total = 1
    for s in new_shape:
        total *= s
    if total == 0:
        return zeros(new_shape)
    flat = a.flatten()
    result = []
    for i in range(total):
        result.append(float(flat[i % flat.size]))
    return array(result).reshape(new_shape)

def choose(a, choices, out=None, mode="raise"):
    if isinstance(a, ndarray):
        choice_arrays = [c if isinstance(c, ndarray) else array(c) for c in choices]
        return _native.choose(a, choice_arrays)
    return choices[0]

def compress(condition, a, axis=None):
    if isinstance(a, ndarray):
        cond = condition if isinstance(condition, ndarray) else array(condition)
        return _native.compress(cond, a, axis)
    return a

def searchsorted(a, v, side="left", sorter=None):
    if isinstance(a, ndarray) and isinstance(v, ndarray):
        return _native.searchsorted(a, v, side)
    if isinstance(a, ndarray):
        return _native.searchsorted(a, array([v]), side)
    return 0

def outer(a, b, out=None):
    """Compute outer product."""
    a_flat = a.flatten() if isinstance(a, ndarray) else array([float(a)])
    b_flat = b.flatten() if isinstance(b, ndarray) else array([float(b)])
    result = []
    for i in range(a_flat.size):
        for j in range(b_flat.size):
            result.append(float(a_flat[i]) * float(b_flat[j]))
    return array(result).reshape((a_flat.size, b_flat.size))

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Cross product stub (3D only)."""
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    return array([
        float(a.flatten()[1]) * float(b.flatten()[2]) - float(a.flatten()[2]) * float(b.flatten()[1]),
        float(a.flatten()[2]) * float(b.flatten()[0]) - float(a.flatten()[0]) * float(b.flatten()[2]),
        float(a.flatten()[0]) * float(b.flatten()[1]) - float(a.flatten()[1]) * float(b.flatten()[0]),
    ])

def tensordot(a, b, axes=2):
    return dot(a, b)  # simplified stub

def roll(a, shift, axis=None):
    flat = a.flatten()
    n = flat.size
    if n == 0:
        return a.copy()
    shift = shift % n
    result = []
    for i in range(n):
        result.append(float(flat[(i - shift) % n]))
    return array(result).reshape(a.shape)

def rollaxis(a, axis, start=0):
    return a  # stub

def moveaxis(a, source, destination):
    return a  # stub

def swapaxes(a, axis1, axis2):
    if a.ndim == 2 and axis1 != axis2:
        return a.T
    return a

def indices(dimensions, dtype=int64, sparse=False):
    return zeros(dimensions)  # stub

def fromiter(iterable, dtype, count=-1):
    return array(list(iterable))

def array_equal(a1, a2, equal_nan=False):
    try:
        a1 = asarray(a1) if not isinstance(a1, ndarray) else a1
        a2 = asarray(a2) if not isinstance(a2, ndarray) else a2
        if a1.shape != a2.shape:
            return False
        return (a1 == a2).all() if hasattr(a1 == a2, 'all') else a1 == a2
    except Exception:
        return False

def array_equiv(a1, a2):
    return array_equal(a1, a2)

def require(a, dtype=None, requirements=None):
    return asarray(a)

def binary_repr(num, width=None):
    if num >= 0:
        s = bin(num)[2:]
    else:
        if width is None:
            width = 1
        s = bin(2**width + num)[2:]
    if width is not None:
        s = s.zfill(width)
    return s

def base_repr(number, base=2, padding=0):
    if number == 0:
        return "0" * (padding + 1)
    digits = []
    n = __import__("builtins").abs(number)
    while n:
        digits.append(str(n % base) if n % base < 10 else chr(ord('A') + n % base - 10))
        n //= base
    if number < 0:
        digits.append("-")
    return "0" * padding + "".join(reversed(digits))

def sort(a, axis=-1, kind=None, order=None):
    if isinstance(a, ndarray):
        if axis is not None and axis < 0:
            axis = a.ndim + axis
        return a.sort(axis)
    return array(sorted(a))

def argsort(a, axis=-1, kind=None, order=None):
    if isinstance(a, ndarray):
        if axis is not None and axis < 0:
            axis = a.ndim + axis
        return a.argsort(axis)
    flat = a.flatten()
    vals = [float(flat[i]) for i in range(flat.size)]
    indices = sorted(range(len(vals)), key=lambda i: vals[i])
    return array([float(i) for i in indices])

def size(a, axis=None):
    if isinstance(a, ndarray):
        return a.size
    return 1

def take(a, indices, axis=None, out=None, mode="raise"):
    flat = a.flatten()
    if isinstance(indices, ndarray):
        idx = [int(float(indices.flatten()[i])) for i in range(indices.size)]
    else:
        idx = list(indices) if hasattr(indices, '__iter__') else [indices]
    return array([float(flat[i]) for i in idx])

def all(a, axis=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        return a.all()
    return bool(a)

def any(a, axis=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        return a.any()
    return bool(a)

def may_share_memory(a, b, max_work=None):
    return False  # stub

def shares_memory(a, b, max_work=None):
    return False  # stub

def signbit(x):
    if isinstance(x, ndarray):
        flat = x.flatten()
        return array([1.0 if float(flat[i]) < 0 or (float(flat[i]) == 0 and _math.copysign(1, float(flat[i])) < 0) else 0.0 for i in range(flat.size)]).reshape(x.shape)
    return x < 0

def power(x1, x2):
    if isinstance(x1, ndarray):
        flat1 = x1.flatten()
        if isinstance(x2, ndarray):
            flat2 = x2.flatten()
            return array([float(flat1[i]) ** float(flat2[i]) for i in range(flat1.size)]).reshape(x1.shape)
        return array([float(flat1[i]) ** float(x2) for i in range(flat1.size)]).reshape(x1.shape)
    return x1 ** x2

def add(x1, x2, out=None):
    return asarray(x1) + asarray(x2)

def divide(x1, x2, out=None):
    return asarray(x1) / asarray(x2)

def greater(x1, x2):
    return asarray(x1) > asarray(x2)

def less(x1, x2):
    return asarray(x1) < asarray(x2)

def maximum(x1, x2):
    if isinstance(x1, ndarray) and isinstance(x2, ndarray):
        return where(x1 > x2, x1, x2)
    return x1 if x1 > x2 else x2

def minimum(x1, x2):
    if isinstance(x1, ndarray) and isinstance(x2, ndarray):
        return where(x1 < x2, x1, x2)
    return x1 if x1 < x2 else x2

def logical_and(x1, x2):
    return asarray(x1) * asarray(x2)  # rough

def logical_or(x1, x2):
    return asarray(x1) + asarray(x2)  # rough

def logical_not(x):
    if isinstance(x, ndarray):
        flat = x.flatten()
        return array([0.0 if float(flat[i]) else 1.0 for i in range(flat.size)]).reshape(x.shape)
    return not x

def logical_xor(x1, x2):
    return logical_and(logical_or(x1, x2), logical_not(logical_and(x1, x2)))

def matrix_transpose(a):
    return a.T

def astype(a, dtype):
    return a.astype(dtype)

def real(a):
    """Return the real part of the array elements."""
    if isinstance(a, ndarray):
        return a.real
    return a

def imag(a):
    """Return the imaginary part of the array elements."""
    if isinstance(a, ndarray):
        return a.imag
    return 0

def conj(a):
    """Return the complex conjugate."""
    if isinstance(a, ndarray):
        return a.conj()
    return a

conjugate = conj

def angle(a, deg=False):
    """Return the angle (argument) of complex elements."""
    if isinstance(a, ndarray):
        # angle not exposed as a native method yet; stub
        return a
    return 0

def einsum(*operands, **kwargs):
    if len(operands) < 2:
        raise ValueError("einsum requires at least a subscript string and one operand")
    subscripts = operands[0]
    arrays = operands[1:]
    return _native.einsum(subscripts, *arrays)

# --- String (char) operations -----------------------------------------------
class _char_mod:
    @staticmethod
    def upper(a):
        return _native.char_upper(a)

    @staticmethod
    def lower(a):
        return _native.char_lower(a)

    @staticmethod
    def capitalize(a):
        return _native.char_capitalize(a)

    @staticmethod
    def strip(a):
        return _native.char_strip(a)

    @staticmethod
    def str_len(a):
        return _native.char_str_len(a)

    @staticmethod
    def startswith(a, prefix):
        return _native.char_startswith(a, prefix)

    @staticmethod
    def endswith(a, suffix):
        return _native.char_endswith(a, suffix)

    @staticmethod
    def replace(a, old, new):
        return _native.char_replace(a, old, new)

char = _char_mod()

# --- dtypes module stub -----------------------------------------------------
class _dtypes_mod:
    pass
dtypes = _dtypes_mod()
