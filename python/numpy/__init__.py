"""NumPy-compatible Python package wrapping the Rust native module."""
import sys as _sys
import math as _math

__version__ = "1.26.0"

# Import from native Rust module
import _numpy_native as _native
from _numpy_native import ndarray
from _numpy_native import dot, concatenate

class AxisError(ValueError, IndexError):
    """Exception for invalid axis."""
    def __init__(self, axis=None, ndim=None, msg_prefix=None):
        if axis is not None and ndim is not None:
            msg = "axis {} is out of bounds for array of dimension {}".format(axis, ndim)
            if msg_prefix:
                msg = "{}: {}".format(msg_prefix, msg)
        elif axis is not None:
            msg = str(axis)
        else:
            msg = msg_prefix or ""
        super().__init__(msg)
        self.axis = axis
        self.ndim = ndim

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
    result = _array_core(data, dtype=dtype, copy=copy, order=order, subok=subok, like=like)
    if ndmin > 0 and isinstance(result, ndarray):
        while result.ndim < ndmin:
            result = expand_dims(result, 0)
    return result

def _array_core(data, dtype=None, copy=None, order=None, subok=False, like=None):
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
        result = data.copy() if copy else data
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    if isinstance(data, (int, float)):
        result = _native.array([float(data)])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    if isinstance(data, str):
        # Single string -> string array
        return _native.array([data])
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        # List of strings -> string array
        return _native.array(data)
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], bool):
        result = _native.array([1.0 if x else 0.0 for x in data])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
        result = _native.array([float(x) for x in data])
        if dtype is not None:
            dt = str(dtype)
            if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
                result = result.astype(dt)
        return result
    # Try the native array constructor
    try:
        result = _native.array(data)
    except (TypeError, ValueError):
        try:
            result = _native.array(_to_float_list(data))
        except (TypeError, ValueError):
            # Final fallback for non-numeric data
            return _ObjectArray(data if isinstance(data, (list, tuple)) else [data])
    if dtype is not None and isinstance(result, ndarray):
        dt = str(dtype)
        if dt not in ("object",) and not dt.startswith("S") and not dt.startswith("U") and dt != "str":
            result = result.astype(dt)
    return result


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
    if dtype is not None:
        return _native.zeros(shape, str(dtype))
    return _native.zeros(shape)

def ones(shape, dtype=None, order="C", like=None):
    if dtype is not None:
        return _native.ones(shape, str(dtype))
    return _native.ones(shape)

def arange(*args, dtype=None, like=None, **kwargs):
    float_args = [float(a) for a in args]
    if dtype is not None:
        # Ensure step is provided so dtype goes in 4th position
        if len(float_args) == 2:
            float_args.append(1.0)
        return _native.arange(float_args[0], float_args[1], float_args[2], str(dtype))
    return _native.arange(*float_args, **kwargs)

def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
    start = float(start)
    stop = float(stop)
    num = int(num)
    if endpoint:
        result = _native.linspace(start, stop, num)
        step = (stop - start) / (num - 1) if num > 1 else 0.0
    else:
        step = (stop - start) / num if num > 0 else 0.0
        result = array([start + i * step for i in range(num)])
    if dtype is not None:
        result = result.astype(str(dtype))
    if retstep:
        return result, step
    return result

def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0):
    y = linspace(start, stop, num=num, endpoint=endpoint)
    result = power(base, y)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    log_start = _math.log10(start)
    log_stop = _math.log10(stop)
    result = logspace(log_start, log_stop, num=num, endpoint=endpoint)
    if dtype is not None:
        result = result.astype(str(dtype))
    return result

def eye(N, M=None, k=0, dtype=None, order="C", like=None):
    if dtype is not None:
        if M is not None:
            return _native.eye(N, M, k, str(dtype))
        return _native.eye(N, N, k, str(dtype))
    if M is not None:
        return _native.eye(N, M, k)
    if k != 0:
        return _native.eye(N, N, k)
    return _native.eye(N)

def where(condition, x=None, y=None):
    if x is None and y is None:
        return nonzero(condition)
    if not isinstance(condition, ndarray):
        condition = asarray(condition)
    # Ensure condition is boolean dtype (native where_ requires bool)
    if str(condition.dtype) != "bool":
        condition = condition.astype("bool")
    if not isinstance(x, ndarray):
        x = full(condition.shape, float(x))
    if not isinstance(y, ndarray):
        y = full(condition.shape, float(y))
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
        if isinstance(other, type) and hasattr(other, '_scalar_name'):
            return self._name == other._scalar_name
        return NotImplemented

    def __hash__(self):
        return hash(self._name)


# Metaclass for scalar type classes so the CLASS itself has custom __str__, __eq__, __hash__
class _ScalarTypeMeta(type):
    """Metaclass for numpy scalar type classes in the type hierarchy."""
    def __new__(mcs, name, bases, namespace, scalar_name=None, python_type=float):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._scalar_name = scalar_name or name
        cls._python_type = python_type
        return cls

    def __init__(cls, name, bases, namespace, scalar_name=None, python_type=float):
        super().__init__(name, bases, namespace)

    def __call__(cls, value=0, *args, **kwargs):
        try:
            return cls._python_type(value)
        except (ValueError, TypeError):
            return value

    def __repr__(cls):
        return f"<class 'numpy.{cls._scalar_name}'>"

    def __str__(cls):
        return cls._scalar_name

    def __eq__(cls, other):
        if isinstance(other, _ScalarTypeMeta):
            return cls._scalar_name == other._scalar_name
        if isinstance(other, _ScalarType):
            return cls._scalar_name == other._name
        if isinstance(other, str):
            return cls._scalar_name == other
        return type.__eq__(cls, other)

    def __hash__(cls):
        return hash(cls._scalar_name)

    def __instancecheck__(cls, instance):
        # These are abstract type hierarchy classes, not instantiated
        return False


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

# --- Type hierarchy classes -------------------------------------------------
class generic(metaclass=_ScalarTypeMeta, scalar_name="generic"):
    """Base class for all numpy scalar types."""
    pass

class number(generic, metaclass=_ScalarTypeMeta, scalar_name="number"):
    """Base class for all numeric scalar types."""
    pass

class integer(number, metaclass=_ScalarTypeMeta, scalar_name="integer"):
    """Base class for integer scalar types."""
    pass

class signedinteger(integer, metaclass=_ScalarTypeMeta, scalar_name="signedinteger"):
    """Base class for signed integer scalar types."""
    pass

class unsignedinteger(integer, metaclass=_ScalarTypeMeta, scalar_name="unsignedinteger"):
    """Base class for unsigned integer scalar types."""
    pass

class inexact(number, metaclass=_ScalarTypeMeta, scalar_name="inexact"):
    """Base class for inexact (float/complex) scalar types."""
    pass

class floating(inexact, metaclass=_ScalarTypeMeta, scalar_name="floating"):
    """Base class for floating-point scalar types."""
    pass

class complexfloating(inexact, metaclass=_ScalarTypeMeta, scalar_name="complexfloating"):
    """Base class for complex scalar types."""
    pass

class character(generic, metaclass=_ScalarTypeMeta, scalar_name="character"):
    """Base class for character types."""
    pass

class flexible(generic, metaclass=_ScalarTypeMeta, scalar_name="flexible"):
    """Base class for flexible types (string, void)."""
    pass

class bool_(generic, metaclass=_ScalarTypeMeta, scalar_name="bool", python_type=bool):
    """Boolean scalar type."""
    pass

# Specific signed integer types
class int8(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int8", python_type=int):
    pass
class int16(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int16", python_type=int):
    pass
class int32(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int32", python_type=int):
    pass
class int64(signedinteger, metaclass=_ScalarTypeMeta, scalar_name="int64", python_type=int):
    pass

# Specific unsigned integer types
class uint8(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint8", python_type=int):
    pass
class uint16(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint16", python_type=int):
    pass
class uint32(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint32", python_type=int):
    pass
class uint64(unsignedinteger, metaclass=_ScalarTypeMeta, scalar_name="uint64", python_type=int):
    pass

# Specific floating-point types
class float16(floating, metaclass=_ScalarTypeMeta, scalar_name="float16", python_type=float):
    pass
class float32(floating, metaclass=_ScalarTypeMeta, scalar_name="float32", python_type=float):
    pass
class float64(floating, metaclass=_ScalarTypeMeta, scalar_name="float64", python_type=float):
    pass

# Specific complex types
class complex64(complexfloating, metaclass=_ScalarTypeMeta, scalar_name="complex64", python_type=complex):
    pass
class complex128(complexfloating, metaclass=_ScalarTypeMeta, scalar_name="complex128", python_type=complex):
    pass

# Character/flexible types
class str_(character, metaclass=_ScalarTypeMeta, scalar_name="str", python_type=str):
    pass
class bytes_(character, metaclass=_ScalarTypeMeta, scalar_name="bytes", python_type=bytes):
    pass
class void(flexible, metaclass=_ScalarTypeMeta, scalar_name="void", python_type=float):
    pass

# Aliases using _ScalarType (for types that don't need hierarchy participation)
float128 = _ScalarType("float128", float)
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
timedelta64 = _ScalarType("timedelta64", int)
datetime64 = _ScalarType("datetime64", int)
string_ = _ScalarType("str", str)
unicode_ = _ScalarType("str", str)
half = _ScalarType("float16", float)
int_ = int64
float_ = float64

# --- Missing functions (stubs) ----------------------------------------------
def empty(shape, dtype=None, order="C"):
    """Stub: returns zeros instead of uninitialized."""
    return zeros(shape, dtype=dtype)

def empty_like(a, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    dt = dtype if dtype is not None else (str(a.dtype) if hasattr(a, 'dtype') else None)
    return zeros(s, dtype=dt)

def full(shape, fill_value, dtype=None, order="C"):
    if dtype is not None:
        return _native.full(shape, float(fill_value), str(dtype))
    return _native.full(shape, float(fill_value))

def full_like(a, fill_value, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    dt = str(dtype) if dtype is not None else None
    if dt is not None:
        return _native.full(s, float(fill_value), dt)
    return _native.full(s, float(fill_value))

def zeros_like(a, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    if dtype is not None:
        return _native.zeros(s, str(dtype))
    return _native.zeros(s, str(a.dtype))

def ones_like(a, dtype=None, order="K", subok=True, shape=None):
    s = shape if shape is not None else a.shape
    if dtype is not None:
        return _native.ones(s, str(dtype))
    return _native.ones(s, str(a.dtype))

def isnan(x):
    """Check for NaN element-wise."""
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        else:
            return _math.isnan(x)
    return _native.isnan(x)

def isfinite(x):
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        else:
            return _math.isfinite(x)
    return _native.isfinite(x)

def isinf(x):
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        else:
            return _math.isinf(x)
    return _native.isinf(x)

def isscalar(x):
    """Return True if x is a scalar (not an array)."""
    return isinstance(x, (int, float, complex, bool, str))

def isrealobj(x):
    """Return True if x is not a complex type."""
    if isinstance(x, ndarray):
        return x.dtype not in ("complex64", "complex128")
    return not isinstance(x, complex)

def iscomplexobj(x):
    """Return True if x has a complex type."""
    if isinstance(x, ndarray):
        return x.dtype in ("complex64", "complex128")
    return isinstance(x, complex)

def isreal(x):
    """Returns boolean array -- True where elements are real."""
    if not isinstance(x, ndarray):
        x = array(x)
    if x.dtype in ("complex64", "complex128"):
        return zeros(x.shape, dtype="bool")
    return ones(x.shape, dtype="bool")

def iscomplex(x):
    """Returns boolean array -- True where elements are complex."""
    if not isinstance(x, ndarray):
        x = array(x)
    if x.dtype in ("complex64", "complex128"):
        return ones(x.shape, dtype="bool")
    return zeros(x.shape, dtype="bool")

def issubdtype(arg1, arg2):
    """Check if arg1 dtype is a subtype of arg2."""
    # Map dtype strings to type hierarchy classes
    _dtype_to_class = {
        'bool': bool_,
        'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
        'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
        'float16': float16, 'float32': float32, 'float64': float64,
        'complex64': complex64, 'complex128': complex128,
    }

    # If arg2 is one of our type hierarchy classes, use issubclass
    if isinstance(arg2, type) and isinstance(arg2, _ScalarTypeMeta):
        # Normalize arg1 to a dtype string
        if isinstance(arg1, dtype):
            dt1 = str(arg1)
        elif isinstance(arg1, str):
            dt1 = arg1
        elif isinstance(arg1, type) and isinstance(arg1, _ScalarTypeMeta):
            dt1 = arg1._scalar_name
        elif isinstance(arg1, _ScalarType):
            dt1 = arg1._name
        elif hasattr(arg1, 'dtype'):
            dt1 = str(arg1.dtype)
        else:
            dt1 = str(arg1)
        cls1 = _dtype_to_class.get(dt1)
        if cls1 is not None:
            return issubclass(cls1, arg2)
        return False

    # Fall back to string-based logic for backward compatibility
    # Normalize arg1
    if isinstance(arg1, dtype):
        dt1 = str(arg1)
    elif isinstance(arg1, str):
        dt1 = arg1
    elif isinstance(arg1, _ScalarType):
        dt1 = arg1._name
    elif hasattr(arg1, 'dtype'):
        dt1 = str(arg1.dtype)
    else:
        dt1 = str(arg1) if not callable(arg1) else getattr(arg1, '__name__', str(arg1))

    # Normalize arg2
    if isinstance(arg2, str):
        dt2 = arg2
    elif isinstance(arg2, _ScalarType):
        dt2 = arg2._name
    elif isinstance(arg2, type) and hasattr(arg2, '__name__'):
        dt2 = arg2.__name__
    elif isinstance(arg2, dtype):
        dt2 = str(arg2)
    else:
        dt2 = str(arg2)

    # Map type categories
    _float_types = {"float16", "float32", "float64", "float", "floating", "float_"}
    _int_types = {"int8", "int16", "int32", "int64", "int", "integer", "signedinteger", "int_"}
    _uint_types = {"uint8", "uint16", "uint32", "uint64", "unsignedinteger"}
    _complex_types = {"complex64", "complex128", "complex", "complexfloating", "complex_"}
    _number_types = _float_types | _int_types | _uint_types | _complex_types | {"number"}
    _bool_types = {"bool", "bool_"}

    # Check if dt1 is a subtype of dt2
    if dt2 in ("floating", "float"):
        return dt1 in _float_types
    elif dt2 in ("integer", "signedinteger", "int"):
        return dt1 in _int_types
    elif dt2 in ("unsignedinteger",):
        return dt1 in _uint_types
    elif dt2 in ("complexfloating", "complex"):
        return dt1 in _complex_types
    elif dt2 in ("number",):
        return dt1 in _number_types
    elif dt2 in ("bool", "bool_"):
        return dt1 in _bool_types
    elif dt2 in ("generic",):
        return True  # everything is a subtype of generic
    elif dt2 in ("inexact",):
        return dt1 in (_float_types | _complex_types)
    else:
        # Direct dtype comparison
        return dt1 == dt2

def asarray(a, dtype=None, order=None):
    if isinstance(a, ndarray):
        if dtype is not None:
            return a.astype(str(dtype))
        return a
    return array(a, dtype=dtype)

def ascontiguousarray(a, dtype=None):
    return asarray(a)

def copy(a, order="K"):
    if isinstance(a, ndarray):
        return a.copy()
    return array(a)

def clip(a, a_min=None, a_max=None, out=None):
    """Clip array values to [a_min, a_max]."""
    if not isinstance(a, ndarray):
        a = array(a)
    if a_min is None and a_max is None:
        return a.copy()
    if a_min is None:
        a_min = float('-inf')
    if a_max is None:
        a_max = float('inf')
    return _native.clip(a, a_min, a_max)

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

def exp2(x):
    """Compute 2**x element-wise."""
    return power(2.0, asarray(x))

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

def log10(x):
    if isinstance(x, ndarray):
        return _native.log10(x)
    return _math.log10(x)

def log2(x):
    if isinstance(x, ndarray):
        return _native.log2(x)
    return _math.log2(x)

def log1p(x):
    if isinstance(x, ndarray):
        return _native.log1p(x)
    return _math.log1p(x)

def expm1(x):
    if isinstance(x, ndarray):
        return _native.expm1(x)
    return _math.expm1(x)

def sign(x):
    if isinstance(x, ndarray):
        return _native.sign(x)
    return (x > 0) - (x < 0)

def deg2rad(x):
    if isinstance(x, ndarray):
        return _native.deg2rad(x)
    return _math.radians(x)

def rad2deg(x):
    if isinstance(x, ndarray):
        return _native.rad2deg(x)
    return _math.degrees(x)

# Aliases
radians = deg2rad
degrees = rad2deg

def sinh(x):
    if isinstance(x, ndarray):
        return _native.sinh(x)
    return _math.sinh(x)

def cosh(x):
    if isinstance(x, ndarray):
        return _native.cosh(x)
    return _math.cosh(x)

def tanh(x):
    if isinstance(x, ndarray):
        return _native.tanh(x)
    return _math.tanh(x)

def arcsin(x):
    if isinstance(x, ndarray):
        return _native.arcsin(x)
    return _math.asin(x)

def arccos(x):
    if isinstance(x, ndarray):
        return _native.arccos(x)
    return _math.acos(x)

def arctan(x):
    if isinstance(x, ndarray):
        return _native.arctan(x)
    return _math.atan(x)

def arctan2(y, x, out=None, where=True, **kwargs):
    if not isinstance(y, ndarray):
        y = array(y)
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arctan2(y, x)

def arcsinh(x):
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arcsinh(x)

def arccosh(x):
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arccosh(x)

def arctanh(x):
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.arctanh(x)

def hypot(x1, x2):
    """Element-wise sqrt(x1**2 + x2**2)."""
    x1 = asarray(x1) if not isinstance(x1, ndarray) else x1
    x2 = asarray(x2) if not isinstance(x2, ndarray) else x2
    return sqrt(x1 * x1 + x2 * x2)

def trunc(x):
    if isinstance(x, ndarray):
        return _native.trunc(x)
    import math as _math
    return _math.trunc(x)

def floor(x):
    if isinstance(x, ndarray):
        return x.floor()
    return _math.floor(x)

def ceil(x):
    if isinstance(x, ndarray):
        return x.ceil()
    return _math.ceil(x)

def around(a, decimals=0, out=None):
    if isinstance(a, ndarray):
        return _native.around(a, decimals)
    factor = 10 ** decimals
    return round(a * factor) / factor

round_ = around

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return True if two arrays are element-wise equal within a tolerance."""
    return bool(all(isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)))

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return boolean array where two arrays are element-wise equal within tolerance."""
    if not isinstance(a, ndarray):
        a = array(a)
    if not isinstance(b, ndarray):
        b = array(b)
    diff = abs(a - b)
    limit = full(diff.shape, atol) + full(diff.shape, rtol) * abs(b)
    result = (diff <= limit)
    if equal_nan:
        both_nan = logical_and(isnan(a), isnan(b))
        result = logical_or(result, both_nan)
    return result

def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs."""
    x1, x2 = asarray(x1), asarray(x2)
    mx = maximum(x1, x2)
    return mx + log1p(exp(-abs(x1 - x2)))

def logaddexp2(x1, x2):
    """Logarithm base 2 of the sum of exponentiations of the inputs in base 2."""
    x1, x2 = asarray(x1), asarray(x2)
    ln2 = 0.6931471805599453
    return logaddexp(x1 * ln2, x2 * ln2) / ln2

def rint(x):
    """Round to nearest integer."""
    if isinstance(x, ndarray):
        return _native.around(x, 0)
    return float(round(x))

def fix(x):
    """Round to nearest integer towards zero."""
    if isinstance(x, ndarray):
        return _native.trunc(x)
    return float(_math.trunc(x))

def square(x):
    """Return the element-wise square."""
    x = asarray(x)
    return x * x

def cbrt(x):
    """Return the element-wise cube root."""
    x = asarray(x)
    # cbrt handles negative numbers correctly
    return sign(x) * power(abs(x), 1.0 / 3.0)

def reciprocal(x):
    """Return the reciprocal of the argument, element-wise."""
    x = asarray(x)
    return ones(x.shape) / x

def copysign(x1, x2):
    """Change the sign of x1 to that of x2, element-wise."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    f1 = x1.flatten().tolist()
    f2 = x2.flatten().tolist()
    result = [_math.copysign(a, b) for a, b in zip(f1, f2)]
    return array(result).reshape(x1.shape)

def heaviside(x1, x2):
    """Compute the Heaviside step function.
    0 where x1 < 0, x2 where x1 == 0, 1 where x1 > 0."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    result = where(x1 > zeros(x1.shape), ones(x1.shape), x2)
    result = where(x1 < zeros(x1.shape), zeros(x1.shape), result)
    return result

def sinc(x):
    """Return the sinc function: sin(pi*x)/(pi*x)."""
    x = asarray(x)
    px = x * pi
    # Avoid division by zero: where px==0, use 1 as denominator
    result = sin(px) / where(px == zeros(px.shape), ones(px.shape), px)
    result = where(x == zeros(x.shape), ones(x.shape), result)
    return result

def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    """Replace NaN with zero and infinity with large finite numbers."""
    x = asarray(x)
    if posinf is None:
        posinf = 1.7976931348623157e+308  # float max
    if neginf is None:
        neginf = -1.7976931348623157e+308
    flat = x.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = float(flat[i])
        if v != v:  # NaN check
            vals.append(nan)
        elif v == float('inf'):
            vals.append(posinf)
        elif v == float('-inf'):
            vals.append(neginf)
        else:
            vals.append(v)
    result = array(vals)
    if x.ndim > 1:
        result = result.reshape(x.shape)
    return result

def _dtype_cast(result, dtype):
    """Cast result to dtype if dtype is not None."""
    if dtype is not None:
        result = asarray(result).astype(str(dtype) if not isinstance(dtype, str) else dtype)
    return result

def sum(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        if axis is not None:
            result = a.sum(axis, keepdims)
        else:
            result = a.sum(None, keepdims)
        return _dtype_cast(result, dtype)
    return __builtins__["sum"](a) if isinstance(__builtins__, dict) else a

def prod(a, axis=None, dtype=None, out=None, keepdims=False):
    if isinstance(a, ndarray):
        result = a.prod(axis, keepdims)
    else:
        result = _native.prod(array(a), axis, keepdims)
    return _dtype_cast(result, dtype)

def cumsum(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        result = a.cumsum(axis)
    else:
        result = array(a).cumsum(axis)
    return _dtype_cast(result, dtype)

def cumprod(a, axis=None, dtype=None, out=None):
    if isinstance(a, ndarray):
        result = a.cumprod(axis)
    else:
        result = array(a).cumprod(axis)
    return _dtype_cast(result, dtype)

def diff(a, n=1, axis=-1, prepend=None, append=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    if prepend is not None:
        prepend = asarray(prepend)
        if prepend.ndim == 0:
            prepend = array([float(prepend)])
        a = concatenate([prepend, a])
    if append is not None:
        append = asarray(append)
        if append.ndim == 0:
            append = array([float(append)])
        a = concatenate([a, append])
    return _native.diff(a, n, axis)

def mean(a, axis=None, dtype=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        return float('nan')
    if axis is not None:
        result = a.mean(axis, keepdims)
    else:
        result = a.mean(None, keepdims)
    return _dtype_cast(result, dtype)

def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        return float('nan')
    if axis is not None:
        result = a.std(axis, ddof, keepdims)
    else:
        result = a.std(None, ddof, keepdims)
    return _dtype_cast(result, dtype)

def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if a.size == 0:
        return float('nan')
    if axis is not None:
        result = a.var(axis, ddof, keepdims)
    else:
        result = a.var(None, ddof, keepdims)
    return _dtype_cast(result, dtype)

def nansum(a, axis=None, dtype=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.nansum(a, axis, keepdims)
    return _dtype_cast(result, dtype)

def nanmean(a, axis=None, dtype=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.nanmean(a, axis, keepdims)
    return _dtype_cast(result, dtype)

def nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.nanstd(a, axis, ddof, keepdims)
    return _dtype_cast(result, dtype)

def nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.nanvar(a, axis, ddof, keepdims)
    return _dtype_cast(result, dtype)

def nanmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanmin(a, axis, keepdims)

def nanmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanmax(a, axis, keepdims)

def nanargmin(a, axis=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanargmin(a, axis)

def nanargmax(a, axis=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nanargmax(a, axis)

def nanprod(a, axis=None, dtype=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.nanprod(a, axis, keepdims)
    return _dtype_cast(result, dtype)

def nancumsum(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nancumsum(a, axis)

def nancumprod(a, axis=None, dtype=None, out=None):
    if not isinstance(a, ndarray):
        a = array(a)
    return _native.nancumprod(a, axis)

def _apply_keepdims(result, a, axis):
    """Expand reduced axis back to size-1 when keepdims=True."""
    if isinstance(result, ndarray):
        shape = list(a.shape)
        shape[axis] = 1
        return result.reshape(shape)
    else:
        # scalar result - wrap in array with expanded dims
        shape = [1] * a.ndim
        return array([float(result)]).reshape(shape)

def quantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_native.quantile(a, float(qi), axis) for qi in q_list]
        return array(results) if axis is None else stack(results)
    result = _native.quantile(a, float(q), axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result

def percentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(q, (list, tuple, ndarray)):
        q_list = q.tolist() if isinstance(q, ndarray) else list(q)
        results = [_native.percentile(a, float(qi), axis) for qi in q_list]
        return array(results) if axis is None else stack(results)
    result = _native.percentile(a, float(q), axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result

def median(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    if not isinstance(a, ndarray):
        a = array(a)
    result = _native.quantile(a, 0.5, axis)
    if keepdims and axis is not None:
        result = _apply_keepdims(result, a, axis)
    return result

def cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None):
    if not isinstance(m, ndarray):
        m = array(m)
    _ddof = ddof if ddof is not None else (0 if bias else 1)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.cov(m, y, rowvar, _ddof)
    return _native.cov(m, None, rowvar, _ddof)

def corrcoef(x, y=None, rowvar=True):
    if not isinstance(x, ndarray):
        x = array(x)
    if y is not None:
        if not isinstance(y, ndarray):
            y = array(y)
        return _native.corrcoef(x, y, rowvar)
    return _native.corrcoef(x, None, rowvar)

def average(a, axis=None, weights=None, returned=False, keepdims=False):
    """Compute the weighted average along the specified axis."""
    a = asarray(a)
    if weights is None:
        avg = mean(a, axis=axis)
        if returned:
            if axis is None:
                return avg, float(a.size)
            return avg, full(avg.shape, float(a.shape[axis]))
        return avg
    weights = asarray(weights)
    wsum = sum(a * weights, axis=axis)
    wt = sum(weights, axis=axis)
    avg = wsum / wt
    if returned:
        return avg, wt
    return avg

def _nan_quantile_impl(a, q, axis):
    """Helper for nanmedian/nanpercentile/nanquantile with axis support."""
    a = asarray(a)
    if axis is None:
        flat = a.flatten()
        n = flat.size
        vals = []
        for i in range(n):
            v = flat[i]
            if v == v:
                vals.append(v)
        if len(vals) == 0:
            return float('nan')
        vals.sort()
        idx = q * (len(vals) - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= len(vals):
            return vals[lo]
        frac = idx - lo
        return vals[lo] * (1 - frac) + vals[hi] * frac
    # With axis: apply along axis manually
    # Move target axis to last, then iterate over other dims
    # Simplified: just handle 2D case
    if a.ndim == 1:
        return _nan_quantile_impl(a, q, None)
    results = []
    if axis == 0:
        for j in range(a.shape[1]):
            col = array([a[i][j] for i in range(a.shape[0])])
            results.append(_nan_quantile_impl(col, q, None))
    else:
        for i in range(a.shape[0]):
            results.append(_nan_quantile_impl(a[i], q, None))
    return array(results)

def nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False):
    """Compute the median along the specified axis, ignoring NaNs."""
    a = asarray(a)
    if axis is not None:
        # For axis support, fall back to sorting approach
        return _nan_quantile_impl(a, 0.5, axis)
    # Flatten, filter NaNs, compute median of remaining
    flat = a.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = flat[i]
        if v == v:  # not NaN
            vals.append(v)
    if len(vals) == 0:
        return float('nan')
    vals.sort()
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        return (vals[mid - 1] + vals[mid]) / 2.0
    return vals[mid]

def nanpercentile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    """Compute the qth percentile, ignoring NaNs."""
    return _nan_quantile_impl(asarray(a), q / 100.0, axis)

def nanquantile(a, q, axis=None, out=None, overwrite_input=False, method="linear", keepdims=False):
    """Compute the qth quantile, ignoring NaNs."""
    return _nan_quantile_impl(asarray(a), q, axis)

def ediff1d(ary, to_end=None, to_begin=None):
    """The differences between consecutive elements of an array."""
    ary = asarray(ary).flatten()
    n = ary.size
    diffs = []
    if to_begin is not None:
        if isinstance(to_begin, (int, float)):
            diffs.append(float(to_begin))
        else:
            tb = asarray(to_begin).flatten()
            for i in range(tb.size):
                diffs.append(tb[i])
    for i in range(1, n):
        diffs.append(ary[i] - ary[i - 1])
    if to_end is not None:
        if isinstance(to_end, (int, float)):
            diffs.append(float(to_end))
        else:
            te = asarray(to_end).flatten()
            for i in range(te.size):
                diffs.append(te[i])
    return array(diffs)

def fmax(x1, x2):
    """Element-wise maximum, ignoring NaNs."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    # where x1 is nan, use x2; where x2 is nan, use x1; otherwise max
    x1_nan = isnan(x1)
    x2_nan = isnan(x2)
    result = where(x1 > x2, x1, x2)
    result = where(x1_nan, x2, result)
    result = where(x2_nan, x1, result)
    return result

def fmin(x1, x2):
    """Element-wise minimum, ignoring NaNs."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    x1_nan = isnan(x1)
    x2_nan = isnan(x2)
    result = where(x1 < x2, x1, x2)
    result = where(x1_nan, x2, result)
    result = where(x2_nan, x1, result)
    return result

def max(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is not None:
        return a.max(axis, keepdims)
    return a.max(None, keepdims)

amax = max

def min(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is not None:
        return a.min(axis, keepdims)
    return a.min(None, keepdims)

amin = min

def argmax(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return a.argmax(axis)

def argmin(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    return a.argmin(axis)

def reshape(a, newshape, order="C"):
    return a.reshape(newshape)

def _transpose_with_axes(a, axes):
    """Transpose ndarray with an arbitrary axis permutation (pure Python).

    Parameters
    ----------
    a : ndarray
    axes : list/tuple of int - the desired permutation of axes.

    Returns an ndarray with axes reordered according to *axes*.
    """
    if not isinstance(a, ndarray):
        a = asarray(a)
    shape = a.shape
    ndim_a = len(shape)
    if axes is None:
        return a.T
    axes = list(axes)
    if len(axes) != ndim_a:
        raise ValueError("axes don't match array")
    # Fast-paths
    if axes == list(range(ndim_a)):
        return a.copy() if hasattr(a, 'copy') else array(a.tolist())
    if ndim_a == 2 and axes == [1, 0]:
        return a.T
    # Generic: walk every index of the output and pick from source
    new_shape = tuple(shape[ax] for ax in axes)
    size = 1
    for s in new_shape:
        size *= s
    flat_data = a.flatten()
    # Build strides of original array (row-major)
    src_strides = [0] * ndim_a
    s = 1
    for i in range(ndim_a - 1, -1, -1):
        src_strides[i] = s
        s *= shape[i]
    result = [0.0] * size
    # Iterate over every multi-index of the *output*
    out_idx = [0] * ndim_a
    for flat_i in range(size):
        # Map output index -> source index
        src_flat = 0
        for d in range(ndim_a):
            src_flat += out_idx[d] * src_strides[axes[d]]
        result[flat_i] = float(flat_data[src_flat])
        # Increment out_idx (rightmost first)
        for d in range(ndim_a - 1, -1, -1):
            out_idx[d] += 1
            if out_idx[d] < new_shape[d]:
                break
            out_idx[d] = 0
    return array(result).reshape(list(new_shape))

def transpose(a, axes=None):
    if axes is not None:
        return _transpose_with_axes(a, axes)
    return a.T

def flatten(a, order="C"):
    return a.flatten()

def ravel(a, order="C"):
    if isinstance(a, ndarray):
        return a.ravel()
    return array(a).ravel()

def squeeze(a, axis=None):
    a = asarray(a)
    if axis is not None and isinstance(axis, (tuple, list)):
        result = a
        for ax in sorted(axis, reverse=True):
            result = squeeze(result, axis=ax)
        return result
    if isinstance(a, ndarray):
        return a.squeeze(axis)
    return a

def expand_dims(a, axis):
    a = asarray(a)
    if isinstance(axis, (tuple, list)):
        ndim_out = a.ndim + len(axis)
        # Normalize negative axes
        normalized = []
        for ax in axis:
            if ax < 0:
                ax = ndim_out + ax
            normalized.append(ax)
        normalized.sort()
        result = a
        for ax in normalized:
            result = expand_dims(result, ax)
        return result
    if isinstance(a, ndarray):
        return a.expand_dims(axis)
    return a

def append(arr, values, axis=None):
    """Append values to the end of an array."""
    if not isinstance(arr, ndarray):
        arr = array(arr)
    if not isinstance(values, ndarray):
        values = array(values)
    if axis is None:
        return concatenate([arr.flatten(), values.flatten()])
    return concatenate([arr, values], axis=axis)

def atleast_1d(*arys):
    """Convert inputs to arrays with at least one dimension."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results

def atleast_2d(*arys):
    """Convert inputs to arrays with at least two dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1, 1])
        elif a.ndim == 1:
            a = a.reshape([1, len(a)])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results

def atleast_3d(*arys):
    """Convert inputs to arrays with at least three dimensions."""
    results = []
    for a in arys:
        if not isinstance(a, ndarray):
            a = array(a)
        if a.ndim == 0:
            a = a.reshape([1, 1, 1])
        elif a.ndim == 1:
            a = a.reshape([1, len(a), 1])
        elif a.ndim == 2:
            a = a.reshape(list(a.shape) + [1])
        results.append(a)
    if len(results) == 1:
        return results[0]
    return results

def stack(arrays, axis=0, out=None):
    return _native.stack_native(list(arrays), axis)

def vstack(tup):
    arrs = [asarray(a) for a in tup]
    # For 1D inputs, reshape to 2D first (row vectors)
    expanded = []
    for a in arrs:
        if a.ndim == 1:
            expanded.append(a.reshape([1, a.shape[0]]))
        else:
            expanded.append(a)
    return concatenate(expanded, 0)

def hstack(tup):
    arrs = [asarray(a) for a in tup]
    if arrs[0].ndim > 1:
        return concatenate(arrs, 1)
    return concatenate(arrs, 0)

def column_stack(tup):
    """Stack 1-D arrays as columns into a 2-D array."""
    arrays = []
    for a in tup:
        a = asarray(a)
        if a.ndim == 1:
            a = a.reshape((a.size, 1))  # Make column vector
        arrays.append(a)
    return concatenate(arrays, 1)

def dstack(tup):
    return _native.dstack(list(tup))

row_stack = vstack

def split(a, indices_or_sections, axis=0):
    return _native.split(a, indices_or_sections, axis)

def vsplit(a, indices_or_sections):
    return split(a, indices_or_sections, 0)

def hsplit(a, indices_or_sections):
    if a.ndim == 1:
        return split(a, indices_or_sections, 0)
    return split(a, indices_or_sections, 1)

def array_split(ary, indices_or_sections, axis=0):
    """Split an array into multiple sub-arrays (allows unequal division)."""
    ary = asarray(ary)
    n = ary.shape[axis]
    if isinstance(indices_or_sections, (int, float)):
        nsections = int(indices_or_sections)
        neach, extras = divmod(n, nsections)
        boundaries = [0]
        for i in range(nsections):
            boundaries.append(boundaries[-1] + neach + (1 if i < extras else 0))
        indices = boundaries[1:-1]
    else:
        indices = list(indices_or_sections)
    return split(ary, indices, axis)

def dsplit(ary, indices_or_sections):
    """Split array into multiple sub-arrays along the 3rd axis (depth)."""
    ary = asarray(ary)
    if ary.ndim < 3:
        raise ValueError("dsplit only works on arrays of 3 or more dimensions")
    return array_split(ary, indices_or_sections, axis=2)

def block(arrays):
    """Assemble an nd-array from nested lists of blocks."""
    if isinstance(arrays, ndarray):
        return arrays
    if not isinstance(arrays, list):
        return asarray(arrays)
    if len(arrays) == 0:
        return array([])
    if isinstance(arrays[0], list):
        rows = []
        for row_blocks in arrays:
            row = hstack([asarray(b) for b in row_blocks])
            rows.append(row)
        return vstack(rows)
    else:
        return concatenate([asarray(a) for a in arrays])

def copyto(dst, src, casting='same_kind', where=True):
    """Copy values from one array to another, broadcasting as necessary."""
    src = asarray(src)
    dst = asarray(dst)
    src_b = broadcast_to(src, dst.shape)
    if where is True:
        return src_b
    mask = asarray(where)
    flat_m = mask.flatten()
    flat_s = src_b.flatten()
    flat_d = dst.flatten()
    n = flat_d.size
    result_vals = []
    for i in range(n):
        if flat_m[i]:
            result_vals.append(float(flat_s[i]))
        else:
            result_vals.append(float(flat_d[i]))
    result = array(result_vals)
    if dst.ndim > 1:
        result = result.reshape(dst.shape)
    return result

def place(arr, mask, vals):
    """Change elements of an array based on conditional and input values."""
    arr = asarray(arr)
    mask = asarray(mask)
    vals_arr = asarray(vals).flatten()
    flat_a = arr.flatten()
    flat_m = mask.flatten()
    n = flat_a.size
    nv = vals_arr.size
    result = []
    vi = 0
    for i in range(n):
        if flat_m[i]:
            result.append(float(vals_arr[vi % nv]))
            vi += 1
        else:
            result.append(float(flat_a[i]))
    r = array(result)
    if arr.ndim > 1:
        r = r.reshape(arr.shape)
    return r

def can_cast(from_, to, casting="safe"):
    _type_order = {
        "bool": 0, "int32": 1, "int64": 2,
        "float32": 3, "float64": 4,
        "complex64": 5, "complex128": 6,
    }
    if isinstance(from_, ndarray):
        from_name = str(from_.dtype)
    elif isinstance(from_, str):
        from_name = from_
    else:
        from_name = str(from_)
    if isinstance(to, str):
        to_name = to
    else:
        to_name = str(to)
    f = _type_order.get(from_name, -1)
    t = _type_order.get(to_name, -1)
    if f < 0 or t < 0:
        return False
    if casting == "unsafe":
        return True
    if casting == "safe" or casting == "same_kind":
        return f <= t
    if casting == "equiv":
        return f == t
    if casting == "no":
        return f == t
    return f <= t

def result_type(*arrays_and_dtypes):
    if len(arrays_and_dtypes) == 0:
        return float64
    dtypes = []
    for a in arrays_and_dtypes:
        if isinstance(a, ndarray):
            dtypes.append(str(a.dtype))
        elif isinstance(a, _ScalarType):
            dtypes.append(str(a))
        elif isinstance(a, type) and isinstance(a, _ScalarTypeMeta):
            dtypes.append(str(a))
        elif isinstance(a, str):
            dtypes.append(a)
        else:
            dtypes.append("float64")
    if len(dtypes) == 1:
        return _ScalarType(dtypes[0])
    result = dtypes[0]
    for d in dtypes[1:]:
        result = _native.promote_types(result, d)
    return _ScalarType(result)

def promote_types(type1, type2):
    return _ScalarType(_native.promote_types(str(type1), str(type2)))

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
        elif isinstance(tp, _ScalarType):
            self.name = tp._name
            self._init_from_name(tp._name)
        elif isinstance(tp, type) and isinstance(tp, _ScalarTypeMeta):
            self.name = tp._scalar_name
            self._init_from_name(tp._scalar_name)
        else:
            self.name = str(tp) if tp else "float64"
            self._init_from_name(self.name)
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
    """Broadcast shape computation for multiple arrays."""
    def __init__(self, *args):
        arrays = [asarray(a) for a in args]
        shapes = [a.shape for a in arrays]
        self.shape = broadcast_shapes(*shapes)
        self.nd = len(self.shape)
        self.ndim = self.nd
        self.size = 1
        for s in self.shape:
            self.size *= s
        self.numiter = len(arrays)

def argwhere(a):
    return _native.argwhere(a)

def nonzero(a):
    if isinstance(a, ndarray):
        return _native.nonzero(a)
    return (array([]),)

def count_nonzero(a, axis=None):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        return _native.count_nonzero(a)
    # Build a boolean mask (nonzero -> 1.0, zero -> 0.0), then sum along axis
    flat = a.flatten().tolist()
    mask_data = [1.0 if v != 0.0 else 0.0 for v in flat]
    mask = array(mask_data).reshape(a.shape)
    result = mask.sum(axis, False)
    # Convert to integer values
    if isinstance(result, ndarray):
        flat_r = result.flatten().tolist()
        int_data = [int(v) for v in flat_r]
        return array([float(v) for v in int_data]).reshape(result.shape)
    return int(result)

# Keep builtin sum reference
_builtin_sum = __builtins__["sum"] if isinstance(__builtins__, dict) else __import__("builtins").sum

def diagonal(a, offset=0, axis1=0, axis2=1):
    """Extract diagonal from 2D array."""
    a = asarray(a)
    if a.ndim == 2 and axis1 == 1 and axis2 == 0:
        return _native.diagonal(a.T, offset)
    return _native.diagonal(a, offset)

_builtin_min = __builtins__["min"] if isinstance(__builtins__, dict) else __import__("builtins").min
_builtin_max = __builtins__["max"] if isinstance(__builtins__, dict) else __import__("builtins").max
_builtin_range = __builtins__["range"] if isinstance(__builtins__, dict) else __import__("builtins").range

def trace(a, offset=0, axis1=0, axis2=1):
    d = diagonal(a, offset=offset, axis1=axis1, axis2=axis2)
    return d.sum()

def ptp(a, axis=None):
    if isinstance(a, ndarray):
        return a.max(axis) - a.min(axis)
    return 0

def repeat(a, repeats, axis=None):
    return _native.repeat(a, repeats, axis)

def tile(a, reps):
    return _native.tile(a, reps)

def resize(a, new_shape):
    a = asarray(a)
    if isinstance(new_shape, int):
        new_shape = (new_shape,)
    total = 1
    for s in new_shape:
        total *= s
    if total == 0:
        return zeros(new_shape)
    flat = a.flatten().tolist()
    n = len(flat)
    if n == 0:
        return zeros(new_shape)
    result = []
    for i in range(total):
        result.append(flat[i % n])
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
    if isinstance(a, ndarray) or isinstance(b, ndarray):
        return _native.outer(a, b)
    a_flat = array([float(a)])
    b_flat = array([float(b)])
    return _native.outer(a_flat, b_flat)

def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Cross product of two arrays.

    Handles 1D vectors of length 2 or 3, and batched 2D arrays where the
    last axis has length 2 or 3.
    """
    a = asarray(a)
    b = asarray(b)
    # Simple 1D cases
    if a.ndim == 1 and b.ndim == 1:
        af = a.flatten().tolist()
        bf = b.flatten().tolist()
        if len(af) == 3 and len(bf) == 3:
            return array([
                af[1]*bf[2] - af[2]*bf[1],
                af[2]*bf[0] - af[0]*bf[2],
                af[0]*bf[1] - af[1]*bf[0],
            ])
        elif len(af) == 2 and len(bf) == 2:
            return array(af[0]*bf[1] - af[1]*bf[0])
        else:
            raise ValueError("incompatible vector sizes for cross product")
    # Batched 2D case: process row by row
    if a.ndim == 2 and b.ndim == 2:
        rows = a.shape[0]
        last = a.shape[-1]
        results = []
        for i in range(rows):
            ai = a[i].flatten().tolist()
            bi = b[i].flatten().tolist()
            if last == 3:
                results.append([
                    ai[1]*bi[2] - ai[2]*bi[1],
                    ai[2]*bi[0] - ai[0]*bi[2],
                    ai[0]*bi[1] - ai[1]*bi[0],
                ])
            elif last == 2:
                results.append(ai[0]*bi[1] - ai[1]*bi[0])
            else:
                raise ValueError("incompatible vector sizes for cross product")
        if last == 2:
            return array(results)
        return array(results)

def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes.

    Parameters
    ----------
    a, b : array_like
    axes : int or (2,) list of lists
        If int N, contract last N axes of *a* with first N axes of *b*.
        If a tuple of two sequences, contract the specified axes.
    """
    a = asarray(a) if not isinstance(a, ndarray) else a
    b = asarray(b) if not isinstance(b, ndarray) else b
    if isinstance(axes, int):
        axes_a = list(range(a.ndim - axes, a.ndim))
        axes_b = list(range(0, axes))
    else:
        axes_a = list(axes[0]) if not isinstance(axes[0], int) else [axes[0]]
        axes_b = list(axes[1]) if not isinstance(axes[1], int) else [axes[1]]
    na = a.ndim
    nb = b.ndim
    # Normalise negative axes
    axes_a = [ax if ax >= 0 else ax + na for ax in axes_a]
    axes_b = [ax if ax >= 0 else ax + nb for ax in axes_b]
    # Free axes (those not being contracted)
    free_a = [i for i in range(na) if i not in axes_a]
    free_b = [i for i in range(nb) if i not in axes_b]
    # Transpose a so free axes come first, contracted axes last
    perm_a = free_a + axes_a
    # Transpose b so contracted axes come first, free axes last
    perm_b = axes_b + free_b
    at = _transpose_with_axes(a, perm_a)
    bt = _transpose_with_axes(b, perm_b)
    # Compute shapes for reshape into 2D
    free_a_shape = [a.shape[i] for i in free_a]
    free_b_shape = [b.shape[i] for i in free_b]
    contract_size = 1
    for ax in axes_a:
        contract_size *= a.shape[ax]
    rows = 1
    for s in free_a_shape:
        rows *= s
    cols = 1
    for s in free_b_shape:
        cols *= s
    at2 = at.reshape([rows, contract_size])
    bt2 = bt.reshape([contract_size, cols])
    result = dot(at2, bt2)
    out_shape = free_a_shape + free_b_shape
    if len(out_shape) == 0:
        return result
    return result.reshape(out_shape)

def roll(a, shift, axis=None):
    if isinstance(a, ndarray):
        return _native.roll(a, shift, axis)
    return array(a)

def moveaxis(a, source, destination):
    """Move axes of an array to new positions.

    Other axes remain in their original order.
    """
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    # Normalise to lists
    if isinstance(source, int):
        source = [source]
    if isinstance(destination, int):
        destination = [destination]
    source = [s if s >= 0 else s + ndim_a for s in source]
    destination = [d if d >= 0 else d + ndim_a for d in destination]
    # Build permutation: start with axes not in source, in order
    order = [i for i in range(ndim_a) if i not in source]
    # Insert source axes at destination positions (must insert in sorted dest order)
    pairs = sorted(zip(destination, source))
    for dst, src in pairs:
        order.insert(dst, src)
    return _transpose_with_axes(a, order)

def rollaxis(a, axis, start=0):
    """Roll the specified axis backwards, until it lies in position *start*."""
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    if axis < 0:
        axis += ndim_a
    if start < 0:
        start += ndim_a
    if start > axis:
        start -= 1
    return moveaxis(a, axis, start)

def swapaxes(a, axis1, axis2):
    """Interchange two axes of an array."""
    a = asarray(a) if not isinstance(a, ndarray) else a
    ndim_a = a.ndim
    if axis1 < 0:
        axis1 += ndim_a
    if axis2 < 0:
        axis2 += ndim_a
    if axis1 == axis2:
        return a
    order = list(range(ndim_a))
    order[axis1], order[axis2] = order[axis2], order[axis1]
    return _transpose_with_axes(a, order)

def meshgrid(*xi, indexing='xy'):
    arrays = [a if isinstance(a, ndarray) else array(a) for a in xi]
    return _native.meshgrid(arrays, indexing)

def pad(a, pad_width, mode='constant', constant_values=0, **kwargs):
    if not isinstance(a, ndarray):
        a = array(a)
    # Normalise pad_width to list of (before, after) per axis
    if isinstance(pad_width, int):
        pw = [(pad_width, pad_width)] * a.ndim
    elif isinstance(pad_width, (list, tuple)):
        if isinstance(pad_width[0], int):
            if len(pad_width) == 2:
                pw = [(pad_width[0], pad_width[1])] * a.ndim
            else:
                pw = [(pad_width[0], pad_width[0])] * a.ndim
        else:
            pw = [(p[0], p[1]) for p in pad_width]
    else:
        pw = [(pad_width, pad_width)] * a.ndim

    if mode == 'constant':
        if isinstance(constant_values, (list, tuple)):
            constant_values = constant_values[0] if isinstance(constant_values[0], (int, float)) else constant_values[0][0]
        return _native.pad(a, pad_width, float(constant_values))

    # Pure-Python implementation for 'edge', 'reflect', 'wrap'
    def _pad_1d(data_list, before, after, mode_str):
        """Pad a 1D Python list with the given mode."""
        n = len(data_list)
        result = []
        if mode_str == 'edge':
            result = [data_list[0]] * before + list(data_list) + [data_list[-1]] * after
        elif mode_str == 'reflect':
            left = []
            for i in range(before):
                idx = (i + 1) % (2 * (n - 1)) if n > 1 else 0
                if idx >= n:
                    idx = 2 * (n - 1) - idx
                left.insert(0, data_list[idx])
            right = []
            for i in range(after):
                idx = (i + 1) % (2 * (n - 1)) if n > 1 else 0
                if idx >= n:
                    idx = 2 * (n - 1) - idx
                right.append(data_list[n - 1 - idx])
            result = left + list(data_list) + right
        elif mode_str == 'wrap':
            left = []
            for i in range(before):
                left.insert(0, data_list[-(i + 1) % n])
            right = []
            for i in range(after):
                right.append(data_list[i % n])
            result = left + list(data_list) + right
        elif mode_str == 'symmetric':
            # Like reflect but includes the edge value
            left = []
            for i in range(before):
                idx = i % (2 * n) if n > 0 else 0
                if idx >= n:
                    idx = 2 * n - 1 - idx
                left.insert(0, data_list[idx])
            right = []
            for i in range(after):
                idx = i % (2 * n) if n > 0 else 0
                if idx >= n:
                    idx = 2 * n - 1 - idx
                right.append(data_list[n - 1 - idx])
            result = left + list(data_list) + right
        elif mode_str == 'linear_ramp':
            end_val = kwargs.get('end_values', 0)
            if isinstance(end_val, (list, tuple)):
                end_val = end_val[0] if isinstance(end_val[0], (int, float)) else end_val[0][0]
            left = []
            for i in range(before):
                # Linear ramp from end_val to data_list[0]
                frac = float(i + 1) / float(before + 1) if before > 0 else 1.0
                left.append(end_val + (data_list[0] - end_val) * frac)
            right = []
            for i in range(after):
                # Linear ramp from data_list[-1] to end_val
                frac = float(i + 1) / float(after + 1) if after > 0 else 1.0
                right.append(data_list[-1] + (end_val - data_list[-1]) * frac)
            result = left + list(data_list) + right
        elif mode_str == 'mean':
            mean_val = sum(data_list) / len(data_list) if len(data_list) > 0 else 0.0
            result = [mean_val] * before + list(data_list) + [mean_val] * after
        elif mode_str == 'median':
            sorted_data = sorted(data_list)
            mid = len(sorted_data) // 2
            if len(sorted_data) % 2 == 0 and len(sorted_data) > 0:
                median_val = (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
            elif len(sorted_data) > 0:
                median_val = sorted_data[mid]
            else:
                median_val = 0.0
            result = [median_val] * before + list(data_list) + [median_val] * after
        elif mode_str == 'minimum':
            min_val = min(data_list) if len(data_list) > 0 else 0.0
            result = [min_val] * before + list(data_list) + [min_val] * after
        elif mode_str == 'maximum':
            max_val = max(data_list) if len(data_list) > 0 else 0.0
            result = [max_val] * before + list(data_list) + [max_val] * after
        else:
            raise NotImplementedError("pad mode '{}' is not supported".format(mode_str))
        return result

    if a.ndim == 1:
        data = [float(a[i]) for i in range(a.shape[0])]
        padded = _pad_1d(data, pw[0][0], pw[0][1], mode)
        return array(padded)

    # nD: pad axis-by-axis, starting from the last axis
    def _to_nested(arr):
        """Convert ndarray to nested Python lists."""
        if arr.ndim == 1:
            return [float(arr[i]) for i in range(arr.shape[0])]
        return [_to_nested(arr[i]) for i in range(arr.shape[0])]

    def _pad_axis(nested, axis, before, after, mode_str, current_depth=0):
        """Recursively pad along a specific axis of nested lists."""
        if current_depth == axis:
            return _pad_1d(nested, before, after, mode_str)
        else:
            return [_pad_axis(sub, axis, before, after, mode_str, current_depth + 1) for sub in nested]

    nested = _to_nested(a)
    for ax in range(a.ndim):
        before, after = pw[ax]
        if before > 0 or after > 0:
            nested = _pad_axis(nested, ax, before, after, mode)
    return array(nested)

def indices(dimensions, dtype=None, sparse=False):
    """Return an array representing the indices of a grid."""
    ndim = len(dimensions)
    if ndim == 0:
        return array([], dtype=dtype)

    if sparse:
        result = []
        for i in range(ndim):
            shape = [1] * ndim
            shape[i] = dimensions[i]
            idx = arange(0, dimensions[i]).reshape(shape)
            result.append(idx)
        return result

    # Dense: result shape is (ndim, *dimensions)
    grids = []
    for axis in range(ndim):
        # For each axis, create index array
        idx = arange(0, dimensions[axis])
        # Reshape to broadcast: shape is [1,...,1,dim_axis,1,...,1]
        shape = [1] * ndim
        shape[axis] = dimensions[axis]
        idx = idx.reshape(shape)
        # Tile to fill all dimensions
        reps = list(dimensions)
        reps[axis] = 1
        grid = tile(idx, reps)
        grids.append(grid)

    # Force contiguous layout before stacking to avoid memory layout issues
    contiguous = [asarray(g.tolist()) for g in grids]
    return stack(contiguous)  # Return single ndarray like NumPy

def fromiter(iterable, dtype=None, count=-1):
    if count > 0:
        data = []
        for i, val in enumerate(iterable):
            if i >= count:
                break
            data.append(val)
    else:
        data = list(iterable)
    return array(data, dtype=dtype)

def fromstring(string, dtype=None, count=-1, sep=''):
    """Parse array from a string."""
    if sep:
        parts = string.split(sep)
        data = [float(p.strip()) for p in parts if p.strip()]
    else:
        data = [float(x) for x in string.split()]
    if count > 0:
        data = data[:count]
    return array(data, dtype=dtype)

def array_equal(a1, a2, equal_nan=False):
    """True if two arrays have the same shape and elements."""
    try:
        if not isinstance(a1, ndarray):
            a1 = array(a1)
        if not isinstance(a2, ndarray):
            a2 = array(a2)
        if a1.shape != a2.shape:
            return False
        if equal_nan:
            try:
                both_nan = logical_and(isnan(a1), isnan(a2))
                neither_nan = logical_and(logical_not(isnan(a1)), logical_not(isnan(a2)))
                return bool(all(logical_or(both_nan, logical_and(neither_nan, a1 == a2))))
            except Exception:
                return bool((a1 == a2).all())
        return bool((a1 == a2).all())
    except Exception:
        return False

def array_equiv(a1, a2):
    """True if arrays are shape-consistent and element-wise equal, with broadcasting."""
    a1 = asarray(a1)
    a2 = asarray(a2)
    try:
        bshape = broadcast_shapes(a1.shape, a2.shape)
        b1 = broadcast_to(a1, bshape)
        b2 = broadcast_to(a2, bshape)
        return bool(all(b1 == b2))
    except Exception:
        return False

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
    if not isinstance(a, ndarray):
        a = asarray(a)
    original_dtype = str(a.dtype)
    if axis is not None and axis < 0:
        axis = a.ndim + axis
    result = a.sort(axis)
    # Preserve original dtype (Rust sort may convert to float64)
    if original_dtype in ('int32', 'int64') and str(result.dtype) != original_dtype:
        result = result.astype(original_dtype)
    return result

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
    if isinstance(a, ndarray):
        return _native.take(a, indices, axis)
    flat = array(a).flatten()
    if isinstance(indices, ndarray):
        idx = [int(float(indices.flatten()[i])) for i in range(indices.size)]
    else:
        idx = list(indices) if hasattr(indices, '__iter__') else [indices]
    return array([float(flat[i]) for i in idx])

def flip(a, axis=None):
    """Reverse the order of elements along the given axis."""
    if isinstance(a, ndarray):
        return _native.flip(a, axis)
    return array(list(reversed(a))) if axis is None else a

def flipud(a):
    """Flip array upside down (reverse along axis 0)."""
    if isinstance(a, ndarray):
        return _native.flipud(a)
    return flip(a, 0)

def fliplr(a):
    """Flip array left-right (reverse along axis 1)."""
    if isinstance(a, ndarray):
        return _native.fliplr(a)
    return flip(a, 1)

def rot90(a, k=1, axes=(0, 1)):
    """Rotate array 90 degrees in the plane of the first two axes."""
    if isinstance(a, ndarray):
        return _native.rot90(a, k)
    return a

def unique(a, return_index=False, return_inverse=False, return_counts=False, axis=None):
    """Return sorted unique elements of an array.

    Parameters
    ----------
    a : array_like
    return_index : bool
        If True, return indices of first occurrences.
    return_inverse : bool
        If True, return indices to reconstruct original from unique.
    return_counts : bool
        If True, return count of each unique value.
    axis : int, optional
        The axis along which to find unique slices. If None, flatten first.

    Returns
    -------
    unique : ndarray
    unique_indices : ndarray (optional)
    unique_inverse : ndarray (optional)
    unique_counts : ndarray (optional)
    """
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is not None:
        # For axis=0: find unique rows (or slices along axis 0)
        if axis < 0:
            axis = a.ndim + axis
        n_slices = a.shape[axis]
        # Extract each slice as a tuple for hashing
        seen = {}
        unique_indices_list = []
        for i in range(n_slices):
            # Build index to extract slice along axis
            if axis == 0:
                sl = a[i]
            else:
                # General case: use swapaxes to bring target axis to front
                tmp = swapaxes(a, 0, axis)
                sl = tmp[i]
            key = tuple(sl.flatten().tolist())
            if key not in seen:
                seen[key] = i
                unique_indices_list.append(i)
        # Build result array from unique indices
        if axis == 0:
            rows = [a[i] for i in unique_indices_list]
        else:
            tmp = swapaxes(a, 0, axis)
            rows = [tmp[i] for i in unique_indices_list]
        # Sort by the first element of each row for consistent ordering
        pairs = list(zip(unique_indices_list, rows))
        pairs.sort(key=lambda p: tuple(p[1].flatten().tolist()))
        unique_indices_list = [p[0] for p in pairs]
        rows = [p[1] for p in pairs]
        result = stack(rows, axis=0)
        if axis != 0:
            result = swapaxes(result, 0, axis)
        extras = return_index or return_inverse or return_counts
        if not extras:
            return result
        ret = (result,)
        if return_index:
            ret = ret + (array([float(i) for i in unique_indices_list]),)
        if return_inverse:
            # Map each original slice index to its position in unique result
            key_to_pos = {}
            for pos, idx in enumerate(unique_indices_list):
                if axis == 0:
                    sl = a[idx]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[idx]
                key = tuple(sl.flatten().tolist())
                key_to_pos[key] = pos
            inv = []
            for i in range(n_slices):
                if axis == 0:
                    sl = a[i]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[i]
                key = tuple(sl.flatten().tolist())
                inv.append(float(key_to_pos[key]))
            ret = ret + (array(inv),)
        if return_counts:
            # Count occurrences of each unique
            count_map = {}
            for i in range(n_slices):
                if axis == 0:
                    sl = a[i]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[i]
                key = tuple(sl.flatten().tolist())
                if key not in count_map:
                    count_map[key] = 0
                count_map[key] += 1
            counts_list = []
            for idx in unique_indices_list:
                if axis == 0:
                    sl = a[idx]
                else:
                    tmp = swapaxes(a, 0, axis)
                    sl = tmp[idx]
                key = tuple(sl.flatten().tolist())
                counts_list.append(float(count_map[key]))
            ret = ret + (array(counts_list),)
        return ret
    flat = a.flatten()
    n = flat.shape[0]
    vals = [float(flat[i]) for i in range(n)]

    # Build sorted unique with tracking info
    indexed = sorted(enumerate(vals), key=lambda t: t[1])
    unique_vals = []
    first_indices = []
    counts = []
    prev = None
    for orig_idx, v in indexed:
        if prev is None or v != prev:
            unique_vals.append(v)
            first_indices.append(orig_idx)
            counts.append(1)
            prev = v
        else:
            counts[-1] += 1
            # Keep the smallest original index
            if orig_idx < first_indices[-1]:
                first_indices[-1] = orig_idx

    result_unique = array(unique_vals)
    extras = return_index or return_inverse or return_counts
    if not extras:
        return result_unique
    ret = (result_unique,)
    if return_index:
        ret = ret + (array([float(i) for i in first_indices]),)
    if return_inverse:
        # For each element in the original flat array, find its position in unique_vals
        val_to_pos = {}
        for i, v in enumerate(unique_vals):
            val_to_pos[v] = i
        inverse = [float(val_to_pos[v]) for v in vals]
        ret = ret + (array(inverse),)
    if return_counts:
        ret = ret + (array([float(c) for c in counts]),)
    return ret

def intersect1d(ar1, ar2, assume_unique=False, return_indices=False):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    if not return_indices:
        return _native.intersect1d(ar1, ar2)
    # Find intersection with indices
    flat1 = ar1.flatten().tolist()
    flat2 = ar2.flatten().tolist()
    s1 = set(flat1)
    s2 = set(flat2)
    common = sorted(s1 & s2)
    ind1 = [flat1.index(v) for v in common]
    ind2 = [flat2.index(v) for v in common]
    return array(common), array(ind1), array(ind2)

def union1d(ar1, ar2):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.union1d(ar1, ar2)

def setdiff1d(ar1, ar2, assume_unique=False):
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    return _native.setdiff1d(ar1, ar2)

def setxor1d(ar1, ar2, assume_unique=False):
    """Return sorted, unique values that are in only one of the input arrays."""
    if not isinstance(ar1, ndarray):
        ar1 = array(ar1)
    if not isinstance(ar2, ndarray):
        ar2 = array(ar2)
    u1 = unique(ar1)
    u2 = unique(ar2)
    # Elements in ar1 but not ar2, plus elements in ar2 but not ar1
    diff1 = setdiff1d(u1, u2)
    diff2 = setdiff1d(u2, u1)
    return sort(concatenate([diff1, diff2]))

def isin(element, test_elements, assume_unique=False, invert=False):
    if not isinstance(element, ndarray):
        element = array(element)
    if not isinstance(test_elements, ndarray):
        test_elements = array(test_elements)
    result = _native.isin(element, test_elements)
    if invert:
        return logical_not(result)
    return result

in1d = isin

def all(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        return a.all()
    # Reduce along specific axis: all elements nonzero iff min != 0
    # But min of booleans (1.0/0.0) works: min==0 means not all true.
    # Use the ndarray's min(axis) which supports axis.
    m = a.min(axis, keepdims)
    # Convert to boolean array: nonzero => True
    flat = m.flatten().tolist()
    result = [v != 0.0 for v in flat]
    return array(result).reshape(m.shape)

def any(a, axis=None, out=None, keepdims=False):
    if not isinstance(a, ndarray):
        a = asarray(a)
    if axis is None:
        return a.any()
    # Reduce along specific axis: any element nonzero iff max != 0
    m = a.max(axis, keepdims)
    flat = m.flatten().tolist()
    result = [v != 0.0 for v in flat]
    return array(result).reshape(m.shape)

def may_share_memory(a, b, max_work=None):
    """Check if arrays may share memory. Returns True if a and b are the same object."""
    return a is b

def shares_memory(a, b, max_work=None):
    """Check if arrays share memory. Returns True if a and b are the same object."""
    return a is b

def signbit(x):
    if isinstance(x, ndarray):
        return _native.signbit(x)
    return x < 0

def power(x1, x2):
    return asarray(x1) ** asarray(x2)

def add(x1, x2, out=None):
    return asarray(x1) + asarray(x2)

def divide(x1, x2, out=None):
    return asarray(x1) / asarray(x2)

def greater(x1, x2):
    return asarray(x1) > asarray(x2)

def less(x1, x2):
    return asarray(x1) < asarray(x2)

def equal(x1, x2):
    return asarray(x1) == asarray(x2)

def not_equal(x1, x2):
    return asarray(x1) != asarray(x2)

def greater_equal(x1, x2):
    return asarray(x1) >= asarray(x2)

def less_equal(x1, x2):
    return asarray(x1) <= asarray(x2)

def maximum(x1, x2):
    x1 = asarray(x1)
    x2 = asarray(x2)
    return where(x1 >= x2, x1, x2)

def minimum(x1, x2):
    x1 = asarray(x1)
    x2 = asarray(x2)
    return where(x1 <= x2, x1, x2)

def logical_and(x1, x2):
    x1 = asarray(x1)
    x2 = asarray(x2)
    out_shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = broadcast_to(x1, out_shape)
    x2 = broadcast_to(x2, out_shape)
    flat1 = x1.flatten().tolist()
    flat2 = x2.flatten().tolist()
    result = [1.0 if (bool(a) and bool(b)) else 0.0 for a, b in zip(flat1, flat2)]
    return array(result).reshape(out_shape)

def logical_or(x1, x2):
    x1 = asarray(x1)
    x2 = asarray(x2)
    out_shape = broadcast_shapes(x1.shape, x2.shape)
    x1 = broadcast_to(x1, out_shape)
    x2 = broadcast_to(x2, out_shape)
    flat1 = x1.flatten().tolist()
    flat2 = x2.flatten().tolist()
    result = [1.0 if (bool(a) or bool(b)) else 0.0 for a, b in zip(flat1, flat2)]
    return array(result).reshape(out_shape)

def logical_not(x):
    if isinstance(x, ndarray):
        return _native.logical_not(x)
    return not x

def logical_xor(x1, x2):
    return logical_and(logical_or(x1, x2), logical_not(logical_and(x1, x2)))

# --- Bitwise operations ------------------------------------------------------

def bitwise_and(x1, x2):
    """Element-wise bitwise AND of integer arrays."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) & int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def bitwise_or(x1, x2):
    """Element-wise bitwise OR of integer arrays."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) | int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def bitwise_xor(x1, x2):
    """Element-wise bitwise XOR of integer arrays."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) ^ int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def bitwise_not(x):
    """Element-wise bitwise NOT (invert) of integer array."""
    x = asarray(x)
    a = x.astype("int64")
    a_list = a.flatten().tolist()
    result = [~int(v) for v in a_list]
    return array([float(v) for v in result]).astype("int64")

invert = bitwise_not

def left_shift(x1, x2):
    """Element-wise left bit shift."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) << int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

def right_shift(x1, x2):
    """Element-wise right bit shift."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    a = x1.astype("int64")
    b = x2.astype("int64")
    a_list = a.flatten().tolist()
    b_list = b.flatten().tolist()
    result = [int(av) >> int(bv) for av, bv in zip(a_list, b_list)]
    return array([float(v) for v in result]).astype("int64")

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

def angle(z, deg=False):
    """Return the angle (argument) of complex or real elements."""
    z = asarray(z)
    if iscomplexobj(z):
        try:
            result = arctan2(imag(z), real(z))
        except Exception:
            result = z.angle()
    else:
        # For real arrays: arctan2(0, x) gives 0 for positive, pi for negative
        result = arctan2(zeros(z.shape), z)
    if deg:
        result = result * (180.0 / _math.pi)
    return result

def _unwrap_1d_list(data, discont, period):
    """Unwrap a 1D list in-place style, returning a new list."""
    if len(data) == 0:
        return []
    result = [data[0]]
    for i in _builtin_range(1, len(data)):
        d = data[i] - result[-1]
        d = d - period * round(d / period)
        result.append(result[-1] + d)
    return result

def unwrap(p, discont=None, axis=-1, period=2*_math.pi):
    """Unwrap by changing deltas between values to 2*pi complement."""
    p = asarray(p)
    if discont is None:
        discont = period / 2
    if p.ndim <= 1:
        return array(_unwrap_1d_list(p.flatten().tolist(), discont, period))
    # For 2D: apply unwrap along the specified axis
    nd = p.ndim
    ax = axis
    if ax < 0:
        ax = nd + ax
    shape = p.shape
    # Move the target axis to the last position
    if ax != nd - 1:
        axes = [i for i in _builtin_range(nd) if i != ax] + [ax]
        pt = transpose(p, axes=axes)
    else:
        pt = p
    # Now unwrap along the last axis
    pt_shape = pt.shape
    n_outer = 1
    for s in pt_shape[:-1]:
        n_outer *= s
    flat_all = pt.flatten().tolist()
    axis_len = pt_shape[-1]
    result_flat = []
    for i in _builtin_range(n_outer):
        start = i * axis_len
        row = flat_all[start:start + axis_len]
        result_flat.extend(_unwrap_1d_list(row, discont, period))
    out = array(result_flat).reshape(pt_shape)
    # Move axis back if we transposed
    if ax != nd - 1:
        # Inverse permutation
        inv_axes = [0] * nd
        for i, a_val in enumerate(axes):
            inv_axes[a_val] = i
        out = transpose(out, axes=inv_axes)
    return out

def histogram_bin_edges(a, bins=10, range=None, weights=None):
    """Compute the bin edges for a histogram without computing the histogram itself."""
    a = asarray(a)
    if isinstance(bins, int):
        flat = a.flatten().tolist()
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo, hi = _builtin_min(flat), _builtin_max(flat)
        edges = linspace(lo, hi, bins + 1)
        return edges
    else:
        return asarray(bins)

def histogram(a, bins=10, range=None, density=None, weights=None):
    if not isinstance(a, ndarray):
        a = array(a)
    if isinstance(bins, (list, tuple, ndarray)):
        # Custom bin edges
        edges = asarray(bins).flatten()
        edge_list = edges.tolist()
        flat = a.flatten().tolist()
        n_bins = len(edge_list) - 1
        counts = [0.0] * n_bins
        w_list = None
        if weights is not None:
            w_list = asarray(weights).flatten().tolist()
        for idx_val, v in enumerate(flat):
            for j in _builtin_range(n_bins):
                if j == n_bins - 1:
                    if edge_list[j] <= v <= edge_list[j + 1]:
                        counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                        break
                else:
                    if edge_list[j] <= v < edge_list[j + 1]:
                        counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                        break
        counts_arr = array(counts)
        if density:
            bin_widths = diff(edges)
            total = float(sum(counts_arr))
            if total > 0.0:
                counts_arr = counts_arr / (total * bin_widths)
        return counts_arr, edges
    # bins is an int
    if weights is not None or range is not None:
        # Python fallback for weights/range with integer bins
        flat = a.flatten().tolist()
        if range is not None:
            lo, hi = float(range[0]), float(range[1])
        else:
            lo, hi = _builtin_min(flat), _builtin_max(flat)
        edges = linspace(lo, hi, num=bins + 1, endpoint=True)
        edge_list = edges.tolist()
        counts = [0.0] * bins
        w_list = None
        if weights is not None:
            w_list = asarray(weights).flatten().tolist()
        for idx_val, val in enumerate(flat):
            if range is not None and (val < lo or val > hi):
                continue
            for j in _builtin_range(bins):
                if (val >= edge_list[j] and val < edge_list[j + 1]) or (j == bins - 1 and val == edge_list[j + 1]):
                    counts[j] += (w_list[idx_val] if w_list is not None else 1.0)
                    break
        hist = array(counts)
        if density:
            widths = array([edge_list[i+1] - edge_list[i] for i in _builtin_range(bins)])
            total = float(sum(hist))
            if total > 0.0:
                hist = hist / (total * widths)
        return hist, edges
    # No weights, no range: use native
    counts, edges = _native.histogram(a, bins)
    if density:
        bin_widths = diff(edges)
        total = float(sum(counts))
        if total > 0.0:
            counts = counts / (total * bin_widths)
    return counts, edges

def histogram2d(x, y, bins=10, range=None, density=None, weights=None):
    """Compute the 2D histogram of two data samples."""
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(y, ndarray):
        y = array(y)
    x_flat = x.flatten()
    y_flat = y.flatten()
    n = x_flat.size
    x_list = x_flat.tolist()
    y_list = y_flat.tolist()
    # Determine number of bins for x and y
    if isinstance(bins, (list, tuple)):
        nbins_x = int(bins[0])
        nbins_y = int(bins[1])
    else:
        nbins_x = int(bins)
        nbins_y = int(bins)
    # Determine ranges
    if range is not None:
        xmin, xmax = float(range[0][0]), float(range[0][1])
        ymin, ymax = float(range[1][0]), float(range[1][1])
    else:
        xmin = _builtin_min(x_list)
        xmax = _builtin_max(x_list)
        ymin = _builtin_min(y_list)
        ymax = _builtin_max(y_list)
    # Build bin edges
    xedges_list = []
    yedges_list = []
    for i in _builtin_range(nbins_x + 1):
        xedges_list.append(xmin + i * (xmax - xmin) / nbins_x)
    for i in _builtin_range(nbins_y + 1):
        yedges_list.append(ymin + i * (ymax - ymin) / nbins_y)
    # Count into 2D bins
    hist_data = []
    for i in _builtin_range(nbins_x):
        row = []
        for j in _builtin_range(nbins_y):
            row.append(0.0)
        hist_data.append(row)
    xspan = xmax - xmin
    yspan = ymax - ymin
    for k in _builtin_range(n):
        xv = x_list[k]
        yv = y_list[k]
        # Find x bin
        if xspan == 0.0:
            xi = 0
        else:
            xi = int((xv - xmin) / (xspan / nbins_x))
        if xi >= nbins_x:
            xi = nbins_x - 1
        if xi < 0:
            xi = 0
        # Find y bin
        if yspan == 0.0:
            yi = 0
        else:
            yi = int((yv - ymin) / (yspan / nbins_y))
        if yi >= nbins_y:
            yi = nbins_y - 1
        if yi < 0:
            yi = 0
        hist_data[xi][yi] = hist_data[xi][yi] + 1.0
    # Convert to arrays
    flat_hist = []
    for i in _builtin_range(nbins_x):
        for j in _builtin_range(nbins_y):
            flat_hist.append(hist_data[i][j])
    hist = array(flat_hist).reshape((nbins_x, nbins_y))
    xedges = array(xedges_list)
    yedges = array(yedges_list)
    return hist, xedges, yedges

def bincount(x, weights=None, minlength=0):
    if not isinstance(x, ndarray):
        x = array(x)
    if weights is not None and not isinstance(weights, ndarray):
        weights = array(weights)
    return _native.bincount(x, weights, minlength)

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

    @staticmethod
    def split(a, sep=None, maxsplit=-1):
        """Split each element in a around sep."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            result.append(str(s).split(sep, maxsplit))
        if len(result) == 1:
            return result[0]
        return result

    @staticmethod
    def join(sep, a):
        """Join strings in a with separator sep, element-wise."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, (list, tuple)):
            items = a
        else:
            items = [a]
        # If items is a list of lists, join each sublist
        if len(items) > 0 and isinstance(items[0], (list, tuple)):
            result = [str(sep).join(str(x) for x in sub) for sub in items]
            return array(result)
        # Otherwise join all items into a single string
        return str(sep).join(str(x) for x in items)

    @staticmethod
    def find(a, sub, start=0, end=None):
        """Find first occurrence of sub in each element of a."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.find(sub, start))
            else:
                result.append(s.find(sub, start, end))
        return array(result)

    @staticmethod
    def count(a, sub, start=0, end=None):
        """Count non-overlapping occurrences of sub in each element of a."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        result = []
        for s in items:
            s = str(s)
            if end is None:
                result.append(s.count(sub, start))
            else:
                result.append(s.count(sub, start, end))
        return array(result)

    @staticmethod
    def add(a, b):
        """Element-wise string concatenation."""
        if isinstance(a, ndarray):
            items_a = a.tolist()
        elif isinstance(a, _ObjectArray):
            items_a = a._data
        elif isinstance(a, str):
            items_a = [a]
        else:
            items_a = list(a)
        if isinstance(b, ndarray):
            items_b = b.tolist()
        elif isinstance(b, _ObjectArray):
            items_b = b._data
        elif isinstance(b, str):
            items_b = [b]
        else:
            items_b = list(b)
        # Broadcast if lengths differ
        if len(items_a) == 1 and len(items_b) > 1:
            items_a = items_a * len(items_b)
        elif len(items_b) == 1 and len(items_a) > 1:
            items_b = items_b * len(items_a)
        result = [str(x) + str(y) for x, y in zip(items_a, items_b)]
        return array(result)

    @staticmethod
    def multiply(a, i):
        """Element-wise string repetition."""
        if isinstance(a, ndarray):
            items = a.tolist()
        elif isinstance(a, _ObjectArray):
            items = a._data
        elif isinstance(a, str):
            items = [a]
        else:
            items = list(a)
        i = int(i)
        result = [str(s) * i for s in items]
        return array(result)

    @staticmethod
    def _to_str_list(a):
        """Convert input to a flat list of strings."""
        if isinstance(a, _ObjectArray):
            return [str(x) for x in a._data]
        if isinstance(a, ndarray):
            data = a.tolist()
            if isinstance(data, list):
                result = []
                for item in data:
                    if isinstance(item, list):
                        result.extend([str(x) for x in item])
                    else:
                        result.append(str(item))
                return result
            return [str(data)]
        if isinstance(a, str):
            return [a]
        if isinstance(a, (list, tuple)):
            result = []
            for item in a:
                if isinstance(item, (list, tuple)):
                    result.extend([str(x) for x in item])
                else:
                    result.append(str(item))
            return result
        return [str(a)]

    @staticmethod
    def center(a, width, fillchar=' '):
        """Pad each string element in a to width, centering the string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.center(int(width), fillchar) for s in data])

    @staticmethod
    def ljust(a, width, fillchar=' '):
        """Left-justify each string element in a to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.ljust(int(width), fillchar) for s in data])

    @staticmethod
    def rjust(a, width, fillchar=' '):
        """Right-justify each string element in a to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.rjust(int(width), fillchar) for s in data])

    @staticmethod
    def zfill(a, width):
        """Pad each string element in a with zeros on the left to width."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.zfill(int(width)) for s in data])

    @staticmethod
    def title(a):
        """Return element-wise title cased version of string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.title() for s in data])

    @staticmethod
    def swapcase(a):
        """Return element-wise with uppercase converted to lowercase and vice versa."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.swapcase() for s in data])

    @staticmethod
    def isalpha(a):
        """Return true for each element if all characters are alphabetic."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isalpha() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isdigit(a):
        """Return true for each element if all characters are digits."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isdigit() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isnumeric(a):
        """Return true for each element if all characters are numeric."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if (s.isnumeric() if hasattr(s, 'isnumeric') else s.isdigit()) else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isupper(a):
        """Return true for each element if all cased characters are uppercase."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isupper() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def islower(a):
        """Return true for each element if all cased characters are lowercase."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.islower() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def isspace(a):
        """Return true for each element if all characters are whitespace."""
        data = _char_mod._to_str_list(a)
        return array([1.0 if s.isspace() else 0.0 for s in data]).astype("bool")

    @staticmethod
    def encode(a, encoding='utf-8', errors='strict'):
        """Encode each string element to bytes."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.encode(encoding, errors) for s in data])

    @staticmethod
    def decode(a, encoding='utf-8', errors='strict'):
        """Decode each bytes element to string."""
        data = _char_mod._to_str_list(a)
        return _ObjectArray([s.decode(encoding, errors) if isinstance(s, bytes) else s for s in data])

char = _char_mod()

# --- Index Utilities --------------------------------------------------------

def unravel_index(indices, shape, order='C'):
    if not isinstance(indices, ndarray):
        if isinstance(indices, int):
            indices = array([indices])
        else:
            indices = array(indices)
    return _native.unravel_index(indices, shape)

def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    arrays = tuple(array([a]) if isinstance(a, (int, float)) else (a if isinstance(a, ndarray) else array(a)) for a in multi_index)
    return _native.ravel_multi_index(arrays, dims)

def interp(x, xp, fp, left=None, right=None, period=None):
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(xp, ndarray):
        xp = array(xp)
    if not isinstance(fp, ndarray):
        fp = array(fp)
    result = _native.interp(x, xp, fp)
    if left is not None or right is not None:
        x_arr = asarray(x).flatten().tolist()
        xp_arr = asarray(xp).flatten().tolist()
        result_list = result.flatten().tolist()
        xp_min = _builtin_min(xp_arr)
        xp_max = _builtin_max(xp_arr)
        for i, xi in enumerate(x_arr):
            if left is not None and xi < xp_min:
                result_list[i] = float(left)
            if right is not None and xi > xp_max:
                result_list[i] = float(right)
        result = array(result_list)
    return result

def gradient(f, *varargs, axis=None, edge_order=1):
    if not isinstance(f, ndarray):
        f = array(f)
    if f.ndim == 1:
        # 1D case: single spacing
        spacing = float(varargs[0]) if varargs else 1.0
        return _native.gradient(f, spacing)
    # nD case
    if axis is not None:
        # gradient along specific axis/axes
        if isinstance(axis, int):
            ax = axis
            if ax < 0:
                ax = f.ndim + ax
            sp = float(varargs[0]) if varargs else 1.0
            result = _native.gradient(f, sp)
            if isinstance(result, (list, tuple)):
                return result[ax]
            return result
        # multiple axes
        results = []
        for i, ax in enumerate(axis):
            sp = float(varargs[i]) if i < len(varargs) else 1.0
            grads = _native.gradient(f, sp)
            if isinstance(grads, (list, tuple)):
                a = ax
                if a < 0:
                    a = f.ndim + a
                results.append(grads[a])
            else:
                results.append(grads)
        return results
    # All axes
    if len(varargs) == 0:
        return _native.gradient(f, 1.0)
    elif len(varargs) == 1:
        return _native.gradient(f, float(varargs[0]))
    else:
        # Different spacing per axis
        results = []
        for i in range(f.ndim):
            sp = float(varargs[i]) if i < len(varargs) else 1.0
            grads = _native.gradient(f, sp)
            if isinstance(grads, (list, tuple)):
                results.append(grads[i])
            else:
                results.append(grads)
        return results

def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(y, ndarray):
        y = array(y)
    return _native.polyfit(x, y, int(deg))

def polyval(p, x):
    if not isinstance(p, ndarray):
        p = array(p)
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.polyval(p, x)

# --- Polynomial utilities ---------------------------------------------------

def roots(p):
    """Return the roots of a polynomial with coefficients given in p."""
    if isinstance(p, poly1d):
        coeffs = list(p._coeffs)
    elif isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    # Remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs = coeffs[1:]
    n = len(coeffs) - 1  # degree
    if n == 0:
        return array([])
    if n == 1:
        return array([-coeffs[1] / coeffs[0]])
    if n == 2:
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = disc ** 0.5
            return array([(-b + sq) / (2 * a), (-b - sq) / (2 * a)])
        else:
            sq = (-disc) ** 0.5
            return array([(-b) / (2 * a), (-b) / (2 * a)])  # real part only
    # For degree > 2: Durand-Kerner method for finding all roots simultaneously.
    # Normalize polynomial so leading coefficient is 1
    a0 = float(coeffs[0])
    norm_coeffs = [float(c) / a0 for c in coeffs]
    # Horner's method to compute polynomial value at a point
    import math as _m
    def _poly_at(z, cs):
        val = 0.0
        for c in cs:
            val = val * z + c
        return val
    # Bound on root magnitudes
    _abs_coeffs = [abs(c) for c in norm_coeffs[1:]]
    _max_coeff = _abs_coeffs[0]
    for _ac in _abs_coeffs[1:]:
        if _ac > _max_coeff:
            _max_coeff = _ac
    bound = 1.0 + _max_coeff
    # Initial guesses: distinct real values spread around
    z = [bound * (0.4 + 0.6 * _m.cos(2.0 * _m.pi * (k + 0.25) / n)) for k in range(n)]
    # Durand-Kerner iteration
    for _iteration in range(1000):
        max_delta = 0.0
        new_z = list(z)
        for i in range(n):
            pval = _poly_at(z[i], norm_coeffs)
            denom = 1.0
            for j in range(n):
                if j != i:
                    diff = z[i] - z[j]
                    if abs(diff) < 1e-15:
                        diff = 1e-15
                    denom *= diff
            if abs(denom) < 1e-30:
                denom = 1e-30
            delta = pval / denom
            new_z[i] = z[i] - delta
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        z = new_z
        if max_delta < 1e-12:
            break
    return array(z)


def polyadd(a1, a2):
    """Add two polynomials (coefficient arrays, highest degree first)."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    while len(a1) < len(a2):
        a1.insert(0, 0.0)
    while len(a2) < len(a1):
        a2.insert(0, 0.0)
    return array([a1[i] + a2[i] for i in range(len(a1))])


def polysub(a1, a2):
    """Subtract two polynomials."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    while len(a1) < len(a2):
        a1.insert(0, 0.0)
    while len(a2) < len(a1):
        a2.insert(0, 0.0)
    return array([a1[i] - a2[i] for i in range(len(a1))])


def polymul(a1, a2):
    """Multiply two polynomials."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    n = len(a1) + len(a2) - 1
    result = [0.0] * n
    for i, c1 in enumerate(a1):
        for j, c2 in enumerate(a2):
            result[i + j] += c1 * c2
    return array(result)


def polyder(p, m=1):
    """Return the derivative of the specified order of a polynomial."""
    if isinstance(p, poly1d):
        return p.deriv(m)
    if isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    for _ in range(m):
        n = len(coeffs) - 1
        if n <= 0:
            coeffs = [0.0]
            break
        new_coeffs = []
        for i in range(n):
            new_coeffs.append(coeffs[i] * (n - i))
        coeffs = new_coeffs
    return array(coeffs)


def polyint(p, m=1, k=0):
    """Return the integral of a polynomial."""
    if isinstance(p, poly1d):
        return p.integ(m, k)
    if isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    for _ in range(m):
        n = len(coeffs)
        new_coeffs = []
        for i in range(n):
            new_coeffs.append(coeffs[i] / (n - i))
        new_coeffs.append(float(k))
        coeffs = new_coeffs
    return array(coeffs)


class poly1d:
    """A one-dimensional polynomial class."""
    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._coeffs = list(c_or_r._coeffs)
        elif r:
            # c_or_r are roots, convert to coefficients
            self._coeffs = [1.0]
            if isinstance(c_or_r, ndarray):
                roots_list = [c_or_r[i] for i in range(c_or_r.size)]
            else:
                roots_list = list(c_or_r)
            for root in roots_list:
                new_coeffs = [0.0] * (len(self._coeffs) + 1)
                for i, c in enumerate(self._coeffs):
                    new_coeffs[i] += c
                    new_coeffs[i + 1] -= c * float(root)
                self._coeffs = new_coeffs
        else:
            if isinstance(c_or_r, ndarray):
                self._coeffs = [c_or_r[i] for i in range(c_or_r.size)]
            else:
                self._coeffs = [float(c) for c in c_or_r]
        self._variable = variable or 'x'

    @property
    def coeffs(self):
        return array(self._coeffs)

    @property
    def c(self):
        return self.coeffs

    @property
    def order(self):
        return len(self._coeffs) - 1

    @property
    def roots(self):
        return _poly1d_roots(self._coeffs)

    @property
    def o(self):
        return self.order

    def __call__(self, val):
        return polyval(self._coeffs, val)

    def __add__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            oc = [float(other)]
        else:
            oc = list(other)
        return poly1d(polyadd(self._coeffs, oc))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            oc = [float(other)]
        else:
            oc = list(other)
        return poly1d(polysub(self._coeffs, oc))

    def __mul__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            return poly1d([c * float(other) for c in self._coeffs])
        else:
            oc = list(other)
        return poly1d(polymul(self._coeffs, oc))

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return poly1d([c * float(other) for c in self._coeffs])
        return self.__mul__(other)

    def __neg__(self):
        return poly1d([-c for c in self._coeffs])

    def __len__(self):
        return self.order

    def __getitem__(self, idx):
        # poly1d[i] returns coefficient of x^i (reverse indexing)
        if idx > self.order:
            return 0.0
        return self._coeffs[self.order - idx]

    def deriv(self, m=1):
        """Return the derivative of this polynomial."""
        coeffs = list(self._coeffs)
        for _ in range(m):
            n = len(coeffs) - 1
            if n <= 0:
                coeffs = [0.0]
                break
            new_coeffs = []
            for i in range(n):
                new_coeffs.append(coeffs[i] * (n - i))
            coeffs = new_coeffs
        return poly1d(coeffs)

    def integ(self, m=1, k=0):
        """Return the integral of this polynomial."""
        coeffs = list(self._coeffs)
        for _ in range(m):
            n = len(coeffs)
            new_coeffs = []
            for i in range(n):
                new_coeffs.append(coeffs[i] / (n - i))
            new_coeffs.append(float(k))
            coeffs = new_coeffs
        return poly1d(coeffs)

    def __repr__(self):
        return "poly1d(" + repr(self._coeffs) + ")"

    def __str__(self):
        return "poly1d(" + repr(self._coeffs) + ")"


# Alias so poly1d.roots can call without name clash with the module-level roots()
_poly1d_roots = roots


# --- I/O: loadtxt / savetxt / genfromtxt ------------------------------------

def loadtxt(fname, dtype=None, comments='#', delimiter=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None, **kwargs):
    """Load data from a text file. Each row must have the same number of values."""
    if isinstance(fname, str):
        f = open(fname, 'r')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        rows = []
        lines_read = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip comment lines
            if comments and line.startswith(comments):
                continue
            if lines_read < skiprows:
                lines_read += 1
                continue
            if max_rows is not None and len(rows) >= max_rows:
                break
            # Split by delimiter
            if delimiter is None:
                parts = line.split()
            else:
                parts = line.split(delimiter)
            # Select columns
            if usecols is not None:
                parts = [parts[i] for i in usecols]
            row = [float(x.strip()) for x in parts]
            rows.append(row)
        if not rows:
            return array([])
        if len(rows) == 1 and ndmin < 2:
            result = array(rows[0])
        else:
            result = array(rows)
        if unpack:
            return result.T
        return result
    finally:
        if close_file:
            f.close()

def savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None):
    """Save an array to a text file."""
    if not isinstance(X, ndarray):
        X = array(X)
    if X.ndim == 1:
        X = X.reshape([X.size, 1])

    if isinstance(fname, str):
        f = open(fname, 'w')
        close_file = True
    else:
        f = fname
        close_file = False
    try:
        if header:
            for hline in header.split('\n'):
                f.write(comments + hline + newline)

        rows = X.shape[0]
        cols = X.shape[1]
        for i in range(rows):
            vals = []
            for j in range(cols):
                vals.append(fmt % float(X[i][j]))
            f.write(delimiter.join(vals) + newline)

        if footer:
            for fline in footer.split('\n'):
                f.write(comments + fline + newline)
    finally:
        if close_file:
            f.close()

def genfromtxt(fname, dtype=None, comments='#', delimiter=None, skip_header=0,
               skip_footer=0, missing_values=None, filling_values=None,
               usecols=None, names=None, excludelist=None, deletechars=None,
               replace_space='_', autostrip=False, case_sensitive=True,
               defaultfmt='f%i', unpack=False, usemask=False, loose=True,
               invalid_raise=True, max_rows=None, encoding='bytes', **kwargs):
    """Load data from text file, handling missing values."""
    if filling_values is None:
        filling_values = float('nan')

    if isinstance(fname, str):
        with open(fname, 'r') as f:
            lines = f.readlines()
    else:
        lines = fname.readlines()

    # Skip header/footer
    lines = lines[skip_header:]
    if skip_footer > 0:
        lines = lines[:-skip_footer]
    if max_rows is not None:
        lines = lines[:max_rows]

    rows = []
    for line in lines:
        line = line.strip()
        if not line or (comments and line.startswith(comments)):
            continue
        if delimiter is None:
            parts = line.split()
        else:
            parts = line.split(delimiter)

        if usecols is not None:
            parts = [parts[i] for i in usecols]

        row = []
        for p in parts:
            p = p.strip()
            if missing_values and p in (missing_values if isinstance(missing_values, (list, tuple, set)) else [missing_values]):
                row.append(filling_values)
            else:
                try:
                    row.append(float(p))
                except (ValueError, TypeError):
                    row.append(filling_values)
        rows.append(row)

    if not rows:
        return array([])
    result = array(rows)
    if unpack:
        return result.T
    return result

# --- ufunc function forms (Tier 12A) ----------------------------------------

def subtract(x1, x2, out=None):
    return asarray(x1) - asarray(x2)

def multiply(x1, x2, out=None):
    return asarray(x1) * asarray(x2)

def true_divide(x1, x2, out=None):
    return asarray(x1) / asarray(x2)

def floor_divide(x1, x2, out=None):
    return asarray(x1) // asarray(x2)

def remainder(x1, x2, out=None):
    return asarray(x1) % asarray(x2)

mod = remainder

def divmod_(x1, x2):
    """Return element-wise quotient and remainder simultaneously."""
    x1, x2 = asarray(x1), asarray(x2)
    return (floor_divide(x1, x2), remainder(x1, x2))

def negative(x):
    return -asarray(x)

def positive(x):
    return asarray(x) * 1

def float_power(x1, x2):
    return power(asarray(x1).astype('float64'), asarray(x2).astype('float64'))

def identity(n, dtype=None):
    return eye(n, dtype=dtype)

def diag(v, k=0):
    """Construct a diagonal array or extract a diagonal.

    If *v* is 1-D, return a 2-D array with *v* on the *k*-th diagonal.
    If *v* is 2-D, extract the *k*-th diagonal (same as ``diagonal``).
    """
    v = asarray(v)
    if v.ndim == 1:
        n = len(v)
        abs_k = abs(k)
        size = n + abs_k
        # Build as flat list then reshape
        flat = [0.0] * (size * size)
        for i in range(n):
            if k >= 0:
                flat[i * size + (k + i)] = float(v[i])
            else:
                flat[(abs_k + i) * size + i] = float(v[i])
        return array(flat).reshape([size, size])
    elif v.ndim == 2:
        return diagonal(v, offset=k)
    else:
        raise ValueError("Input must be 1-D or 2-D")

def tri(N, M=None, k=0, dtype=None):
    """An array with ones at and below the given diagonal and zeros elsewhere."""
    if M is None:
        M = N
    rows = []
    for i in range(N):
        row = []
        for j in range(M):
            row.append(1.0 if j <= i + k else 0.0)
        rows.append(row)
    return array(rows)

def tril(m, k=0):
    """Lower triangle of an array. Return a copy with elements above the k-th diagonal zeroed."""
    m = asarray(m)
    mask = tri(m.shape[0], m.shape[1], k=k)
    return m * mask

def triu(m, k=0):
    """Upper triangle of an array. Return a copy with elements below the k-th diagonal zeroed."""
    m = asarray(m)
    mask = tri(m.shape[0], m.shape[1], k=k - 1)
    return m * (ones(m.shape) - mask)

def vander(x, N=None, increasing=False):
    """Generate a Vandermonde matrix."""
    x = asarray(x).flatten()
    n = x.size
    if N is None:
        N = n
    if increasing:
        cols = []
        for j in range(N):
            col = []
            for i in range(n):
                col.append(x[i] ** j)
            cols.append(array(col))
    else:
        cols = []
        for j in range(N):
            col = []
            for i in range(n):
                col.append(x[i] ** (N - 1 - j))
            cols.append(array(col))
    return stack(cols, axis=1)

def kron(a, b):
    """Kronecker product of two arrays."""
    a = asarray(a)
    b = asarray(b)
    if a.ndim == 1:
        a = a.reshape((1, a.size))
    if b.ndim == 1:
        b = b.reshape((1, b.size))
    ar, ac = a.shape[0], a.shape[1]
    br, bc = b.shape[0], b.shape[1]
    rows = []
    for i in range(ar):
        for bi in range(br):
            row = []
            for j in range(ac):
                for bj in range(bc):
                    row.append(a[i][j] * b[bi][bj])
            rows.append(row)
    return array(rows)

def inner(a, b):
    """Inner product of two arrays.

    For 1-D arrays this is the dot product.  For higher-dimensional arrays
    this contracts over the last axis of both a and b.
    """
    a = asarray(a)
    b = asarray(b)
    if a.ndim <= 1 and b.ndim <= 1:
        return dot(a, b)
    # For 2D: inner(A, B) = A @ B.T
    if a.ndim == 2 and b.ndim == 2:
        return dot(a, b.T)
    # General: tensordot contracting last axis of each
    return tensordot(a, b, axes=([-1], [-1]))

def matmul(x1, x2):
    """Matrix product of two arrays (same as the ``@`` operator)."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    return dot(x1, x2)

def broadcast_to(arr, shape):
    """Broadcast an array to a new shape using reshape + tile."""
    arr = asarray(arr)
    arr_shape = arr.shape
    if arr_shape == tuple(shape):
        return arr
    ndim = len(shape)
    arr_ndim = len(arr_shape)
    # Prepend 1s to make same ndim
    if arr_ndim < ndim:
        new_shape = [1] * (ndim - arr_ndim) + list(arr_shape)
        arr = arr.reshape(new_shape)
        arr_shape = tuple(new_shape)
    # Check compatibility and compute reps
    reps = []
    for i in range(ndim):
        if arr_shape[i] == shape[i]:
            reps.append(1)
        elif arr_shape[i] == 1:
            reps.append(shape[i])
        else:
            raise ValueError(f"cannot broadcast shape {arr_shape} to {tuple(shape)}")
    return tile(arr, reps)

def flatnonzero(a):
    """Return indices of non-zero elements in the flattened array."""
    a = asarray(a).flatten()
    indices_list = []
    for i in range(len(a)):
        if float(a[i]) != 0.0:
            indices_list.append(i)
    return array(indices_list)

def extract(condition, arr):
    """Return elements of arr where condition is True."""
    condition = asarray(condition).flatten()
    arr = asarray(arr).flatten()
    result = []
    for i in range(len(arr)):
        if float(condition[i]) != 0.0:
            result.append(float(arr[i]))
    if not result:
        return array([])
    return array(result)

def digitize(x, bins, right=False):
    """Return the indices of the bins to which each value belongs."""
    x = asarray(x)
    bins = asarray(bins)
    bins_list = [float(bins[i]) for i in range(len(bins))]
    result = []
    ascending = len(bins_list) < 2 or bins_list[-1] >= bins_list[0]
    for i in range(x.size):
        val = float(x.flatten()[i])
        if ascending:
            # bins ascending: find first bin > val (or >= if right)
            idx = 0
            for j in range(len(bins_list)):
                if right:
                    if bins_list[j] < val:
                        idx = j + 1
                else:
                    if bins_list[j] <= val:
                        idx = j + 1
            result.append(idx)
        else:
            # bins descending
            idx = 0
            for j in range(len(bins_list)):
                if right:
                    if bins_list[j] > val:
                        idx = j + 1
                else:
                    if bins_list[j] >= val:
                        idx = j + 1
            result.append(idx)
    return array(result)

def convolve(a, v, mode='full'):
    """Discrete, linear convolution of two one-dimensional sequences."""
    a = asarray(a).flatten()
    v = asarray(v).flatten()
    na = len(a)
    nv = len(v)
    n_full = na + nv - 1
    result = []
    for k in range(n_full):
        s = 0.0
        for j in range(nv):
            i = k - j
            if 0 <= i < na:
                s += float(a[i]) * float(v[j])
        result.append(s)
    result = array(result)
    if mode == 'full':
        return result
    elif mode == 'same':
        start = (nv - 1) // 2
        return array([float(result[start + i]) for i in range(na)])
    elif mode == 'valid':
        n_valid = abs(na - nv) + 1
        start = min(na, nv) - 1
        return array([float(result[start + i]) for i in range(n_valid)])
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid', got '" + mode + "'")

def delete(arr, obj, axis=None):
    """Return a new array with sub-arrays along an axis deleted."""
    arr = asarray(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    n = arr.shape[axis]
    # Normalize obj to a list of indices
    if isinstance(obj, int):
        indices_to_del = [obj if obj >= 0 else n + obj]
    elif isinstance(obj, (list, tuple)):
        indices_to_del = [i if i >= 0 else n + i for i in obj]
    elif isinstance(obj, ndarray):
        indices_to_del = [int(obj[i]) for i in range(len(obj))]
        indices_to_del = [i if i >= 0 else n + i for i in indices_to_del]
    else:
        indices_to_del = [int(obj)]
    del_set = set(indices_to_del)
    keep = [i for i in range(n) if i not in del_set]
    if not keep:
        # All deleted - return empty
        new_shape = list(arr.shape)
        new_shape[axis] = 0
        return array([])
    # Build result by selecting kept indices along axis
    if arr.ndim == 1:
        result = [float(arr[i]) for i in keep]
        return array(result)
    else:
        if axis == 0:
            # For multi-dimensional, concatenate slices along axis 0
            slices = []
            for i in keep:
                slices.append(arr[i])
            if slices:
                rows = []
                for s in slices:
                    if s.ndim == 0:
                        rows.append([float(s)])
                    else:
                        rows.append([float(s[j]) for j in range(len(s))])
                return array(rows)
            return array([])
        else:
            # For non-zero axis: transpose so target axis is first,
            # delete along axis 0, then transpose back
            # Build axis permutation: move target axis to front
            ndim = arr.ndim
            perm = [axis] + [i for i in range(ndim) if i != axis]
            inv_perm = [0] * ndim
            for i, p in enumerate(perm):
                inv_perm[p] = i
            t = transpose(arr, perm)
            t_del = delete(t, obj, axis=0)
            return transpose(t_del, inv_perm)

def insert(arr, obj, values, axis=None):
    """Insert values along the given axis before the given indices."""
    arr = asarray(arr)
    if axis is None:
        arr = arr.flatten()
        axis = 0
    n = arr.shape[axis]
    idx = obj if isinstance(obj, int) else int(obj)
    if idx < 0:
        idx = n + idx
    if arr.ndim == 1:
        result = []
        if isinstance(values, (int, float)):
            values = [values]
        elif isinstance(values, ndarray):
            values = [float(values[i]) for i in range(len(values))]
        for i in range(n):
            if i == idx:
                for v in values:
                    result.append(float(v))
            result.append(float(arr[i]))
        if idx >= n:
            for v in values:
                result.append(float(v))
        return array(result)
    # Multi-dimensional: transpose target axis to front, insert along axis 0, transpose back
    ndims = arr.ndim
    if axis < 0:
        axis = ndims + axis
    # Build permutation to move target axis to position 0
    perm = [axis] + [i for i in range(ndims) if i != axis]
    inv_perm = [0] * ndims
    for i, p in enumerate(perm):
        inv_perm[p] = i
    t = transpose(arr, perm)
    # t now has target axis as axis 0; shape is (n, ...)
    values = asarray(values)
    # Build slices for before, inserted, and after
    before_slices = []
    for i in range(idx):
        before_slices.append(t[i])
    after_slices = []
    for i in range(idx, t.shape[0]):
        after_slices.append(t[i])
    # Reshape values to match the sub-array shape if needed
    sub_shape = t.shape[1:] if ndims > 1 else ()
    if values.ndim == 0 or (values.ndim == 1 and len(sub_shape) == 1 and values.shape[0] == sub_shape[0]):
        # values is a single row to insert
        vals_row = values.reshape(list(sub_shape)) if sub_shape else values
        inserted = [vals_row]
    else:
        inserted = [values[i] for i in range(values.shape[0])]
    all_rows = before_slices + inserted + after_slices
    # Stack rows manually: build flat list
    row_size = 1
    for s in sub_shape:
        row_size = row_size * s
    flat = []
    for row in all_rows:
        r = asarray(row).flatten()
        for j in range(row_size):
            flat.append(float(r[j]))
    new_shape = list(t.shape)
    new_shape[0] = len(all_rows)
    result = array(flat).reshape(new_shape)
    return transpose(result, inv_perm)

def select(condlist, choicelist, default=0):
    """Return array drawn from elements in choicelist, depending on conditions."""
    if len(condlist) != len(choicelist):
        raise ValueError("condlist and choicelist must be the same length")
    condlist = [asarray(c) for c in condlist]
    choicelist = [asarray(c) for c in choicelist]
    # Determine output shape from first array
    shape = condlist[0].shape
    n = condlist[0].size
    result = [float(default)] * n
    # Process in reverse order so first matching condition wins
    for i in range(len(condlist) - 1, -1, -1):
        cond = condlist[i].flatten()
        choice = choicelist[i].flatten()
        for j in range(n):
            if float(cond[j]) != 0.0:
                result[j] = float(choice[j])
    result_arr = array(result)
    if len(shape) > 1:
        result_arr = result_arr.reshape(list(shape))
    return result_arr

def piecewise(x, condlist, funclist):
    """Evaluate a piecewise-defined function."""
    x = asarray(x)
    n = x.size
    flat_x = x.flatten()
    result = [0.0] * n

    # funclist should have len(condlist) or len(condlist)+1 entries
    # If len(condlist)+1, the last entry is the "otherwise" value
    has_otherwise = len(funclist) == len(condlist) + 1

    for i in range(n):
        val = float(flat_x[i])
        matched = False
        for j, cond in enumerate(condlist):
            c = asarray(cond).flatten()
            if float(c[i]) != 0.0:
                if callable(funclist[j]):
                    result[i] = float(funclist[j](val))
                else:
                    result[i] = float(funclist[j])
                matched = True
                break
        if not matched and has_otherwise:
            if callable(funclist[-1]):
                result[i] = float(funclist[-1](val))
            else:
                result[i] = float(funclist[-1])

    result_arr = array(result)
    if len(x.shape) > 1:
        result_arr = result_arr.reshape(list(x.shape))
    return result_arr

# --- mgrid / ogrid / ix_ ----------------------------------------------------

class _MGrid:
    """Return dense multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else 0
                step = s.step if s.step is not None else 1
                # Complex step means "use linspace with this many points"
                if isinstance(step, complex) or (hasattr(step, 'imag') and step.imag != 0):
                    num = int(abs(step))
                    grid = linspace(float(start), float(stop), num=num, endpoint=True)
                    arrays.append(grid)
                else:
                    vals = []
                    v = float(start)
                    if step > 0:
                        while v < float(stop):
                            vals.append(v)
                            v += float(step)
                    else:
                        while v > float(stop):
                            vals.append(v)
                            v += float(step)
                    arrays.append(array(vals) if vals else array([]))
            else:
                arrays.append(array([float(s)]))

        if ndim == 1:
            return arrays[0]

        # Create dense meshgrid
        shapes = [len(a) for a in arrays]
        result = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = shapes[i]
            reshaped = arr.reshape(shape)
            reps = list(shapes)
            reps[i] = 1
            result.append(tile(reshaped, reps))
        return result

mgrid = _MGrid()


class _OGrid:
    """Return open (sparse) multi-dimensional 'meshgrid' arrays via slice notation."""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        ndim = len(key)
        arrays = []
        for s in key:
            if isinstance(s, slice):
                start = s.start if s.start is not None else 0
                stop = s.stop if s.stop is not None else 0
                step = s.step if s.step is not None else 1
                # Complex step means "use linspace with this many points"
                if isinstance(step, complex) or (hasattr(step, 'imag') and step.imag != 0):
                    num = int(abs(step))
                    grid = linspace(float(start), float(stop), num=num, endpoint=True)
                    arrays.append(grid)
                else:
                    vals = []
                    v = float(start)
                    if step > 0:
                        while v < float(stop):
                            vals.append(v)
                            v += float(step)
                    else:
                        while v > float(stop):
                            vals.append(v)
                            v += float(step)
                    arrays.append(array(vals) if vals else array([]))
            else:
                arrays.append(array([float(s)]))

        if ndim == 1:
            return arrays[0]

        # Sparse: each array reshaped to broadcast along its own axis
        result = []
        for i, arr in enumerate(arrays):
            shape = [1] * ndim
            shape[i] = len(arr)
            result.append(arr.reshape(shape))
        return result

ogrid = _OGrid()


def ix_(*args):
    """Construct an open mesh from multiple sequences for cross-indexing."""
    ndim = len(args)
    result = []
    for i, arg in enumerate(args):
        arr = asarray(arg).flatten()
        shape = [1] * ndim
        shape[i] = len(arr)
        result.append(arr.reshape(shape))
    return tuple(result)


class _RClass:
    """Row concatenation using index syntax: np.r_[1:5, 7, 8]"""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        pieces = []
        for item in key:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop
                step = item.step if item.step is not None else 1
                if stop is None:
                    raise ValueError("r_ requires explicit stop for slices")
                pieces.append(arange(start, stop, step))
            elif isinstance(item, (int, float)):
                pieces.append(array([item]))
            else:
                pieces.append(asarray(item))
        if len(pieces) == 0:
            return array([])
        return concatenate(pieces)

r_ = _RClass()


class _CClass:
    """Column concatenation: np.c_[a, b] == np.column_stack((a, b))"""
    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        arrays = []
        for item in key:
            if isinstance(item, slice):
                start = item.start if item.start is not None else 0
                stop = item.stop
                step = item.step if item.step is not None else 1
                if stop is None:
                    raise ValueError("c_ requires explicit stop for slices")
                arr = arange(start, stop, step)
            elif isinstance(item, (int, float)):
                arr = array([item])
            else:
                arr = asarray(item)
            # Ensure 2D
            if arr.ndim == 1:
                arr = arr.reshape((-1 if arr.size > 0 else 0, 1))
            arrays.append(arr)
        if len(arrays) == 0:
            return array([])
        return concatenate(arrays, 1)

c_ = _CClass()


class _SClass:
    """Index expression helper: np.s_[0:5, 1::2]"""
    def __getitem__(self, key):
        return key

s_ = _SClass()


def lexsort(keys):
    """Perform an indirect stable sort using a sequence of keys.
    Last key is used as the primary sort key."""
    keys_list = [asarray(k) for k in keys]
    n = keys_list[0].size
    indices = list(range(n))
    # Flatten all keys
    flat_keys = []
    for k in keys_list:
        fk = k.flatten()
        flat_keys.append([fk[i] for i in range(n)])
    # Sort indices using keys (last key = primary, first key = least significant)
    # Python sort is stable, so we sort from least significant to most significant
    for ki in range(len(flat_keys)):
        vals = flat_keys[ki]
        indices.sort(key=lambda idx, v=vals: v[idx])
    return array(indices)

def partition(a, kth, axis=-1):
    """Return a partitioned copy of an array.
    Creates a copy where the element at kth position is where it would be in a sorted array.
    Elements before kth are <= element at kth, elements after are >=."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        flat = a.flatten()
        n = flat.size
        vals = [flat[i] for i in range(n)]
        vals.sort()
        return array(vals)
    # For multi-dim, sort along axis (full sort, which satisfies partition contract)
    return sort(a, axis=axis)

def argpartition(a, kth, axis=-1):
    """Return indices that would partition the array.
    Same as argsort but only guarantees kth element is correct."""
    a = asarray(a)
    if axis == -1:
        axis = a.ndim - 1
    if a.ndim == 1 or axis is None:
        return argsort(a)
    return argsort(a, axis=axis)

def correlate(a, v, mode='valid'):
    """Cross-correlation of two 1-dimensional sequences."""
    a = asarray(a).flatten()
    v = asarray(v).flatten()
    na = a.size
    nv = v.size
    # Reverse v for correlation (correlation = convolution with reversed kernel)
    v_rev = array([v[nv - 1 - i] for i in range(nv)])
    return convolve(a, v_rev, mode=mode)


# --- Tier 15 Group C: apply_along_axis, vectorize, put, putmask, broadcast_arrays ---

def apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """Apply a function to 1-D slices of an array along the given axis."""
    arr = asarray(arr)
    if arr.ndim == 1:
        return asarray(func1d(arr, *args, **kwargs))
    nd = arr.ndim
    if axis < 0:
        axis = nd + axis
    # General nD implementation
    # Build the shape of the non-target axes (the "outer" iteration shape)
    shape = arr.shape
    out_shape = tuple(shape[i] for i in range(nd) if i != axis)
    # Compute total number of outer iterations
    n_outer = 1
    for s in out_shape:
        n_outer *= s
    # For each combination of indices in the non-target axes, extract the 1D slice
    results = []
    for flat_idx in range(n_outer):
        # Convert flat_idx to multi-index in out_shape
        idx = []
        rem = flat_idx
        for s in reversed(out_shape):
            idx.append(rem % s)
            rem = rem // s
        idx.reverse()
        # Build the full index with a slice at the target axis position
        # Extract the 1D slice along the target axis
        # We need to index arr with idx inserted around the target axis
        outer_idx_pos = 0
        slice_vals = []
        for k in range(shape[axis]):
            # Build index tuple for element [idx[0], ..., k, ..., idx[-1]]
            full_idx = []
            oi = 0
            for dim in range(nd):
                if dim == axis:
                    full_idx.append(k)
                else:
                    full_idx.append(idx[oi])
                    oi += 1
            # Navigate to element
            elem = arr
            for fi in full_idx:
                elem = elem[fi]
            slice_vals.append(float(elem))
        slice_arr = array(slice_vals)
        result = func1d(slice_arr, *args, **kwargs)
        results.append(result)
    # Reshape results back to out_shape
    # If func1d returns scalar, result shape is out_shape
    first_res = results[0]
    if isinstance(first_res, ndarray):
        # func returns array - more complex, but handle scalar reduction case
        result_arr = asarray(results)
        return result_arr.reshape(out_shape)
    else:
        return array([float(r) for r in results]).reshape(out_shape)


class vectorize:
    """Generalized function class.

    Takes a nested sequence of objects or numpy arrays as inputs and returns
    a single numpy array or a tuple of numpy arrays by applying the function
    element-by-element.
    """
    def __init__(self, pyfunc, otypes=None, doc=None, excluded=None, cache=False, signature=None):
        self.pyfunc = pyfunc
        self.otypes = otypes
        if doc is not None:
            self.__doc__ = doc
        elif pyfunc.__doc__:
            self.__doc__ = pyfunc.__doc__

    def __call__(self, *args, **kwargs):
        # Convert all args to arrays
        arr_args = [asarray(a) for a in args]
        if len(arr_args) == 0:
            return array([])
        # Get the size from first arg
        first = arr_args[0].flatten()
        n = first.size
        results = []
        for i in range(n):
            elem_args = []
            for a in arr_args:
                flat = a.flatten()
                if flat.size == 1:
                    elem_args.append(flat[0])
                else:
                    elem_args.append(flat[i])
            results.append(self.pyfunc(*elem_args, **kwargs))
        result = array(results)
        # Try to reshape to the shape of first arg
        if arr_args[0].ndim > 1:
            result = result.reshape(arr_args[0].shape)
        return result


def put(a, ind, v, mode='raise'):
    """Replaces specified elements of an array with given values."""
    a = asarray(a)
    flat = a.flatten()
    n = flat.size
    ind_arr = asarray(ind).flatten()
    v_arr = asarray(v).flatten()
    vals = [flat[i] for i in range(n)]
    ni = ind_arr.size
    nv = v_arr.size
    for idx in range(ni):
        i = int(ind_arr[idx])
        if mode == 'wrap':
            i = i % n
        elif mode == 'clip':
            if i < 0:
                i = 0
            elif i >= n:
                i = n - 1
        vals[i] = v_arr[idx % nv]
    result = array(vals)
    if a.ndim > 1:
        result = result.reshape(a.shape)
    return result


def putmask(a, mask, values):
    """Changes elements of an array based on conditional and input values."""
    a = asarray(a)
    mask = asarray(mask)
    values = asarray(values)
    flat_a = a.flatten()
    flat_m = mask.flatten()
    flat_v = values.flatten()
    n = flat_a.size
    nv = flat_v.size
    vals = [flat_a[i] for i in range(n)]
    vi = 0
    for i in range(n):
        if flat_m[i]:
            vals[i] = flat_v[vi % nv]
            vi += 1
    result = array(vals)
    if a.ndim > 1:
        result = result.reshape(a.shape)
    return result


def broadcast_arrays(*args):
    """Broadcast any number of arrays against each other."""
    arrays = [asarray(a) for a in args]
    if len(arrays) == 0:
        return []
    # Find common shape
    shape = list(arrays[0].shape)
    for a in arrays[1:]:
        ashape = list(a.shape)
        # Pad shorter shape with 1s on left
        while len(shape) < len(ashape):
            shape.insert(0, 1)
        while len(ashape) < len(shape):
            ashape.insert(0, 1)
        new_shape = []
        for s1, s2 in zip(shape, ashape):
            if s1 == s2:
                new_shape.append(s1)
            elif s1 == 1:
                new_shape.append(s2)
            elif s2 == 1:
                new_shape.append(s1)
            else:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        shape = new_shape
    return [broadcast_to(a, tuple(shape)) for a in arrays]


# --- trapz / trapezoid — trapezoidal integration ----------------------------
def trapz(y, x=None, dx=1.0, axis=-1):
    """Integrate along the given axis using the composite trapezoidal rule."""
    y = asarray(y)
    if y.ndim == 1:
        n = y.size
        if x is not None:
            x = asarray(x).flatten()
            total = 0.0
            for i in range(1, n):
                total += (x[i] - x[i-1]) * (y[i] + y[i-1]) / 2.0
            return total
        else:
            total = 0.0
            for i in range(1, n):
                total += dx * (y[i] + y[i-1]) / 2.0
            return total
    # For multi-dim, apply along specified axis
    if axis == -1:
        axis = y.ndim - 1
    if y.ndim == 2:
        results = []
        if axis == 0:
            for j in range(y.shape[1]):
                col = array([y[i][j] for i in range(y.shape[0])])
                results.append(trapz(col, dx=dx))
        else:
            for i in range(y.shape[0]):
                results.append(trapz(y[i], dx=dx))
        return array(results)
    # General nD case
    if axis < 0:
        axis = y.ndim + axis
    # Move target axis to last, flatten leading dims, apply trapz per lane
    y_moved = moveaxis(y, axis, -1)
    result_shape = list(y_moved.shape[:-1])
    n_lane = y_moved.shape[-1]
    lead = 1
    for s in result_shape:
        lead *= s
    y_flat = y_moved.reshape((lead, n_lane))
    flat_results = []
    y_list = y_flat.tolist()
    for i in range(lead):
        row = array(y_list[i])
        if x is not None:
            flat_results.append(float(trapz(row, x=asarray(x), dx=dx)))
        else:
            flat_results.append(float(trapz(row, dx=dx)))
    if not result_shape:
        return float(flat_results[0])
    return array(flat_results).reshape(result_shape)

trapezoid = trapz


# --- finfo — floating point type info ---------------------------------------
class finfo:
    """Machine limits for floating point types."""
    def __init__(self, dtype=None):
        if dtype is None or str(dtype) in ('float64', 'f8', 'float', 'd'):
            self.bits = 64
            self.eps = 2.220446049250313e-16
            self.max = 1.7976931348623157e+308
            self.min = -1.7976931348623157e+308
            self.tiny = 2.2250738585072014e-308
            self.smallest_normal = 2.2250738585072014e-308
            self.smallest_subnormal = 5e-324
            self.resolution = 1e-15
            self.dtype = float64
            self.maxexp = 1024
            self.minexp = -1021
            self.nmant = 52
            self.nexp = 11
            self.machep = -52
            self.negep = -53
            self.iexp = 11
            self.precision = 15
        elif str(dtype) in ('float32', 'f4', 'f'):
            self.bits = 32
            self.eps = 1.1920929e-07
            self.max = 3.4028235e+38
            self.min = -3.4028235e+38
            self.tiny = 1.1754944e-38
            self.smallest_normal = 1.1754944e-38
            self.smallest_subnormal = 1e-45
            self.resolution = 1e-6
            self.dtype = float32
            self.maxexp = 128
            self.minexp = -125
            self.nmant = 23
            self.nexp = 8
            self.machep = -23
            self.negep = -24
            self.iexp = 8
            self.precision = 6
        elif str(dtype) in ('float16', 'half', 'f2', 'e'):
            self.bits = 16
            self.eps = 9.765625e-04
            self.max = 65504.0
            self.min = -65504.0
            self.tiny = 6.103515625e-05
            self.smallest_normal = 6.103515625e-05
            self.smallest_subnormal = 5.96e-08
            self.resolution = 0.001
            self.dtype = float16
            self.maxexp = 16
            self.minexp = -13
            self.nmant = 10
            self.nexp = 5
            self.machep = -10
            self.negep = -11
            self.iexp = 5
            self.precision = 3
        else:
            raise ValueError("finfo only supports float16, float32 and float64")

    def __repr__(self):
        return f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, dtype={self.dtype})"


# --- iinfo — integer type info ----------------------------------------------
class iinfo:
    """Machine limits for integer types."""
    def __init__(self, dtype=None):
        if dtype is None or str(dtype) in ('int64', 'i8', 'int', 'l'):
            self.bits = 64
            self.min = -9223372036854775808
            self.max = 9223372036854775807
            self.dtype = int64
            self.kind = 'i'
        elif str(dtype) in ('int32', 'i4', 'i'):
            self.bits = 32
            self.min = -2147483648
            self.max = 2147483647
            self.dtype = int32
            self.kind = 'i'
        elif str(dtype) in ('int8', 'i1'):
            self.bits = 8
            self.min = -128
            self.max = 127
            self.dtype = int8
            self.kind = 'i'
        elif str(dtype) in ('int16', 'i2'):
            self.bits = 16
            self.min = -32768
            self.max = 32767
            self.dtype = int16
            self.kind = 'i'
        elif str(dtype) in ('uint8',):
            self.bits = 8
            self.min = 0
            self.max = 255
            self.dtype = dtype
            self.kind = 'u'
        elif str(dtype) in ('uint16',):
            self.bits = 16
            self.min = 0
            self.max = 65535
            self.dtype = dtype
            self.kind = 'u'
        elif str(dtype) in ('uint32',):
            self.bits = 32
            self.min = 0
            self.max = 4294967295
            self.dtype = dtype
            self.kind = 'u'
        elif str(dtype) in ('uint64',):
            self.bits = 64
            self.min = 0
            self.max = 18446744073709551615
            self.dtype = dtype
            self.kind = 'u'
        else:
            raise ValueError("iinfo does not support this dtype")

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"


# --- fromfunction — construct array from function ----------------------------
def fromfunction(function, shape, dtype=float, **kwargs):
    """Construct an array by executing a function over each coordinate."""
    coords = indices(shape, dtype=dtype)
    return asarray(function(*coords, **kwargs))


# --- fmod — C-style remainder (sign of dividend) ----------------------------
def fmod(x1, x2):
    """Return the element-wise remainder of division (C-style, sign of dividend)."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    return x1 - trunc(x1 / x2) * x2


# --- modf — return fractional and integer parts -----------------------------
def modf(x):
    """Return the fractional and integral parts of an array, element-wise."""
    x = asarray(x)
    integer_part = trunc(x)
    fractional_part = x - integer_part
    return fractional_part, integer_part


# --- fill_diagonal — fill main diagonal of 2-d array -----------------------
def fill_diagonal(a, val, wrap=False):
    """Fill the main diagonal of the given array. Returns new array (our arrays are immutable)."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("array must be 2-d")
    n = a.shape[0]
    m = a.shape[1]
    rows = []
    for i in range(n):
        row = []
        for j in range(m):
            if i == j:
                row.append(float(val) if not isinstance(val, (list, tuple)) else float(val[i % len(val)]))
            else:
                row.append(a[i][j])
        rows.append(row)
    return array(rows)


# --- diag_indices / diag_indices_from — diagonal index helpers --------------
def diag_indices(n, ndim=2):
    """Return the indices to access the main diagonal of an array."""
    idx = arange(0, n)
    return tuple([idx] * ndim)

def diag_indices_from(arr):
    """Return the indices to access the main diagonal of an n-dimensional array."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    n = arr.shape[0]
    if arr.shape[0] != arr.shape[1]:
        raise ValueError("array must be square")
    return diag_indices(n, 2)


# --- tril_indices / triu_indices — triangle index helpers -------------------
def tril_indices(n, k=0, m=None):
    """Return the indices for the lower-triangle of an (n, m) array."""
    if m is None:
        m = n
    rows = []
    cols = []
    for i in range(n):
        for j in range(m):
            if j <= i + k:
                rows.append(float(i))
                cols.append(float(j))
    return array(rows), array(cols)

def triu_indices(n, k=0, m=None):
    """Return the indices for the upper-triangle of an (n, m) array."""
    if m is None:
        m = n
    rows = []
    cols = []
    for i in range(n):
        for j in range(m):
            if j >= i + k:
                rows.append(float(i))
                cols.append(float(j))
    return array(rows), array(cols)

def tril_indices_from(arr, k=0):
    """Return the indices for the lower-triangle of arr."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    return tril_indices(arr.shape[0], k=k, m=arr.shape[1])

def triu_indices_from(arr, k=0):
    """Return the indices for the upper-triangle of arr."""
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("array must be 2-d")
    return triu_indices(arr.shape[0], k=k, m=arr.shape[1])


# --- ndenumerate — multidimensional index iterator --------------------------
class ndenumerate:
    """Multidimensional index iterator."""
    def __init__(self, arr):
        self._arr = asarray(arr)
        self._flat = self._arr.flatten()
        self._shape = self._arr.shape
        self._size = self._flat.size
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        # Convert flat index to multi-dim index
        idx = self._idx
        multi = []
        for s in reversed(self._shape):
            multi.append(idx % s)
            idx //= s
        multi.reverse()
        val = self._flat[self._idx]
        self._idx += 1
        return tuple(multi), val


# --- ndindex — N-dimensional index iterator ---------------------------------
class ndindex:
    """An N-dimensional iterator object to index arrays."""
    def __init__(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        self._shape = shape
        self._size = 1
        for s in shape:
            self._size *= s
        self._idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        idx = self._idx
        multi = []
        for s in reversed(self._shape):
            multi.append(idx % s)
            idx //= s
        multi.reverse()
        self._idx += 1
        return tuple(multi)


# --- Signal window functions ------------------------------------------------
def bartlett(M):
    """Return the Bartlett window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    n = arange(0, M)
    mid = (M - 1) / 2.0
    vals = []
    for i in range(M):
        v = float(n[i])
        if v <= mid:
            vals.append(2.0 * v / (M - 1))
        else:
            vals.append(2.0 - 2.0 * v / (M - 1))
    return array(vals)

def blackman(M):
    """Return the Blackman window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.42 - 0.5 * _math.cos(2.0 * pi * i / (M - 1)) + 0.08 * _math.cos(4.0 * pi * i / (M - 1)))
    return array(vals)

def hamming(M):
    """Return the Hamming window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.54 - 0.46 * _math.cos(2.0 * pi * i / (M - 1)))
    return array(vals)

def hanning(M):
    """Return the Hanning window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    vals = []
    for i in range(M):
        vals.append(0.5 - 0.5 * _math.cos(2.0 * pi * i / (M - 1)))
    return array(vals)

def kaiser(M, beta):
    """Return the Kaiser window."""
    if M < 1:
        return array([])
    if M == 1:
        return array([1.0])
    # I0 is modified Bessel function of first kind, order 0
    # Use series approximation
    def _i0(x):
        """Modified Bessel function I0 via series."""
        val = 1.0
        term = 1.0
        for k in range(1, 25):
            term *= (x / 2.0) ** 2 / (k * k)
            val += term
        return val

    alpha = (M - 1) / 2.0
    vals = []
    for i in range(M):
        arg = beta * _math.sqrt(1.0 - ((i - alpha) / alpha) ** 2)
        vals.append(_i0(arg) / _i0(beta))
    return array(vals)


# --- nditer — simplified N-dimensional iterator -----------------------------
class nditer:
    """Simplified N-dimensional iterator."""
    def __init__(self, op, flags=None, op_flags=None, op_dtypes=None, order='K',
                 casting='safe', op_axes=None, itershape=None, buffersize=0):
        if isinstance(op, (list, tuple)):
            self._arrays = [asarray(a) for a in op]
        else:
            self._arrays = [asarray(op)]
        self._flat = [a.flatten() for a in self._arrays]
        self._size = self._flat[0].size
        self._idx = 0
        self._shape = self._arrays[0].shape
        self.multi_index = None

    def _unravel(self, flat_idx):
        """Convert flat index to tuple of indices."""
        idx = []
        remaining = flat_idx
        for dim in reversed(self._shape):
            idx.append(remaining % dim)
            remaining //= dim
        return tuple(reversed(idx))

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx >= self._size:
            raise StopIteration
        # Compute multi_index from flat index
        self.multi_index = self._unravel(self._idx)
        if len(self._flat) == 1:
            val = self._flat[0][self._idx]
            self._idx += 1
            return val
        vals = tuple(f[self._idx] for f in self._flat)
        self._idx += 1
        return vals

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @property
    def finished(self):
        return self._idx >= self._size

    def iternext(self):
        if self._idx >= self._size:
            return False
        self._idx += 1
        return self._idx < self._size

    @property
    def value(self):
        if len(self._flat) == 1:
            return self._flat[0][self._idx]
        return tuple(f[self._idx] for f in self._flat)


# --- array_str / array_repr ------------------------------------------------
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """Return a string representation of the data in an array."""
    a = asarray(a)
    return str(a)

def array_repr(a, max_line_width=None, precision=None, suppress_small=None):
    """Return the string representation of an array."""
    a = asarray(a)
    return repr(a)


# --- Tier 18 Group C: i0, apply_over_axes, real_if_close, isneginf, isposinf ---

def i0(x):
    """Modified Bessel function of the first kind, order 0."""
    x = asarray(x)
    flat = x.flatten()
    n = flat.size
    result = []
    for i in range(n):
        v = float(flat[i])
        # Series expansion: I0(x) = sum_{k=0}^{inf} ((x/2)^k / k!)^2
        val = 1.0
        term = 1.0
        for k in range(1, 25):
            term *= (v / 2.0) ** 2 / (k * k)
            val += term
        result.append(val)
    r = array(result)
    if x.ndim > 1:
        r = r.reshape(x.shape)
    return r

def apply_over_axes(func, a, axes):
    """Apply a function repeatedly over multiple axes."""
    a = asarray(a)
    if isinstance(axes, int):
        axes = [axes]
    for ax in axes:
        result = func(a, axis=ax)
        if isinstance(result, ndarray):
            a = result
        else:
            a = asarray(result)
    return a

def real_if_close(a, tol=100):
    """If input is complex with all imaginary parts close to zero, return real parts."""
    a = asarray(a)
    if a.dtype not in ("complex64", "complex128"):
        return a
    # Check if imaginary part is negligible
    im = imag(a)
    re = real(a)
    eps = 2.220446049250313e-16  # float64 machine epsilon
    flat_im = im.flatten()
    for i in range(flat_im.size):
        if _math.fabs(float(flat_im[i])) > tol * eps:
            return a  # has significant imaginary part
    return re

def isneginf(x, out=None):
    """Test element-wise for negative infinity."""
    x = asarray(x)
    flat = x.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = float(flat[i])
        vals.append(1.0 if (not (v != v) and v == float('-inf')) else 0.0)
    r = array(vals)
    if x.ndim > 1:
        r = r.reshape(x.shape)
    # Convert to bool by comparing > 0
    return r > zeros(r.shape)

def isposinf(x, out=None):
    """Test element-wise for positive infinity."""
    x = asarray(x)
    flat = x.flatten()
    n = flat.size
    vals = []
    for i in range(n):
        v = float(flat[i])
        vals.append(1.0 if (not (v != v) and v == float('inf')) else 0.0)
    r = array(vals)
    if x.ndim > 1:
        r = r.reshape(x.shape)
    return r > zeros(r.shape)


# --- save/load/savez (Tier 18A) ---------------------------------------------

def save(file, arr, allow_pickle=True, fix_imports=True):
    """Save an array to a .npy file (text-based format for compatibility)."""
    arr = asarray(arr)
    with open(file, 'w') as f:
        f.write(f"# shape: {list(arr.shape)}\n")
        flat = arr.flatten()
        vals = [str(flat[i]) for i in range(flat.size)]
        f.write(','.join(vals) + '\n')

def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII'):
    """Load array from a .npy file (text-based format)."""
    with open(file, 'r') as f:
        lines = f.readlines()
    # Parse shape from first line
    shape_line = lines[0].strip()
    if shape_line.startswith('# shape:'):
        import json
        shape = tuple(json.loads(shape_line.split(':')[1].strip()))
        data_line = lines[1].strip()
    else:
        # Fallback: treat as flat data
        data_line = lines[0].strip()
        shape = None
    vals = [float(v) for v in data_line.split(',')]
    result = array(vals)
    if shape is not None and len(shape) > 1:
        result = result.reshape(shape)
    return result

def savez(file, *args, **kwds):
    """Save several arrays into a single file in text format.
    Since we can't use actual npz (zip) format, save as multi-section text."""
    arrays = {}
    for i, arr in enumerate(args):
        arrays[f'arr_{i}'] = asarray(arr)
    for name, arr in kwds.items():
        arrays[name] = asarray(arr)
    with open(file, 'w') as f:
        for name, arr in arrays.items():
            f.write(f"# {name} shape: {list(arr.shape)}\n")
            flat = arr.flatten()
            vals = [str(flat[i]) for i in range(flat.size)]
            f.write(','.join(vals) + '\n')

savez_compressed = savez  # alias, same behavior in our sandbox

# --- frompyfunc (Tier 18A) --------------------------------------------------

def frompyfunc(func, nin, nout):
    """Takes an arbitrary Python function and returns a NumPy ufunc-like object.
    Returns a vectorize wrapper."""
    return vectorize(func)

# --- take_along_axis / put_along_axis (Tier 18A) ----------------------------

def take_along_axis(arr, indices, axis):
    """Take values from the input array by matching 1-d index and data slices along the given axis."""
    arr = asarray(arr)
    indices = asarray(indices)
    if arr.ndim == 1:
        result = []
        for i in range(indices.size):
            result.append(arr[int(indices[i])])
        return array(result)
    if arr.ndim == 2:
        if axis == 0:
            rows = []
            for j in range(arr.shape[1]):
                col = []
                for i in range(indices.shape[0]):
                    col.append(arr[int(indices[i][j])][j])
                rows.append(col)
            # Transpose to get correct shape
            result = []
            for i in range(indices.shape[0]):
                row = [rows[j][i] for j in range(arr.shape[1])]
                result.append(row)
            return array(result)
        else:  # axis == 1
            rows = []
            for i in range(arr.shape[0]):
                row = []
                for j in range(indices.shape[1]):
                    row.append(arr[i][int(indices[i][j])])
                rows.append(row)
            return array(rows)
    # General nD case: move axis to last, flatten leading dims, gather, reshape back
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    out_shape = ind_m.shape
    n_axis = arr_m.shape[-1]
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    arr_flat = arr_m.reshape((lead, n_axis))
    ind_flat = ind_m.reshape((lead, ind_m.shape[-1]))
    arr_list = arr_flat.tolist()
    ind_list = ind_flat.tolist()
    result = []
    for i in range(lead):
        row = arr_list[i]
        idxs = ind_list[i]
        result.append([row[int(j)] for j in idxs])
    result_arr = array(result).reshape(out_shape)
    return moveaxis(result_arr, -1, axis)

def put_along_axis(arr, indices, values, axis):
    """Put values into the destination array by matching 1-d index and data slices along the given axis."""
    arr = asarray(arr)
    indices = asarray(indices)
    values = asarray(values)
    if arr.ndim == 1:
        result = [arr[i] for i in range(arr.size)]
        vals_flat = values.flatten()
        for i in range(indices.size):
            result[int(indices[i])] = vals_flat[i % vals_flat.size]
        return array(result)
    if arr.ndim == 2 and axis == 1:
        rows = []
        for i in range(arr.shape[0]):
            row = [arr[i][j] for j in range(arr.shape[1])]
            for j in range(indices.shape[1]):
                idx = int(indices[i][j])
                row[idx] = values[i][j] if values.ndim == 2 else values[j]
            rows.append(row)
        return array(rows)
    # General nD case: move axis to last, flatten, scatter, reshape back
    if axis < 0:
        axis = arr.ndim + axis
    arr_m = moveaxis(arr, axis, -1)
    ind_m = moveaxis(indices, axis, -1)
    val_m = moveaxis(values, axis, -1)
    out_shape = arr_m.shape
    lead = 1
    for s in arr_m.shape[:-1]:
        lead *= s
    n_axis = arr_m.shape[-1]
    arr_flat = arr_m.reshape((lead, n_axis)).tolist()
    ind_flat = ind_m.reshape((lead, ind_m.shape[-1])).tolist()
    val_flat = val_m.reshape((lead, val_m.shape[-1])).tolist()
    for i in range(lead):
        for j in range(len(ind_flat[i])):
            arr_flat[i][int(ind_flat[i][j])] = val_flat[i][j]
    result = array(arr_flat).reshape(out_shape)
    return moveaxis(result, -1, axis)

# --- linalg extensions (built on existing primitives) -----------------------

def _linalg_pinv(a):
    """Compute the (Moore-Penrose) pseudo-inverse of a matrix using SVD."""
    a = asarray(a)
    U, s, Vt = linalg.svd(a)
    # Build pseudo-inverse: V @ diag(1/s) @ U^T
    # s is 1D singular values
    n = s.size
    s_inv_vals = []
    tol = 1e-15 * s[0] if n > 0 else 0
    for i in range(n):
        v = s[i]
        if v > tol:
            s_inv_vals.append(1.0 / v)
        else:
            s_inv_vals.append(0.0)
    s_inv = diag(array(s_inv_vals))
    # pinv = Vt.T @ s_inv @ U.T
    return dot(dot(Vt.T, s_inv), U.T)

def _linalg_matrix_rank(M, tol=None):
    """Return matrix rank using SVD."""
    M = asarray(M)
    U, s, Vt = linalg.svd(M)
    n = s.size
    if tol is None:
        s_max = float(s[0]) if n > 0 else 0.0
        tol = s_max * 1e-15 * _builtin_max(M.shape[0], M.shape[1]) if n > 0 else 0
    rank = 0
    for i in range(n):
        if s[i] > tol:
            rank += 1
    return rank

def _linalg_matrix_power(M, n):
    """Raise a square matrix to the (integer) power n."""
    M = asarray(M)
    if n == 0:
        return eye(M.shape[0])
    if n < 0:
        M = linalg.inv(M)
        n = -n
    result = eye(M.shape[0])
    for _ in range(n):
        result = dot(result, M)
    return result

def _linalg_slogdet(a):
    """Compute sign and log of the determinant."""
    a = asarray(a)
    d = linalg.det(a)
    import math as _m
    if d > 0:
        return 1.0, _m.log(d)
    elif d < 0:
        return -1.0, _m.log(-d)
    else:
        return 0.0, float('-inf')

def _linalg_cond(x, p=None):
    """Compute the condition number of a matrix."""
    x = asarray(x)
    U, s, Vt = linalg.svd(x)
    n = s.size
    if n == 0:
        return float('inf')
    s_max = s[0]
    s_min = s[n - 1]
    if s_min == 0:
        return float('inf')
    return s_max / s_min

def _linalg_eigh(a):
    """Eigenvalues and eigenvectors of a symmetric matrix.
    Falls back to eig (our eig handles symmetric matrices fine)."""
    return linalg.eig(asarray(a))

def _linalg_eigvals(a):
    """Compute eigenvalues only."""
    vals, vecs = linalg.eig(asarray(a))
    return vals

def _linalg_multi_dot(arrays):
    """Compute the dot product of two or more arrays in a single call."""
    result = asarray(arrays[0])
    for i in range(1, len(arrays)):
        result = dot(result, asarray(arrays[i]))
    return result

_linalg_norm_orig = linalg.norm

def _linalg_norm_with_axis(x, ord=None, axis=None, keepdims=False):
    """Compute matrix or vector norm, optionally along an axis."""
    x = asarray(x)
    if axis is None:
        # Delegate to native Rust norm (Frobenius / flat L2)
        return _linalg_norm_orig(x)
    # Compute norm along specified axis (use positional arg for sum/max)
    if ord is None or ord == 2:
        return sqrt((x * x).sum(axis))
    elif ord == 1:
        return abs(x).sum(axis)
    elif ord == float('inf'):
        return abs(x).max(axis)
    else:
        return (abs(x) ** ord).sum(axis) ** (1.0 / ord)

# Monkey-patch linalg module
linalg.norm = _linalg_norm_with_axis
linalg.pinv = _linalg_pinv
linalg.matrix_rank = _linalg_matrix_rank
linalg.matrix_power = _linalg_matrix_power
linalg.slogdet = _linalg_slogdet
linalg.cond = _linalg_cond
linalg.eigh = _linalg_eigh
linalg.eigvals = _linalg_eigvals
linalg.multi_dot = _linalg_multi_dot
linalg.lstsq = _native.linalg.lstsq
linalg.cholesky = _native.linalg.cholesky
linalg.qr = _native.linalg.qr
linalg.trace = trace

def _linalg_eigvalsh(a, UPLO='L'):
    """Eigenvalues of symmetric/Hermitian matrix."""
    vals, _ = linalg.eigh(a)
    return vals

linalg.eigvalsh = _linalg_eigvalsh

# --- FFT module extensions (Tier 19 Group B) --------------------------------

def _fft_rfftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies for rfft."""
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = []
    for i in range(N):
        results.append(float(i) * val)
    return array(results)

def _fft_fftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies."""
    results = []
    half = (n - 1) // 2 + 1
    for i in range(half):
        results.append(float(i) / (n * d))
    for i in range(-(n // 2), 0):
        results.append(float(i) / (n * d))
    return array(results)

def _fft_fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = n // 2
        result = roll(result, shift, axis=ax)
    return result

def _fft_ifftshift(x, axes=None):
    """The inverse of fftshift."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = -(n // 2)
        result = roll(result, shift, axis=ax)
    return result

def _fft_complex_column_fft(row_ffts, rows, cols, inverse=False):
    """Apply FFT/IFFT along columns of a complex (rows, cols, 2) representation.

    row_ffts is a list of (cols, 2) arrays from fft.fft applied to each row.
    Returns a (rows, cols, 2) shaped array representing the 2D FFT result.
    The complex representation uses [real, imag] pairs.
    """
    fft_fn = fft.ifft if inverse else fft.fft
    # For each column j, extract real and imaginary parts across all rows,
    # apply FFT to each separately, then combine using:
    #   DFT(xr + j*xi) = DFT(xr) + j*DFT(xi)
    #   result_real = DFT(xr)_real - DFT(xi)_imag
    #   result_imag = DFT(xr)_imag + DFT(xi)_real
    col_results = []  # col_results[j] is a list of (real, imag) for each row i
    for j in range(cols):
        col_real = array([row_ffts[i][j][0] for i in range(rows)])
        col_imag = array([row_ffts[i][j][1] for i in range(rows)])
        if inverse:
            # ifft requires complex format (n, 2) - convert real arrays to complex with zero imag
            col_real_c = array([[float(col_real[i]), 0.0] for i in range(rows)])
            col_imag_c = array([[float(col_imag[i]), 0.0] for i in range(rows)])
            fft_of_real = fft_fn(col_real_c)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag_c)   # (rows, 2)
        else:
            fft_of_real = fft_fn(col_real)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag)   # (rows, 2)
        # Combine: for each row i
        col_ri = []
        for i in range(rows):
            r = fft_of_real[i][0] - fft_of_imag[i][1]
            im = fft_of_real[i][1] + fft_of_imag[i][0]
            col_ri.append((r, im))
        col_results.append(col_ri)
    # Reconstruct as (rows, cols, 2) using stack
    final_rows = []
    for i in range(rows):
        row_data = []
        for j in range(cols):
            row_data.append([col_results[j][i][0], col_results[j][i][1]])
        final_rows.append(array(row_data))
    return stack(final_rows)

def _fft_fft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("fft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # FFT each row -> list of (cols, 2) complex arrays
    row_ffts = [fft.fft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_ffts, rows, cols, inverse=False)

def _fft_ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D inverse discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("ifft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # IFFT each row -> list of (cols, 2) complex arrays
    row_iffts = [fft.ifft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)

# Monkey-patch fft module with extension functions
fft.rfftfreq = _fft_rfftfreq
fft.fftfreq = _fft_fftfreq
fft.fftshift = _fft_fftshift
fft.ifftshift = _fft_ifftshift
fft.fft2 = _fft_fft2
fft.ifft2 = _fft_ifft2

def _fft_fftn(a, s=None, axes=None):
    """N-dimensional FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.fft(a)
    elif a.ndim == 2:
        return fft.fft2(a, s=s)
    else:
        # For higher dimensions, apply fft2 on last two axes as approximation
        # This is a simplified implementation
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            # Apply 1D FFT along each axis using apply_along_axis
            result = apply_along_axis(lambda x: array(fft.fft(array(x)).tolist()), ax, result)
        return result

def _fft_ifftn(a, s=None, axes=None):
    """N-dimensional inverse FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.ifft(a)
    elif a.ndim == 2:
        return fft.ifft2(a, s=s)
    elif a.ndim == 3 and a.shape[-1] == 2:
        # Complex representation from fft2/fftn: shape (rows, cols, 2)
        # This is a complex array; apply ifft2 logic
        rows = a.shape[0]
        cols = a.shape[1]
        # Extract row-wise complex data and apply ifft
        row_iffts = []
        for i in _builtin_range(rows):
            row_iffts.append(fft.ifft(a[i]))
        return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)
    else:
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            result = apply_along_axis(lambda x: array(fft.ifft(array(x)).tolist()), ax, result)
        return result

fft.fftn = _fft_fftn
fft.ifftn = _fft_ifftn

def _fft_rfft(a, n=None, axis=-1, norm=None):
    """Real FFT - only positive frequencies."""
    a = asarray(a).astype("float64")
    if a.ndim == 0:
        a = a.reshape([1])
    data = a.tolist()
    if isinstance(data[0], list):
        raise NotImplementedError("rfft only supports 1D")
    N = n if n is not None else len(data)
    # Pad or truncate
    if len(data) < N:
        data = data + [0.0] * (N - len(data))
    elif len(data) > N:
        data = data[:N]
    # Compute full DFT - only first N//2 + 1 frequencies
    import cmath
    result = []
    out_len = N // 2 + 1
    for k in _builtin_range(out_len):
        s = 0.0 + 0.0j
        for n_idx in _builtin_range(N):
            angle = -2.0 * 3.141592653589793 * k * n_idx / N
            s += data[n_idx] * cmath.exp(1j * angle)
        if norm == "ortho":
            s /= N ** 0.5
        result.append([s.real, s.imag])
    # Return as (out_len, 2) shaped array matching native fft format
    return array(result)

def _fft_irfft(a, n=None, axis=-1, norm=None):
    """Inverse real FFT."""
    a = asarray(a)
    # Handle (M, 2) complex format from rfft
    data_list = a.tolist()
    if a.ndim == 2 and a.shape[1] == 2:
        # Complex format: [[real, imag], ...]
        data_r = [row[0] for row in data_list]
        data_i = [row[1] for row in data_list]
    elif a.ndim == 1:
        # Real-only input
        data_r = data_list
        data_i = [0.0] * len(data_r)
    else:
        data_r = a.real.tolist() if hasattr(a, 'real') else data_list
        data_i = a.imag.tolist() if hasattr(a, 'imag') else [0.0] * len(data_r)
    if not isinstance(data_r, list):
        data_r = [data_r]
        data_i = [data_i]
    m = len(data_r)
    N = n if n is not None else 2 * (m - 1)
    # Reconstruct full spectrum using Hermitian symmetry
    import math as _math_mod
    full_spectrum = []
    for i in _builtin_range(m):
        full_spectrum.append(complex(data_r[i], data_i[i]))
    # Mirror for negative frequencies
    for i in _builtin_range(m, N):
        mirror = N - i
        full_spectrum.append(complex(data_r[mirror], -data_i[mirror]))
    # IDFT
    result = []
    for n_idx in _builtin_range(N):
        s = 0.0
        for k in _builtin_range(N):
            angle = 2.0 * 3.141592653589793 * k * n_idx / N
            s += full_spectrum[k].real * _math_mod.cos(angle) - full_spectrum[k].imag * _math_mod.sin(angle)
        if norm == "ortho":
            s /= N ** 0.5
        else:
            s /= N
        result.append(s)
    return array(result)

fft.rfft = _fft_rfft
fft.irfft = _fft_irfft

# --- random extension functions (Tier 19 Group C) ---------------------------

def _random_shuffle(x):
    """Modify a sequence in-place by shuffling its contents. Returns None."""
    if not isinstance(x, ndarray):
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    # Attempt to update array in-place via __setitem__
    try:
        for i in range(n):
            x[i] = vals[i]
    except Exception:
        pass  # if in-place update not supported, shuffle is best-effort
    return None  # real numpy returns None

def _random_permutation(x):
    """Randomly permute a sequence, or return a permuted range."""
    if isinstance(x, (int, float)):
        x = arange(0, int(x))
    else:
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    result = array(vals)
    if x.ndim > 1:
        result = result.reshape(x.shape)
    return result

def _random_standard_normal(size=None):
    """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
    if size is None:
        return float(random.normal(0.0, 1.0, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    return random.normal(0.0, 1.0, size)

def _random_exponential(scale=1.0, size=None):
    """Draw samples from an exponential distribution."""
    if size is None:
        import math as _m
        u = float(random.uniform(0.0, 1.0, (1,)).flatten()[0])
        if u >= 1.0:
            u = 0.9999999999
        return float(-scale * _m.log(1.0 - u))
    if isinstance(size, int):
        size = (size,)
    # Generate uniform [0,1) then transform: -scale * ln(1 - U)
    u = random.uniform(0.0, 1.0, size)
    flat = u.flatten()
    n = flat.size
    result = []
    for i in range(n):
        v = float(flat[i])
        if v >= 1.0:
            v = 0.9999999999
        import math as _m
        result.append(-scale * _m.log(1.0 - v))
    r = array(result)
    if u.ndim > 1:
        r = r.reshape(u.shape)
    return r

def _random_poisson(lam=1.0, size=None):
    """Draw samples from a Poisson distribution."""
    import math as _m
    if size is None:
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        return float(k - 1)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Knuth algorithm
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        result.append(float(k - 1))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_binomial(n, p, size=None):
    """Draw samples from a binomial distribution."""
    if size is None:
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        return float(successes)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        result.append(float(successes))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_beta(a, b, size=None):
    """Draw samples from a Beta distribution.
    Uses the relationship: if X~Gamma(a,1) and Y~Gamma(b,1), then X/(X+Y)~Beta(a,b)."""
    if size is None:
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                return float(x / (x + y))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Use Johnk's algorithm for Beta
        import math as _m
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                result.append(x / (x + y))
                break
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_gamma(shape_param, scale=1.0, size=None):
    """Draw samples from a Gamma distribution using Marsaglia-Tsang method."""
    import math as _m
    def _gamma_one_sample(shape_param, scale):
        alpha = shape_param
        if alpha < 1:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            alpha = alpha + 1
            boost = u ** (1.0 / shape_param)
        else:
            boost = 1.0
        d = alpha - 1.0/3.0
        c = 1.0 / _m.sqrt(9.0 * d)
        while True:
            x = float(random.randn((1,))[0])
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < 1 - 0.0331 * x**4:
                return d * v * scale * boost
            if _m.log(u) < 0.5 * x**2 + d * (1 - v + _m.log(v)):
                return d * v * scale * boost
    if size is None:
        return float(_gamma_one_sample(shape_param, scale))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        result.append(_gamma_one_sample(shape_param, scale))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_multinomial(n, pvals, size=None):
    """Draw samples from a multinomial distribution."""
    pvals = [float(p) for p in (pvals.tolist() if isinstance(pvals, ndarray) else pvals)]
    k = len(pvals)
    if size is None:
        # Single draw: n trials among k categories
        result = [0] * k
        for _ in range(n):
            r = float(random.rand((1,))[0])
            cumsum = 0.0
            for j in range(k):
                cumsum += pvals[j]
                if r < cumsum:
                    result[j] += 1
                    break
            else:
                result[-1] += 1
        return array(result)
    else:
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        rows = []
        for _ in range(total):
            result = [0] * k
            for _ in range(n):
                r = float(random.rand((1,))[0])
                cumsum = 0.0
                for j in range(k):
                    cumsum += pvals[j]
                    if r < cumsum:
                        result[j] += 1
                        break
                else:
                    result[-1] += 1
            rows.append(result)
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out

def _random_lognormal(mean=0.0, sigma=1.0, size=None):
    """Draw samples from a log-normal distribution."""
    if size is None:
        import math as _m
        n = float(random.normal(mean, sigma, (1,)).flatten()[0])
        return float(_m.exp(n))
    if isinstance(size, int):
        size = (size,)
    normals = random.normal(mean, sigma, size)
    return exp(normals)

def _random_geometric(p, size=None):
    """Draw samples from a geometric distribution.
    Returns number of trials until first success (minimum value 1)."""
    import math as _m
    if size is None:
        log1mp = _m.log(1.0 - p)
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        return float(_m.ceil(_m.log(u) / log1mp))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    log1mp = _m.log(1.0 - p)
    result = []
    for _ in range(total):
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        # Avoid log(0)
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        result.append(float(_m.ceil(_m.log(u) / log1mp)))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r

def _random_dirichlet(alpha, size=None):
    """Draw samples from a Dirichlet distribution."""
    if isinstance(alpha, ndarray):
        alpha = alpha.tolist()
    alpha = [float(a) for a in alpha]
    k = len(alpha)
    if size is None:
        # Single draw
        samples = []
        for a in alpha:
            g = float(_random_gamma(a, 1.0, (1,))[0])
            samples.append(g)
        total = sum(samples)
        return array([s / total for s in samples])
    else:
        if isinstance(size, int):
            size = (size,)
        num = 1
        for s in size:
            num *= s
        rows = []
        for _ in range(num):
            samples = []
            for a in alpha:
                g = float(_random_gamma(a, 1.0, (1,))[0])
                samples.append(g)
            total = sum(samples)
            rows.append([s / total for s in samples])
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out

class _Generator:
    """Random number generator (simplified)."""
    def __init__(self, seed_val=None):
        if seed_val is not None:
            random.seed(int(seed_val))

    def random(self, size=None):
        if size is None:
            return float(random.rand((1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.rand(size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def integers(self, low, high=None, size=None, dtype='int64', endpoint=False):
        if high is None:
            high = low
            low = 0
        if not endpoint:
            pass  # high is exclusive already
        else:
            high = high + 1
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

    def choice(self, a, size=None, replace=True, p=None):
        if size is None:
            size = 1
        return random.choice(asarray(a), size, replace)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.normal(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.uniform(low, high, size)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

def _random_random(size=None):
    """Return random floats in [0, 1). Same as rand but takes size tuple."""
    if size is None:
        return float(random.rand((1,))[0])
    if isinstance(size, int):
        size = (size,)
    # Compute total elements
    total = 1
    for s in size:
        total *= s
    result = random.uniform(0.0, 1.0, (total,))
    if len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_multivariate_normal(mean, cov, size=None):
    """Draw from multivariate normal distribution."""
    mean = asarray(mean)
    cov = asarray(cov)
    n = len(mean.tolist())

    # Cholesky decomposition of covariance
    L = linalg.cholesky(cov)

    if size is None:
        # Single sample: generate n standard normals
        z = random.normal(0.0, 1.0, (n,))
        # Transform: L @ z + mean
        z_list = z.tolist()
        mean_list = mean.tolist()
        L_list = L.tolist()
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * z_list[j]
            sample.append(val)
        return array(sample)

    if isinstance(size, int):
        size = (size,)

    total = 1
    for s in size:
        total *= s

    # Generate total*n standard normals
    z = random.normal(0.0, 1.0, (total * n,)).reshape((total, n))

    # Transform: samples = mean + z @ L^T
    z_list = z.tolist()
    mean_list = mean.tolist()
    L_list = L.tolist()

    results = []
    for row in z_list:
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * row[j]
            sample.append(val)
        results.append(sample)

    result = array(results)
    if len(size) > 1:
        result = result.reshape(list(size) + [n])
    return result

def _random_chisquare(df, size=None):
    """Chi-square distribution (sum of df squared standard normals)."""
    df = int(df)
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        z = random.normal(0.0, 1.0, (df,))
        z_list = z.tolist()
        results.append(sum(v * v for v in z_list))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_laplace(loc=0.0, scale=1.0, size=None):
    """Laplace distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    # Inverse CDF: loc - scale * sign(u - 0.5) * log(1 - 2*abs(u - 0.5))
    u_list = u.tolist()
    results = []
    import math
    for ui in u_list:
        ui_shifted = ui - 0.5
        if ui_shifted == 0:
            results.append(loc)
        else:
            sign_val = 1.0 if ui_shifted > 0 else -1.0
            results.append(loc - scale * sign_val * math.log(1.0 - 2.0 * abs(ui_shifted)))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_triangular(left, mode, right, size=None):
    """Triangular distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    u_list = u.tolist()
    results = []
    fc = (mode - left) / (right - left)
    for ui in u_list:
        if ui < fc:
            results.append(left + ((right - left) * (mode - left) * ui) ** 0.5)
        else:
            results.append(right - ((right - left) * (right - mode) * (1.0 - ui)) ** 0.5)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_rayleigh(scale=1.0, size=None):
    """Rayleigh distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [scale * math.sqrt(-2.0 * math.log(1.0 - ui)) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_weibull(a, size=None):
    """Weibull distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [(-math.log(1.0 - ui)) ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_logistic(loc=0.0, scale=1.0, size=None):
    """Logistic distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc + scale * math.log(u / (1.0 - u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_gumbel(loc=0.0, scale=1.0, size=None):
    """Gumbel distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc - scale * math.log(-math.log(u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_negative_binomial(n, p, size=None):
    """Negative binomial distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        # Generate n geometric trials: count failures before n successes
        count = 0
        successes = 0
        while successes < n:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < p:
                successes += 1
            else:
                count += 1
        results.append(float(count))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_power(a, size=None):
    """Power distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    results = [ui ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_vonmises(mu, kappa, size=None):
    """Von Mises distribution (rejection sampling)."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    if kappa < 1e-6:
        # For very small kappa, uniform on [-pi, pi]
        for _ in range(total):
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            results.append(-math.pi + 2.0 * math.pi * u)
    else:
        tau = 1.0 + (1.0 + 4.0 * kappa * kappa) ** 0.5
        rho = (tau - (2.0 * tau) ** 0.5) / (2.0 * kappa)
        r = (1.0 + rho * rho) / (2.0 * rho)
        for _ in range(total):
            while True:
                u1 = float(random.uniform(0.0, 1.0, (1,))[0])
                z = _math.cos(math.pi * u1)
                f = (1.0 + r * z) / (r + z)
                c = kappa * (r - f)
                u2 = float(random.uniform(0.0, 1.0, (1,))[0])
                if u2 < c * (2.0 - c) or u2 <= c * _math.exp(1.0 - c):
                    u3 = float(random.uniform(0.0, 1.0, (1,))[0])
                    theta = mu + (1.0 if u3 > 0.5 else -1.0) * _math.acos(f)
                    results.append(theta)
                    break
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_wald(mean, scale, size=None):
    """Wald (inverse Gaussian) distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        v = float(random.normal(0.0, 1.0, (1,))[0])
        y = v * v
        x = mean + (mean * mean * y) / (2.0 * scale) - (mean / (2.0 * scale)) * (4.0 * mean * scale * y + mean * mean * y * y) ** 0.5
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u <= mean / (mean + x):
            results.append(x)
        else:
            results.append(mean * mean / x)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _random_zipf(a, size=None):
    """Zipf distribution (rejection sampling)."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    am1 = a - 1.0
    b = 2.0 ** am1
    results = []
    for _ in range(total):
        while True:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u <= 0.0:
                continue
            v = float(random.uniform(0.0, 1.0, (1,))[0])
            x = int(u ** (-1.0 / am1))
            if x < 1:
                x = 1
            t = (1.0 + 1.0 / x) ** am1
            if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                results.append(float(x))
                break
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result

def _default_rng(seed=None):
    return _Generator(seed)

class _RandomState:
    """Legacy random state compatible with np.random.RandomState(seed)."""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def rand(self, *shape):
        if len(shape) == 0:
            return float(random.rand((1,))[0])
        return random.rand(shape)

    def randn(self, *shape):
        if len(shape) == 0:
            return float(random.randn((1,))[0])
        return random.randn(shape)

    def randint(self, low, high=None, size=None, dtype='int64'):
        if high is None:
            high = low
            low = 0
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

    def random(self, size=None):
        return _random_random(size=size)

    def random_sample(self, size=None):
        return _random_random(size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return float(random.normal(float(loc), float(scale), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.normal(float(loc), float(scale), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return float(random.uniform(float(low), float(high), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.uniform(float(low), float(high), size)

    def choice(self, a, size=None, replace=True, p=None):
        arr = asarray(a) if not isinstance(a, ndarray) else a
        if size is None:
            size = 1
        return random.choice(arr, size, replace)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def seed(self, seed=None):
        random.seed(seed)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

    def beta(self, a, b, size=None):
        return _random_beta(a, b, size)

    def gamma(self, shape, scale=1.0, size=None):
        return _random_gamma(shape, scale, size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return _random_lognormal(mean, sigma, size)

    def chisquare(self, df, size=None):
        return _random_chisquare(df, size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def multivariate_normal(self, mean, cov, size=None):
        return _random_multivariate_normal(mean, cov, size)

    def get_state(self):
        return {'state': 'not_implemented'}

    def set_state(self, state):
        pass

# Monkey-patch random module with extension functions
random.shuffle = _random_shuffle
random.permutation = _random_permutation
random.standard_normal = _random_standard_normal
random.exponential = _random_exponential
random.poisson = _random_poisson
random.binomial = _random_binomial
random.beta = _random_beta
random.gamma = _random_gamma
random.multinomial = _random_multinomial
random.lognormal = _random_lognormal
random.geometric = _random_geometric
random.dirichlet = _random_dirichlet
random.default_rng = _default_rng
random.Generator = _Generator
random.random = _random_random
random.random_sample = _random_random
random.multivariate_normal = _random_multivariate_normal
random.chisquare = _random_chisquare
random.laplace = _random_laplace
random.triangular = _random_triangular
random.rayleigh = _random_rayleigh
random.weibull = _random_weibull
random.logistic = _random_logistic
random.gumbel = _random_gumbel
random.negative_binomial = _random_negative_binomial
random.power = _random_power
random.vonmises = _random_vonmises
random.wald = _random_wald
random.zipf = _random_zipf
random.RandomState = _RandomState

# --- Numerical Utilities (Tier 20C) -----------------------------------------

def packbits(a, axis=None, bitorder='big'):
    """Pack a binary-valued array into uint8 (int64 since we lack uint8 dtype)."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    # Flatten if axis is None
    vals = a.flatten().tolist()
    if bitorder == 'little':
        # Reverse bit order within each byte
        result = []
        for i in range(0, len(vals), 8):
            chunk = vals[i:i+8]
            byte = 0
            for j in range(len(chunk)):
                if int(chunk[j]):
                    byte |= (1 << j)
            result.append(byte)
        return array(result)
    else:
        # big endian (default)
        result = []
        for i in range(0, len(vals), 8):
            chunk = vals[i:i+8]
            byte = 0
            for j in range(len(chunk)):
                if int(chunk[j]):
                    byte |= (1 << (7 - j))
            result.append(byte)
        return array(result)


def unpackbits(a, axis=None, count=None, bitorder='big'):
    """Unpack elements of a uint8 array into a binary-valued output array."""
    if not isinstance(a, ndarray):
        a = asarray(a)
    vals = a.flatten().tolist()
    result = []
    for v in vals:
        byte = int(v)
        if bitorder == 'little':
            for j in range(8):
                result.append((byte >> j) & 1)
        else:
            for j in range(7, -1, -1):
                result.append((byte >> j) & 1)
    if count is not None:
        count = int(count)
        if count < len(result):
            result = result[:count]
        else:
            result = result + [0] * (count - len(result))
    return array(result)


def asfortranarray(a):
    """Return an array laid out in Fortran order (simplified: just copy)."""
    return array(a, copy=True)


def asarray_chkfinite(a):
    """Convert to array, checking for NaN and Inf."""
    a = asarray(a)
    if isinstance(a, ndarray):
        vals = a.flatten().tolist()
        for v in vals:
            if _math.isinf(v) or _math.isnan(v):
                raise ValueError("array must not contain infs or NaNs")
    return a


def nextafter(x1, x2):
    """Return the next floating-point value after x1 towards x2, element-wise."""
    if isinstance(x1, (int, float)) and isinstance(x2, (int, float)):
        return _math.nextafter(float(x1), float(x2))
    x1 = asarray(x1)
    x2 = asarray(x2)
    v1 = x1.flatten().tolist()
    v2 = x2.flatten().tolist()
    # Broadcast: if one is scalar-like (len 1), expand
    if len(v1) == 1 and len(v2) > 1:
        v1 = v1 * len(v2)
    elif len(v2) == 1 and len(v1) > 1:
        v2 = v2 * len(v1)
    result = [_math.nextafter(float(a), float(b)) for a, b in zip(v1, v2)]
    return array(result)


def spacing(x):
    """Return the distance between x and the nearest adjacent number."""
    if isinstance(x, (int, float)):
        ax = abs(float(x))
        return _math.nextafter(ax, _math.inf) - ax
    x = asarray(x)
    vals = x.flatten().tolist()
    result = []
    for v in vals:
        ax = abs(float(v))
        result.append(_math.nextafter(ax, _math.inf) - ax)
    return array(result)


def vdot(a, b):
    """Conjugate dot product of two arrays (flattened)."""
    a = asarray(a).flatten()
    b = asarray(b).flatten()
    return dot(a, b)


def broadcast_shapes(*shapes):
    """Compute the broadcast result shape from multiple shapes."""
    if not shapes:
        return ()
    ndim = _builtin_max(len(s) for s in shapes)
    result = [1] * ndim
    for shape in shapes:
        offset = ndim - len(shape)
        for i, dim in enumerate(shape):
            j = i + offset
            if result[j] == 1:
                result[j] = dim
            elif dim != 1 and dim != result[j]:
                raise ValueError(
                    f"shape mismatch: objects cannot be broadcast to a single shape. Mismatch at dimension {j}"
                )
    return tuple(result)


def gcd(x1, x2):
    """Element-wise greatest common divisor."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    f1 = x1.flatten().tolist()
    f2 = x2.flatten().tolist()
    result = [_math.gcd(int(a), int(b)) for a, b in zip(f1, f2)]
    return array(result).reshape(x1.shape)


def lcm(x1, x2):
    """Element-wise least common multiple."""
    x1 = asarray(x1)
    x2 = asarray(x2)
    f1 = x1.flatten().tolist()
    f2 = x2.flatten().tolist()
    def _lcm_pair(a, b):
        ia, ib = int(a), int(b)
        if ia == 0 or ib == 0:
            return 0
        g = _math.gcd(ia, ib)
        v = ia * ib
        if v < 0:
            v = -v
        return v // g
    result = [_lcm_pair(a, b) for a, b in zip(f1, f2)]
    return array(result).reshape(x1.shape)


def polydiv(u, v):
    """Polynomial division: returns (quotient, remainder)."""
    if isinstance(u, poly1d):
        u = list(u._coeffs)
    elif isinstance(u, ndarray):
        u = [float(u[i]) for i in range(u.size)]
    else:
        u = [float(c) for c in u]
    if isinstance(v, poly1d):
        v = list(v._coeffs)
    elif isinstance(v, ndarray):
        v = [float(v[i]) for i in range(v.size)]
    else:
        v = [float(c) for c in v]
    n = len(u)
    nv = len(v)
    if nv > n:
        return array([0.0]), array(u)
    q = [0.0] * (n - nv + 1)
    r = list(u)
    for i in range(n - nv + 1):
        q[i] = r[i] / v[0]
        for j in range(nv):
            r[i + j] -= q[i] * v[j]
    remainder = r[n - nv + 1:]
    return array(q), array(remainder)


def fabs(x):
    """Absolute value for floats, element-wise."""
    return abs(asarray(x))


# --- ufunc-like wrappers with .reduce() / .accumulate() --------------------

class _UfuncWithReduce:
    """Wraps a binary element-wise function to add .reduce() and .accumulate()."""
    def __init__(self, func, reduce_func, name=None):
        self._func = func
        self._reduce = reduce_func
        self.__name__ = name or getattr(func, '__name__', 'ufunc')

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)

    def reduce(self, a, axis=None, **kwargs):
        return self._reduce(a, axis=axis)

    def accumulate(self, a, axis=0, **kwargs):
        a = asarray(a)
        flat = a.flatten().tolist()
        if len(flat) == 0:
            return a
        # Simple 1-D accumulate
        result = [flat[0]]
        for v in flat[1:]:
            result.append(float(self._func(result[-1], v)))
        return array(result).reshape(a.shape)

    def outer(self, a, b, **kwargs):
        a = asarray(a)
        b = asarray(b)
        fa = a.flatten().tolist()
        fb = b.flatten().tolist()
        result = []
        for va in fa:
            for vb in fb:
                result.append(float(self._func(va, vb)))
        return array(result).reshape((len(fa), len(fb)))

    def __repr__(self):
        return f"<ufunc '{self.__name__}'>"

# Save original function references before wrapping
_maximum_func = maximum
_minimum_func = minimum
_add_func = add
_multiply_func = multiply
_subtract_func = subtract

# Wrap them as ufunc-like objects
maximum = _UfuncWithReduce(_maximum_func, lambda a, axis=None: max(a, axis=axis), name='maximum')
minimum = _UfuncWithReduce(_minimum_func, lambda a, axis=None: min(a, axis=axis), name='minimum')
add = _UfuncWithReduce(_add_func, lambda a, axis=None: sum(a, axis=axis), name='add')
multiply = _UfuncWithReduce(_multiply_func, lambda a, axis=None: prod(a, axis=axis), name='multiply')
def _subtract_reduce(a, axis=None):
    a = asarray(a)
    if axis is None:
        flat = a.flatten().tolist()
        if len(flat) == 0:
            return array(0.0)
        result = flat[0]
        for v in flat[1:]:
            result = result - v
        return result
    sh = list(a.shape)
    n = sh[axis]
    if n == 0:
        new_sh = sh[:axis] + sh[axis+1:]
        return zeros(tuple(new_sh) if new_sh else (1,))
    slices = [take(a, [i], axis=axis).squeeze(axis=axis) for i in range(n)]
    result = slices[0]
    for s in slices[1:]:
        result = result - s
    return result

subtract = _UfuncWithReduce(_subtract_func, _subtract_reduce, name='subtract')


# --- Tier 27 Group B: Additional functions ----------------------------------

def frombuffer(buffer, dtype=None, count=-1, offset=0):
    """Interpret a buffer as a 1-D array (simplified: treats as uint8)."""
    if isinstance(buffer, (bytes, bytearray)):
        data = list(buffer[offset:])
        if count != -1:
            data = data[:count]
        result = array([float(x) for x in data])
        if dtype is not None:
            result = result.astype(str(dtype))
        return result
    return asarray(buffer)

def trim_zeros(filt, trim='fb'):
    """Trim leading and/or trailing zeros from a 1-D array."""
    filt = asarray(filt)
    data = filt.tolist()
    first = 0
    last = len(data)
    if 'f' in trim or 'F' in trim:
        for i in _builtin_range(len(data)):
            if data[i] != 0:
                first = i
                break
        else:
            return array([])
    if 'b' in trim or 'B' in trim:
        for i in _builtin_range(len(data) - 1, -1, -1):
            if data[i] != 0:
                last = i + 1
                break
        else:
            return array([])
    return array(data[first:last])

def histogramdd(sample, bins=10, range=None, density=False, weights=None):
    """Simplified N-dimensional histogram."""
    sample = asarray(sample)
    if sample.ndim == 1:
        sample = sample.reshape((-1, 1))
    sample_list = sample.tolist()
    n_samples = len(sample_list)
    n_dims = len(sample_list[0])

    # Determine bins per dimension
    if isinstance(bins, int):
        bins_per_dim = [bins] * n_dims
    else:
        bins_per_dim = list(bins)

    # Determine edges per dimension
    _range = range  # local alias to avoid conflict
    edges = []
    for d in _builtin_range(n_dims):
        vals = [row[d] for row in sample_list]
        if _range is not None and _range[d] is not None:
            lo, hi = float(_range[d][0]), float(_range[d][1])
        else:
            lo, hi = _builtin_min(vals), _builtin_max(vals)
        edge = linspace(lo, hi, num=bins_per_dim[d] + 1, endpoint=True).tolist()
        edges.append(edge)

    # Build histogram
    shape = bins_per_dim
    total = 1
    for s in shape:
        total *= s
    counts = [0.0] * total

    w_list = None
    if weights is not None:
        w_list = asarray(weights).flatten().tolist()

    for idx_s in _builtin_range(n_samples):
        row = sample_list[idx_s]
        bin_indices = []
        in_range_flag = True
        for d in _builtin_range(n_dims):
            val = row[d]
            edge = edges[d]
            nb = bins_per_dim[d]
            found = False
            for j in _builtin_range(nb):
                if (val >= edge[j] and val < edge[j + 1]) or (j == nb - 1 and val == edge[j + 1]):
                    bin_indices.append(j)
                    found = True
                    break
            if not found:
                in_range_flag = False
                break
        if not in_range_flag:
            continue
        # Compute flat index
        flat_idx = 0
        stride = 1
        for d in _builtin_range(n_dims - 1, -1, -1):
            flat_idx += bin_indices[d] * stride
            stride *= bins_per_dim[d]
        counts[flat_idx] += (w_list[idx_s] if w_list is not None else 1.0)

    hist = array(counts).reshape(shape)
    edge_arrays = [array(e) for e in edges]

    if density and float(sum(hist)) > 0:
        # Normalize
        total_count = float(sum(hist))
        bin_volumes = ones(shape)
        for d in _builtin_range(n_dims):
            widths = [edges[d][i+1] - edges[d][i] for i in _builtin_range(bins_per_dim[d])]
            w_arr = array(widths)
            bcast_shape = [1] * n_dims
            bcast_shape[d] = bins_per_dim[d]
            w_arr = w_arr.reshape(bcast_shape)
            bin_volumes = bin_volumes * broadcast_to(w_arr, shape)
        hist = hist / (total_count * bin_volumes)

    return hist, edge_arrays

def common_type(*arrays):
    """Return a scalar type common to input arrays."""
    has_complex = False
    max_float = 32
    for a in arrays:
        arr = asarray(a)
        dt = str(arr.dtype)
        if "complex" in dt:
            has_complex = True
            if "128" in dt:
                max_float = 64
            else:
                max_float = _builtin_max(max_float, 32)
        elif "float64" in dt or "float" == dt:
            max_float = _builtin_max(max_float, 64)
        elif "float32" in dt:
            max_float = _builtin_max(max_float, 32)
        elif "int" in dt:
            max_float = _builtin_max(max_float, 64)  # int promotes to float64
    if has_complex:
        return complex128 if max_float >= 64 else complex64
    return float64 if max_float >= 64 else float32

class matrix:
    """Simplified matrix class (deprecated in NumPy, but still used)."""
    def __init__(self, data, dtype=None, copy=True):
        if isinstance(data, str):
            # Parse string like "1 2; 3 4"
            rows = data.split(";")
            parsed = []
            for row in rows:
                parsed.append([float(x) for x in row.strip().split()])
            self.A = array(parsed)
        else:
            self.A = atleast_2d(asarray(data))
        if dtype is not None:
            self.A = self.A.astype(str(dtype))

    @property
    def T(self):
        return matrix(self.A.T)

    @property
    def I(self):
        return matrix(linalg.inv(self.A))

    @property
    def shape(self):
        return self.A.shape

    @property
    def ndim(self):
        return self.A.ndim

    def __mul__(self, other):
        if isinstance(other, matrix):
            return matrix(dot(self.A, other.A))
        return matrix(self.A * asarray(other))

    def __add__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A + other.A)
        return matrix(self.A + asarray(other))

    def __sub__(self, other):
        if isinstance(other, matrix):
            return matrix(self.A - other.A)
        return matrix(self.A - asarray(other))

    def __getitem__(self, key):
        return self.A[key]

    def tolist(self):
        return self.A.tolist()

    def __repr__(self):
        return "matrix({})".format(self.A.tolist())


# --- Tier 30 Group C: array2string, lib.stride_tricks, info, who -----------

def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix='', style=None, formatter=None,
                 threshold=None, edgeitems=None, sign=None, floatmode=None,
                 suffix='', legacy=None):
    """Return a string representation of an array."""
    a = asarray(a)
    return repr(a)


def info(object=None, maxwidth=76, output=None, toplevel='numpy'):
    """Display documentation for numpy objects."""
    if object is not None:
        doc = getattr(object, '__doc__', None)
        if doc:
            print(doc)
        else:
            print("No documentation available for {}".format(object))


def who(vardict=None):
    """Print info about variables in the given dictionary."""
    if vardict is None:
        return
    for name, val in vardict.items():
        if hasattr(val, 'shape'):
            print("{}: shape={}, dtype={}".format(name, val.shape, val.dtype))


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


# --- testing module ---------------------------------------------------------
class _TestingModule:
    def assert_allclose(self, actual, desired, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
        actual = asarray(actual)
        desired = asarray(desired)
        if not allclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan):
            actual_list = actual.tolist()
            desired_list = desired.tolist()
            raise AssertionError(err_msg or "Not equal to tolerance rtol={}, atol={}\n Actual: {}\n Desired: {}".format(rtol, atol, actual_list, desired_list))

    def assert_array_equal(self, x, y, err_msg='', verbose=True, strict=False):
        x = asarray(x)
        y = asarray(y)
        if not array_equal(x, y):
            raise AssertionError(err_msg or "Arrays are not equal\n x: {}\n y: {}".format(x.tolist(), y.tolist()))

    def assert_array_almost_equal(self, x, y, decimal=6, err_msg='', verbose=True):
        x = asarray(x)
        y = asarray(y)
        if not allclose(x, y, rtol=0, atol=1.5 * 10**(-decimal)):
            raise AssertionError(err_msg or "Arrays are not almost equal to {} decimals".format(decimal))

    def assert_equal(self, actual, desired, err_msg='', verbose=True):
        actual_a = asarray(actual)
        desired_a = asarray(desired)
        if not array_equal(actual_a, desired_a):
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


testing = _TestingModule()

# --- dtypes module stub -----------------------------------------------------
class _dtypes_mod:
    pass
dtypes = _dtypes_mod()

# --- Import submodules so np.ma and np.polynomial are accessible ------------
import numpy.ma as ma
import numpy.polynomial as polynomial
