"""Element-wise math, type checking, comparison, arithmetic operators."""
import sys as _sys
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    _ObjectArray, _copy_into, _CLIP_UNSET,
    _builtin_range, _builtin_min, _builtin_max,
)
from ._core_types import (
    _ScalarType, _ScalarTypeMeta, _NumpyIntScalar, _NumpyFloatScalar, _NumpyComplexScalar,
    dtype, finfo, iinfo, _normalize_dtype, can_cast, result_type,
    bool_, int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    float16, float32, float64,
    complex64, complex128,
)
from ._creation import array, asarray, zeros, ones, empty, full, concatenate, linspace, where
from ._datetime import _datetime64_cls, _timedelta64_cls

__all__ = [
    # Trig
    'sin', 'cos', 'tan',
    'arcsin', 'arccos', 'arctan', 'arctan2',
    'sinh', 'cosh', 'tanh',
    'arcsinh', 'arccosh', 'arctanh',
    # Exp/log
    'exp', 'exp2', 'log', 'log2', 'log10', 'log1p', 'expm1',
    'logaddexp', 'logaddexp2',
    # Rounding
    'floor', 'ceil', 'trunc', 'rint', 'around', 'fix', 'round_',
    # Power
    'sqrt', 'cbrt', 'square', 'reciprocal', 'power', 'float_power',
    # Arithmetic
    'add', 'subtract', 'multiply', 'divide', 'true_divide', 'floor_divide',
    'remainder', 'mod', 'divmod_', 'divmod', 'negative', 'positive',
    'fmod', 'modf', 'fabs',
    # Extrema
    'maximum', 'minimum', 'fmax', 'fmin',
    # Other
    'abs', 'absolute', 'sign', 'signbit', 'copysign', 'heaviside',
    'ldexp', 'frexp', 'hypot', 'sinc', 'nan_to_num', 'clip',
    'nextafter', 'spacing',
    # Angle
    'deg2rad', 'rad2deg', 'degrees', 'radians',
    # Special
    'gamma', 'lgamma', 'erf', 'erfc', 'j0', 'j1', 'y0', 'y1', 'i0',
    # Complex
    'real', 'imag', 'conj', 'conjugate', 'angle', 'unwrap', 'real_if_close',
    # Type checking
    'isnan', 'isinf', 'isfinite', 'isneginf', 'isposinf',
    'isscalar', 'isreal', 'iscomplex', 'isrealobj', 'iscomplexobj',
    'issubdtype', 'issubclass_',
    # Comparison
    'allclose', 'isclose',
    'greater', 'less', 'equal', 'not_equal', 'greater_equal', 'less_equal',
    # GCD/LCM
    'gcd', 'lcm',
    # Casting
    'astype',
    # Internal helpers
    '_builtin_divmod',
    '_cmath_isnan', '_cmath_isinf', '_cmath_isfinite',
    '_unwrap_1d_list',
]

# --- Save builtin divmod before shadowing -----------------------------------
_builtin_divmod = __builtins__["divmod"] if isinstance(__builtins__, dict) else __import__("builtins").divmod

# --- Private helpers for complex math checks --------------------------------

def _cmath_isnan(v):
    import cmath
    if isinstance(v, complex):
        return cmath.isnan(v)
    try:
        return _math.isnan(v)
    except (TypeError, ValueError):
        return False

def _cmath_isinf(v):
    import cmath
    if isinstance(v, complex):
        return cmath.isinf(v)
    try:
        return _math.isinf(v)
    except (TypeError, ValueError):
        return False

def _cmath_isfinite(v):
    import cmath
    if isinstance(v, complex):
        return cmath.isfinite(v)
    try:
        return _math.isfinite(v)
    except (TypeError, ValueError):
        return True

# --- Type checking -----------------------------------------------------------

def isnan(x):
    """Check for NaN element-wise."""
    if isinstance(x, _ObjectArray):
        return array([_cmath_isnan(v) for v in x._data])
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        elif isinstance(x, complex):
            import cmath
            return cmath.isnan(x)
        else:
            return _math.isnan(x)
    return _native.isnan(x)

def isfinite(x):
    if isinstance(x, _ObjectArray):
        return array([_cmath_isfinite(v) for v in x._data])
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        elif isinstance(x, complex):
            import cmath
            return cmath.isfinite(x)
        else:
            return _math.isfinite(x)
    return _native.isfinite(x)

def isinf(x):
    if isinstance(x, _ObjectArray):
        return array([_cmath_isinf(v) for v in x._data])
    if not isinstance(x, ndarray):
        if isinstance(x, (list, tuple)):
            x = asarray(x)
        else:
            return _math.isinf(x)
    return _native.isinf(x)

def isscalar(x):
    """Return True if x is a scalar (not an array)."""
    if isinstance(x, (int, float, complex, bool, str)):
        return True
    if isinstance(x, ndarray):
        return False
    if isinstance(x, (list, tuple)):
        return False
    if x is None:
        return False
    # Check for PEP 3141 Number types and numpy scalar types
    try:
        import numbers
        if isinstance(x, numbers.Number):
            return True
    except ImportError:
        pass
    # Check for numpy scalar types (0-d arrays or scalar wrappers)
    if hasattr(x, 'ndim') and x.ndim == 0:
        return True
    return False

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

def issubclass_(arg1, arg2):
    """Determine if a class is a subclass of a second class.
    (Wrapper around Python's issubclass that returns False instead of raising TypeError.)"""
    try:
        return issubclass(arg1, arg2)
    except TypeError:
        return False

# --- Clip --------------------------------------------------------------------

def clip(a, a_min=_CLIP_UNSET, a_max=_CLIP_UNSET, out=None, **kwargs):
    """Clip array values to [a_min, a_max]."""
    _conflict_msg = ("Passing `min` or `max` keyword argument when `a_min` and "
                     "`a_max` are provided is forbidden.")
    # Validate casting kwarg
    _clip_casting = "same_kind"  # default
    if 'casting' in kwargs:
        c = kwargs.pop('casting')
        if c is None:
            _clip_casting = "same_kind"
        else:
            _valid = ('no', 'equiv', 'safe', 'same_kind', 'unsafe')
            if c not in _valid:
                raise ValueError("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'")
            _clip_casting = c
    # Conflict checks: both positional AND keyword
    if a_min is not _CLIP_UNSET and a_max is not _CLIP_UNSET:
        if 'min' in kwargs or 'max' in kwargs:
            raise ValueError(_conflict_msg)
    # Support min=/max= as aliases for a_min/a_max
    _min_from_kw = False
    _max_from_kw = False
    if 'min' in kwargs:
        _min_from_kw = True
        a_min = kwargs.pop('min')
    if 'max' in kwargs:
        _max_from_kw = True
        a_max = kwargs.pop('max')
    # np.clip(arr) with no bounds → return copy
    if a_min is _CLIP_UNSET and a_max is _CLIP_UNSET:
        if not isinstance(a, (ndarray, _ObjectArray)):
            a = array(a)
        return a.copy() if isinstance(a, ndarray) else a
    # Check required args. Keyword-only min/max may omit one bound.
    if a_max is _CLIP_UNSET:
        if _min_from_kw:
            a_max = None
        else:
            raise TypeError("clip() missing 1 required positional argument: 'a_max'")
    if a_min is _CLIP_UNSET:
        if _max_from_kw:
            a_min = None
        else:
            raise TypeError("clip() missing 1 required positional argument: 'a_min'")
    if not isinstance(a, (ndarray, _ObjectArray)):
        a = array(a)
    if a_min is None and a_max is None:
        return a.copy() if isinstance(a, ndarray) else a
    # Check casting constraint when out is provided
    if out is not None and isinstance(a, ndarray) and isinstance(out, ndarray):
        # Determine result dtype of clipping
        _bound_dt = None
        for b in (a_min, a_max):
            if b is not None:
                if isinstance(b, ndarray):
                    bdt = str(b.dtype)
                elif isinstance(b, float):
                    bdt = "float64"
                elif isinstance(b, int):
                    bdt = "int64"
                else:
                    bdt = None
                if bdt is not None:
                    _bound_dt = bdt
        if _bound_dt is not None:
            a_dt = str(a.dtype)
            out_dt = str(out.dtype)
            # If bounds are float and array/out are int, check casting
            if _bound_dt in ("float32", "float64", "float16") and out_dt in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"):
                if _clip_casting != "unsafe":
                    raise TypeError(
                        "Cannot cast ufunc 'clip' output from dtype('{}') to "
                        "dtype('{}') with casting rule '{}'".format(
                            _bound_dt, out_dt, _clip_casting))
    # _ObjectArray (complex) fallback
    if isinstance(a, _ObjectArray):
        data = list(a._data)
        for i, v in enumerate(data):
            if a_min is not None:
                try:
                    if v < a_min:
                        data[i] = a_min
                except TypeError:
                    pass
            if a_max is not None:
                try:
                    if v > a_max:
                        data[i] = a_max
                except TypeError:
                    pass
        return _ObjectArray(data, a._dtype)
    # Check if min/max are arrays — need element-wise clipping
    min_is_array = isinstance(a_min, ndarray)
    max_is_array = isinstance(a_max, ndarray)
    if min_is_array or max_is_array:
        result = a.copy()
        if a_min is not None:
            a_min_arr = a_min if min_is_array else full(a.shape, float(a_min))
            result = where(result < a_min_arr, a_min_arr, result)
        if a_max is not None:
            a_max_arr = a_max if max_is_array else full(a.shape, float(a_max))
            result = where(result > a_max_arr, a_max_arr, result)
        if out is not None:
            _copy_into(out, result)
            return out
        return result
    # Complex dtype: delegate to ndarray.clip which has Rust-level complex support
    dt_name = str(a.dtype) if isinstance(a, ndarray) else ""
    if "complex" in dt_name and isinstance(a, ndarray):
        return a.clip(a_min, a_max, out=out)
    # For integer dtypes, clamp bounds to dtype range to preserve dtype
    _int_dtypes = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    _int_dtypes = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    if dt_name in _int_dtypes:
        info = iinfo(dt_name)
        if a_min is None:
            a_min_f = float('-inf')
        else:
            a_min_f = float(a_min)
            if a_min_f < info.min:
                a_min_f = float(info.min)
            elif a_min_f > info.max:
                a_min_f = float(info.max)
        if a_max is None:
            a_max_f = float('inf')
        else:
            a_max_f = float(a_max)
            if a_max_f > info.max:
                a_max_f = float(info.max)
            elif a_max_f < info.min:
                a_max_f = float(info.min)
        if not (_math.isnan(a_min_f) or _math.isnan(a_max_f)):
            result = _native.clip(a, a_min_f, a_max_f)
            result = result.astype(dt_name)
            if out is not None:
                _copy_into(out, result)
                return out
            return result
    if a_min is None:
        a_min = float('-inf')
    else:
        if isinstance(a_min, complex):
            a_min = a_min.real
        a_min = float(a_min)
    if a_max is None:
        a_max = float('inf')
    else:
        if isinstance(a_max, complex):
            a_max = a_max.real
        a_max = float(a_max)
    # NaN propagation: if either bound is NaN, result is all NaN
    if _math.isnan(a_min) or _math.isnan(a_max):
        result = full(a.shape, float('nan'), dtype=str(a.dtype))
        if out is not None:
            _copy_into(out, result)
            return out
        return result
    result = _native.clip(a, a_min, a_max)
    if out is not None:
        _copy_into(out, result)
        return out
    return result

# --- abs / absolute ----------------------------------------------------------

def abs(x, out=None):
    if isinstance(x, ndarray):
        result = x.abs()
        if out is not None:
            _copy_into(out, result)
            return out
        return result
    if isinstance(x, _ObjectArray):
        return x.__abs__()
    return __builtins__["abs"](x) if isinstance(__builtins__, dict) else _math.fabs(x)

absolute = abs

# --- sqrt, exp, log family ---------------------------------------------------

def sqrt(x):
    if isinstance(x, ndarray):
        return x.sqrt()
    if isinstance(x, complex):
        import cmath
        return cmath.sqrt(x)
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

# --- sign --------------------------------------------------------------------

def sign(x):
    if isinstance(x, ndarray):
        return _native.sign(x)
    return (x > 0) - (x < 0)

# --- Angle conversions -------------------------------------------------------

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

# --- Trig --------------------------------------------------------------------

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
    return _native.hypot(asarray(x1), asarray(x2))

# --- Rounding ----------------------------------------------------------------

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
    _builtin_round = __import__("builtins").round
    if not isinstance(a, ndarray):
        if isinstance(a, (list, tuple)):
            # Use Python's builtin round for banker's rounding
            result = array([_builtin_round(float(x), decimals) for x in a])
            if out is not None:
                out[:] = result
                return out
            return result
        else:
            return _builtin_round(float(a), decimals)
    return _native.around(a, decimals)

round_ = around
round = around

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

# --- ldexp / frexp -----------------------------------------------------------

def ldexp(x1, x2):
    """Return x1 * 2**x2, element-wise."""
    return _native.ldexp(asarray(x1), asarray(x2))

def frexp(x):
    """Decompose elements of x into mantissa and twos exponent."""
    if isinstance(x, ndarray):
        flat = x.flatten().tolist()
        mantissa = []
        exponent = []
        for v in flat:
            m, e = _math.frexp(float(v))
            mantissa.append(m)
            exponent.append(e)
        return array(mantissa).reshape(x.shape), array([float(e) for e in exponent]).reshape(x.shape)
    m, e = _math.frexp(float(x))
    return m, e

# --- logaddexp ---------------------------------------------------------------

def logaddexp(x1, x2):
    """Logarithm of the sum of exponentiations of the inputs."""
    return _native.logaddexp(asarray(x1), asarray(x2))

def logaddexp2(x1, x2):
    """Logarithm base 2 of the sum of exponentiations of the inputs in base 2."""
    return _native.logaddexp2(asarray(x1), asarray(x2))

# --- Power / square / cbrt / reciprocal -------------------------------------

def power(x1, x2):
    return asarray(x1) ** asarray(x2)

def float_power(x1, x2):
    return power(asarray(x1).astype('float64'), asarray(x2).astype('float64'))

def square(x):
    """Return the element-wise square."""
    x = asarray(x)
    return x * x

def cbrt(x):
    """Return the element-wise cube root."""
    return _native.cbrt(asarray(x))

def reciprocal(x):
    """Return the reciprocal of the argument, element-wise."""
    x = asarray(x)
    return ones(x.shape) / x

# --- copysign / heaviside / sinc / nan_to_num --------------------------------

def copysign(x1, x2):
    """Change the sign of x1 to that of x2, element-wise."""
    return _native.copysign(asarray(x1), asarray(x2))

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
    import numpy as _np
    x = asarray(x)
    px = x * _np.pi
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

# --- Special functions -------------------------------------------------------

def gamma(x):
    """Gamma function."""
    return _native.gamma(asarray(x))

def lgamma(x):
    """Log of the absolute value of the gamma function."""
    return _native.lgamma(asarray(x))

def erf(x):
    """Error function."""
    return _native.erf(asarray(x))

def erfc(x):
    """Complementary error function: 1 - erf(x)."""
    return _native.erfc(asarray(x))

def j0(x):
    """Bessel function of the first kind, order 0."""
    return _native.j0(asarray(x))

def j1(x):
    """Bessel function of the first kind, order 1."""
    return _native.j1(asarray(x))

def y0(x):
    """Bessel function of the second kind, order 0."""
    return _native.y0(asarray(x))

def y1(x):
    """Bessel function of the second kind, order 1."""
    return _native.y1(asarray(x))

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

# --- Comparison / allclose / isclose -----------------------------------------

def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return True if two arrays are element-wise equal within a tolerance."""
    _builtin_all = __import__("builtins").all
    result = isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if isinstance(result, ndarray):
        return bool(_builtin_all(result.flatten().tolist()))
    return bool(result)

def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    """Return boolean array where two arrays are element-wise equal within tolerance."""
    import numpy as _np
    # Handle MaskedArray inputs — delegate to data, preserve mask
    from numpy.ma import MaskedArray as _MA
    _a_ma = isinstance(a, _MA)
    _b_ma = isinstance(b, _MA)
    if _a_ma or _b_ma:
        a_data = a.data if _a_ma else (asarray(a) if not isinstance(a, ndarray) else a)
        b_data = b.data if _b_ma else (asarray(b) if not isinstance(b, ndarray) else b)
        result_data = isclose(a_data, b_data, rtol=rtol, atol=atol, equal_nan=equal_nan)
        mask = a.mask if _a_ma else (b.mask if _b_ma else None)
        return _MA(result_data, mask=mask)
    # Fast path for _ObjectArray (complex, object, temporal dtypes)
    if isinstance(a, _ObjectArray) or isinstance(b, _ObjectArray):
        _babs = __import__("builtins").abs
        a_obj = a if isinstance(a, _ObjectArray) else None
        b_obj = b if isinstance(b, _ObjectArray) else None
        out_shape = (a_obj or b_obj)._shape
        a_data = a._data if isinstance(a, _ObjectArray) else (a.flatten().tolist() if isinstance(a, ndarray) else [a])
        b_data = b._data if isinstance(b, _ObjectArray) else (b.flatten().tolist() if isinstance(b, ndarray) else [b])
        # Broadcast scalar b to match a_data length
        if len(b_data) == 1 and len(a_data) > 1:
            b_data = b_data * len(a_data)
        # Get atol as a numeric value (handle timedelta64 atol)
        atol_val = atol._value if isinstance(atol, _timedelta64_cls) else atol
        results = []
        for av, bv in zip(a_data, b_data):
            # NaT check (treated like NaN)
            av_nat = isinstance(av, (_datetime64_cls, _timedelta64_cls)) and av._is_nat
            bv_nat = isinstance(bv, (_datetime64_cls, _timedelta64_cls)) and bv._is_nat
            if equal_nan and (av_nat or _cmath_isnan(av if not av_nat else 0)) and \
                             (bv_nat or _cmath_isnan(bv if not bv_nat else 0)):
                results.append(av_nat and bv_nat)
            elif av_nat or bv_nat:
                results.append(False)
            else:
                try:
                    diff = av - bv
                    diff_val = diff._value if isinstance(diff, (_datetime64_cls, _timedelta64_cls)) else diff
                    bv_val = bv._value if isinstance(bv, (_datetime64_cls, _timedelta64_cls)) else bv
                    results.append(_babs(diff_val) <= atol_val + rtol * _babs(bv_val))
                except (TypeError, ValueError):
                    results.append(av == bv)
        arr = _native.array([1.0 if r else 0.0 for r in results]).astype("bool")
        if len(out_shape) > 1:
            arr = arr.reshape(list(out_shape))
        return arr
    scalar_input = not isinstance(a, ndarray) and not isinstance(b, ndarray) and not isinstance(a, (list, tuple)) and not isinstance(b, (list, tuple))
    if not isinstance(a, ndarray):
        a = array(a) if isinstance(a, (list, tuple)) else array([a])
    if not isinstance(b, ndarray):
        b = array(b) if isinstance(b, (list, tuple)) else array([b])
    # Handle infinities: inf == inf (same sign) should be True
    a_inf = isinf(a)
    b_inf = isinf(b)
    both_inf = _np.logical_and(a_inf, b_inf)
    # Same-sign infinities are "close"
    same_inf = _np.logical_and(both_inf, (a == b))
    # For the general case, replace inf with 0 to avoid inf-inf=nan
    a_safe = where(a_inf, zeros(a.shape), a)
    b_safe = where(b_inf, zeros(b.shape), b)
    diff = abs(a_safe - b_safe)
    if isinstance(atol, (list, tuple, ndarray)):
        atol_arr = asarray(atol)
    else:
        atol_arr = full(diff.shape, atol)
    if isinstance(rtol, (list, tuple, ndarray)):
        rtol_arr = asarray(rtol)
    else:
        rtol_arr = full(diff.shape, rtol)
    limit = atol_arr + rtol_arr * abs(b_safe)
    result = _np.logical_or(diff <= limit, same_inf)
    # Different-sign infinities are never close
    diff_inf = _np.logical_and(both_inf, _np.logical_not(same_inf))
    result = _np.logical_and(result, _np.logical_not(diff_inf))
    # One inf, one finite: not close
    one_inf = _np.logical_and(_np.logical_or(a_inf, b_inf), _np.logical_not(both_inf))
    result = _np.logical_and(result, _np.logical_not(one_inf))
    if equal_nan:
        both_nan = _np.logical_and(isnan(a), isnan(b))
        result = _np.logical_or(result, both_nan)
    if scalar_input and result.size == 1:
        return bool(result.flatten()[0])
    return result

# --- Comparison operators ----------------------------------------------------

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

# --- Extrema -----------------------------------------------------------------

def maximum(x1, x2):
    return _native.maximum(asarray(x1), asarray(x2))

def minimum(x1, x2):
    return _native.minimum(asarray(x1), asarray(x2))

def fmax(x1, x2):
    """Element-wise maximum, ignoring NaNs."""
    return _native.fmax(asarray(x1), asarray(x2))

def fmin(x1, x2):
    """Element-wise minimum, ignoring NaNs."""
    return _native.fmin(asarray(x1), asarray(x2))

# --- signbit -----------------------------------------------------------------

def signbit(x):
    if isinstance(x, ndarray):
        return _native.signbit(x)
    return x < 0

# --- Arithmetic operators ----------------------------------------------------

def add(x1, x2, out=None):
    a = x1 if isinstance(x1, ndarray) else (asarray(x1) if isinstance(x1, (list, tuple, _ObjectArray)) else x1)
    b = x2 if isinstance(x2, ndarray) else (asarray(x2) if isinstance(x2, (list, tuple, _ObjectArray)) else x2)

    scalar_scalar = (
        not isinstance(x1, (ndarray, list, tuple, _ObjectArray))
        and not isinstance(x2, (ndarray, list, tuple, _ObjectArray))
    )
    if scalar_scalar:
        target = str(result_type(x1, x2))
        val = x1 + x2
        if target.startswith("complex"):
            return _ObjectArray([complex(val)], target)
        return array([val], dtype=target)

    if isinstance(a, _ObjectArray) or isinstance(b, _ObjectArray):
        target = str(result_type(x1, x2))
        if not isinstance(a, _ObjectArray):
            a = _ObjectArray([a] * len(b._data), target)
        if not isinstance(b, _ObjectArray):
            b = _ObjectArray([b] * len(a._data), target)
        n = len(a._data) if len(a._data) < len(b._data) else len(b._data)
        vals = [a._data[i] + b._data[i] for i in range(n)]
        return _ObjectArray(vals, target)

    def _scalar_for_array_op(v):
        if hasattr(v, "_numpy_dtype_name"):
            dn = str(getattr(v, "_numpy_dtype_name"))
            if dn == "bool":
                return bool(v)
            if dn.startswith("int") or dn.startswith("uint"):
                return int(v)
            if dn.startswith("float"):
                return float(v)
            if dn.startswith("complex"):
                return complex(v)
        return v

    if isinstance(a, ndarray) and not isinstance(b, ndarray):
        b = _scalar_for_array_op(b)
        if not isinstance(b, (int, float, complex, bool)):
            b = asarray(b)
    elif isinstance(b, ndarray) and not isinstance(a, ndarray):
        a = _scalar_for_array_op(a)
        if not isinstance(a, (int, float, complex, bool)):
            a = asarray(a)

    if isinstance(a, ndarray) or isinstance(b, ndarray):
        r = a + b
    else:
        r = asarray(a) + asarray(b)
    if hasattr(r, "dtype"):
        target = str(result_type(x1, x2))
        if str(r.dtype) != target:
            try:
                r = r.astype(target)
            except Exception:
                pass
    return r

def divide(x1, x2, out=None):
    return asarray(x1) / asarray(x2)

def subtract(x1, x2, out=None):
    return asarray(x1) - asarray(x2)

def multiply(x1, x2, out=None):
    r = asarray(x1) * asarray(x2)
    if hasattr(r, "dtype"):
        target = str(result_type(x1, x2))
        if str(r.dtype) != target:
            try:
                r = r.astype(target)
            except Exception:
                pass
    return r

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

divmod = divmod_

def negative(x):
    return -asarray(x)

def positive(x):
    return asarray(x) * 1

def fmod(x1, x2):
    """Return the element-wise remainder of division (C-style)."""
    return _native.fmod(asarray(x1), asarray(x2))

def modf(x):
    """Return the fractional and integral parts of an array, element-wise."""
    x = asarray(x)
    integer_part = trunc(x)
    fractional_part = x - integer_part
    return fractional_part, integer_part

def fabs(x):
    """Absolute value for floats, element-wise."""
    return abs(asarray(x))

# --- nextafter / spacing -----------------------------------------------------

def nextafter(x1, x2):
    """Return the next floating-point value after x1 towards x2, element-wise."""
    return _native.nextafter(asarray(x1), asarray(x2))

def spacing(x):
    """Return the distance between x and the nearest adjacent number."""
    if isinstance(x, (int, float)):
        ax = __builtins__["abs"](float(x)) if isinstance(__builtins__, dict) else _math.fabs(float(x))
        return _math.nextafter(ax, _math.inf) - ax
    x = asarray(x)
    vals = x.flatten().tolist()
    result = []
    for v in vals:
        ax = _math.fabs(float(v))
        result.append(_math.nextafter(ax, _math.inf) - ax)
    return array(result)

# --- GCD / LCM ---------------------------------------------------------------

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

# --- Casting -----------------------------------------------------------------

def astype(a, dtype, casting='unsafe', copy=True):
    # Accept numpy scalar types (e.g. np.int64(10)) and plain scalars
    if isinstance(a, (int, float)):
        a = asarray(a)
    elif isinstance(a, _ScalarType):
        a = asarray(float(a))
    elif not isinstance(a, ndarray):
        raise TypeError("Input should be a NumPy array")
    dtype_str = _normalize_dtype(str(dtype))
    if casting != 'unsafe':
        from_dtype = str(a.dtype)
        if not can_cast(from_dtype, dtype_str, casting=casting):
            raise TypeError("Cannot cast array data from {} to {} according to the rule '{}'".format(
                from_dtype, dtype_str, casting))
    if copy is False and str(a.dtype) == dtype_str:
        return a
    return a.astype(dtype_str)

# --- Complex -----------------------------------------------------------------

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
    import numpy as _np
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
        pt = _np.transpose(p, axes=axes)
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
        out = _np.transpose(out, axes=inv_axes)
    return out

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

# --- isneginf / isposinf -----------------------------------------------------

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
