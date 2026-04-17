"""Scalar type machinery: _ScalarType, numpy scalar classes, type hierarchy, and arithmetic."""
import sys as _sys
import math as _math
import operator as _operator
import cmath as _cmath
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    _unsupported_numeric_dtype, _is_temporal_dtype, _temporal_dtype_info,
    _ObjectArray, _builtin_max, _builtin_min,
)

__all__ = [
    # Scalar types
    'float64', 'float32', 'float16', 'int8', 'int16', 'int32', 'int64',
    'uint8', 'uint16', 'uint32', 'uint64', 'complex64', 'complex128',
    'bool_', 'str_', 'bytes_', 'void', 'object_',
    # Type hierarchy
    'generic', 'number', 'integer', 'signedinteger', 'unsignedinteger',
    'inexact', 'floating', 'complexfloating', 'character', 'flexible',
    # Metaclass and base
    '_ScalarType', '_NumpyIntScalar', '_NumpyFloatScalar', '_NumpyComplexScalar',
    '_NumpyVoidScalar', '_ScalarTypeMeta',
    # Constants and aliases
    'True_', 'False_', 'int_', 'typecodes', 'sctypes', 'sctypeDict',
    'float128', 'intp', 'intc', 'uintp', 'byte', 'ubyte', 'short', 'ushort',
    'longlong', 'ulonglong', 'single', 'double', 'longdouble',
    'csingle', 'cdouble', 'clongdouble',
    'string_', 'unicode_', 'half', 'float_', 'complex_', 'uint', 'long', 'ulong',
    'longfloat', 'clongfloat', 'longcomplex',
    # dtype char map and itemsize
    '_DTYPE_CHAR_MAP', '_DTYPE_ITEMSIZE',
]

# ---------------------------------------------------------------------------
# Dtype character map
# ---------------------------------------------------------------------------
_DTYPE_CHAR_MAP = {
    '?': 'bool', 'b': 'int8', 'B': 'uint8',
    'h': 'int16', 'H': 'uint16',
    'i': 'int32', 'I': 'uint32',
    'l': 'int64', 'L': 'uint64',
    'q': 'int64', 'Q': 'uint64',
    'e': 'float16', 'f': 'float32', 'd': 'float64', 'g': 'float64',
    'F': 'complex64', 'D': 'complex128', 'G': 'complex128',
    # Python type class names
    "<class 'bool'>": 'bool', "<class 'int'>": 'int64', "<class 'float'>": 'float64',
    "<class 'complex'>": 'complex128', "<class 'str'>": 'str',
    'f4': 'float32', 'f8': 'float64', 'f2': 'float16',
    'i1': 'int8', 'i2': 'int16', 'i4': 'int32', 'i8': 'int64',
    'u1': 'uint8', 'u2': 'uint16', 'u4': 'uint32', 'u8': 'uint64',
    'c8': 'complex64', 'c16': 'complex128',
    'b1': 'bool',
    '<f4': 'float32', '<f8': 'float64', '<f2': 'float16',
    '<i1': 'int8', '<i2': 'int16', '<i4': 'int32', '<i8': 'int64',
    '<u1': 'uint8', '<u2': 'uint16', '<u4': 'uint32', '<u8': 'uint64',
    '<c8': 'complex64', '<c16': 'complex128',
    '>f4': 'float32', '>f8': 'float64', '>f2': 'float16',
    '>i1': 'int8', '>i2': 'int16', '>i4': 'int32', '>i8': 'int64',
    '>u1': 'uint8', '>u2': 'uint16', '>u4': 'uint32', '>u8': 'uint64',
    '>c8': 'complex64', '>c16': 'complex128',
    '=f4': 'float32', '=f8': 'float64',
    '=i4': 'int32', '=i8': 'int64',
    # Unicode string aliases (all map to 'str')
    '<U': 'str', 'U': 'str', '<U1': 'str', '<U2': 'str', '<U4': 'str',
    '<U8': 'str', '<U16': 'str', '<U32': 'str', '<U64': 'str',
    '>U1': 'str', '>U2': 'str', '>U4': 'str',
    # Python type class names for bytes
    "<class 'bytes'>": 'bytes',
    # Object dtype aliases
    'O': 'object', '|O': 'object', 'object': 'object',
    "<class 'object'>": 'object',
    # Pointer types (map to int64 on 64-bit)
    'p': 'int64', 'P': 'uint64',
    # Void/string fallback aliases
    'V0': 'void', 'V3': 'void', 'V10': 'void',
    'S': 'bytes', 'S0': 'bytes', 'U0': 'str',
    # Byte string aliases (all map to 'bytes')
    '|S0': 'bytes', '|S1': 'bytes', '|S2': 'bytes',
    '|S4': 'bytes', '|S8': 'bytes',
}


# ---------------------------------------------------------------------------
# Scalar type helpers
# ---------------------------------------------------------------------------

_DTYPE_ITEMSIZE = {
    'bool': 1, 'int8': 1, 'uint8': 1, 'int16': 2, 'uint16': 2,
    'int32': 4, 'uint32': 4, 'int64': 8, 'uint64': 8,
    'float16': 2, 'float32': 4, 'float64': 8,
    'complex64': 8, 'complex128': 16,
}


class _ScalarType:
    """A callable dtype alias that can construct scalars and be used as a dtype string."""
    def __init__(self, name, python_type=float):
        self._name = name
        self._type = python_type

    def __call__(self, value=0, *args, **kwargs):
        try:
            return self._type(value, *args, **kwargs)
        except (ValueError, TypeError):
            # bytes_(x) converts via str first (e.g. bytes_(-2) == b'-2')
            if self._type is bytes and not args and not kwargs:
                return str(value).encode('ascii')
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


def _get_numpy_dtype_name(x):
    """Get the numpy dtype name of a scalar value, or None for non-numpy scalars."""
    return getattr(x, '_numpy_dtype_name', None)


def _wrap_scalar_result(value, dtype_name):
    """Wrap a Python value in the appropriate numpy scalar type for the given dtype."""
    if dtype_name == 'bool':
        return bool_(value)
    if dtype_name in ('int8', 'int16', 'int32', 'int64',
                      'uint8', 'uint16', 'uint32', 'uint64'):
        # Handle integer overflow wrapping
        try:
            iv = int(value)
        except (TypeError, ValueError, OverflowError):
            iv = value
        else:
            _INT_OVERFLOW = {
                'int8': (-128, 127, 256),
                'int16': (-32768, 32767, 65536),
                'int32': (-2147483648, 2147483647, 4294967296),
                'int64': (-9223372036854775808, 9223372036854775807, 18446744073709551616),
                'uint8': (0, 255, 256),
                'uint16': (0, 65535, 65536),
                'uint32': (0, 4294967295, 4294967296),
                'uint64': (0, 18446744073709551615, 18446744073709551616),
                'bool': (0, 1, 2),
            }
            if dtype_name in _INT_OVERFLOW:
                lo, hi, mod = _INT_OVERFLOW[dtype_name]
                if iv < lo or iv > hi:
                    iv = iv % mod
                    if dtype_name.startswith('int') and iv > hi:
                        iv -= mod
                value = iv
        return _NumpyIntScalar(value, dtype_name)
    if dtype_name in ('float16', 'float32', 'float64'):
        return _NumpyFloatScalar(value, dtype_name)
    if dtype_name in ('complex64', 'complex128'):
        return _NumpyComplexScalar(value, dtype_name)
    return value


def _scalar_promote(dn1, dn2):
    """Promote two dtype names and return the result dtype name string."""
    # Inlined fast promotion table to avoid constructing dtype objects.
    _PROMOTE = {
        # same type -> same type
    }
    key = (dn1, dn2)
    # Fast path: same dtype (except bool+bool which promotes to int8 for arithmetic)
    if dn1 == dn2:
        if dn1 == 'bool':
            return 'int8'
        return dn1

    # Integer promotion tables (manual for speed)
    _INT_RANK = {
        'bool': 0, 'int8': 1, 'uint8': 2, 'int16': 3, 'uint16': 4,
        'int32': 5, 'uint32': 6, 'int64': 7, 'uint64': 8,
    }
    _FLOAT_RANK = {'float16': 0, 'float32': 1, 'float64': 2}
    _COMPLEX_RANK = {'complex64': 0, 'complex128': 1}

    # Both integers
    if dn1 in _INT_RANK and dn2 in _INT_RANK:
        # NumPy: bool + bool -> int8 (not bool) for arithmetic
        if dn1 == 'bool' and dn2 == 'bool':
            return 'int8'
        if dn1 == 'bool':
            return dn2
        if dn2 == 'bool':
            return dn1
        a_signed = not dn1.startswith('uint')
        b_signed = not dn2.startswith('uint')
        _bits = {'int8': 8, 'uint8': 8, 'int16': 16, 'uint16': 16,
                 'int32': 32, 'uint32': 32, 'int64': 64, 'uint64': 64}
        ab = _bits[dn1]
        bb = _bits[dn2]
        bmax = ab if ab > bb else bb
        if a_signed == b_signed:
            _names = {8: 'int8', 16: 'int16', 32: 'int32', 64: 'int64'} if a_signed else \
                     {8: 'uint8', 16: 'uint16', 32: 'uint32', 64: 'uint64'}
            return _names[bmax]
        # signed/unsigned mix
        sbits = ab if a_signed else bb
        ubits = ab if not a_signed else bb
        if sbits > ubits:
            return {8: 'int8', 16: 'int16', 32: 'int32', 64: 'int64'}[sbits]
        for b in (8, 16, 32, 64):
            if b > ubits:
                return {8: 'int8', 16: 'int16', 32: 'int32', 64: 'int64'}[b]
        return 'float64'

    # Both floats
    if dn1 in _FLOAT_RANK and dn2 in _FLOAT_RANK:
        return dn1 if _FLOAT_RANK[dn1] >= _FLOAT_RANK[dn2] else dn2

    # Both complex
    if dn1 in _COMPLEX_RANK and dn2 in _COMPLEX_RANK:
        return dn1 if _COMPLEX_RANK[dn1] >= _COMPLEX_RANK[dn2] else dn2

    # Complex + anything -> complex
    if dn1 in _COMPLEX_RANK or dn2 in _COMPLEX_RANK:
        def _real_bits(d):
            if d in _COMPLEX_RANK:
                return 32 if d == 'complex64' else 64
            if d in _FLOAT_RANK:
                return {'float16': 16, 'float32': 32, 'float64': 64}[d]
            # Integers: map to the float precision needed
            _int_to_float = {'bool': 16, 'int8': 16, 'uint8': 16,
                             'int16': 16, 'uint16': 16,
                             'int32': 64, 'uint32': 64,
                             'int64': 64, 'uint64': 64}
            return _int_to_float.get(d, 64)
        rb = _real_bits(dn1)
        rc = _real_bits(dn2)
        m = rb if rb > rc else rc
        return 'complex64' if m <= 32 else 'complex128'

    # Float + int -> float (at least float32)
    if dn1 in _FLOAT_RANK or dn2 in _FLOAT_RANK:
        fd = dn1 if dn1 in _FLOAT_RANK else dn2
        other = dn2 if dn1 in _FLOAT_RANK else dn1
        fb = {'float16': 16, 'float32': 32, 'float64': 64}[fd]
        if other in _INT_RANK:
            ob = {'bool': 1, 'int8': 8, 'uint8': 8, 'int16': 16, 'uint16': 16,
                  'int32': 32, 'uint32': 32, 'int64': 64, 'uint64': 64}[other]
            if fb == 16:
                return 'float16' if ob <= 8 else 'float32'
            if fb == 32:
                return 'float64' if ob >= 32 else 'float32'
            return 'float64'
        return fd

    return 'float64'


def _complex_pow(a, b):
    """Complex power that matches numpy semantics for inf/nan.

    Uses C99 cpow semantics: z^w = exp(w * log(z)).
    This handles inf/nan cases that Python's complex.__pow__ gets wrong.
    Also uses repeated multiplication for small integer exponents to avoid
    precision loss from log/exp in RustPython.
    """
    if isinstance(a, complex) and isinstance(b, complex):
        ar, ai = a.real, a.imag
        br, bi = b.real, b.imag
        # If base or exponent contains nan, result is nan+nanj
        if _math.isnan(ar) or _math.isnan(ai) or _math.isnan(br) or _math.isnan(bi):
            return complex(float('nan'), float('nan'))
        # If base contains inf, use log/exp form (C99 cpow)
        if _math.isinf(ar) or _math.isinf(ai) or _math.isinf(br) or _math.isinf(bi):
            try:
                la = _cmath.log(a)
                return _cmath.exp(b * la)
            except (OverflowError, ValueError):
                return complex(float('nan'), float('nan'))
        # For small non-negative integer exponents, use repeated multiplication
        # to avoid precision loss from RustPython's log/exp-based complex pow.
        if bi == 0.0 and br == int(br) and 0 <= int(br) <= 100:
            n = int(br)
            if n == 0:
                return complex(1, 0)
            result = complex(1, 0)
            base = a
            while n > 0:
                if n % 2 == 1:
                    result = complex(result.real * base.real - result.imag * base.imag,
                                     result.real * base.imag + result.imag * base.real)
                base = complex(base.real * base.real - base.imag * base.imag,
                               2 * base.real * base.imag)
                n //= 2
            return result
        # For negative integer exponents, compute positive then invert
        if bi == 0.0 and br == int(br) and -100 <= int(br) < 0:
            pos = _complex_pow(a, complex(-br, 0))
            return complex(1, 0) / pos
        try:
            return _operator.pow(a, b)
        except (OverflowError, ValueError):
            try:
                la = _cmath.log(a)
                return _cmath.exp(b * la)
            except (OverflowError, ValueError):
                return complex(float('nan'), float('nan'))
    return _operator.pow(a, b)


def _safe_pow(a, b):
    """Power that uses complex pow for complex operands, else operator.pow."""
    if isinstance(a, complex) or isinstance(b, complex):
        return _complex_pow(complex(a), complex(b))
    return _operator.pow(a, b)


_SENTINEL = object()  # unique sentinel for getattr defaults

_BINOP_MAP = {
    '__add__': _operator.add, '__sub__': _operator.sub, '__mul__': _operator.mul,
    '__truediv__': _operator.truediv, '__floordiv__': _operator.floordiv,
    '__mod__': _operator.mod, '__pow__': _safe_pow,
    '__lshift__': _operator.lshift, '__rshift__': _operator.rshift,
    '__and__': _operator.and_, '__or__': _operator.or_, '__xor__': _operator.xor,
}


def _coerce_for_op(val, res_dn):
    """Coerce a value to the Python type matching res_dn."""
    if res_dn in ('complex64', 'complex128'):
        return complex(val)
    if res_dn in ('float16', 'float32', 'float64'):
        if isinstance(val, complex):
            return float(val.real)
        return float(val)
    if isinstance(val, complex):
        return int(val.real)
    return int(val)


def _scalar_binop_result(self_val, self_dn, other, op_name):
    """Perform self OP other, return wrapped numpy scalar or NotImplemented."""
    other_dn = _get_numpy_dtype_name(other)
    if other_dn is not None:
        # Both are numpy scalars: strong-typed promotion
        res_dn = _scalar_promote(self_dn, other_dn)
    elif isinstance(other, (ndarray, _ObjectArray)):
        return NotImplemented
    elif getattr(other, '__array_ufunc__', _SENTINEL) is None:
        # __array_ufunc__ = None means defer to other's reflected op
        return NotImplemented
    elif isinstance(other, bool):
        # NEP 50: Python bool is weak, promote with bool
        res_dn = _scalar_promote(self_dn, 'bool')
    elif isinstance(other, int):
        # NEP 50: Python int is weak — adopt the numpy scalar's dtype
        if self_dn == 'bool':
            res_dn = 'int8'
        elif self_dn in ('float16', 'float32', 'float64', 'complex64', 'complex128'):
            res_dn = self_dn
        else:
            res_dn = self_dn  # int types keep their dtype
    elif isinstance(other, float):
        # NEP 50: Python float is weak — but float wins over int
        if self_dn in ('float16', 'float32', 'float64'):
            res_dn = self_dn
        elif self_dn in ('complex64', 'complex128'):
            res_dn = self_dn
        else:
            # int/bool + python float -> float64
            res_dn = 'float64'
    elif isinstance(other, complex):
        # NEP 50: Python complex with zero imaginary is treated as float64 (matches array behavior).
        # Python complex with nonzero imaginary promotes to complex128.
        if other.imag == 0:
            if self_dn in ('complex64', 'complex128'):
                res_dn = 'complex128'
            else:
                res_dn = 'float64'
        else:
            res_dn = 'complex128'
    else:
        return NotImplemented

    # truediv of integers always returns float
    if op_name == '__truediv__' and res_dn in ('bool', 'int8', 'int16', 'int32', 'int64',
                                                'uint8', 'uint16', 'uint32', 'uint64'):
        res_dn = 'float64'

    op_func = _BINOP_MAP.get(op_name)
    if op_func is None:
        return NotImplemented

    a = _coerce_for_op(self_val, res_dn)
    b = _coerce_for_op(other, res_dn)
    try:
        val = op_func(a, b)
    except ZeroDivisionError:
        if op_name in ('__floordiv__', '__mod__') and res_dn in (
                'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
            import warnings
            warnings.warn("divide by zero encountered in floor_divide", RuntimeWarning, stacklevel=4)
            return _wrap_scalar_result(0, res_dn)
        raise
    except TypeError:
        return NotImplemented
    if val is NotImplemented:
        return NotImplemented
    return _wrap_scalar_result(val, res_dn)


def _scalar_rbinop_result(self_val, self_dn, other, op_name):
    """Perform other OP self (reverse), return wrapped numpy scalar or NotImplemented."""
    other_dn = _get_numpy_dtype_name(other)
    if other_dn is not None:
        # Both numpy scalars: strong-typed promotion
        res_dn = _scalar_promote(other_dn, self_dn)
    elif isinstance(other, (ndarray, _ObjectArray)):
        return NotImplemented
    elif getattr(other, '__array_ufunc__', _SENTINEL) is None:
        return NotImplemented
    elif isinstance(other, bool):
        res_dn = _scalar_promote('bool', self_dn)
    elif isinstance(other, int):
        # NEP 50: Python int is weak — adopt self's dtype
        if self_dn == 'bool':
            res_dn = 'int8'
        elif self_dn in ('float16', 'float32', 'float64', 'complex64', 'complex128'):
            res_dn = self_dn
        else:
            res_dn = self_dn
    elif isinstance(other, float):
        # NEP 50: Python float is weak — but float wins over int
        if self_dn in ('float16', 'float32', 'float64'):
            res_dn = self_dn
        elif self_dn in ('complex64', 'complex128'):
            res_dn = self_dn
        else:
            res_dn = 'float64'
    elif isinstance(other, complex):
        # NEP 50: Python complex with zero imaginary is treated as float64 (matches array behavior).
        if other.imag == 0:
            if self_dn in ('complex64', 'complex128'):
                res_dn = 'complex128'
            else:
                res_dn = 'float64'
        else:
            res_dn = 'complex128'
    else:
        return NotImplemented

    if op_name == '__truediv__' and res_dn in ('bool', 'int8', 'int16', 'int32', 'int64',
                                                'uint8', 'uint16', 'uint32', 'uint64'):
        res_dn = 'float64'

    op_func = _BINOP_MAP.get(op_name)
    if op_func is None:
        return NotImplemented

    a = _coerce_for_op(other, res_dn)
    b = _coerce_for_op(self_val, res_dn)
    try:
        val = op_func(a, b)
    except ZeroDivisionError:
        if op_name in ('__floordiv__', '__mod__') and res_dn in (
                'bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
            import warnings
            warnings.warn("divide by zero encountered in floor_divide", RuntimeWarning, stacklevel=4)
            return _wrap_scalar_result(0, res_dn)
        raise
    except TypeError:
        return NotImplemented
    if val is NotImplemented:
        return NotImplemented
    return _wrap_scalar_result(val, res_dn)


def _scalar_cmp_result(self_val, self_dn, other, op_name):
    """Perform a comparison operation and return a numpy bool scalar (with .dtype = bool)."""
    import operator as _op_mod
    _CMP_MAP = {
        '__lt__': _op_mod.lt, '__le__': _op_mod.le,
        '__gt__': _op_mod.gt, '__ge__': _op_mod.ge,
        '__eq__': _op_mod.eq, '__ne__': _op_mod.ne,
    }
    op_func = _CMP_MAP.get(op_name)
    if op_func is None:
        return NotImplemented
    other_dn = _get_numpy_dtype_name(other)
    if other_dn is not None:
        other_val = _coerce_for_op(other, other_dn)
    elif isinstance(other, (ndarray, _ObjectArray)):
        return NotImplemented
    elif isinstance(other, bool):
        other_val = other
    elif isinstance(other, int):
        other_val = other
    elif isinstance(other, float):
        other_val = other
    elif isinstance(other, complex):
        if op_name in ('__lt__', '__le__', '__gt__', '__ge__'):
            # Match array behavior: if imag==0, treat as real; else not supported
            if other.imag != 0:
                return NotImplemented
            other_val = other.real
        else:
            other_val = other
    else:
        return NotImplemented
    try:
        if (
            self_dn in ('float16', 'float32')
            and other_dn is None
            and isinstance(other_val, (int, float))
            and op_name in ('__eq__', '__ne__')
        ):
            other_val = float(_wrap_scalar_result(other_val, self_dn))
        result = op_func(self_val, other_val)
    except TypeError:
        return NotImplemented
    return bool_(result)


class _NumpyIntScalar(int):
    _numpy_dtype_name: str

    def __new__(cls, value=0, dtype_name="int64"):
        obj = int.__new__(cls, int(value))
        obj._numpy_dtype_name = dtype_name
        return obj

    def __array_namespace__(self, *, api_version=None):
        import numpy
        return numpy

    @property
    def dtype(self):
        from ._dtype import dtype
        return dtype(self._numpy_dtype_name)

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def size(self):
        return 1

    def __str__(self):
        if self._numpy_dtype_name == 'bool':
            return 'True' if int(self) else 'False'
        return int.__repr__(int(self))

    def __repr__(self):
        return self.__str__()

    def __format__(self, fmt):
        if not fmt and self._numpy_dtype_name == 'bool':
            return self.__str__()
        return int.__format__(int(self), fmt)

    def __round__(self, ndigits=None):
        if ndigits is None:
            return int(self)
        rounded = __import__("builtins").round(int(self), ndigits)
        # Keep int64 round() as numpy scalar (compat tests rely on this).
        if self._numpy_dtype_name == "int64":
            return _NumpyIntScalar(rounded, self._numpy_dtype_name)
        return int(rounded)

    def round(self, ndigits=0):
        return _NumpyIntScalar(__import__("builtins").round(int(self), ndigits), self._numpy_dtype_name)

    @property
    def itemsize(self):
        return _DTYPE_ITEMSIZE.get(self._numpy_dtype_name, 8)

    @property
    def nbytes(self):
        return self.itemsize

    def astype(self, dtype_arg, *args, **kwargs):
        """Cast scalar to a different type."""
        from ._dtype import dtype
        dt = dtype(dtype_arg)
        dn = dt.name
        if dn in ('float16', 'float32', 'float64'):
            return _NumpyFloatScalar(float(self), dn)
        if dn in ('complex64', 'complex128'):
            return _NumpyComplexScalar(complex(self), dn)
        if dn in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'):
            return _NumpyIntScalar(int(self), dn)
        return int(self)

    def is_integer(self):
        return True

    def bit_count(self):
        return int.bit_count(int(self))

    @property
    def device(self):
        return "cpu"

    def to_device(self, device):
        if device == "cpu":
            return self
        raise ValueError(f"Unsupported device: {device}")

    def view(self, dtype_arg):
        """Reinterpret the bytes of this integer scalar as another dtype."""
        import struct as _struct
        dn = str(dtype_arg).replace('numpy.', '')
        src_name = self._numpy_dtype_name
        val = int(self)
        _int_fmts = {'int64': ('q', 8), 'intp': ('q', 8), 'int32': ('i', 4),
                      'int16': ('h', 2), 'int8': ('b', 1),
                      'uint64': ('Q', 8), 'uint32': ('I', 4),
                      'uint16': ('H', 2), 'uint8': ('B', 1)}
        if src_name not in _int_fmts:
            raise TypeError(f"Cannot view {src_name} as {dn}")
        src_fmt, src_size = _int_fmts[src_name]
        raw = _struct.pack(src_fmt, val)
        _dst_map = {
            'float64': ('d', 8, 'float64', True), 'f8': ('d', 8, 'float64', True),
            'float32': ('f', 4, 'float32', True), 'f4': ('f', 4, 'float32', True),
            'float16': ('e', 2, 'float16', True), 'f2': ('e', 2, 'float16', True),
            'int64': ('q', 8, 'int64', False), 'i8': ('q', 8, 'int64', False),
            'int32': ('i', 4, 'int32', False), 'i4': ('i', 4, 'int32', False),
            'int16': ('h', 2, 'int16', False), 'i2': ('h', 2, 'int16', False),
            'int8': ('b', 1, 'int8', False), 'i1': ('b', 1, 'int8', False),
            'uint64': ('Q', 8, 'uint64', False), 'u8': ('Q', 8, 'uint64', False),
            'uint32': ('I', 4, 'uint32', False), 'u4': ('I', 4, 'uint32', False),
            'uint16': ('H', 2, 'uint16', False), 'u2': ('H', 2, 'uint16', False),
            'uint8': ('B', 1, 'uint8', False), 'u1': ('B', 1, 'uint8', False),
        }
        if dn not in _dst_map:
            raise TypeError(f"Cannot view {src_name} as {dn}")
        dst_fmt, dst_size, dst_name, is_float = _dst_map[dn]
        if src_size != dst_size:
            raise TypeError(f"Cannot view {src_name} ({src_size} bytes) as {dn} ({dst_size} bytes)")
        bits = _struct.unpack(dst_fmt, raw)[0]
        if is_float:
            return _NumpyFloatScalar(bits, dst_name)
        return _NumpyIntScalar(bits, dst_name)

    # Arithmetic operators
    def __add__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__add__')

    def __radd__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__add__')

    def __sub__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__sub__')

    def __rsub__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__sub__')

    def __mul__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__mul__')

    def __rmul__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__mul__')

    def __truediv__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__truediv__')

    def __rtruediv__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__truediv__')

    def __floordiv__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__floordiv__')

    def __rfloordiv__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__floordiv__')

    def __mod__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__mod__')

    def __rmod__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__mod__')

    def __pow__(self, other, mod=None):
        if mod is not None:
            return int.__pow__(self, other, mod)
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__pow__')

    def __rpow__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__pow__')

    def __lshift__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__lshift__')

    def __rlshift__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__lshift__')

    def __rshift__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__rshift__')

    def __rrshift__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__rshift__')

    def __and__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__and__')

    def __rand__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__and__')

    def __or__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__or__')

    def __ror__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__or__')

    def __xor__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__xor__')

    def __rxor__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__xor__')

    def __neg__(self):
        return _NumpyIntScalar(int.__neg__(self), self._numpy_dtype_name)

    def __pos__(self):
        return _NumpyIntScalar(int.__pos__(self), self._numpy_dtype_name)

    def __abs__(self):
        return _NumpyIntScalar(int.__abs__(self), self._numpy_dtype_name)

    def __invert__(self):
        return _NumpyIntScalar(int.__invert__(self), self._numpy_dtype_name)

    # Comparison operators — return numpy bool scalar (with .dtype) instead of Python bool
    def __lt__(self, other): return _scalar_cmp_result(int(self), self._numpy_dtype_name, other, '__lt__')
    def __le__(self, other): return _scalar_cmp_result(int(self), self._numpy_dtype_name, other, '__le__')
    def __gt__(self, other): return _scalar_cmp_result(int(self), self._numpy_dtype_name, other, '__gt__')
    def __ge__(self, other): return _scalar_cmp_result(int(self), self._numpy_dtype_name, other, '__ge__')
    def __eq__(self, other): return _scalar_cmp_result(int(self), self._numpy_dtype_name, other, '__eq__')
    def __ne__(self, other): return _scalar_cmp_result(int(self), self._numpy_dtype_name, other, '__ne__')
    __hash__ = int.__hash__


def _float_to_str(val, max_digits, dtype_name='float64'):
    """Format a float with limited significant digits, matching numpy output.
    Uses Dragon4-like shortest-unique representation for float16/float32."""
    import math as _m
    import struct as _struct
    if _m.isnan(val):
        return 'nan'
    if _m.isinf(val):
        return '-inf' if val < 0 else 'inf'
    if val == 0.0:
        if _m.copysign(1.0, val) < 0:
            return '-0.0'
        return '0.0'

    if dtype_name == 'float16':
        # Find shortest representation that round-trips through float16
        pack_fmt, unpack_fmt = 'e', 'e'
        try:
            b1 = _struct.pack(pack_fmt, val)
        except (OverflowError, _struct.error):
            return repr(val)
        for ndig in range(1, 12):
            s = f'{val:.{ndig}g}'
            sv = float(s)
            try:
                b2 = _struct.pack(pack_fmt, sv)
                if b1 == b2:
                    # Ensure decimal point
                    if '.' not in s and 'e' not in s and 'E' not in s:
                        s += '.0'
                    return s
            except (OverflowError, _struct.error):
                continue
        formatted = f'{val:.{max_digits}g}'
    elif dtype_name == 'float32':
        # Find shortest representation that round-trips through float32
        try:
            b1 = _struct.pack('f', val)
        except (OverflowError, _struct.error):
            return repr(val)
        for ndig in range(1, 20):
            s = f'{val:.{ndig}g}'
            sv = float(s)
            try:
                b2 = _struct.pack('f', sv)
                if b1 == b2:
                    if '.' not in s and 'e' not in s and 'E' not in s:
                        s += '.0'
                    return s
            except (OverflowError, _struct.error):
                continue
        formatted = f'{val:.{max_digits}g}'
    else:
        formatted = f'{val:.{max_digits}g}'

    # Ensure we always have a decimal point for float
    if '.' not in formatted and 'e' not in formatted and 'E' not in formatted:
        formatted += '.0'
    return formatted


def _truncate_float(value, dtype_name):
    """Truncate a float value to the precision of the target dtype."""
    import struct as _struct
    fval = float(value)
    if dtype_name == "float32":
        return _struct.unpack('f', _struct.pack('f', fval))[0]
    elif dtype_name == "float16":
        # Use struct 'e' format for IEEE 754 half-precision
        try:
            return _struct.unpack('e', _struct.pack('e', fval))[0]
        except (OverflowError, _struct.error):
            return fval
    return fval


class _NumpyFloatScalar(float):
    _numpy_dtype_name: str

    def __new__(cls, value=0.0, dtype_name="float64"):
        fval = _truncate_float(float(value), dtype_name)
        obj = float.__new__(cls, fval)
        obj._numpy_dtype_name = dtype_name
        return obj

    def __array_namespace__(self, *, api_version=None):
        import numpy
        return numpy

    def __repr__(self):
        dn = self._numpy_dtype_name
        s = self.__str__()
        return f'np.{dn}({s})'

    def __str__(self):
        val = float(self)
        dn = self._numpy_dtype_name
        # Use appropriate precision for dtype
        if dn == 'float16':
            return _float_to_str(val, 5, dtype_name='float16')
        elif dn == 'float32':
            return _float_to_str(val, 8, dtype_name='float32')
        # float64 uses Python's default str
        return float.__repr__(val)

    def __format__(self, fmt):
        if not fmt:
            return self.__str__()
        return float.__format__(float(self), fmt)

    @property
    def dtype(self):
        from ._dtype import dtype
        return dtype(self._numpy_dtype_name)

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def size(self):
        return 1

    def __round__(self, ndigits=None):
        _builtin_round = __import__("builtins").round
        if ndigits is None:
            return int(_builtin_round(float(self)))
        return _NumpyFloatScalar(_builtin_round(float(self), ndigits), self._numpy_dtype_name)

    def round(self, ndigits=0):
        _builtin_round = __import__("builtins").round
        return _NumpyFloatScalar(_builtin_round(float(self), ndigits), self._numpy_dtype_name)

    @property
    def itemsize(self):
        return _DTYPE_ITEMSIZE.get(self._numpy_dtype_name, 8)

    @property
    def nbytes(self):
        return self.itemsize

    def astype(self, dtype_arg, *args, **kwargs):
        """Cast scalar to a different type."""
        from ._dtype import dtype
        dt = dtype(dtype_arg)
        dn = dt.name
        if dn in ('float16', 'float32', 'float64'):
            return _NumpyFloatScalar(float(self), dn)
        if dn in ('complex64', 'complex128'):
            return _NumpyComplexScalar(complex(self), dn)
        if dn in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'):
            return _NumpyIntScalar(int(self), dn)
        return float(self)

    def as_integer_ratio(self):
        """Return (numerator, denominator) pair."""
        return float.as_integer_ratio(float(self))

    def is_integer(self):
        return float.is_integer(float(self))

    @property
    def device(self):
        return "cpu"

    def to_device(self, device):
        if device == "cpu":
            return self
        raise ValueError(f"Unsupported device: {device}")

    def view(self, dtype_arg):
        """Reinterpret the bytes of this float scalar as another dtype."""
        import struct as _struct
        dn = str(dtype_arg).replace('numpy.', '')
        src_name = self._numpy_dtype_name
        val = float(self)
        _float_fmts = {'float64': ('d', 8), 'float32': ('f', 4), 'float16': ('e', 2)}
        if src_name not in _float_fmts:
            raise TypeError(f"Cannot view {src_name} as {dn}")
        src_fmt, src_size = _float_fmts[src_name]
        raw = _struct.pack(src_fmt, val)
        _dst_map = {
            'float64': ('d', 8, 'float64', True), 'f8': ('d', 8, 'float64', True),
            'float32': ('f', 4, 'float32', True), 'f4': ('f', 4, 'float32', True),
            'float16': ('e', 2, 'float16', True), 'f2': ('e', 2, 'float16', True),
            'int64': ('q', 8, 'int64', False), 'i8': ('q', 8, 'int64', False),
            'int32': ('i', 4, 'int32', False), 'i4': ('i', 4, 'int32', False),
            'int16': ('h', 2, 'int16', False), 'i2': ('h', 2, 'int16', False),
            'int8': ('b', 1, 'int8', False), 'i1': ('b', 1, 'int8', False),
            'uint64': ('Q', 8, 'uint64', False), 'u8': ('Q', 8, 'uint64', False),
            'uint32': ('I', 4, 'uint32', False), 'u4': ('I', 4, 'uint32', False),
            'uint16': ('H', 2, 'uint16', False), 'u2': ('H', 2, 'uint16', False),
            'uint8': ('B', 1, 'uint8', False), 'u1': ('B', 1, 'uint8', False),
        }
        if dn not in _dst_map:
            raise TypeError(f"Cannot view {src_name} as {dn}")
        dst_fmt, dst_size, dst_name, is_float = _dst_map[dn]
        if src_size != dst_size:
            raise TypeError(f"Cannot view {src_name} ({src_size} bytes) as {dn} ({dst_size} bytes)")
        bits = _struct.unpack(dst_fmt, raw)[0]
        if is_float:
            return _NumpyFloatScalar(bits, dst_name)
        return _NumpyIntScalar(bits, dst_name)

    # Arithmetic operators
    def __add__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__add__')

    def __radd__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__add__')

    def __sub__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__sub__')

    def __rsub__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__sub__')

    def __mul__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__mul__')

    def __rmul__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__mul__')

    def __truediv__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__truediv__')

    def __rtruediv__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__truediv__')

    def __floordiv__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__floordiv__')

    def __rfloordiv__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__floordiv__')

    def __mod__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__mod__')

    def __rmod__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__mod__')

    def __pow__(self, other, mod=None):
        if mod is not None:
            return NotImplemented
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__pow__')

    def __rpow__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__pow__')

    def __neg__(self):
        return _NumpyFloatScalar(float.__neg__(self), self._numpy_dtype_name)

    def __pos__(self):
        return _NumpyFloatScalar(float.__pos__(self), self._numpy_dtype_name)

    def __abs__(self):
        return _NumpyFloatScalar(float.__abs__(self), self._numpy_dtype_name)

    # Comparison operators — return numpy bool scalar (with .dtype) instead of Python bool
    def __lt__(self, other): return _scalar_cmp_result(float(self), self._numpy_dtype_name, other, '__lt__')
    def __le__(self, other): return _scalar_cmp_result(float(self), self._numpy_dtype_name, other, '__le__')
    def __gt__(self, other): return _scalar_cmp_result(float(self), self._numpy_dtype_name, other, '__gt__')
    def __ge__(self, other): return _scalar_cmp_result(float(self), self._numpy_dtype_name, other, '__ge__')
    def __eq__(self, other): return _scalar_cmp_result(float(self), self._numpy_dtype_name, other, '__eq__')
    def __ne__(self, other): return _scalar_cmp_result(float(self), self._numpy_dtype_name, other, '__ne__')
    __hash__ = float.__hash__


class _NumpyComplexScalar(complex):
    _numpy_dtype_name: str

    def __new__(cls, value=0j, dtype_name="complex128"):
        cval = complex(value)
        obj = complex.__new__(cls, cval)
        obj._numpy_dtype_name = dtype_name
        return obj

    @property
    def dtype(self):
        from ._dtype import dtype
        return dtype(self._numpy_dtype_name)

    @property
    def shape(self):
        return ()

    @property
    def ndim(self):
        return 0

    @property
    def size(self):
        return 1

    def __array_namespace__(self, *, api_version=None):
        import numpy
        return numpy

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return complex.__repr__(complex(self))

    def __repr__(self):
        return complex.__repr__(complex(self))

    def __format__(self, fmt):
        if not fmt:
            return self.__str__()
        return complex.__format__(complex(self), fmt)

    @property
    def itemsize(self):
        return _DTYPE_ITEMSIZE.get(self._numpy_dtype_name, 16)

    @property
    def nbytes(self):
        return self.itemsize

    def astype(self, dtype_arg, *args, **kwargs):
        """Cast scalar to a different type."""
        from ._dtype import dtype
        dt = dtype(dtype_arg)
        dn = dt.name
        if dn in ('float16', 'float32', 'float64'):
            return _NumpyFloatScalar(float(self.real), dn)
        if dn in ('complex64', 'complex128'):
            return _NumpyComplexScalar(complex(self), dn)
        if dn in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'):
            return _NumpyIntScalar(int(self.real), dn)
        return complex(self)

    @property
    def device(self):
        return "cpu"

    def to_device(self, device):
        if device == "cpu":
            return self
        raise ValueError(f"Unsupported device: {device}")

    # Arithmetic operators
    def __add__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__add__')

    def __radd__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__add__')

    def __sub__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__sub__')

    def __rsub__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__sub__')

    def __mul__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__mul__')

    def __rmul__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__mul__')

    def __truediv__(self, other):
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__truediv__')

    def __rtruediv__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__truediv__')

    def __pow__(self, other, mod=None):
        if mod is not None:
            return NotImplemented
        return _scalar_binop_result(self, self._numpy_dtype_name, other, '__pow__')

    def __rpow__(self, other):
        return _scalar_rbinop_result(self, self._numpy_dtype_name, other, '__pow__')

    def __neg__(self):
        return _NumpyComplexScalar(complex.__neg__(self), self._numpy_dtype_name)

    def __pos__(self):
        return _NumpyComplexScalar(complex.__pos__(self), self._numpy_dtype_name)

    def __abs__(self):
        # abs(complex) returns float
        res_dn = 'float32' if self._numpy_dtype_name == 'complex64' else 'float64'
        return _NumpyFloatScalar(complex.__abs__(self), res_dn)

    # Comparison operators: eq/ne return numpy bool scalar; lt/le/gt/ge raise TypeError
    def __eq__(self, other):
        return _scalar_cmp_result(complex(self), self._numpy_dtype_name, other, '__eq__')
    def __ne__(self, other):
        return _scalar_cmp_result(complex(self), self._numpy_dtype_name, other, '__ne__')
    def __lt__(self, other): raise TypeError(f"'<' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    def __le__(self, other): raise TypeError(f"'<=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    def __gt__(self, other): raise TypeError(f"'>' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    def __ge__(self, other): raise TypeError(f"'>=' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
    __hash__ = complex.__hash__

    def to_device(self, device):
        if device == "cpu":
            return self
        raise ValueError(f"Unsupported device: {device}")


# Void scalar placeholder
class _NumpyVoidScalar:
    def __init__(self, value=None):
        self._value = value
        self._numpy_dtype_name = "void"
        self._is_void = True
    def __bool__(self):
        return False
    def __repr__(self):
        return "numpy.void()"


# Metaclass for scalar type classes so the CLASS itself has custom __str__, __eq__, __hash__
class _ScalarTypeMeta(type):
    """Metaclass for numpy scalar type classes in the type hierarchy."""
    _scalar_name: str
    _python_type: type

    def __new__(mcs, name, bases, namespace, scalar_name=None, python_type=float):
        cls = super().__new__(mcs, name, bases, namespace)
        cls._scalar_name = scalar_name or name
        cls._python_type = python_type
        return cls

    def __init__(cls, name, bases, namespace, scalar_name=None, python_type=float):
        super().__init__(name, bases, namespace)

    def __subclasscheck__(cls, subclass):
        """Allow issubclass(type(scalar), np.floating) etc."""
        # Standard check first
        if super().__subclasscheck__(subclass):
            return True
        # Map Python scalar types to abstract numpy types
        _abstract_map = {
            'generic': (int, float, complex, bool),
            'number': (int, float, complex),
            'integer': (int,),
            'signedinteger': (int,),
            'inexact': (float, complex),
            'floating': (float,),
            'complexfloating': (complex,),
        }
        name = getattr(cls, '_scalar_name', '')
        if name in _abstract_map:
            for pytype in _abstract_map[name]:
                if issubclass(subclass, pytype):
                    return True
        return False

    def __instancecheck__(cls, instance):
        """Allow isinstance(scalar, np.floating) etc."""
        if super().__instancecheck__(instance):
            return True
        return cls.__subclasscheck__(type(instance))

    def __call__(cls, value=0, *args, **kwargs):
        scalar_name = cls._scalar_name
        # Reject unexpected keyword arguments for non-str/bytes types
        if kwargs and scalar_name not in ('str', 'bytes'):
            if scalar_name == 'void':
                # void accepts dtype keyword
                unexpected = set(kwargs) - {'dtype'}
            elif scalar_name in ('datetime64',):
                unexpected = set(kwargs)
            else:
                unexpected = set(kwargs)
            if unexpected:
                raise TypeError(
                    f"{scalar_name}() takes no keyword arguments")
        # If given a list/tuple/ndarray, create an array with this dtype
        if isinstance(value, (list, tuple)):
            import numpy as _np
            return _np.array(value, dtype=scalar_name)
        if isinstance(value, ndarray):
            return value.astype(scalar_name)
        if scalar_name in ('complex64', 'complex128') and len(args) == 1:
            real_part = value
            imag_part = args[0]
            # NumPy requires both parts to be real numbers (int/float), not
            # complex or None.
            if isinstance(real_part, complex) or isinstance(imag_part, complex):
                raise TypeError(
                    "complex() can't take second arg if first is a complex")
            if real_part is None or imag_part is None:
                raise TypeError(
                    "complex() first argument must be a string or a number, "
                    "not 'NoneType'")
            try:
                value = complex(real_part, imag_part)
            except (ValueError, TypeError):
                raise
        # Unsigned integer wrapping (negative values wrap around)
        if scalar_name in ('uint8', 'uint16', 'uint32', 'uint64'):
            try:
                iv = int(value)
            except (ValueError, TypeError):
                pass
            else:
                if iv < 0:
                    bits = {'uint8': 8, 'uint16': 16, 'uint32': 32, 'uint64': 64}[scalar_name]
                    value = iv & ((1 << bits) - 1)
        # For str/bytes, pass through extra args (e.g. encoding)
        if scalar_name in ('str', 'bytes'):
            try:
                base_value = cls._python_type(value, *args, **kwargs)
            except UnicodeError:
                raise  # Let encoding errors propagate
            except (ValueError, TypeError):
                if cls._python_type is bytes and not args and not kwargs:
                    return str(value).encode('ascii')
                return value
            return base_value
        if scalar_name == 'bool':
            try:
                base_value = cls._python_type(value)
            except (ValueError, TypeError):
                return value
            return cls.__new__(cls, base_value)
        try:
            base_value = cls._python_type(value)
        except (ValueError, TypeError):
            return value
        if scalar_name in ('int8', 'int16', 'int32', 'int64',
                           'uint8', 'uint16', 'uint32', 'uint64'):
            return _NumpyIntScalar(base_value, scalar_name)
        if scalar_name in ('float16', 'float32', 'float64'):
            return _NumpyFloatScalar(base_value, scalar_name)
        if scalar_name in ('complex64', 'complex128'):
            return _NumpyComplexScalar(base_value, scalar_name)
        if scalar_name == 'void':
            return _NumpyVoidScalar(value)
        return base_value

    def __repr__(cls):
        return f"<class 'numpy.{cls._scalar_name}'>"

    def __str__(cls):
        return cls._scalar_name

    def fromhex(cls, string):
        """Create a scalar from a hexadecimal string (like float.fromhex)."""
        fval = float.fromhex(string)
        return cls(fval)

    def __eq__(cls, other):
        if isinstance(other, _ScalarTypeMeta):
            return cls._scalar_name == other._scalar_name
        if isinstance(other, _ScalarType):
            return cls._scalar_name == other._name
        if isinstance(other, str):
            return cls._scalar_name == other
        # Handle comparison with dtype (use local import to avoid circular)
        try:
            from ._dtype import dtype as _dtype_cls
            if isinstance(other, _dtype_cls):
                return cls._scalar_name == other.name
        except ImportError:
            pass
        # Handle comparison with concrete scalar type classes
        # e.g., type(np.int64(1)) == np.int64  →  _NumpyIntScalar == int64
        if isinstance(other, type):
            _name_map = {
                '_NumpyIntScalar': {'int8', 'int16', 'int32', 'int64',
                                    'uint8', 'uint16', 'uint32', 'uint64',
                                    'bool', 'intp'},
                '_NumpyFloatScalar': {'float16', 'float32', 'float64'},
                '_NumpyComplexScalar': {'complex64', 'complex128'},
                '_NumpyVoidScalar': {'void'},
            }
            other_name = getattr(other, '__name__', '')
            if other_name in _name_map:
                return cls._scalar_name in _name_map[other_name]
        return type.__eq__(cls, other)

    def __hash__(cls):
        return hash(cls._scalar_name)

    def __instancecheck__(cls, instance):
        """Allow isinstance(3, np.integer) etc. to work."""
        scalar_name = cls._scalar_name

        # Map Python types to numpy type hierarchy
        if isinstance(instance, bool):
            # bool is a subclass of int in Python, check it first
            return scalar_name in ('bool', 'generic', 'number', 'integer', 'signedinteger')
        if isinstance(instance, int):
            return scalar_name in ('generic', 'number', 'integer', 'signedinteger',
                                   'int8', 'int16', 'int32', 'int64',
                                   'uint8', 'uint16', 'uint32', 'uint64',
                                   'unsignedinteger', 'intp')
        if isinstance(instance, float):
            return scalar_name in ('generic', 'number', 'inexact', 'floating',
                                   'float16', 'float32', 'float64')
        if isinstance(instance, complex):
            return scalar_name in ('generic', 'number', 'inexact', 'complexfloating',
                                   'complex64', 'complex128')
        if isinstance(instance, str):
            return scalar_name in ('generic', 'character', 'str')
        if isinstance(instance, bytes):
            return scalar_name in ('generic', 'character', 'bytes')
        return False


# ---------------------------------------------------------------------------
# typecodes
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Type hierarchy classes
# ---------------------------------------------------------------------------
_ABSTRACT_SCALAR_TYPES = {
    'number', 'integer', 'signedinteger', 'unsignedinteger',
    'inexact', 'floating', 'complexfloating',
}
# Concrete types that are intentionally subscriptable
_SUBSCRIPTABLE_CONCRETE = {'bool', 'datetime64', 'timedelta64'}


class generic(metaclass=_ScalarTypeMeta, scalar_name="generic"):
    """Base class for all numpy scalar types."""
    def __class_getitem__(cls, item):
        import types as _types
        name = cls._scalar_name
        if name in _ABSTRACT_SCALAR_TYPES:
            # complexfloating accepts 1 or 2 type args
            if name == 'complexfloating':
                if isinstance(item, tuple):
                    if len(item) < 1 or len(item) > 2:
                        what = 'few' if len(item) == 0 else 'many'
                        raise TypeError(
                            f"Too {what} arguments for "
                            f"{cls}")
                else:
                    item = (item,)
            else:
                # Other abstract types accept exactly 1 type arg
                if isinstance(item, tuple):
                    if len(item) != 1:
                        raise TypeError(
                            f"Too {'few' if len(item) == 0 else 'many'} "
                            f"arguments for {cls}")
                    item = item[0]
            return _types.GenericAlias(cls, item)
        if name in _SUBSCRIPTABLE_CONCRETE:
            return _types.GenericAlias(cls, item)
        raise TypeError(
            f"There are no type variables left in {cls}")

    # Dunder stubs for ndarray interface compatibility
    def __array_namespace__(self, *, api_version=None):
        import numpy; return numpy
    def __copy__(self): return self
    def __deepcopy__(self, memo=None): return self
    def __array__(self, dtype=None): import numpy; return numpy.array(self, dtype=dtype)
    def __array_wrap__(self, array, context=None, return_scalar=False): return array

    # Stub methods matching ndarray interface (for scalar_methods tests)
    def all(self, axis=None, out=None, keepdims=False): return bool(self)
    def any(self, axis=None, out=None, keepdims=False): return bool(self)
    def argmax(self, axis=None, out=None, **kw): return 0
    def argmin(self, axis=None, out=None, **kw): return 0
    def argsort(self, axis=-1, kind=None, order=None): import numpy; return numpy.array([0])
    def clip(self, min=None, max=None, out=None, **kw):
        v = self
        if min is not None and v < min: v = type(self)(min)
        if max is not None and v > max: v = type(self)(max)
        return v
    def compress(self, condition, axis=None, out=None): import numpy; return numpy.asarray(self).compress(condition, axis)
    def conjugate(self): return self
    conj = conjugate
    def copy(self, order='C'): return self
    def cumprod(self, axis=None, dtype=None, out=None): import numpy; return numpy.array([self])
    def cumsum(self, axis=None, dtype=None, out=None): import numpy; return numpy.array([self])
    def diagonal(self, offset=0, axis1=0, axis2=1): return self
    def flatten(self, order='C'): import numpy; return numpy.array([self])
    def item(self, *args): return self.__class__(self)
    def max(self, axis=None, out=None, keepdims=False): return self
    def mean(self, axis=None, dtype=None, out=None, keepdims=False): return float(self)
    def min(self, axis=None, out=None, keepdims=False): return self
    def nonzero(self):
        import numpy
        if bool(self):
            return (numpy.array([0]),)
        return (numpy.array([], dtype='int64'),)
    def prod(self, axis=None, dtype=None, out=None, keepdims=False): return self
    def ptp(self, axis=None, out=None, keepdims=False): return type(self)(0)
    def put(self, indices, values, mode='raise'): pass
    def ravel(self, order='C'): import numpy; return numpy.array([self])
    def repeat(self, repeats, axis=None): import numpy; return numpy.array([self] * int(repeats))
    def reshape(self, *shape, order='C'): import numpy; return numpy.array(self).reshape(*shape)
    def round(self, decimals=0, out=None): return type(self)(__import__('builtins').round(float(self), decimals))
    def searchsorted(self, v, side='left', sorter=None): import numpy; return numpy.searchsorted(numpy.array([self]), v, side=side)
    def squeeze(self, axis=None): return self
    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False): return type(self)(0)
    def sum(self, axis=None, dtype=None, out=None, keepdims=False): return self
    def swapaxes(self, axis1, axis2): return self
    def take(self, indices, axis=None, out=None, mode='raise'): import numpy; return numpy.array([self]).take(indices, axis=axis)
    def tobytes(self, order='C'): import struct; return struct.pack('d', float(self))
    def tolist(self): return self.__class__(self)
    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None): return self
    def transpose(self, *axes): return self
    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False): return type(self)(0)

    # Additional ndarray-compatible stubs
    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        import numpy; return numpy.array(self).astype(dtype)
    def byteswap(self, inplace=False): return self
    def choose(self, choices, out=None, mode='raise'):
        import numpy; return numpy.array(self).choose(choices, out=out, mode=mode)
    def dump(self, file): pass
    def dumps(self): return b''
    def fill(self, value): pass
    def getfield(self, dtype, offset=0):
        import numpy; return numpy.array(self).getfield(dtype, offset)
    def resize(self, new_shape, refcheck=True):
        import numpy; return numpy.array(self).resize(new_shape, refcheck=refcheck)
    def setfield(self, val, dtype, offset=0): pass
    def setflags(self, write=None, align=None, uic=None): pass
    def sort(self, axis=-1, kind=None, order=None): pass
    def to_device(self, device, /, stream=None): return self
    def tofile(self, fid, sep='', format='%s'): pass
    def view(self, dtype=None, type=None):
        import numpy; return numpy.array(self).view(dtype)

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

class bool_(_NumpyIntScalar, generic, metaclass=_ScalarTypeMeta, scalar_name="bool", python_type=bool):
    """Boolean scalar type."""
    _TRUE_SINGLETON = None
    _FALSE_SINGLETON = None

    def __new__(cls, value=0, dtype_name="bool"):
        truth = 1 if bool(value) else 0
        if truth:
            if cls._TRUE_SINGLETON is None:
                obj = _NumpyIntScalar.__new__(cls, 1, 'bool')
                cls._TRUE_SINGLETON = obj
            return cls._TRUE_SINGLETON
        if cls._FALSE_SINGLETON is None:
            obj = _NumpyIntScalar.__new__(cls, 0, 'bool')
            cls._FALSE_SINGLETON = obj
        return cls._FALSE_SINGLETON

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

# Aliases — point to the _ScalarTypeMeta classes so that ``dtype.type is alias``
# holds (identity check).  ``_ScalarType`` instances are only kept for types
# that have no _ScalarTypeMeta class counterpart (float128, object_, string_…).
float128 = _ScalarType("float128", float)
intp = int64
intc = int32
uintp = uint64
byte = int8
ubyte = uint8
short = int16
ushort = uint16
longlong = int64
ulonglong = uint64
single = float32
double = float64
longdouble = float64
csingle = complex64
cdouble = complex128
clongdouble = complex128
longfloat = float64
clongfloat = complex128
longcomplex = complex128
object_ = _ScalarType("object", object)

# More scalar aliases (set after datetime section in original __init__.py)
string_ = _ScalarType("str", str)
unicode_ = _ScalarType("str", str)
half = float16
int_ = int64
float_ = float64
complex_ = complex128
uint = uint64
long = int64
ulong = uint64

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
True_ = bool_(True)
False_ = bool_(False)

# ---------------------------------------------------------------------------
# sctypes and sctypeDict
# ---------------------------------------------------------------------------
sctypes = {
    'int': [int8, int16, int32, int64],
    'uint': [uint8, uint16, uint32, uint64],
    'float': [float16, float32, float64],
    'complex': [complex64, complex128],
    'others': [bool_, object_, str_, bytes_, void],
}
sctypeDict = {
    'int8': int8, 'int16': int16, 'int32': int32, 'int64': int64,
    'uint8': uint8, 'uint16': uint16, 'uint32': uint32, 'uint64': uint64,
    'float16': float16, 'float32': float32, 'float64': float64,
    'complex64': complex64, 'complex128': complex128,
    'bool': bool_, 'object': object_, 'str': str_, 'bytes': bytes_,
    'i1': int8, 'i2': int16, 'i4': int32, 'i8': int64,
    'u1': uint8, 'u2': uint16, 'u4': uint32, 'u8': uint64,
    'f2': float16, 'f4': float32, 'f8': float64,
    'c8': complex64, 'c16': complex128,
}
