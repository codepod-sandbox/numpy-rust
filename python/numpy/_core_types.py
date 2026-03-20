"""Type system: scalar types, dtype class, type hierarchy, finfo/iinfo, type casting."""
import sys as _sys
import math as _math
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
    # dtype
    'dtype', 'StructuredDtype', '_DTypeClassMeta',
    # Per-dtype DType classes
    'Float64DType', 'Float32DType', 'Float16DType',
    'Int8DType', 'Int16DType', 'Int32DType', 'Int64DType',
    'UInt8DType', 'UInt16DType', 'UInt32DType', 'UInt64DType',
    'Complex64DType', 'Complex128DType', 'BoolDType', 'StrDType',
    'BytesDType', 'VoidDType', 'ObjectDType',
    # Info
    'finfo', 'iinfo', '_MachAr',
    # Type casting + dtype normalization
    'can_cast', 'result_type', 'promote_types', 'find_common_type',
    'common_type', 'mintypecode', '_normalize_dtype', '_normalize_dtype_with_size',
    '_string_dtype_info', '_DTYPE_CHAR_MAP',
    # Constants and aliases
    'True_', 'False_', 'int_', 'typecodes', 'sctypes', 'sctypeDict',
    'float128', 'intp', 'intc', 'uintp', 'byte', 'ubyte', 'short', 'ushort',
    'longlong', 'ulonglong', 'single', 'double', 'longdouble',
    'csingle', 'cdouble', 'clongdouble',
    'string_', 'unicode_', 'half', 'float_', 'complex_', 'uint', 'long', 'ulong',
    'longfloat', 'clongfloat', 'longcomplex',
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
    if dtype_name in ('bool', 'int8', 'int16', 'int32', 'int64',
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


import operator as _operator
import cmath as _cmath
import math as _math


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
        return float(val)
    return int(val)


def _scalar_binop_result(self_val, self_dn, other, op_name):
    """Perform self OP other, return wrapped numpy scalar or NotImplemented."""
    other_dn = _get_numpy_dtype_name(other)
    if other_dn is not None:
        # Both are numpy scalars: strong-typed promotion
        res_dn = _scalar_promote(self_dn, other_dn)
    elif isinstance(other, (ndarray, _ObjectArray)):
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
        # NEP 50: Python complex is weak
        if self_dn in ('complex64', 'complex128'):
            res_dn = self_dn
        elif self_dn in ('float16', 'float32'):
            res_dn = 'complex64'
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
        # NEP 50: Python complex is weak
        if self_dn in ('complex64', 'complex128'):
            res_dn = self_dn
        elif self_dn in ('float16', 'float32'):
            res_dn = 'complex64'
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
    except TypeError:
        return NotImplemented
    if val is NotImplemented:
        return NotImplemented
    return _wrap_scalar_result(val, res_dn)


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
        except (OverflowError, struct.error):
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


class _NumpyComplexScalar(complex):
    _numpy_dtype_name: str

    def __new__(cls, value=0j, dtype_name="complex128"):
        cval = complex(value)
        obj = complex.__new__(cls, cval)
        obj._numpy_dtype_name = dtype_name
        return obj

    @property
    def dtype(self):
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
        # Unsigned integer overflow check
        if scalar_name in ('uint8', 'uint16', 'uint32', 'uint64'):
            try:
                iv = int(value)
            except (ValueError, TypeError):
                pass
            else:
                if iv < 0:
                    raise OverflowError(
                        "can't convert negative value to " + scalar_name)
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
        try:
            base_value = cls._python_type(value)
        except (ValueError, TypeError):
            return value
        if scalar_name in ('bool', 'int8', 'int16', 'int32', 'int64',
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
        if isinstance(other, dtype):
            return cls._scalar_name == other.name
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
# StructuredDtype
# ---------------------------------------------------------------------------
class StructuredDtype:
    """Minimal structured dtype for field-based array access."""
    def __init__(self, fields):
        # fields is list of (name, dtype) or (name, dtype, shape)
        self._fields = []
        self._names = []
        for f in fields:
            name = f[0]
            dt = dtype(f[1]) if not isinstance(f[1], str) else f[1]
            self._fields.append((name, dt))
            self._names.append(name)
        self.names = tuple(self._names)
        self.fields = {}
        offset = 0
        for name, dt in self._fields:
            if isinstance(dt, str):
                dt_obj = dtype(dt)
            else:
                dt_obj = dt
            self.fields[name] = (dt_obj, offset)
            offset += dt_obj.itemsize if hasattr(dt_obj, 'itemsize') else 8
        self.itemsize = offset
        self.kind = 'V'
        self.char = 'V'
        self.name = 'void'
        self.str = '|V{}'.format(self.itemsize)
        self.descr = [(name, str(self.fields[name][0])) for name in self._names]

    def __repr__(self):
        parts = ', '.join("('{}', '{}')".format(n, d) for n, d in self._fields)
        return 'dtype([{}])'.format(parts)

    def __str__(self):
        return repr(self)

    def _normalized_fields(self):
        """Return fields with type strings normalized to canonical names."""
        result = []
        for name, dt in self._fields:
            # Normalize: strip endian prefix and map to canonical name
            dt_s = str(dt)
            if isinstance(dt_s, str) and dt_s and dt_s[0] in '<>|=':
                dt_s = dt_s[1:]
            # Map short codes to canonical names for comparison
            _short_to_canonical = {
                'i1': 'int8', 'i2': 'int16', 'i4': 'int32', 'i8': 'int64',
                'u1': 'uint8', 'u2': 'uint16', 'u4': 'uint32', 'u8': 'uint64',
                'f2': 'float16', 'f4': 'float32', 'f8': 'float64',
                'c8': 'complex64', 'c16': 'complex128', 'b1': 'bool',
            }
            canonical = _short_to_canonical.get(dt_s, dt_s)
            result.append((name, canonical))
        return result

    def __eq__(self, other):
        if isinstance(other, StructuredDtype):
            return self._normalized_fields() == other._normalized_fields()
        return False

    def __hash__(self):
        return hash(tuple(self._normalized_fields()))


# ---------------------------------------------------------------------------
# Metaclass for per-dtype DType classes
# ---------------------------------------------------------------------------
# Gives classes like Float64DType a custom __str__ so str(Float64DType)='float64'.
class _DTypeClassMeta(type):
    def __str__(cls):
        return cls._dtype_class_name
    def __repr__(cls):
        return f"numpy.dtypes.{cls.__name__}"
    def __eq__(cls, other):
        if isinstance(other, _DTypeClassMeta):
            return cls._dtype_class_name == other._dtype_class_name
        return NotImplemented
    def __hash__(cls):
        return hash(cls._dtype_class_name)


def _parse_comma_dtype(s):
    """Parse a comma-separated dtype string like 'i4,f8' into a StructuredDtype.

    Auto-generates field names f0, f1, f2, ...
    Each comma-separated part is parsed as a dtype string.
    """
    parts = s.split(',')
    fields = []
    for i, part in enumerate(parts):
        part = part.strip()
        fields.append(('f{}'.format(i), part))
    return StructuredDtype(fields)


# ---------------------------------------------------------------------------
# dtype class
# ---------------------------------------------------------------------------
class dtype:
    """Stub for numpy dtype objects."""

    _dtype_class_map = {}  # filled after DType subclasses are defined below

    def __new__(cls, tp=None, metadata=None, align=False):
        if cls is dtype:
            # Determine canonical dtype name to pick the right subclass
            name = None
            if isinstance(tp, type) and hasattr(tp, '_dtype_class_name'):
                name = tp._dtype_class_name
            elif isinstance(tp, str):
                # Comma-separated dtype strings like 'i4,f8' -> structured dtype
                if ',' in tp:
                    return object.__new__(cls)
                name = _DTYPE_CHAR_MAP.get(tp, tp)
                # Handle arbitrary |Sn/Sn -> 'bytes', |Vn/Vn -> 'void', <Un/Un -> 'str'
                if isinstance(name, str):
                    if name.startswith('|S') and len(name) > 2 and name[2:].isdigit():
                        name = 'bytes'
                    elif name.startswith('S') and len(name) > 1 and name[1:].isdigit():
                        name = 'bytes'
                    elif name.startswith('|V') and len(name) > 2 and name[2:].isdigit():
                        name = 'void'
                    elif name.startswith('V') and len(name) > 1 and name[1:].isdigit():
                        name = 'void'
                    elif len(name) > 1 and (name.startswith('<U') or name.startswith('>U')):
                        if name[2:].isdigit():
                            name = 'str'
                    elif name.startswith('U') and len(name) > 1 and name[1:].isdigit():
                        name = 'str'
                    elif len(name) >= 2 and name[0].isdigit():
                        name = 'void'
            elif tp is object:
                name = 'object'
            elif tp is float or tp is float64:
                name = 'float64'
            elif tp is int or tp is int64:
                name = 'int64'
            elif tp is bool or tp is bool_:
                name = 'bool'
            elif isinstance(tp, _ScalarType):
                name = tp._name
            elif isinstance(tp, type) and isinstance(tp, _ScalarTypeMeta):
                name = tp._scalar_name
            elif isinstance(tp, dtype):
                name = tp.name
            if name and name in dtype._dtype_class_map:
                target_cls = dtype._dtype_class_map[name]
                if target_cls is not cls:
                    return object.__new__(target_cls)
        return object.__new__(cls)

    def __init__(self, tp=None, metadata=None, align=False):
        if isinstance(tp, list):
            # List-of-tuples structured dtype: delegate to StructuredDtype
            sd = StructuredDtype(tp)
            self.name = sd.name
            self.kind = sd.kind
            self.itemsize = sd.itemsize
            self.char = sd.char
            self.names = sd.names
            self.fields = sd.fields
            self._structured = sd
            self.descr = sd.descr
        elif isinstance(tp, StructuredDtype):
            self.name = tp.name
            self.kind = tp.kind
            self.itemsize = tp.itemsize
            self.char = tp.char
            self.names = tp.names
            self.fields = tp.fields
            self._structured = tp
            self.descr = tp.descr
        elif isinstance(tp, dtype):
            self.name = tp.name
            self.kind = tp.kind
            self.itemsize = tp.itemsize
            self.char = tp.char
            if hasattr(tp, '_structured') and tp._structured is not None:
                self.names = tp.names
                self.fields = tp.fields
                self._structured = tp._structured
                self.descr = tp.descr
        elif isinstance(tp, tuple):
            # Subarray dtype: (base_dtype, shape)
            base_dt = dtype(tp[0])
            sub_shape = tp[1]
            if isinstance(sub_shape, int):
                sub_shape = (sub_shape,)
            elif not isinstance(sub_shape, tuple):
                sub_shape = tuple(sub_shape)
            sub_size = 1
            for s in sub_shape:
                sub_size *= s
            self.name = base_dt.name
            self.kind = base_dt.kind
            self.char = base_dt.char
            self.itemsize = base_dt.itemsize * sub_size
            self.subdtype = (base_dt, sub_shape)
            self.base = base_dt
            self.shape = sub_shape
            self.names = None
            self.fields = None
            self.byteorder = base_dt.byteorder
            self.str = base_dt.str
            self.type = base_dt.type
            self.metadata = metadata
            self.alignment = base_dt.itemsize
            self.isalignedstruct = False
            self.isnative = True
            self.hasobject = False
            self.num = getattr(base_dt, 'num', 0)
            self.descr = base_dt.descr if hasattr(base_dt, 'descr') else [('', self.str)]
            return
        elif isinstance(tp, str) and ',' in tp:
            # Comma-separated dtype string like 'i4,f8' -> structured dtype
            sd = _parse_comma_dtype(tp)
            self.name = sd.name
            self.kind = sd.kind
            self.itemsize = sd.itemsize
            self.char = sd.char
            self.names = sd.names
            self.fields = sd.fields
            self._structured = sd
            self.descr = sd.descr
        elif isinstance(tp, str):
            tp = _DTYPE_CHAR_MAP.get(tp, tp)
            # Handle arbitrary |Sn or Sn byte strings -> 'bytes'
            if tp.startswith('|S') and len(tp) > 2 and tp[2:].isdigit():
                tp = 'bytes'
            elif tp.startswith('S') and len(tp) > 1 and tp[1:].isdigit():
                tp = 'bytes'
            # Handle arbitrary |Vn or Vn void dtypes -> 'void'
            elif tp.startswith('|V') and len(tp) > 2 and tp[2:].isdigit():
                tp = 'void'
            elif tp.startswith('V') and len(tp) > 1 and tp[1:].isdigit():
                tp = 'void'
            # Handle repeat-count dtypes like '2i' -> 'void'
            elif len(tp) >= 2 and tp[0].isdigit():
                tp = 'void'
            # Handle arbitrary <Un / Un unicode strings -> 'str'
            elif len(tp) > 1 and (tp.startswith('<U') or tp.startswith('>U')):
                suffix = tp[2:]
                if suffix.isdigit():
                    tp = 'str'
            elif tp.startswith('U') and len(tp) > 1 and tp[1:].isdigit():
                tp = 'str'
            # Handle temporal dtypes (m8/M8/timedelta64/datetime64)
            elif _is_temporal_dtype(tp):
                kind, unit, canonical, str_form = _temporal_dtype_info(tp)
                self.name = canonical
                self.kind = kind
                self.itemsize = 8
                self.char = kind  # 'm' or 'M'
                self.str = str_form
                self.byteorder = '<'
                self.names = None
                self.fields = None
                self.subdtype = None
                self.base = self
                self.shape = ()
                self.metadata = metadata
                self.alignment = 8
                self.isalignedstruct = False
                self.isnative = True
                self.hasobject = False
                self.num = 21 if kind == 'm' else 22  # matching numpy loosely
                self.descr = [('', str_form)]
                # Lazy references to temporal scalar classes to avoid circular import
                import numpy as _np
                self.type = _np.timedelta64 if kind == 'm' else _np.datetime64
                return
            self.name = tp
            self._init_from_name(tp)
        elif tp is object:
            self.name = "object"
            self.kind = "O"
            self.itemsize = 8
            self.char = "O"
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
        elif tp is complex or tp is complex128:
            self.name = "complex128"
            self.kind = "c"
            self.itemsize = 16
            self.char = "D"
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
        elif isinstance(tp, type) and hasattr(tp, '_dtype_class_name'):
            self.name = tp._dtype_class_name
            self._init_from_name(tp._dtype_class_name)
        else:
            self.name = str(tp) if tp else "float64"
            self._init_from_name(self.name)
        # self.type: numpy scalar type class
        _type_map = {
            "float64": float64, "float32": float32, "float16": float16,
            "int64": int64, "int32": int32, "int16": int16, "int8": int8,
            "uint64": uint64, "uint32": uint32, "uint16": uint16, "uint8": uint8,
            "bool": bool_, "complex128": complex128, "complex64": complex64,
            "str": str_, "bytes": bytes_, "object": object_,
        }
        self.type = _type_map.get(self.name, float64)
        # self.str: typestring format (e.g., "<f8")
        _typestr = {
            "float64": "<f8", "float32": "<f4", "float16": "<f2",
            "int64": "<i8", "int32": "<i4", "int16": "<i2", "int8": "|i1",
            "uint64": "<u8", "uint32": "<u4", "uint16": "<u2", "uint8": "|u1",
            "bool": "|b1",
            "complex128": "<c16", "complex64": "<c8",
            "object": "|O", "str": "<U", "bytes": "|S0", "void": "|V0",
        }
        self.str = _typestr.get(self.name, "<f8")
        # names/fields: None for non-structured dtypes
        if not hasattr(self, 'names'):
            self.names = None
        if not hasattr(self, 'fields'):
            self.fields = None
        # byteorder
        self.byteorder = '=' if self.name in ('bool',) else '<'
        if self.kind == 'b':
            self.byteorder = '|'
        elif self.itemsize == 1:
            self.byteorder = '|'
        self.subdtype = None
        self.base = self
        self.shape = ()  # shape of each element (empty tuple for non-structured)
        self.metadata = metadata
        self.alignment = self.itemsize
        self.isalignedstruct = False
        self.isnative = True
        self.hasobject = False
        # num: unique dtype number (matching numpy convention loosely)
        _num_map = {
            "bool": 0, "int8": 1, "uint8": 2, "int16": 3, "uint16": 4,
            "int32": 5, "uint32": 6, "int64": 7, "uint64": 8,
            "float16": 23, "float32": 11, "float64": 12,
            "complex64": 14, "complex128": 15, "object": 17, "str": 19,
        }
        self.num = _num_map.get(self.name, 12)
        # descr: list of (name, typestr) tuples for structured arrays, or [('', typestr)]
        if hasattr(self, '_structured') and self._structured is not None:
            self.descr = [(n, str(self.fields[n][0])) for n in self.names]
        else:
            self.descr = [('', self.str)]

    def _init_from_name(self, name):
        _info = {
            "float64": ("f", 8, "d"), "float32": ("f", 4, "f"),
            "float16": ("f", 2, "e"), "int64": ("i", 8, "q"),
            "int32": ("i", 4, "l"), "int16": ("i", 2, "h"),
            "int8": ("i", 1, "b"), "uint64": ("u", 8, "Q"),
            "uint32": ("u", 4, "L"), "uint16": ("u", 2, "H"),
            "uint8": ("u", 1, "B"), "bool": ("b", 1, "?"),
            "complex128": ("c", 16, "D"), "complex64": ("c", 8, "F"),
            "object": ("O", 8, "O"),
        }
        if name in _info:
            self.kind, self.itemsize, self.char = _info[name]
        else:
            self.kind, self.itemsize, self.char = "f", 8, "d"

    def __repr__(self):
        if hasattr(self, '_structured') and self._structured is not None:
            return repr(self._structured)
        return f"dtype('{self.name}')"

    def __str__(self):
        if hasattr(self, '_structured') and self._structured is not None:
            return str(self._structured)
        return self.name

    @staticmethod
    def _parse_structured_name(name):
        """Try to parse a structured dtype name string into a StructuredDtype."""
        if not (isinstance(name, str) and name.startswith('[')):
            return None
        import ast as _ast_local
        try:
            fields = _ast_local.literal_eval(name)
            return StructuredDtype(fields)
        except Exception:
            return None

    def __eq__(self, other):
        if isinstance(other, dtype):
            s_struct = getattr(self, '_structured', None)
            o_struct = getattr(other, '_structured', None)
            if s_struct is not None and o_struct is not None:
                return s_struct == o_struct
            # Try to parse structured dtype from name strings when _structured is None
            if s_struct is None and o_struct is None:
                s_from_name = dtype._parse_structured_name(getattr(self, 'name', ''))
                o_from_name = dtype._parse_structured_name(getattr(other, 'name', ''))
                if s_from_name is not None and o_from_name is not None:
                    return s_from_name == o_from_name
            # Mixed: one has _structured, other might be parseable
            ss = s_struct or dtype._parse_structured_name(getattr(self, 'name', ''))
            os = o_struct or dtype._parse_structured_name(getattr(other, 'name', ''))
            if ss is not None and os is not None:
                return ss == os
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other or self.name == _normalize_dtype(other)
        if isinstance(other, type) and isinstance(other, _ScalarTypeMeta):
            return self.name == other._scalar_name
        if isinstance(other, type):
            # Handle Python builtin types: bool, int, float
            import builtins as _bi
            _type_map = {_bi.bool: "bool", _bi.int: "int64", _bi.float: "float64", _bi.complex: "complex128"}
            if other in _type_map:
                return self.name == _type_map[other]
        return NotImplemented

    def __hash__(self):
        return hash(self.name)

    def newbyteorder(self, new_order="S"):
        d = dtype(self.name)
        d.metadata = self.metadata
        return d


# ---------------------------------------------------------------------------
# Per-dtype DType classes (numpy.dtypes.Float64DType etc.)
# ---------------------------------------------------------------------------
# Each is a subclass of dtype with metaclass _DTypeClassMeta so that
# type(np.dtype('float64')) == Float64DType and str(Float64DType) == 'float64'.

class Float64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'float64'
    type = float64

class Float32DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'float32'
    type = float32

class Float16DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'float16'
    type = float16

class Int8DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int8'
    type = int8

class Int16DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int16'
    type = int16

class Int32DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int32'
    type = int32

class Int64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'int64'
    type = int64

class UInt8DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint8'
    type = uint8

class UInt16DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint16'
    type = uint16

class UInt32DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint32'
    type = uint32

class UInt64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'uint64'
    type = uint64

class Complex64DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'complex64'
    type = complex64

class Complex128DType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'complex128'
    type = complex128

class BoolDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'bool'
    type = bool_

class StrDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'str'
    type = str_

class BytesDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'bytes'
    type = bytes_

class VoidDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'void'
    type = void

class ObjectDType(dtype, metaclass=_DTypeClassMeta):
    _dtype_class_name = 'object'
    type = object_

# Populate the map so dtype.__new__ can dispatch to the right subclass
dtype._dtype_class_map = {
    'float64': Float64DType, 'float32': Float32DType, 'float16': Float16DType,
    'int8': Int8DType, 'int16': Int16DType, 'int32': Int32DType, 'int64': Int64DType,
    'uint8': UInt8DType, 'uint16': UInt16DType, 'uint32': UInt32DType, 'uint64': UInt64DType,
    'complex64': Complex64DType, 'complex128': Complex128DType,
    'bool': BoolDType, 'str': StrDType, 'bytes': BytesDType,
    'void': VoidDType, 'object': ObjectDType,
}

# ---------------------------------------------------------------------------
# More constants
# ---------------------------------------------------------------------------
True_ = True
False_ = False
int_ = int64  # re-affirm after any potential shadowing


# ---------------------------------------------------------------------------
# _normalize_dtype  (depends on _DTypeClassMeta, so lives here)
# ---------------------------------------------------------------------------
def _normalize_dtype(dt):
    """Normalize dtype string/type to a canonical name our Rust backend understands."""
    if dt is None:
        return None
    # Structured dtype (comma-separated string or dtype with _structured)
    if isinstance(dt, str) and ',' in dt:
        return 'void'
    if isinstance(dt, dtype) and hasattr(dt, '_structured') and dt._structured is not None:
        return 'void'
    if dt is object:
        return 'object'
    if isinstance(dt, type) and hasattr(dt, '_dtype_class_name'):
        return dt._dtype_class_name
    # Non-numpy type classes (e.g. decimal.Decimal) → treat as object
    if isinstance(dt, type) and not hasattr(dt, '_scalar_name') and not hasattr(dt, '_name'):
        if dt not in (float, int, bool, complex, str, bytes):
            return 'object'
    s = str(dt)
    result = _DTYPE_CHAR_MAP.get(s, None)
    if result is not None:
        return result
    # Handle arbitrary byte string dtypes |Sn -> 'bytes'
    if s.startswith('|S') and len(s) > 2 and s[2:].isdigit():
        return 'bytes'
    # Handle arbitrary void dtypes |Vn -> 'void'
    if s.startswith('|V') and len(s) > 2 and s[2:].isdigit():
        return 'void'
    # Handle arbitrary unicode string dtypes <Un, Un -> 'str'
    if (s.startswith('<U') or s.startswith('>U') or s.startswith('U')) and len(s) > 1:
        suffix = s[2:] if s[0] in '<>' else s[1:]
        if suffix.isdigit():
            return 'str'
    # Temporal dtypes: return as-is (handled by _array_core and dtype class)
    if _is_temporal_dtype(s):
        _, _, canonical, _ = _temporal_dtype_info(s)
        return canonical
    # Handle repeat-count dtypes like '2i', '3f8' → treat as void/structured
    if len(s) >= 2 and s[0].isdigit():
        return 'void'
    # Handle bare Vn (without | prefix)
    if s.startswith('V') and len(s) > 1 and s[1:].isdigit():
        return 'void'
    # Handle bare Sn (without | prefix)
    if s.startswith('S') and len(s) > 1 and s[1:].isdigit():
        return 'bytes'
    # Handle unsupported type objects passed as dtype (e.g. decimal.Decimal)
    if s.startswith("<class '") and s.endswith("'>"):
        return 'object'
    # Long double / long float aliases (no true long double in Rust, use float64/complex128)
    _long_aliases = {
        'longdouble': 'float64', 'longfloat': 'float64',
        'clongdouble': 'complex128', 'clongfloat': 'complex128',
        'longcomplex': 'complex128',
    }
    if s in _long_aliases:
        return _long_aliases[s]
    return s


def _normalize_dtype_with_size(dt):
    """Normalize dtype, preserving size for string/void dtype objects."""
    if dt is None:
        return None
    if dt is object:
        return 'object'
    if isinstance(dt, type) and hasattr(dt, '_dtype_class_name'):
        return dt._dtype_class_name
    if isinstance(dt, dtype):
        if getattr(dt, 'kind', None) == 'S':
            return 'S{}'.format(dt.itemsize)
        if getattr(dt, 'kind', None) == 'U':
            chars = dt.itemsize // 4 if dt.itemsize else 0
            return 'U{}'.format(chars)
        if getattr(dt, 'kind', None) == 'V':
            return 'V{}'.format(dt.itemsize)
        return _normalize_dtype(str(dt))
    # For string inputs, preserve U/S/V size info
    if isinstance(dt, str):
        s = dt.lstrip('<>=|')
        if s.startswith('U') and len(s) > 1 and s[1:].isdigit():
            return s
        if s.startswith('S') and len(s) > 1 and s[1:].isdigit():
            return s
        if s.startswith('V') and len(s) > 1 and s[1:].isdigit():
            return s
    return _normalize_dtype(str(dt))


def _string_dtype_info(dt):
    """Return (kind, itemsize) for string/bytes dtype strings, else (None, None)."""
    if not isinstance(dt, str):
        return None, None
    s = dt.lstrip('<>=|')
    if s.startswith('S'):
        n = s[1:]
        if n.isdigit():
            return 'bytes', int(n)
        if s == 'S':
            return 'bytes', 1
    if s.startswith('U'):
        n = s[1:]
        if n.isdigit():
            return 'str', 4 * int(n)
        if s == 'U':
            return 'str', 4
    if s == 'str':
        return 'str', 4
    if s == 'bytes':
        return 'bytes', 1
    if s.startswith('V'):
        n = s[1:]
        if n.isdigit():
            return 'void', int(n)
        if s == 'V':
            return 'void', 0
    return None, None


# ---------------------------------------------------------------------------
# Type casting functions
# ---------------------------------------------------------------------------

def can_cast(from_=None, to=None, casting="safe"):
    # --- safe-cast ordering: lower number can safely cast to higher number ---
    _type_order = {
        "bool": 0,
        "int8": 1, "uint8": 2,
        "int16": 3, "uint16": 4,
        "int32": 5, "uint32": 6,
        "int64": 7, "uint64": 8,
        "float16": 9,
        "float32": 10, "float64": 11,
        "complex64": 12, "complex128": 13,
    }
    # Safe-cast graph: from -> set of safe targets (transitive via ordering isn't enough
    # because uint8->int16 is safe but uint8 order 2, int16 order 3 works;
    # but uint32->int32 is NOT safe).  Use explicit safe-cast rules.
    _safe_casts = {
        "bool":       {"bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"},
        "int8":       {"int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex64", "complex128"},
        "uint8":      {"uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float16", "float32", "float64", "complex64", "complex128"},
        "int16":      {"int16", "int32", "int64", "float32", "float64", "complex64", "complex128"},
        "uint16":     {"uint16", "int32", "uint32", "int64", "uint64", "float32", "float64", "complex64", "complex128"},
        "int32":      {"int32", "int64", "float64", "complex128"},
        "uint32":     {"uint32", "int64", "uint64", "float64", "complex128"},
        "int64":      {"int64", "float64", "complex128"},
        "uint64":     {"uint64", "float64", "complex128"},
        "float16":    {"float16", "float32", "float64", "complex64", "complex128"},
        "float32":    {"float32", "float64", "complex64", "complex128"},
        "float64":    {"float64", "complex128"},
        "complex64":  {"complex64", "complex128"},
        "complex128": {"complex128"},
    }
    _int_types = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
    _signed_int = {"int8", "int16", "int32", "int64"}
    _unsigned_int = {"uint8", "uint16", "uint32", "uint64"}
    _float_types = {"float16", "float32", "float64"}
    _complex_types = {"complex64", "complex128"}
    # Max string representation lengths for numeric -> string casting
    _str_len = {
        "bool": 5, "int8": 4, "uint8": 3, "int16": 6, "uint16": 5,
        "int32": 11, "uint32": 10, "int64": 21, "uint64": 20,
    }

    # --- NEP 50: reject plain Python scalars ---
    if (
        isinstance(from_, (int, float, complex))
        and not isinstance(from_, bool)
        and not hasattr(from_, "_numpy_dtype_name")
    ):
        raise TypeError("Cannot interpret '{}' as a data type".format(type(from_).__name__))

    # --- None check ---
    if from_ is None or to is None:
        raise TypeError("Cannot interpret 'NoneType' as a data type")

    # --- Helper to detect structured dtypes ---
    def _is_structured(x):
        if isinstance(x, (list, tuple)):
            return True
        if isinstance(x, str) and ',' in x:
            return True
        if isinstance(x, dtype) and ',' in x.name:
            return True
        return False

    def _count_fields(x):
        """Count top-level fields in a structured dtype spec. Returns (n_fields, has_subarray)."""
        if isinstance(x, str) and ',' in x:
            return len(x.split(',')), False
        if isinstance(x, (list, tuple)):
            # List of (name, dtype) or (name, dtype, shape) tuples
            count = 0
            has_sub = False
            for item in x:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    _, field_dt = item[0], item[1]
                    if isinstance(field_dt, (list, tuple)):
                        # Nested structured
                        n, _ = _count_fields(field_dt)
                        if n > 1:
                            has_sub = True
                        else:
                            count += 1
                    elif isinstance(field_dt, str) and ',' in field_dt:
                        # Check if commas are outside parens (multi-field) vs inside (subarray)
                        _dt_no_paren = field_dt
                        _depth = 0
                        _has_outer_comma = False
                        for _ch in field_dt:
                            if _ch == '(':
                                _depth += 1
                            elif _ch == ')':
                                _depth -= 1
                            elif _ch == ',' and _depth == 0:
                                _has_outer_comma = True
                                break
                        if _has_outer_comma:
                            has_sub = True
                        else:
                            count += 1
                    else:
                        count += 1
                    if len(item) >= 3:
                        has_sub = True  # subarray
                else:
                    count += 1
            return count, has_sub
        return 1, False

    from_struct = _is_structured(from_)
    to_struct = _is_structured(to)

    # --- Structured dtype casting rules ---
    if from_struct and to_struct:
        # Both structured: only with unsafe
        return casting == "unsafe"
    if not from_struct and to_struct:
        # Simple -> structured: only unsafe
        return casting == "unsafe"
    if from_struct and not to_struct:
        # Structured -> simple: only unsafe, and only if single field (recursive)
        if casting != "unsafe":
            return False
        n_fields, has_sub_multi = _count_fields(from_)
        if n_fields > 1 or has_sub_multi:
            return False
        return True

    # --- Normalize dtype names ---
    def _to_name(x):
        """Extract canonical dtype name and raw string (for endian checks)."""
        if hasattr(x, "_numpy_dtype_name"):
            name = x._numpy_dtype_name
            return name, name
        if isinstance(x, ndarray):
            return str(x.dtype), str(x.dtype)
        if isinstance(x, dtype):
            return x.name, str(x)
        if isinstance(x, _ScalarType):
            return str(x), str(x)
        if isinstance(x, type):
            if isinstance(x, _ScalarTypeMeta):
                return x._scalar_name, x._scalar_name
            if x is float:
                return "float64", "float64"
            if x is int:
                return "int64", "int64"
            if x is bool:
                return "bool", "bool"
            if x is complex:
                return "complex128", "complex128"
        if isinstance(x, str):
            raw = x
            # Strip endian prefix for normalization but keep raw for endian checks
            norm = _DTYPE_CHAR_MAP.get(x, x)
            if norm != x:
                return norm, raw
            # Try stripping endian prefix
            s = x
            if s and s[0] in '<>=|':
                s = s[1:]
            norm = _DTYPE_CHAR_MAP.get(s, s)
            return norm, raw
        return str(x), str(x)

    from_name, from_raw = _to_name(from_)
    to_name, to_raw = _to_name(to)

    # --- String/bytes dtype handling ---
    def _is_string_dtype(name, raw):
        """Check if dtype is a string (U) or bytes (S) type, return (True, length) or (False, 0)."""
        for s in (raw, name):
            if isinstance(s, str):
                stripped = s.lstrip('<>=|')
                if stripped.startswith('S') and len(stripped) > 1:
                    try:
                        return True, int(stripped[1:]), 'S'
                    except (ValueError, IndexError):
                        pass
                if stripped.startswith('U') and len(stripped) > 1:
                    try:
                        return True, int(stripped[1:]), 'U'
                    except (ValueError, IndexError):
                        pass
                if stripped == 'S' or stripped == 'bytes':
                    return True, 0, 'S'
                if stripped == 'U' or stripped == 'str':
                    return True, 0, 'U'
        return False, 0, ''

    to_is_str, to_str_len, to_str_kind = _is_string_dtype(to_name, to_raw)
    from_is_str, from_str_len, from_str_kind = _is_string_dtype(from_name, from_raw)

    if to_is_str and not from_is_str:
        # Numeric -> string: check if string is long enough
        needed = _str_len.get(from_name, 0)
        if needed == 0:
            # Unknown numeric type -> can't safely cast
            if casting == "unsafe":
                return True
            return False
        if casting == "unsafe":
            return True
        return to_str_len >= needed
    if from_is_str and to_is_str:
        if casting == "unsafe":
            return True
        if from_str_kind == to_str_kind:
            return to_str_len >= from_str_len
        # S -> U promotion: U can hold S
        if from_str_kind == 'S' and to_str_kind == 'U':
            return to_str_len >= from_str_len
        return False
    if from_is_str and not to_is_str:
        return casting == "unsafe"

    # --- Endian-aware checks ---
    def _has_endian(raw):
        return isinstance(raw, str) and len(raw) > 0 and raw[0] in '<>'
    def _get_endian(raw):
        if isinstance(raw, str) and len(raw) > 0 and raw[0] in '<>':
            return raw[0]
        return '='

    # --- Numeric casting ---
    if from_name not in _type_order or to_name not in _type_order:
        if casting == "unsafe":
            return True
        return False

    if casting == "unsafe":
        return True
    if casting == "no":
        if from_name != to_name:
            return False
        # Same base type: check endianness must match exactly
        if _has_endian(from_raw) and _has_endian(to_raw):
            return _get_endian(from_raw) == _get_endian(to_raw)
        return True
    if casting == "equiv":
        # Same base type, possibly different endianness
        return from_name == to_name
    if casting == "safe":
        return to_name in _safe_casts.get(from_name, set())
    if casting == "same_kind":
        if to_name in _safe_casts.get(from_name, set()):
            return True
        # Allow downcast within same kind
        if from_name in _signed_int and to_name in _signed_int:
            return True
        if from_name in _unsigned_int and to_name in _unsigned_int:
            return True
        # bool -> any int is same_kind (bool is a sub-kind of int)
        if from_name == "bool" and to_name in _int_types:
            return True
        if from_name in _float_types and to_name in _float_types:
            return True
        if from_name in _complex_types and to_name in _complex_types:
            return True
        return False
    return to_name in _safe_casts.get(from_name, set())


def result_type(*arrays_and_dtypes):
    if len(arrays_and_dtypes) == 0:
        return float64
    dtypes = []
    for a in arrays_and_dtypes:
        if isinstance(a, ndarray):
            dtypes.append(str(a.dtype))
        elif isinstance(a, _ObjectArray):
            dtypes.append(_normalize_dtype(str(a.dtype)))
        elif hasattr(a, "_numpy_dtype_name"):
            dtypes.append(str(getattr(a, "_numpy_dtype_name")))
        elif isinstance(a, _ScalarType):
            dtypes.append(str(a))
        elif isinstance(a, type) and isinstance(a, _ScalarTypeMeta):
            dtypes.append(str(a))
        elif isinstance(a, str):
            dtypes.append(a)
        elif isinstance(a, bool):
            dtypes.append("bool")
        elif isinstance(a, int):
            dtypes.append("int64")
        elif isinstance(a, float):
            dtypes.append("float64")
        elif isinstance(a, complex):
            dtypes.append("complex128")
        else:
            dtypes.append("float64")
    if len(dtypes) == 1:
        return dtype(dtypes[0])
    result = dtypes[0]
    for d in dtypes[1:]:
        result = str(promote_types(result, d))
    return dtype(result)


def _numeric_to_str_len(name):
    """Return number of characters needed to represent this numeric dtype as string."""
    _sizes = {
        'bool': 5, 'bool_': 5,
        'int8': 4, 'int16': 6, 'int32': 11, 'int64': 21,
        'uint8': 3, 'uint16': 5, 'uint32': 10, 'uint64': 20,
        'float16': 12, 'float32': 12, 'float64': 22,
        'complex64': 24, 'complex128': 48,
    }
    return _sizes.get(name, None)


def _parse_us_dtype(raw):
    """Parse a raw U/S dtype spec (possibly endian-prefixed).
    Returns (char, size) where char is 'U' or 'S', size is int (0 = unsized).
    Returns None if not a U/S dtype.
    """
    if isinstance(raw, str):
        s = raw.lstrip('<>=|')
    elif isinstance(raw, dtype):
        # For S types, name preserves size; U types lose size (become 'str')
        s = raw.name
    else:
        s = str(raw).lstrip('<>=|')
    if s.startswith('U') and (len(s) == 1 or s[1:].isdigit()):
        return ('U', int(s[1:]) if len(s) > 1 else 0)
    if s.startswith('S') and (len(s) == 1 or s[1:].isdigit()):
        return ('S', int(s[1:]) if len(s) > 1 else 0)
    if s == 'str':
        # dtype('U5') loses size -> treat as unsized U
        return ('U', 0)
    return None


def promote_types(type1, type2):
    # Capture original representations for U/S type detection (before dtype conversion)
    _raw1 = type1 if isinstance(type1, (str, dtype)) else str(type1)
    _raw2 = type2 if isinstance(type2, (str, dtype)) else str(type2)
    _us1 = _parse_us_dtype(_raw1)
    _us2 = _parse_us_dtype(_raw2)

    # Ensure both are dtype objects for metadata access
    if not isinstance(type1, dtype):
        try:
            type1 = dtype(type1)
        except (TypeError, ValueError):
            type1_str = str(type1)
            # For unsupported dtypes, if both are equal, return as-is
            if str(type1) == str(type2):
                d = dtype.__new__(dtype)
                d.name = type1_str
                d.kind = 'V'
                d.itemsize = 0
                d.char = 'V'
                d.byteorder = '='
                d.metadata = getattr(type2 if isinstance(type2, dtype) else type1, 'metadata', None)
                return d
            raise
    if not isinstance(type2, dtype):
        try:
            type2 = dtype(type2)
        except (TypeError, ValueError):
            raise

    t1_meta = getattr(type1, 'metadata', None)
    t2_meta = getattr(type2, 'metadata', None)

    # Fast-path: identical types -> return directly (preserves metadata for V, O, etc.)
    if type1.name == type2.name:
        # For structured/void dtypes, check full equality (field names must match)
        t1_names = getattr(type1, 'names', None)
        t2_names = getattr(type2, 'names', None)
        if t1_names is not None or t2_names is not None:
            # Structured: must be fully equal
            if type1 != type2:
                raise TypeError("invalid type promotion")
        # Also check itemsize for void types (V6 vs V10)
        t1_is = getattr(type1, 'itemsize', 0)
        t2_is = getattr(type2, 'itemsize', 0)
        if type1.kind == 'V' and t1_is != t2_is:
            raise TypeError("invalid type promotion")
        result = dtype(type1.name)
        # Only preserve metadata when both are identical
        if t1_meta is not None and t1_meta == t2_meta:
            result.metadata = t1_meta
        return result

    # Strip endian prefixes for Rust backend
    s1 = type1.name
    s2 = type2.name
    if s1 and s1[0] in '<>=|':
        s1 = s1[1:]
    if s2 and s2[0] in '<>=|':
        s2 = s2[1:]
    s1 = _DTYPE_CHAR_MAP.get(s1, s1)
    s2 = _DTYPE_CHAR_MAP.get(s2, s2)

    # If names match after normalization
    if s1 == s2:
        result = dtype(s1)
        if t1_meta is not None and t1_meta == t2_meta:
            result.metadata = t1_meta
        return result

    # String/bytes type promotion
    if _us1 is not None or _us2 is not None:
        # Object promoted with string -> object
        if s1 == 'object' or s2 == 'object':
            return dtype('object')
        if _us1 is not None and _us2 is not None:
            # Both are string/bytes types: U+U, S+S, U+S
            c1, n1 = _us1
            c2, n2 = _us2
            # U wins over S (unicode is wider)
            out_char = 'U' if ('U' in (c1, c2)) else 'S'
            out_size = max(n1, n2)
            return dtype(out_char + str(out_size))
        # One is string/bytes, other is numeric
        if _us1 is not None:
            us_char, us_size = _us1
            numeric_name = s2
        else:
            us_char, us_size = _us2
            numeric_name = s1
        needed = _numeric_to_str_len(numeric_name)
        if needed is None:
            # Unknown numeric type - can't promote
            raise TypeError("Cannot promote string dtype with numeric dtype")
        out_size = max(us_size, needed)
        return dtype(us_char + str(out_size))

    _int_bits = {
        "bool": 1,
        "int8": 8, "int16": 16, "int32": 32, "int64": 64,
        "uint8": 8, "uint16": 16, "uint32": 32, "uint64": 64,
    }
    _float_bits = {"float16": 16, "float32": 32, "float64": 64}
    _complex_bits = {"complex64": 64, "complex128": 128}
    _signed_names = {8: "int8", 16: "int16", 32: "int32", 64: "int64"}
    _unsigned_names = {8: "uint8", 16: "uint16", 32: "uint32", 64: "uint64"}
    _float_names = {16: "float16", 32: "float32", 64: "float64"}
    _complex_names = {64: "complex64", 128: "complex128"}

    def _next_signed(bits):
        for b in (8, 16, 32, 64):
            if b > bits:
                return _signed_names[b]
        return None

    def _promote_numeric(a, b):
        _bmax = __import__("builtins").max
        if a in _complex_bits or b in _complex_bits:
            # Lift real to matching complex precision.
            def _real_float_bits(x):
                if x in _complex_bits:
                    return 32 if x == "complex64" else 64
                if x in _float_bits:
                    return _float_bits[x]
                # integers/bool route through float64 conservatively.
                return 64
            rb = _real_float_bits(a)
            rc = _real_float_bits(b)
            return _complex_names[64 if _bmax(rb, rc) <= 32 else 128]

        if a in _float_bits or b in _float_bits:
            # Scalar promotion behavior close to NumPy's tested cases.
            fa = _float_bits[a] if a in _float_bits else None
            fb = _float_bits[b] if b in _float_bits else None
            fbits = _bmax(fa or 0, fb or 0)
            other = b if fa is not None else a
            if other in _int_bits:
                obits = _int_bits[other]
                if fbits == 16:
                    return "float16" if obits <= 8 else "float32"
                if fbits == 32:
                    if other.startswith("uint"):
                        return "float64" if obits >= 32 else "float32"
                    return "float64" if obits >= 32 else "float32"
                return "float64"
            return _float_names[fbits]

        if a in _int_bits and b in _int_bits:
            if a == "bool":
                return b
            if b == "bool":
                return a

            a_unsigned = a.startswith("uint")
            b_unsigned = b.startswith("uint")
            abit = _int_bits[a]
            bbit = _int_bits[b]

            if a_unsigned == b_unsigned:
                bits = _bmax(abit, bbit)
                return _unsigned_names[bits] if a_unsigned else _signed_names[bits]

            # signed/unsigned mix
            sbits = abit if not a_unsigned else bbit
            ubits = abit if a_unsigned else bbit
            if sbits > ubits:
                return _signed_names[sbits]
            nxt = _next_signed(ubits)
            if nxt is not None:
                return nxt
            return "float64"

        return None

    promoted = _promote_numeric(s1, s2)
    result = dtype(promoted if promoted is not None else _native.promote_types(s1, s2))
    # Preserve metadata only when both have identical metadata
    if t1_meta is not None and t1_meta == t2_meta:
        result.metadata = t1_meta
    elif t1_meta is not None and t2_meta is None:
        result.metadata = t1_meta
    elif t2_meta is not None and t1_meta is None:
        result.metadata = t2_meta
    return result


def find_common_type(array_types, scalar_types):
    """Deprecated in numpy 2.0, but still used by some packages.
    Determines common type following standard coercion rules."""
    from functools import reduce as _reduce
    all_types = list(array_types) + list(scalar_types)
    if not all_types:
        return dtype("float64")
    return _reduce(lambda a, b: dtype(str(result_type(a, b))), all_types)


# ---------------------------------------------------------------------------
# finfo / iinfo / _MachAr
# ---------------------------------------------------------------------------

class finfo:
    """Machine limits for floating point types."""

    _finfo_cache = {}

    def __class_getitem__(cls, item):
        import types as _types
        return _types.GenericAlias(cls, (item,))

    def __new__(cls, dtype=None):
        key = cls._resolve_key(dtype)
        if key in cls._finfo_cache:
            return cls._finfo_cache[key]
        obj = super().__new__(cls)
        obj._init_from_key(key)
        cls._finfo_cache[key] = obj
        return obj

    def __init__(self, dtype=None):
        # __new__ already initialised the instance via _init_from_key
        pass

    @staticmethod
    def _resolve_key(dtype):
        """Normalise *dtype* to one of 'float16', 'float32', 'float64'."""
        if dtype is None or dtype is float:
            return 'float64'
        # Handle plain float/int instances (not types)
        if isinstance(dtype, float) and not isinstance(dtype, _NumpyFloatScalar):
            return 'float64'
        # Handle objects with .dtype attribute (e.g. ndarray, custom objects)
        if hasattr(dtype, 'dtype') and not isinstance(dtype, type):
            dtype = dtype.dtype
        # Handle scalar instances (e.g. float32(1.0))
        if hasattr(dtype, '_numpy_dtype_name') and not isinstance(dtype, type):
            dtype = dtype._numpy_dtype_name
        # Handle _ScalarTypeMeta classes (float64, float32, etc.)
        if isinstance(dtype, type) and hasattr(dtype, '_scalar_name'):
            dtype = dtype._scalar_name
        # Handle dtype objects
        if hasattr(dtype, 'name') and not isinstance(dtype, str):
            dtype = dtype.name
        s = str(dtype)
        _f64 = ('float64', 'f8', 'float', 'd', 'longdouble', 'longfloat', 'g',
                'double', 'float_')
        _f32 = ('float32', 'f4', 'f', 'single')
        _f16 = ('float16', 'half', 'f2', 'e')
        # Complex types map to their float component
        _c128 = ('complex128', 'c16', 'complex', 'cdouble', 'clongdouble',
                 'clongfloat', 'complex_', 'D')
        _c64 = ('complex64', 'c8', 'csingle', 'F')
        if s in _f64:
            return 'float64'
        if s in _f32:
            return 'float32'
        if s in _f16:
            return 'float16'
        if s in _c128:
            return 'float64'
        if s in _c64:
            return 'float32'
        raise ValueError("finfo only supports float16, float32 and float64")

    def _init_from_key(self, key):
        if key == 'float64':
            self.bits = 64
            self.eps = float64(2.220446049250313e-16)
            self.epsneg = float64(1.1102230246251565e-16)
            self.max = 1.7976931348623157e+308
            self.min = -1.7976931348623157e+308
            self.tiny = 2.2250738585072014e-308
            self.smallest_normal = 2.2250738585072014e-308
            self.smallest_subnormal = 5e-324
            self.precision = 15
            self.resolution = float64(10) ** (-self.precision)
            self.dtype = float64
            self.maxexp = 1024
            self.minexp = -1022
            self.nmant = 52
            self.nexp = 11
            self.machep = -52
            self.negep = -53
            self.iexp = 11
        elif key == 'float32':
            self.bits = 32
            self.eps = float32(1.1920929e-07)
            self.epsneg = float32(5.960464477539063e-08)
            import struct as _s
            self.max = _s.unpack('f', b'\xff\xff\x7f\x7f')[0]  # exact float32 max
            self.min = -self.max
            self.tiny = _s.unpack('f', b'\x00\x00\x80\x00')[0]  # exact float32 tiny
            self.smallest_normal = self.tiny
            self.smallest_subnormal = _s.unpack('f', b'\x01\x00\x00\x00')[0]
            self.precision = 6
            self.resolution = float32(10) ** (-self.precision)
            self.dtype = float32
            self.maxexp = 128
            self.minexp = -126
            self.nmant = 23
            self.nexp = 8
            self.machep = -23
            self.negep = -24
            self.iexp = 8
        elif key == 'float16':
            self.bits = 16
            self.eps = float16(9.765625e-04)
            self.epsneg = float16(4.8828125e-04)
            self.max = 65504.0
            self.min = -65504.0
            self.tiny = 6.103515625e-05
            self.smallest_normal = 6.103515625e-05
            self.smallest_subnormal = 5.96e-08
            self.precision = 3
            self.resolution = float16(10) ** (-self.precision)
            self.dtype = float16
            self.maxexp = 16
            self.minexp = -14
            self.nmant = 10
            self.nexp = 5
            self.machep = -10
            self.negep = -11
            self.iexp = 5
        # Legacy _machar attribute (deprecated but accessed by some tests)
        self._machar = _MachAr(self)

    def __repr__(self):
        return f"finfo(resolution={self.resolution}, min={self.min}, max={self.max}, dtype={self.dtype})"

    def __eq__(self, other):
        if not isinstance(other, finfo):
            return NotImplemented
        return self.bits == other.bits

    def __ne__(self, other):
        if not isinstance(other, finfo):
            return NotImplemented
        return self.bits != other.bits

class _MachAr:
    """Legacy MachAr stub (deprecated in numpy 1.22+)."""
    def __init__(self, finfo_obj):
        self.eps = finfo_obj.eps
        self.tiny = finfo_obj.tiny
        self.huge = finfo_obj.max
        self.smallest_normal = finfo_obj.smallest_normal
        self.smallest_subnormal = getattr(finfo_obj, 'smallest_subnormal', 0.0)


class iinfo:
    """Machine limits for integer types."""
    def __class_getitem__(cls, item):
        import types as _types
        return _types.GenericAlias(cls, (item,))

    def __init__(self, dtype=None):
        # Handle scalar instances: extract dtype name from numpy scalars or Python int
        if isinstance(dtype, _NumpyIntScalar):
            dtype = dtype._numpy_dtype_name
        elif dtype is int:
            dtype = 'int64'
        elif isinstance(dtype, int) and not isinstance(dtype, bool):
            dtype = 'int64'
        # Handle _ScalarTypeMeta classes (int8, int16, etc.)
        elif isinstance(dtype, _ScalarTypeMeta):
            dtype = dtype._scalar_name
        if dtype is None or str(dtype) in ('int64', 'i8', 'int', 'l', 'q'):
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
        elif str(dtype) in ('int8', 'i1', 'b'):
            self.bits = 8
            self.min = -128
            self.max = 127
            self.dtype = int8
            self.kind = 'i'
        elif str(dtype) in ('int16', 'i2', 'h'):
            self.bits = 16
            self.min = -32768
            self.max = 32767
            self.dtype = int16
            self.kind = 'i'
        elif str(dtype) in ('uint8', 'u1', 'B'):
            self.bits = 8
            self.min = 0
            self.max = 255
            self.dtype = uint8
            self.kind = 'u'
        elif str(dtype) in ('uint16', 'u2', 'H'):
            self.bits = 16
            self.min = 0
            self.max = 65535
            self.dtype = uint16
            self.kind = 'u'
        elif str(dtype) in ('uint32', 'u4', 'I'):
            self.bits = 32
            self.min = 0
            self.max = 4294967295
            self.dtype = uint32
            self.kind = 'u'
        elif str(dtype) in ('uint64', 'u8', 'Q', 'L'):
            self.bits = 64
            self.min = 0
            self.max = 18446744073709551615
            self.dtype = uint64
            self.kind = 'u'
        else:
            # Try normalizing via dtype()
            try:
                _dt = dtype(str(dtype))
                _name = _dt.name
                if _name in ('int8', 'int16', 'int32', 'int64',
                             'uint8', 'uint16', 'uint32', 'uint64'):
                    self.__init__(_name)
                    return
            except Exception:
                pass
            raise ValueError("iinfo does not support this dtype")

    def __repr__(self):
        return f"iinfo(min={self.min}, max={self.max}, dtype={self.dtype})"


# ---------------------------------------------------------------------------
# mintypecode / common_type
# ---------------------------------------------------------------------------

def mintypecode(typechars, typeset='GDFgdf', default='d'):
    """Return the character for the minimum-size type to which given types can be safely cast.

    Returns the typecode character from typeset that can represent all typechars.
    """
    # Map from type char to (is_complex, float_precision_bits)
    # Precision bits = float precision per component
    # Only include standard type chars; legacy/unknown chars are treated as needing
    # at least the default type.
    _type_info = {
        '?': (False, 0), 'b': (False, 0), 'B': (False, 0),
        '1': (False, 0), 's': (False, 0), 'w': (False, 0),
        'h': (False, 0), 'H': (False, 0),
        'i': (False, 0), 'I': (False, 0), 'l': (False, 0), 'L': (False, 0),
        'u': (False, 0), 'c': (False, 0),
        'e': (False, 16),
        'f': (False, 32), 'd': (False, 64), 'g': (False, 128),
        'F': (True, 32), 'D': (True, 64), 'G': (True, 128),
    }
    # Determine: do we have any complex? what's the max float precision needed?
    has_complex = False
    max_prec = 0
    any_recognized = False
    for tc in typechars:
        info = _type_info.get(tc)
        if info is None:
            continue
        any_recognized = True
        is_cmplx, prec = info
        if is_cmplx:
            has_complex = True
        max_prec = _builtin_max(max_prec, prec)

    if not any_recognized:
        return default

    # If only integer/non-float types seen, return default
    if max_prec == 0 and not has_complex:
        return default

    # Build candidate list from typeset
    # For each candidate, determine if it satisfies:
    # 1. If has_complex, candidate must be complex
    # 2. Candidate precision >= max_prec
    candidates = []
    _prec_order = {'f': 32, 'd': 64, 'g': 128, 'F': 32, 'D': 64, 'G': 128}
    for tc in typeset:
        info = _type_info.get(tc)
        if info is None:
            continue
        is_cmplx, prec = info
        # Must be complex if input has complex, must be real if input is real
        if has_complex and not is_cmplx:
            continue
        if not has_complex and is_cmplx:
            continue
        # Must cover precision
        if prec < max_prec:
            continue
        candidates.append((prec, tc))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    # Fallback: return default
    return default


def common_type(*arrays):
    """Return a scalar type common to input arrays."""
    has_complex = False
    max_float = 0  # track bits: 16, 32, 64
    for a in arrays:
        import numpy as _np
        arr = _np.asarray(a)
        dt = str(arr.dtype)
        if "complex" in dt:
            has_complex = True
            if "128" in dt:
                max_float = _builtin_max(max_float, 64)
            else:
                max_float = _builtin_max(max_float, 32)
        elif "float64" in dt or "float" == dt:
            max_float = _builtin_max(max_float, 64)
        elif "float32" in dt:
            max_float = _builtin_max(max_float, 32)
        elif "float16" in dt:
            max_float = _builtin_max(max_float, 16)
        elif "int" in dt or "uint" in dt:
            max_float = _builtin_max(max_float, 64)  # int promotes to float64
        elif dt == "bool":
            max_float = _builtin_max(max_float, 64)  # bool promotes to float64
    if has_complex:
        if max_float >= 64:
            return complex128
        return complex64
    if max_float >= 64 or max_float == 0:
        return float64
    if max_float >= 32:
        return float32
    return float16


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
