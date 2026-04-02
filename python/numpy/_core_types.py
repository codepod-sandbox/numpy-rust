"""Type system: scalar types, dtype class, type hierarchy, finfo/iinfo, type casting.

This module is now a thin re-export facade. The actual implementations live in:
  - _scalar_types.py  — scalar type machinery and arithmetic
  - _dtype.py         — the dtype class and DType subclasses
  - _type_promotion.py — type casting and promotion rules
  - _type_info.py     — finfo, iinfo, and numeric utilities
"""
from ._scalar_types import *
from ._scalar_types import (
    _DTYPE_CHAR_MAP, _DTYPE_ITEMSIZE,
    _ScalarType, _NumpyIntScalar, _NumpyFloatScalar, _NumpyComplexScalar,
    _NumpyVoidScalar, _ScalarTypeMeta,
    _get_numpy_dtype_name, _wrap_scalar_result, _scalar_promote,
    _complex_pow, _safe_pow, _coerce_for_op,
    _scalar_binop_result, _scalar_rbinop_result, _scalar_cmp_result,
    _float_to_str, _truncate_float,
    _SENTINEL, _BINOP_MAP,
    _ABSTRACT_SCALAR_TYPES, _SUBSCRIPTABLE_CONCRETE,
    typecodes, sctypes, sctypeDict,
    float64, float32, float16,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    complex64, complex128,
    bool_, str_, bytes_, void, object_,
    generic, number, integer, signedinteger, unsignedinteger,
    inexact, floating, complexfloating, character, flexible,
    float128, intp, intc, uintp, byte, ubyte, short, ushort,
    longlong, ulonglong, single, double, longdouble,
    csingle, cdouble, clongdouble, longfloat, clongfloat, longcomplex,
    string_, unicode_, half, int_, float_, complex_, uint, long, ulong,
    True_, False_,
)
from ._dtype import *
from ._dtype import (
    StructuredDtype, _DTypeClassMeta, _parse_comma_dtype, dtype,
    Float64DType, Float32DType, Float16DType,
    Int8DType, Int16DType, Int32DType, Int64DType,
    UInt8DType, UInt16DType, UInt32DType, UInt64DType,
    Complex64DType, Complex128DType, BoolDType, StrDType,
    BytesDType, VoidDType, ObjectDType,
    _normalize_dtype, _normalize_dtype_with_size, _string_dtype_info,
)
from ._type_promotion import *
from ._type_promotion import (
    can_cast, result_type, _numeric_to_str_len, _parse_us_dtype,
    promote_types, find_common_type, mintypecode, common_type,
    min_scalar_type,
)
from ._type_info import *
from ._type_info import finfo, _MachAr, iinfo

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
    'common_type', 'mintypecode', 'min_scalar_type', '_normalize_dtype', '_normalize_dtype_with_size',
    '_string_dtype_info', '_DTYPE_CHAR_MAP',
    # Constants and aliases
    'True_', 'False_', 'int_', 'typecodes', 'sctypes', 'sctypeDict',
    'float128', 'intp', 'intc', 'uintp', 'byte', 'ubyte', 'short', 'ushort',
    'longlong', 'ulonglong', 'single', 'double', 'longdouble',
    'csingle', 'cdouble', 'clongdouble',
    'string_', 'unicode_', 'half', 'float_', 'complex_', 'uint', 'long', 'ulong',
    'longfloat', 'clongfloat', 'longcomplex',
]
