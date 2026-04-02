"""Numeric type info: finfo, iinfo, and _MachAr."""
from ._scalar_types import (
    _NumpyFloatScalar, _NumpyIntScalar, _ScalarTypeMeta,
    float16, float32, float64,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
)

__all__ = [
    'finfo',
    '_MachAr',
    'iinfo',
]


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
                from ._dtype import dtype as _dtype_cls
                _dt = _dtype_cls(str(dtype))
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
