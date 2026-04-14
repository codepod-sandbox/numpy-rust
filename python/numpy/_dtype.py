"""The dtype class, structured dtype, DType subclasses, and dtype normalization utilities."""
import _numpy_native as _native
from ._helpers import _is_temporal_dtype, _temporal_dtype_info
from ._scalar_types import (
    _ScalarType, _ScalarTypeMeta, _DTYPE_CHAR_MAP,
    float64, float32, float16,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    complex64, complex128,
    bool_, str_, bytes_, void, object_,
)

__all__ = [
    'dtype', 'StructuredDtype', '_DTypeClassMeta',
    '_parse_comma_dtype',
    # Per-dtype DType classes
    'Float64DType', 'Float32DType', 'Float16DType',
    'Int8DType', 'Int16DType', 'Int32DType', 'Int64DType',
    'UInt8DType', 'UInt16DType', 'UInt32DType', 'UInt64DType',
    'Complex64DType', 'Complex128DType', 'BoolDType', 'StrDType',
    'BytesDType', 'VoidDType', 'ObjectDType',
    # Normalization utilities
    '_normalize_dtype', '_normalize_dtype_with_size', '_string_dtype_info',
    # Re-export for convenience
    '_DTYPE_CHAR_MAP',
]


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


def _native_byteorder_char():
    import sys as _sys
    return '<' if _sys.byteorder == 'little' else '>'


def _normalize_byteorder(byteorder, itemsize):
    if itemsize <= 1:
        return '|'
    if byteorder == '=':
        return _native_byteorder_char()
    if byteorder in ('<', '>', '|'):
        return byteorder
    return _native_byteorder_char()


def _byteorder_ignored(name, kind=None):
    return name in ('bool', 'bytes', 'void', 'object') or kind in ('b', 'S', 'V', 'O')


def _typestr_for(name, byteorder):
    _typestr = {
        "float64": "f8", "float32": "f4", "float16": "f2",
        "int64": "i8", "int32": "i4", "int16": "i2", "int8": "i1",
        "uint64": "u8", "uint32": "u4", "uint16": "u2", "uint8": "u1",
        "bool": "b1",
        "complex128": "c16", "complex64": "c8",
        "object": "O", "str": "U", "bytes": "S0", "void": "V0",
    }
    suffix = _typestr.get(name, "f8")
    if byteorder == '|':
        return '|' + suffix
    return byteorder + suffix


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
                raw_name = tp[1:] if tp and tp[0] in '<>=|' else tp
                name = _DTYPE_CHAR_MAP.get(raw_name, raw_name)
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
        requested_byteorder = None
        raw_string_spec = None
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
            else:
                self.names = None
                self.fields = None
            self.byteorder = tp.byteorder
            self.str = tp.str
            self.type = tp.type
            self.metadata = metadata if metadata is not None else tp.metadata
            self.alignment = tp.alignment
            self.isalignedstruct = tp.isalignedstruct
            self.isnative = tp.isnative
            self.hasobject = tp.hasobject
            self.num = tp.num
            self.subdtype = getattr(tp, 'subdtype', None)
            self.base = getattr(tp, 'base', self)
            self.shape = getattr(tp, 'shape', ())
            return
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
            if tp and tp[0] in '<>=|':
                requested_byteorder = tp[0]
                tp = tp[1:]
            raw_string_spec = tp
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
        self.byteorder = '|' if _byteorder_ignored(self.name, self.kind) or self.itemsize == 1 else '<'
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

        if raw_string_spec is not None:
            spec = raw_string_spec
            if spec.startswith('S') and (len(spec) == 1 or spec[1:].isdigit()):
                size = int(spec[1:]) if len(spec) > 1 else 1
                self.name = 'bytes'
                self.kind = 'S'
                self.itemsize = size
                self.char = 'S'
                self.type = bytes_
                self.byteorder = '|'
                self.isnative = True
                self.str = '|S{}'.format(size)
                self.alignment = 1
            elif spec.startswith('U') and (len(spec) == 1 or spec[1:].isdigit()):
                size = int(spec[1:]) if len(spec) > 1 else 1
                self.name = 'str'
                self.kind = 'U'
                self.itemsize = 4 * size
                self.char = 'U'
                self.type = str_
                self.byteorder = _normalize_byteorder(requested_byteorder or _native_byteorder_char(), self.itemsize)
                self.isnative = self.byteorder in ('|', _native_byteorder_char())
                self.str = '{}U{}'.format(self.byteorder, size)
                self.alignment = 4
            elif spec.startswith('V') and (len(spec) == 1 or spec[1:].isdigit()):
                size = int(spec[1:]) if len(spec) > 1 else 0
                self.name = 'void'
                self.kind = 'V'
                self.itemsize = size
                self.char = 'V'
                self.type = void
                self.byteorder = '|'
                self.isnative = True
                self.str = '|V{}'.format(size)
                self.alignment = 1
            if spec.startswith(('S', 'U', 'V')):
                self.descr = [('', self.str)]

        if requested_byteorder is not None:
            if _byteorder_ignored(self.name, self.kind):
                self.byteorder = '|'
                self.isnative = True
            else:
                self.byteorder = _normalize_byteorder(requested_byteorder, self.itemsize)
                self.isnative = self.byteorder in ('|', _native_byteorder_char())
            if not (raw_string_spec is not None and raw_string_spec.startswith(('S', 'U', 'V'))):
                self.str = _typestr_for(self.name, self.byteorder)
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
        return f"dtype('{str(self)}')"

    def __str__(self):
        if hasattr(self, '_structured') and self._structured is not None:
            return str(self._structured)
        if self.byteorder not in ('|', _native_byteorder_char()):
            return self.str
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
            if getattr(self, 'kind', None) in ('S', 'U', 'V') or getattr(other, 'kind', None) in ('S', 'U', 'V'):
                if getattr(self, 'kind', None) == 'U' and getattr(other, 'kind', None) == 'U':
                    return (
                        self.name == other.name and
                        getattr(self, 'itemsize', None) == getattr(other, 'itemsize', None)
                    )
                return (
                    self.name == other.name and
                    getattr(self, 'itemsize', None) == getattr(other, 'itemsize', None) and
                    _normalize_byteorder(self.byteorder, self.itemsize) ==
                    _normalize_byteorder(other.byteorder, other.itemsize)
                )
            return (
                self.name == other.name and
                _normalize_byteorder(self.byteorder, self.itemsize) ==
                _normalize_byteorder(other.byteorder, other.itemsize)
            )
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
        return hash((self.name, self.byteorder, getattr(self, 'itemsize', None)))

    def newbyteorder(self, new_order="S"):
        d = dtype(self)
        if _byteorder_ignored(d.name, getattr(d, 'kind', None)) or d.byteorder == '|':
            return d
        current = _normalize_byteorder(d.byteorder, d.itemsize)
        if new_order in ('S', 's'):
            target = '>' if current == '<' else '<'
        elif new_order == '=':
            target = _native_byteorder_char()
        else:
            target = new_order
        d.byteorder = _normalize_byteorder(target, d.itemsize)
        d.isnative = d.byteorder in ('|', _native_byteorder_char())
        d.str = _typestr_for(d.name, d.byteorder)
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
