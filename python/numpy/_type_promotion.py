"""Type casting and promotion rules: can_cast, result_type, promote_types, find_common_type, etc."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray, _builtin_max, _builtin_min
from ._scalar_types import (
    _ScalarType, _ScalarTypeMeta, _DTYPE_CHAR_MAP,
    float64, float32, float16,
    int8, int16, int32, int64,
    uint8, uint16, uint32, uint64,
    complex64, complex128,
    bool_, str_, bytes_, void, object_,
)
from ._dtype import dtype, _normalize_dtype, _normalize_dtype_with_size

__all__ = [
    'can_cast',
    'result_type',
    '_numeric_to_str_len',
    '_parse_us_dtype',
    'promote_types',
    'find_common_type',
    'mintypecode',
    'common_type',
    'min_scalar_type',
]


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

    if "rational" in str(to_name):
        return from_name in {"bool", "int8", "int16", "int32", "int64"} or from_name == to_name
    if "rational" in str(from_name):
        return to_name in {"float64", "double", "complex128"} or from_name == to_name

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
        if getattr(raw, 'kind', None) == 'S':
            return ('S', int(getattr(raw, 'itemsize', 0) or 0))
        if getattr(raw, 'kind', None) == 'U':
            itemsize = int(getattr(raw, 'itemsize', 0) or 0)
            return ('U', itemsize // 4 if itemsize else 0)
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

    def _is_non_native(dt):
        return isinstance(dt, dtype) and not getattr(dt, 'isnative', True)

    def _promote_structured_fields(dt1, dt2):
        n1 = getattr(dt1, 'names', None)
        n2 = getattr(dt2, 'names', None)
        if n1 is None or n2 is None or len(n1) != len(n2):
            raise TypeError("invalid type promotion")
        if dt1 == dt2:
            result = dtype(dt1)
            if getattr(dt1, 'metadata', None) is not None and getattr(dt1, 'metadata', None) == getattr(dt2, 'metadata', None):
                result.metadata = dt1.metadata
            else:
                result.metadata = None
            return result
        fields = []
        for name1, name2 in zip(n1, n2):
            if name1 != name2:
                raise TypeError("invalid type promotion")
            field1 = dt1.fields[name1][0]
            field2 = dt2.fields[name2][0]
            fields.append((name1, promote_types(field1, field2)))
        return dtype(fields)

    t1_meta = getattr(type1, 'metadata', None)
    t2_meta = getattr(type2, 'metadata', None)

    if "rational" in str(getattr(type1, 'name', type1)) or "rational" in str(getattr(type2, 'name', type2)):
        rational_dt = type1 if "rational" in str(getattr(type1, 'name', type1)) else type2
        other = type2 if rational_dt is type1 else type1
        other_name = getattr(other, 'name', str(other))
        if getattr(rational_dt, 'name', None) == other_name:
            result = dtype(getattr(rational_dt, 'name'))
            if t1_meta is not None and t1_meta == t2_meta and not (_is_non_native(type1) or _is_non_native(type2)):
                result.metadata = t1_meta
            return result
        if other_name in {"bool", "int8", "int16", "int32", "int64"}:
            if getattr(rational_dt, 'metadata', None) is None:
                return rational_dt
            return dtype(getattr(rational_dt, 'name'))
        if other_name in {"float64", "double"}:
            if getattr(other, 'metadata', None) is None:
                return other
            return dtype(other_name)
        raise TypeError("invalid type promotion")

    if getattr(type1, 'names', None) is not None and getattr(type2, 'names', None) is not None:
        return _promote_structured_fields(type1, type2)

    def _new_string_result(spec, source=None):
        result = dtype(spec)
        if source is not None and getattr(source, 'metadata', None) is not None:
            result.metadata = source.metadata
        return result

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
        if type1.kind in ('S', 'U') and t1_is != t2_is:
            pass
        else:
            if type1.kind in ('S', 'U', 'V') and not (_is_non_native(type1) or _is_non_native(type2)):
                return type1
            if (type1.kind == 'U' and (_is_non_native(type1) or _is_non_native(type2))):
                result = _new_string_result('U{}'.format(t1_is // 4 if t1_is else 0), type1)
            elif type1.kind in ('S', 'U', 'V') or t1_names is not None:
                result = dtype(type1)
            else:
                result = dtype(type1.name)
            # Byte-swapped identical types promote to native dtype and usually lose metadata
            if t1_meta is not None and t1_meta == t2_meta:
                if _is_non_native(type1) or _is_non_native(type2):
                    if getattr(result, 'char', None) == 'U' or getattr(result, 'name', None) == 'str':
                        result.metadata = t1_meta
                    else:
                        result.metadata = None
                else:
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
    if s1 == s2 and _us1 is None and _us2 is None:
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
            out_spec = out_char + str(out_size)
            t1_matches = _normalize_dtype_with_size(type1) == out_spec
            t2_matches = _normalize_dtype_with_size(type2) == out_spec
            if t1_matches and not _is_non_native(type1):
                return type1
            if t2_matches and not _is_non_native(type2):
                return type2
            if t1_matches:
                return _new_string_result(out_spec, type1 if out_char == 'U' else None)
            if t2_matches:
                return _new_string_result(out_spec, type2 if out_char == 'U' else None)
            return dtype(out_spec)
        # One is string/bytes, other is numeric
        if _us1 is not None:
            us_char, us_size = _us1
            numeric_name = s2
            us_dt = type1
        else:
            us_char, us_size = _us2
            numeric_name = s1
            us_dt = type2
        needed = _numeric_to_str_len(numeric_name)
        if needed is None:
            # Unknown numeric type - can't promote
            raise TypeError("Cannot promote string dtype with numeric dtype")
        out_size = max(us_size, needed)
        out_spec = us_char + str(out_size)
        if _is_non_native(us_dt):
            result = dtype(out_spec)
            if us_char == 'U' and getattr(us_dt, 'metadata', None) is not None and _normalize_dtype_with_size(us_dt) == out_spec:
                result.metadata = us_dt.metadata
            return result
        if str(us_dt) == out_spec or _normalize_dtype_with_size(us_dt) == out_spec:
            return us_dt
        return dtype(out_spec)

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


def min_scalar_type(a):
    """Return the data type with the smallest size and smallest scalar kind
    to which the given scalar 'a' can be safely cast."""
    if isinstance(a, bool):
        return dtype(bool_)
    if isinstance(a, int):
        # Find smallest int type that holds the value
        if 0 <= a <= 255:
            return dtype(uint8)
        if -128 <= a <= 127:
            return dtype(int8)
        if 0 <= a <= 65535:
            return dtype(uint16)
        if -32768 <= a <= 32767:
            return dtype(int16)
        if 0 <= a <= 2**32 - 1:
            return dtype(uint32)
        if -(2**31) <= a <= 2**31 - 1:
            return dtype(int32)
        if 0 <= a <= 2**64 - 1:
            return dtype(uint64)
        return dtype(int64)
    if isinstance(a, float):
        return dtype(float64)
    if isinstance(a, complex):
        return dtype(complex128)
    # For arrays, return array dtype
    if hasattr(a, 'dtype'):
        return a.dtype
    return dtype(float64)
