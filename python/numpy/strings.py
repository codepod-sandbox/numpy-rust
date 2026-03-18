"""numpy.strings - string operations on arrays."""
import numpy as np


def _apply_unary(func, a):
    """Apply a unary string method element-wise."""
    scalar_input = isinstance(a, (str, bytes))
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(func(str(v)))
    if scalar_input and len(result) == 1:
        return result[0]
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out


def _apply_binary(func, a, b):
    """Apply a binary string method element-wise."""
    scalar_input = isinstance(a, (str, bytes))
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    flat_a = arr_a.flatten()
    flat_b = arr_b.flatten()
    result = []
    size = max(flat_a.size, flat_b.size)
    for i in range(size):
        va = flat_a[i % flat_a.size]
        vb = flat_b[i % flat_b.size]
        if isinstance(va, bytes):
            va = va.decode('latin-1')
        if isinstance(vb, bytes):
            vb = vb.decode('latin-1')
        result.append(func(str(va), vb))
    if scalar_input and len(result) == 1:
        return result[0]
    out = np.array(result)
    shape = arr_a.shape if arr_a.size >= arr_b.size else arr_b.shape
    return out.reshape(shape) if len(shape) > 0 else out


def add(x1, x2):
    return _apply_binary(lambda a, b: a + str(b), x1, x2)

def multiply(a, i):
    scalar_input = isinstance(a, (str, bytes))
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    i_val = int(i) if not hasattr(i, '__len__') else None
    for idx in range(flat.size):
        v = flat[idx]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        rep = i_val if i_val is not None else int(i)
        result.append(str(v) * rep)
    if scalar_input and len(result) == 1:
        return result[0]
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def center(a, width, fillchar=' '):
    return _apply_unary(lambda s: s.center(int(width), fillchar), a)

def ljust(a, width, fillchar=' '):
    return _apply_unary(lambda s: s.ljust(int(width), fillchar), a)

def rjust(a, width, fillchar=' '):
    return _apply_unary(lambda s: s.rjust(int(width), fillchar), a)

def zfill(a, width):
    return _apply_unary(lambda s: s.zfill(int(width)), a)

def upper(a):
    return _apply_unary(lambda s: s.upper(), a)

def lower(a):
    return _apply_unary(lambda s: s.lower(), a)

def swapcase(a):
    return _apply_unary(lambda s: s.swapcase(), a)

def capitalize(a):
    return _apply_unary(lambda s: s.capitalize(), a)

def title(a):
    return _apply_unary(lambda s: s.title(), a)

def replace(a, old, new, count=-1):
    if count == -1:
        return _apply_unary(lambda s: s.replace(str(old), str(new)), a)
    return _apply_unary(lambda s: s.replace(str(old), str(new), int(count)), a)

def strip(a, chars=None):
    return _apply_unary(lambda s: s.strip(chars), a)

def lstrip(a, chars=None):
    return _apply_unary(lambda s: s.lstrip(chars), a)

def rstrip(a, chars=None):
    return _apply_unary(lambda s: s.rstrip(chars), a)

def split(a, sep=None, maxsplit=-1):
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(str(v).split(sep, maxsplit))
    return np.array(result, dtype=object)

def join(sep, seq):
    arr = np.asarray(seq)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        result.append(str(sep).join(v))
    return np.array(result)

def count(a, sub, start=0, end=None):
    def _count(s):
        return s.count(str(sub), start, end)
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(_count(str(v)))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def find(a, sub, start=0, end=None):
    def _find(s):
        return s.find(str(sub), start, end)
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(_find(str(v)))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def rfind(a, sub, start=0, end=None):
    def _rfind(s):
        return s.rfind(str(sub), start, end)
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(_rfind(str(v)))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def index(a, sub, start=0, end=None):
    def _index(s):
        return s.index(str(sub), start, end)
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(_index(str(v)))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def rindex(a, sub, start=0, end=None):
    def _rindex(s):
        return s.rindex(str(sub), start, end)
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(_rindex(str(v)))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def startswith(a, prefix, start=0, end=None):
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(str(v).startswith(str(prefix), start, end))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def endswith(a, suffix, start=0, end=None):
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(str(v).endswith(str(suffix), start, end))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def encode(a, encoding='utf-8', errors='strict'):
    return _apply_unary(lambda s: s.encode(encoding, errors), a)

def decode(a, encoding='utf-8', errors='strict'):
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            result.append(v.decode(encoding, errors))
        else:
            result.append(str(v))
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out

def isalpha(a):
    return _apply_unary(lambda s: s.isalpha(), a)

def isdigit(a):
    return _apply_unary(lambda s: s.isdigit(), a)

def isalnum(a):
    return _apply_unary(lambda s: s.isalnum(), a)

def isspace(a):
    return _apply_unary(lambda s: s.isspace(), a)

def isupper(a):
    return _apply_unary(lambda s: s.isupper(), a)

def islower(a):
    return _apply_unary(lambda s: s.islower(), a)

def istitle(a):
    return _apply_unary(lambda s: s.istitle(), a)

def isnumeric(a):
    return _apply_unary(lambda s: s.isnumeric(), a)

def isdecimal(a):
    return _apply_unary(lambda s: s.isdecimal(), a)

def partition(a, sep):
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(str(v).partition(str(sep)))
    return np.array(result, dtype=object)

def rpartition(a, sep):
    arr = np.asarray(a)
    flat = arr.flatten()
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        result.append(str(v).rpartition(str(sep)))
    return np.array(result, dtype=object)

def expandtabs(a, tabsize=8):
    return _apply_unary(lambda s: s.expandtabs(int(tabsize)), a)

def mod(a, values):
    """String formatting with % operator, element-wise."""
    scalar_input = isinstance(a, (str, bytes))
    arr = np.asarray(a)
    flat = arr.flatten()
    if not isinstance(values, (tuple, list)):
        values_list = [values]
    else:
        values_list = list(values)
    result = []
    for i in range(flat.size):
        v = flat[i]
        if isinstance(v, bytes):
            v = v.decode('latin-1')
        try:
            result.append(str(v) % tuple(values_list))
        except TypeError:
            result.append(str(v) % values_list[0] if len(values_list) == 1 else str(v) % tuple(values_list))
    if scalar_input and len(result) == 1:
        return result[0]
    out = np.array(result)
    return out.reshape(arr.shape) if len(arr.shape) > 0 else out


def equal(x1, x2):
    """Return element-wise string equality comparison."""
    arr1 = np.asarray(x1)
    arr2 = np.asarray(x2)
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    size = max(flat1.size, flat2.size)
    result = []
    for i in range(size):
        v1 = flat1[i % flat1.size]
        v2 = flat2[i % flat2.size]
        if isinstance(v1, bytes):
            v1 = v1.decode('latin-1')
        if isinstance(v2, bytes):
            v2 = v2.decode('latin-1')
        result.append(str(v1) == str(v2))
    out = np.array(result)
    shape = arr1.shape if arr1.size >= arr2.size else arr2.shape
    return out.reshape(shape) if len(shape) > 0 else out


def not_equal(x1, x2):
    """Return element-wise string inequality comparison."""
    return ~equal(x1, x2)


def greater(x1, x2):
    """Return element-wise string greater-than comparison."""
    return _apply_binary(lambda a, b: a > str(b), x1, x2)


def greater_equal(x1, x2):
    """Return element-wise string greater-equal comparison."""
    return _apply_binary(lambda a, b: a >= str(b), x1, x2)


def less(x1, x2):
    """Return element-wise string less-than comparison."""
    return _apply_binary(lambda a, b: a < str(b), x1, x2)


def less_equal(x1, x2):
    """Return element-wise string less-equal comparison."""
    return _apply_binary(lambda a, b: a <= str(b), x1, x2)


def str_len(a):
    """Return element-wise string lengths."""
    return _apply_unary(lambda s: len(s), a)


def translate(a, table, deletechars=None):
    def _translate(s):
        if deletechars:
            s = s.translate(str.maketrans('', '', deletechars))
        if table:
            s = s.translate(table)
        return s
    return _apply_unary(_translate, a)
