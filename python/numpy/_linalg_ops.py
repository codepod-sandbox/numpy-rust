"""Linear algebra operations in the flat numpy namespace."""
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import (
    _ObjectArray,
    AxisError,
    _builtin_max,
    _copy_into,
    _ComplexResultArray,
    _flat_arraylike_data,
)
from ._creation import array, asarray
from ._core_types import dtype

__all__ = [
    'tensordot', 'inner', 'kron', 'matmul', 'vdot', 'einsum', 'einsum_path',
    'outer', 'cross',
    '_has_complex', '_scimath_wrap', '_ScimathModule',
]


def _ensure_array(value):
    return value if isinstance(value, ndarray) else asarray(value)


def _ensure_array_pair(a, b):
    return _ensure_array(a), _ensure_array(b)


def _flatten_array(value):
    return _ensure_array(value).flatten()


def _normalize_tensordot_axes(axes, a_ndim, b_ndim):
    if isinstance(axes, int):
        axes_a = list(range(a_ndim - axes, a_ndim))
        axes_b = list(range(0, axes))
    else:
        axes_a = list(axes[0]) if not isinstance(axes[0], int) else [axes[0]]
        axes_b = list(axes[1]) if not isinstance(axes[1], int) else [axes[1]]
    axes_a = [ax if ax >= 0 else ax + a_ndim for ax in axes_a]
    axes_b = [ax if ax >= 0 else ax + b_ndim for ax in axes_b]
    return axes_a, axes_b


def _coerce_einsum_arrays(arrays, dtype=None):
    arrays = [_ensure_array(arr) for arr in arrays]
    if dtype is not None:
        import numpy as _np
        compute_dt = str(_np.dtype(dtype))
        return [arr if str(arr.dtype) == compute_dt else arr.astype(compute_dt) for arr in arrays]
    if len(arrays) > 1:
        import numpy as _np
        common_dt = str(arrays[0].dtype)
        for arr in arrays[1:]:
            common_dt = str(_np.promote_types(common_dt, str(arr.dtype)))
        arrays = [arr if str(arr.dtype) == common_dt else arr.astype(common_dt) for arr in arrays]
    return arrays


def _has_complex(result):
    """Check if any element in result is complex (avoids shadowed builtin any)."""
    for r in result:
        if isinstance(r, complex):
            return True
    return False


def _scimath_wrap(result):
    """Wrap complex128 ndarray results in _ComplexResultArray for compat.
    RustPython returns complex scalars as (re, im) tuples; _ComplexResultArray
    converts them to proper Python complex objects on element access."""
    if hasattr(result, 'dtype') and result.dtype == dtype('complex128'):
        flat = _flat_arraylike_data(result.flatten())
        flat_complex = [complex(v[0], v[1]) if isinstance(v, tuple) else complex(v) for v in flat]
        return _ComplexResultArray(flat_complex, result.shape)
    return result


class _ScimathModule:
    """Complex-safe math functions (numpy.lib.scimath)."""

    @staticmethod
    def _unary(fn, x):
        return _scimath_wrap(fn(_ensure_array(x)))

    @staticmethod
    def _binary(fn, x, y):
        return _scimath_wrap(fn(_ensure_array(x), _ensure_array(y)))

    @staticmethod
    def sqrt(x):
        return _ScimathModule._unary(_native.scimath_sqrt, x)

    @staticmethod
    def log(x):
        return _ScimathModule._unary(_native.scimath_log, x)

    @staticmethod
    def log2(x):
        return _ScimathModule._unary(_native.scimath_log2, x)

    @staticmethod
    def log10(x):
        return _ScimathModule._unary(_native.scimath_log10, x)

    @staticmethod
    def arcsin(x):
        return _ScimathModule._unary(_native.scimath_arcsin, x)

    @staticmethod
    def arccos(x):
        return _ScimathModule._unary(_native.scimath_arccos, x)

    @staticmethod
    def arctanh(x):
        return _ScimathModule._unary(_native.scimath_arctanh, x)

    @staticmethod
    def power(x, p):
        return _ScimathModule._binary(_native.scimath_power, x, p)


# ---------------------------------------------------------------------------
# Linear algebra / product functions
# ---------------------------------------------------------------------------

def outer(a, b, out=None):
    """Compute outer product."""
    a = _flatten_array(a)
    b = _flatten_array(b)
    result = _native.outer(a, b)
    if out is not None:
        _copy_into(out, result)
        return out
    return result


def cross(a, b, axisa=-1, axisb=-1, axisc=-1, axis=None):
    """Cross product of two arrays."""
    from ._manipulation import moveaxis, broadcast_shapes, broadcast_to
    _a_scalar = not isinstance(a, (ndarray, list, tuple))
    _b_scalar = not isinstance(b, (ndarray, list, tuple))
    a, b = _ensure_array_pair(a, b)
    if a.ndim == 0 or b.ndim == 0 or _a_scalar or _b_scalar:
        raise ValueError("At least one array has zero dimension")
    if axis is not None:
        axisa = axisb = axisc = axis
    if axisa < -a.ndim or axisa >= a.ndim:
        raise AxisError(axisa, a.ndim, "axisa")
    if axisb < -b.ndim or axisb >= b.ndim:
        raise AxisError(axisb, b.ndim, "axisb")
    if a.ndim > 1 and axisa != -1 and axisa != a.ndim - 1:
        a = moveaxis(a, axisa, -1)
    if b.ndim > 1 and axisb != -1 and axisb != b.ndim - 1:
        b = moveaxis(b, axisb, -1)
    if a.ndim >= 2 and b.ndim == 1:
        b = b.reshape((1,) * (a.ndim - 1) + (b.shape[0],))
        b_shape = list(a.shape[:-1]) + [b.shape[-1]]
        b_flat = _flat_arraylike_data(b.flatten())
        b_new = []
        batch = 1
        for s in a.shape[:-1]:
            batch *= s
        vec_len = b.shape[-1]
        for i in range(batch):
            b_new.extend(b_flat[:vec_len])
        b = array(b_new).reshape(b_shape)
    elif b.ndim >= 2 and a.ndim == 1:
        a = a.reshape((1,) * (b.ndim - 1) + (a.shape[0],))
        a_shape = list(b.shape[:-1]) + [a.shape[-1]]
        a_flat = _flat_arraylike_data(a.flatten())
        a_new = []
        batch = 1
        for s in b.shape[:-1]:
            batch *= s
        vec_len = a.shape[-1]
        for i in range(batch):
            a_new.extend(a_flat[:vec_len])
        a = array(a_new).reshape(a_shape)
    if a.ndim == 1 and b.ndim == 1:
        af = _flat_arraylike_data(a.flatten())
        bf = _flat_arraylike_data(b.flatten())
        la, lb = len(af), len(bf)
        if la not in (2, 3) or lb not in (2, 3):
            raise ValueError("incompatible vector sizes for cross product")
        if la == 2:
            af = [af[0], af[1], 0.0]
        if lb == 2:
            bf = [bf[0], bf[1], 0.0]
        cx = af[1]*bf[2] - af[2]*bf[1]
        cy = af[2]*bf[0] - af[0]*bf[2]
        cz = af[0]*bf[1] - af[1]*bf[0]
        if la == 2 and lb == 2:
            return array(cz)
        return array([cx, cy, cz])
    if a.ndim >= 2 and b.ndim >= 2:
        a_batch = a.shape[:-1]
        b_batch = b.shape[:-1]
        try:
            out_batch = broadcast_shapes(a_batch, b_batch)
        except Exception:
            out_batch = a_batch
        a_bc = broadcast_to(a, tuple(out_batch) + (a.shape[-1],))
        b_bc = broadcast_to(b, tuple(out_batch) + (b.shape[-1],))
        la = a_bc.shape[-1]
        lb = b_bc.shape[-1]
        batch_size = 1
        for s in out_batch:
            batch_size *= s
        af = _flat_arraylike_data(a_bc.flatten())
        bf = _flat_arraylike_data(b_bc.flatten())
        results = []
        for i in range(batch_size):
            ai = af[i * la:(i + 1) * la]
            bi = bf[i * lb:(i + 1) * lb]
            if la == 2:
                ai = [ai[0], ai[1], 0.0]
            if lb == 2:
                bi = [bi[0], bi[1], 0.0]
            cx = ai[1]*bi[2] - ai[2]*bi[1]
            cy = ai[2]*bi[0] - ai[0]*bi[2]
            cz = ai[0]*bi[1] - ai[1]*bi[0]
            if la == 2 and lb == 2:
                results.append(cz)
            else:
                results.extend([cx, cy, cz])
        if la == 2 and lb == 2:
            result = array(results).reshape(out_batch)
        else:
            result = array(results).reshape(list(out_batch) + [3])
        if axisc != -1 and axisc != result.ndim - 1 and result.ndim > 1 and not (la == 2 and lb == 2):
            result = moveaxis(result, -1, axisc)
        return result


def tensordot(a, b, axes=2):
    """Compute tensor dot product along specified axes."""
    from ._manipulation import _transpose_with_axes
    from _numpy_native import dot
    a, b = _ensure_array_pair(a, b)
    na = a.ndim
    nb = b.ndim
    axes_a, axes_b = _normalize_tensordot_axes(axes, na, nb)
    free_a = [i for i in range(na) if i not in axes_a]
    free_b = [i for i in range(nb) if i not in axes_b]
    perm_a = free_a + axes_a
    perm_b = axes_b + free_b
    at = _transpose_with_axes(a, perm_a)
    bt = _transpose_with_axes(b, perm_b)
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
        # 0-d result — return scalar with shape ()
        val = float(result.flatten()[0])
        return array(val)
    return result.reshape(out_shape)


def inner(a, b):
    """Inner product of two arrays."""
    from _numpy_native import dot
    a, b = _ensure_array_pair(a, b)
    if a.ndim <= 1 and b.ndim <= 1:
        return dot(a, b)
    if a.ndim == 2 and b.ndim == 2:
        return dot(a, b.T)
    return tensordot(a, b, axes=([-1], [-1]))


def kron(a, b):
    """Kronecker product of two arrays."""
    a, b = _ensure_array_pair(a, b)
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


def matmul(x1, x2):
    """Matrix product of two arrays (same as the @ operator)."""
    # Check for __array_ufunc__ dispatch on inputs
    for arg in (x1, x2):
        if not isinstance(arg, ndarray) and not isinstance(arg, (int, float, bool, complex)):
            au = getattr(type(arg), '__array_ufunc__', NotImplemented)
            if au is not NotImplemented and au is not None:
                result = arg.__array_ufunc__(matmul, '__call__', x1, x2)
                if result is not NotImplemented:
                    return result
    from _numpy_native import dot
    x1, x2 = _ensure_array_pair(x1, x2)
    return dot(x1, x2)


def vdot(a, b):
    """Conjugate dot product of two arrays (flattened)."""
    from _numpy_native import dot
    a = _flatten_array(a)
    b = _flatten_array(b)
    return dot(a, b)


def einsum(*operands, **kwargs):
    """Evaluates the Einstein summation convention on the operands."""
    out = kwargs.pop('out', None)
    dtype = kwargs.pop('dtype', None)
    order = kwargs.pop('order', 'K')
    casting = kwargs.pop('casting', 'safe')
    optimize = kwargs.pop('optimize', False)
    if len(operands) == 0:
        raise ValueError("No input operands")
    # Handle interleaved subscript format: einsum(op0, subs0, op1, subs1, ..., [output_subs])
    if not isinstance(operands[0], str):
        # Interleaved format
        arrays = []
        sub_parts = []
        # Map integer subscript labels to letters
        label_map = {}
        label_counter = [0]
        def _int_to_letter(n):
            if n == Ellipsis:
                return '...'
            if n not in label_map:
                label_map[n] = chr(ord('a') + label_counter[0])
                label_counter[0] += 1
            return label_map[n]
        i = 0
        while i < len(operands):
            arr = operands[i]
            i += 1
            if i >= len(operands):
                # Last element is output subscripts
                if isinstance(arr, list):
                    out_sub = ''.join(_int_to_letter(x) for x in arr)
                    sub_parts.append('->' + out_sub)
                break
            subs = operands[i]
            i += 1
            # Convert ndarray subscripts to a list
            if isinstance(subs, ndarray):
                subs = _flat_arraylike_data(subs)
                if not isinstance(subs, list):
                    subs = [subs]
            if isinstance(subs, list):
                sub_str = ''.join(_int_to_letter(x) for x in subs)
            else:
                sub_str = str(subs)
            arrays.append(asarray(arr))
            sub_parts.append(sub_str)
            # Check if next element is a list (output subscripts) with no following operand
            if i < len(operands) and isinstance(operands[i], list) and (i + 1 >= len(operands) or isinstance(operands[i + 1], list)):
                out_sub = ''.join(_int_to_letter(x) for x in operands[i])
                sub_parts.append('->' + out_sub)
                i += 1
        subscripts = ','.join(p for p in sub_parts if not p.startswith('->'))
        out_part = [p for p in sub_parts if p.startswith('->')]
        if out_part:
            subscripts += out_part[0]
    else:
        if len(operands) < 1:
            raise ValueError("No input operands")
        subscripts = operands[0]
        arrays = list(operands[1:])
    if not isinstance(subscripts, str):
        raise TypeError("subscripts must be a string")
    if len(arrays) == 0 and subscripts == '':
        raise ValueError("No input operands")
    # Validate and convert all operands to arrays
    arrays = _coerce_einsum_arrays(arrays, dtype=dtype)
    # Handle implicit output subscripts
    if '->' not in subscripts:
        input_subs = subscripts.replace(' ', '')
        # Handle ellipsis
        parts = input_subs.split(',')
        from collections import Counter
        counts = Counter()
        has_ellipsis = '...' in input_subs
        for p in parts:
            p_clean = p.replace('...', '')
            counts.update(p_clean)
        output = ''.join(sorted(c for c, n in counts.items() if n == 1))
        if has_ellipsis:
            output = '...' + output
        subscripts = input_subs + '->' + output
    # Expand ellipsis to explicit indices before calling native einsum
    if '...' in subscripts:
        parts = subscripts.split('->')
        input_part = parts[0]
        output_part = parts[1] if len(parts) > 1 else None
        input_terms = input_part.split(',')
        # Find used indices
        used_indices = set()
        for t in input_terms:
            used_indices.update(c for c in t if c.isalpha())
        if output_part:
            used_indices.update(c for c in output_part if c.isalpha())
        # Find available indices for ellipsis expansion
        all_letters = [chr(c) for c in range(ord('A'), ord('Z')+1)] + [chr(c) for c in range(ord('a'), ord('z')+1)]
        avail = [c for c in all_letters if c not in used_indices]
        # Determine ellipsis dimensions for each operand
        expanded_terms = []
        ellipsis_ndim = 0
        for idx_t, t in enumerate(input_terms):
            if '...' in t:
                explicit_count = len(t.replace('...', ''))
                if idx_t < len(arrays):
                    arr_ndim = arrays[idx_t].ndim
                    this_ellipsis = arr_ndim - explicit_count
                    if this_ellipsis < 0:
                        this_ellipsis = 0
                    ellipsis_ndim = _builtin_max(ellipsis_ndim, this_ellipsis)
        # Now expand
        ellipsis_labels = avail[:ellipsis_ndim]
        new_input_terms = []
        for idx_t, t in enumerate(input_terms):
            if '...' in t:
                explicit_count = len(t.replace('...', ''))
                if idx_t < len(arrays):
                    arr_ndim = arrays[idx_t].ndim
                    this_ellipsis = arr_ndim - explicit_count
                else:
                    this_ellipsis = ellipsis_ndim
                if this_ellipsis < 0:
                    this_ellipsis = 0
                # Use right-aligned ellipsis labels for broadcasting
                labels = ellipsis_labels[ellipsis_ndim - this_ellipsis:]
                new_input_terms.append(t.replace('...', ''.join(labels)))
            else:
                new_input_terms.append(t)
        new_input = ','.join(new_input_terms)
        if output_part is not None:
            if '...' in output_part:
                new_output = output_part.replace('...', ''.join(ellipsis_labels))
            else:
                new_output = output_part
            subscripts = new_input + '->' + new_output
        else:
            subscripts = new_input
    try:
        result = _native.einsum(subscripts, *arrays)
    except TypeError as e:
        # Handle scalar inputs / type mismatches that native einsum can't handle
        err_msg = str(e)
        if "Expected type" in err_msg:
            # Upcast everything to float64 as a safe fallback
            new_arrays = [arr.astype('float64') for arr in arrays]
            result = _native.einsum(subscripts, *new_arrays)
        else:
            raise
    if dtype is not None and isinstance(result, ndarray):
        import numpy as _np
        _out_dt = str(_np.dtype(dtype))
        if str(result.dtype) != _out_dt:
            result = result.astype(_out_dt)
    if out is not None:
        if isinstance(out, ndarray):
            flat_r = result.flatten()
            for i in range(flat_r.size):
                out.flat[i] = flat_r[i]
            return out
    return result


def einsum_path(*operands, optimize='greedy'):
    """Evaluate optimal contraction order (stub returns naive path).

    Performs the same input validation as einsum so that bad calls
    raise the same errors (ValueError / TypeError).
    """
    if len(operands) == 0:
        raise ValueError("No input operands")
    # Parse subscripts / arrays just like einsum does
    if not isinstance(operands[0], str):
        # Interleaved format – first operand must be array-like
        raise TypeError("subscripts must be a string")
    subscripts = operands[0]
    if not isinstance(subscripts, str):
        raise TypeError("subscripts must be a string")
    arrays = list(operands[1:])
    if len(arrays) == 0 and subscripts == '':
        raise ValueError("No input operands")
    # Convert to ndarrays
    for i in range(len(arrays)):
        arrays[i] = asarray(arrays[i])
    # Check operand count vs subscripts
    if '->' in subscripts:
        input_part = subscripts.split('->')[0]
    else:
        input_part = subscripts
    n_subs = len(input_part.split(','))
    if n_subs != len(arrays):
        raise ValueError(
            "einsum: {} operands but subscripts specify {}".format(
                len(arrays), n_subs))
    # Check dims vs subscript indices
    input_terms = input_part.split(',')
    for idx, term in enumerate(input_terms):
        if idx < len(arrays):
            clean = term.replace('...', '')
            n_explicit = len(clean)
            arr_ndim = arrays[idx].ndim
            has_ellipsis = '...' in term
            if has_ellipsis:
                if arr_ndim < n_explicit:
                    raise ValueError(
                        "einsum: operand has {} dims but subscript "
                        "has {} indices".format(arr_ndim, n_explicit))
            else:
                if n_explicit != arr_ndim:
                    raise ValueError(
                        "einsum: operand has {} dims but subscript "
                        "has {} indices".format(arr_ndim, n_explicit))
    # Return naive path
    n = len(arrays)
    path = [(0, 1)] * _builtin_max(1, n - 1)
    return path, ""
