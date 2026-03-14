"""Linear algebra Python-level wrappers."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray, linalg as _linalg_native, dot
from ._helpers import _builtin_max
from ._creation import array, asarray, eye

__all__ = [
    '_linalg_pinv',
    '_linalg_matrix_rank',
    '_linalg_matrix_power',
    '_linalg_slogdet',
    '_linalg_cond',
    '_linalg_eigh',
    '_linalg_eigvals',
    '_linalg_multi_dot',
    '_linalg_norm_with_axis',
    '_linalg_eigvalsh',
    '_linalg_matrix_norm',
    '_linalg_vector_norm',
    '_linalg_tensorsolve',
]

# Keep a reference to the native linalg module (same object as numpy.linalg)
linalg = _linalg_native

_linalg_norm_orig = linalg.norm


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
    # Need diag from numpy - use lazy import to avoid circular reference
    import numpy as _np
    s_inv = _np.diag(array(s_inv_vals))
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


def _linalg_norm_with_axis(x, ord=None, axis=None, keepdims=False):
    """Compute matrix or vector norm, optionally along an axis."""
    x = asarray(x)
    if axis is None:
        # Delegate to native Rust norm (Frobenius / flat L2)
        return _linalg_norm_orig(x)
    # Compute norm along specified axis (use positional arg for sum/max)
    if ord is None or ord == 2:
        return (x * x).sum(axis) ** 0.5
    elif ord == 1:
        return abs(x).sum(axis)
    elif ord == float('inf'):
        return abs(x).max(axis)
    else:
        return (abs(x) ** ord).sum(axis) ** (1.0 / ord)


def _linalg_eigvalsh(a, UPLO='L'):
    """Eigenvalues of symmetric/Hermitian matrix."""
    vals, _ = linalg.eigh(a)
    return vals


def _linalg_matrix_norm(x, ord='fro', axis=(-2, -1), keepdims=False):
    """Matrix norm."""
    return linalg.norm(x)


def _linalg_vector_norm(x, ord=2, axis=None, keepdims=False):
    """Vector norm."""
    return linalg.norm(x)


def _linalg_tensorsolve(a, b, axes=None):
    """Solve tensor equation (stub)."""
    raise NotImplementedError("tensorsolve not implemented")


# Monkey-patch linalg module with Python-level extensions
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
linalg.eigvalsh = _linalg_eigvalsh
linalg.matrix_norm = _linalg_matrix_norm
linalg.vector_norm = _linalg_vector_norm
linalg.tensorsolve = _linalg_tensorsolve
