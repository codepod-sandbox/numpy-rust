"""Polynomial utilities, convolution, correlation."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray, _builtin_min, _builtin_max
from ._creation import array, asarray

__all__ = [
    'poly1d',
    'roots',
    'polyfit',
    'polyval',
    'polyadd',
    'polysub',
    'polymul',
    'polyder',
    'polyint',
    'polydiv',
    'convolve',
    'correlate',
]


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(y, ndarray):
        y = array(y)
    return _native.polyfit(x, y, int(deg))


def polyval(p, x):
    if not isinstance(p, ndarray):
        p = array(p)
    if not isinstance(x, ndarray):
        x = array(x)
    return _native.polyval(p, x)


# --- Polynomial utilities ---------------------------------------------------

def roots(p):
    """Return the roots of a polynomial with coefficients given in p."""
    if isinstance(p, poly1d):
        coeffs = list(p._coeffs)
    elif isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    # Remove leading zeros
    while len(coeffs) > 1 and coeffs[0] == 0:
        coeffs = coeffs[1:]
    n = len(coeffs) - 1  # degree
    if n == 0:
        return array([])
    if n == 1:
        return array([-coeffs[1] / coeffs[0]])
    if n == 2:
        a, b, c = coeffs[0], coeffs[1], coeffs[2]
        disc = b * b - 4 * a * c
        if disc >= 0:
            sq = disc ** 0.5
            return array([(-b + sq) / (2 * a), (-b - sq) / (2 * a)])
        else:
            sq = (-disc) ** 0.5
            return array([(-b) / (2 * a), (-b) / (2 * a)])  # real part only
    # For degree > 2: Durand-Kerner method for finding all roots simultaneously.
    # Normalize polynomial so leading coefficient is 1
    a0 = float(coeffs[0])
    norm_coeffs = [float(c) / a0 for c in coeffs]
    # Horner's method to compute polynomial value at a point
    def _poly_at(z, cs):
        val = 0.0
        for c in cs:
            val = val * z + c
        return val
    # Bound on root magnitudes
    _abs_coeffs = [abs(c) for c in norm_coeffs[1:]]
    _max_coeff = _abs_coeffs[0]
    for _ac in _abs_coeffs[1:]:
        if _ac > _max_coeff:
            _max_coeff = _ac
    bound = 1.0 + _max_coeff
    # Initial guesses: distinct real values spread around
    z = [bound * (0.4 + 0.6 * _math.cos(2.0 * _math.pi * (k + 0.25) / n)) for k in range(n)]
    # Durand-Kerner iteration
    for _iteration in range(1000):
        max_delta = 0.0
        new_z = list(z)
        for i in range(n):
            pval = _poly_at(z[i], norm_coeffs)
            denom = 1.0
            for j in range(n):
                if j != i:
                    diff = z[i] - z[j]
                    if abs(diff) < 1e-15:
                        diff = 1e-15
                    denom *= diff
            if abs(denom) < 1e-30:
                denom = 1e-30
            delta = pval / denom
            new_z[i] = z[i] - delta
            if abs(delta) > max_delta:
                max_delta = abs(delta)
        z = new_z
        if max_delta < 1e-12:
            break
    return array(z)


def polyadd(a1, a2):
    """Add two polynomials (coefficient arrays, highest degree first)."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    while len(a1) < len(a2):
        a1.insert(0, 0.0)
    while len(a2) < len(a1):
        a2.insert(0, 0.0)
    return array([a1[i] + a2[i] for i in range(len(a1))])


def polysub(a1, a2):
    """Subtract two polynomials."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    while len(a1) < len(a2):
        a1.insert(0, 0.0)
    while len(a2) < len(a1):
        a2.insert(0, 0.0)
    return array([a1[i] - a2[i] for i in range(len(a1))])


def polymul(a1, a2):
    """Multiply two polynomials."""
    if isinstance(a1, poly1d):
        a1 = list(a1._coeffs)
    elif isinstance(a1, ndarray):
        a1 = [a1[i] for i in range(a1.size)]
    else:
        a1 = [float(c) for c in a1]
    if isinstance(a2, poly1d):
        a2 = list(a2._coeffs)
    elif isinstance(a2, ndarray):
        a2 = [a2[i] for i in range(a2.size)]
    else:
        a2 = [float(c) for c in a2]
    n = len(a1) + len(a2) - 1
    result = [0.0] * n
    for i, c1 in enumerate(a1):
        for j, c2 in enumerate(a2):
            result[i + j] += c1 * c2
    return array(result)


def polyder(p, m=1):
    """Return the derivative of the specified order of a polynomial."""
    if isinstance(p, poly1d):
        return p.deriv(m)
    if isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    for _ in range(m):
        n = len(coeffs) - 1
        if n <= 0:
            coeffs = [0.0]
            break
        new_coeffs = []
        for i in range(n):
            new_coeffs.append(coeffs[i] * (n - i))
        coeffs = new_coeffs
    return array(coeffs)


def polyint(p, m=1, k=0):
    """Return the integral of a polynomial."""
    if isinstance(p, poly1d):
        return p.integ(m, k)
    if isinstance(p, ndarray):
        coeffs = [p[i] for i in range(p.size)]
    else:
        coeffs = [float(c) for c in p]
    for _ in range(m):
        n = len(coeffs)
        new_coeffs = []
        for i in range(n):
            new_coeffs.append(coeffs[i] / (n - i))
        new_coeffs.append(float(k))
        coeffs = new_coeffs
    return array(coeffs)


class poly1d:
    """A one-dimensional polynomial class."""
    def __init__(self, c_or_r, r=False, variable=None):
        if isinstance(c_or_r, poly1d):
            self._coeffs = list(c_or_r._coeffs)
        elif r:
            # c_or_r are roots, convert to coefficients
            self._coeffs = [1.0]
            if isinstance(c_or_r, ndarray):
                roots_list = [c_or_r[i] for i in range(c_or_r.size)]
            else:
                roots_list = list(c_or_r)
            for root in roots_list:
                new_coeffs = [0.0] * (len(self._coeffs) + 1)
                for i, c in enumerate(self._coeffs):
                    new_coeffs[i] += c
                    new_coeffs[i + 1] -= c * float(root)
                self._coeffs = new_coeffs
        else:
            if isinstance(c_or_r, ndarray):
                self._coeffs = [c_or_r[i] for i in range(c_or_r.size)]
            else:
                self._coeffs = [float(c) for c in c_or_r]
        self._variable = variable or 'x'

    @property
    def coeffs(self):
        return array(self._coeffs)

    @property
    def c(self):
        return self.coeffs

    @property
    def order(self):
        return len(self._coeffs) - 1

    @property
    def roots(self):
        return _poly1d_roots(self._coeffs)

    @property
    def o(self):
        return self.order

    def __call__(self, val):
        return polyval(self._coeffs, val)

    def __add__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            oc = [float(other)]
        else:
            oc = list(other)
        return poly1d(polyadd(self._coeffs, oc))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            oc = [float(other)]
        else:
            oc = list(other)
        return poly1d(polysub(self._coeffs, oc))

    def __mul__(self, other):
        if isinstance(other, poly1d):
            oc = other._coeffs
        elif isinstance(other, (int, float)):
            return poly1d([c * float(other) for c in self._coeffs])
        else:
            oc = list(other)
        return poly1d(polymul(self._coeffs, oc))

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return poly1d([c * float(other) for c in self._coeffs])
        return self.__mul__(other)

    def __neg__(self):
        return poly1d([-c for c in self._coeffs])

    def __len__(self):
        return self.order

    def __getitem__(self, idx):
        # poly1d[i] returns coefficient of x^i (reverse indexing)
        if idx > self.order:
            return 0.0
        return self._coeffs[self.order - idx]

    def deriv(self, m=1):
        """Return the derivative of this polynomial."""
        coeffs = list(self._coeffs)
        for _ in range(m):
            n = len(coeffs) - 1
            if n <= 0:
                coeffs = [0.0]
                break
            new_coeffs = []
            for i in range(n):
                new_coeffs.append(coeffs[i] * (n - i))
            coeffs = new_coeffs
        return poly1d(coeffs)

    def integ(self, m=1, k=0):
        """Return the integral of this polynomial."""
        coeffs = list(self._coeffs)
        for _ in range(m):
            n = len(coeffs)
            new_coeffs = []
            for i in range(n):
                new_coeffs.append(coeffs[i] / (n - i))
            new_coeffs.append(float(k))
            coeffs = new_coeffs
        return poly1d(coeffs)

    def __repr__(self):
        return "poly1d(" + repr(self._coeffs) + ")"

    def __str__(self):
        return "poly1d(" + repr(self._coeffs) + ")"


# Alias so poly1d.roots can call without name clash with the module-level roots()
_poly1d_roots = roots


def polydiv(u, v):
    """Polynomial division: returns (quotient, remainder)."""
    if isinstance(u, poly1d):
        u = list(u._coeffs)
    elif isinstance(u, ndarray):
        u = [float(u[i]) for i in range(u.size)]
    else:
        u = [float(c) for c in u]
    if isinstance(v, poly1d):
        v = list(v._coeffs)
    elif isinstance(v, ndarray):
        v = [float(v[i]) for i in range(v.size)]
    else:
        v = [float(c) for c in v]
    n = len(u)
    nv = len(v)
    if nv > n:
        return array([0.0]), array(u)
    q = [0.0] * (n - nv + 1)
    r = list(u)
    for i in range(n - nv + 1):
        q[i] = r[i] / v[0]
        for j in range(nv):
            r[i + j] -= q[i] * v[j]
    remainder = r[n - nv + 1:]
    return array(q), array(remainder)


def convolve(a, v, mode='full'):
    """Discrete, linear convolution of two one-dimensional sequences."""
    a = asarray(a).flatten()
    v = asarray(v).flatten()
    # Normalize mode: support integer and abbreviated string modes
    if isinstance(mode, int):
        _mode_map = {0: 'valid', 1: 'same', 2: 'full'}
        if mode not in _mode_map:
            raise ValueError("mode must be 0, 1, or 2")
        mode = _mode_map[mode]
    elif isinstance(mode, str):
        _abbrev = {'v': 'valid', 's': 'same', 'f': 'full'}
        if mode in _abbrev:
            import warnings as _w
            _w.warn("Use of abbreviated mode '{}' is deprecated. Use the full string.".format(mode), DeprecationWarning, stacklevel=3)
            mode = _abbrev[mode]
    elif mode is None:
        raise TypeError("mode must not be None")
    na = len(a)
    nv = len(v)
    n_full = na + nv - 1
    a_list = a.tolist()
    v_list = v.tolist()
    _is_complex = 'complex' in str(a.dtype) or 'complex' in str(v.dtype)
    if not _is_complex:
        # Check if values are actually complex (e.g. _ObjectArray with complex)
        try:
            if isinstance(a_list, list) and len(a_list) > 0 and isinstance(a_list[0], complex):
                _is_complex = True
        except Exception:
            pass
    result = []
    for k in range(n_full):
        s = complex(0) if _is_complex else 0.0
        for j in range(nv):
            i = k - j
            if 0 <= i < na:
                ai = a_list[i] if isinstance(a_list, list) else a_list
                vj = v_list[j] if isinstance(v_list, list) else v_list
                s += ai * vj
        result.append(s)
    if _is_complex:
        return _ObjectArray(result)
    result = array(result)
    if mode == 'full':
        return result
    elif mode == 'same':
        start = (nv - 1) // 2
        return array([float(result[start + i]) for i in range(na)])
    elif mode == 'valid':
        n_valid = abs(na - nv) + 1
        start = _builtin_min(na, nv) - 1
        return array([float(result[start + i]) for i in range(n_valid)])
    else:
        raise ValueError("mode must be 'full', 'same', or 'valid', got '" + str(mode) + "'")


def correlate(a, v, mode='valid'):
    """Cross-correlation of two 1-dimensional sequences."""
    a = asarray(a).flatten()
    v = asarray(v).flatten()
    # Normalize mode first (same rules as convolve)
    if isinstance(mode, int):
        _mode_map = {0: 'valid', 1: 'same', 2: 'full'}
        if mode not in _mode_map:
            raise ValueError("mode must be 0, 1, or 2")
        mode = _mode_map[mode]
    elif isinstance(mode, str):
        _abbrev = {'v': 'valid', 's': 'same', 'f': 'full'}
        if mode in _abbrev:
            import warnings as _w
            _w.warn("Use of abbreviated mode '{}' is deprecated. Use the full string.".format(mode), DeprecationWarning, stacklevel=2)
            mode = _abbrev[mode]
    elif mode is None:
        raise TypeError("mode must not be None")
    na = a.size
    nv = v.size
    if na == 0 or nv == 0:
        raise ValueError("Array arguments cannot be empty")
    # Check for complex dtypes and do correlation manually
    a_dt = str(a.dtype)
    v_dt = str(v.dtype)
    # Also detect complex data in _ObjectArray
    _has_complex = 'complex' in a_dt or 'complex' in v_dt
    if not _has_complex:
        try:
            _d = a.flatten().tolist() if hasattr(a, 'tolist') else list(a)
            if len(_d) > 0 and isinstance(_d[0], complex):
                _has_complex = True
        except Exception:
            pass
    if _has_complex:
        # Pure Python complex correlation
        a_list = a.flatten().tolist()
        v_list = v.flatten().tolist()
        na_l = len(a_list)
        nv_l = len(v_list)
        # Full correlation length
        full_len = na_l + nv_l - 1

        def _to_cplx(v):
            """Convert value to Python complex (handles (re,im) tuples from Rust)."""
            if isinstance(v, complex):
                return v
            if isinstance(v, (tuple, list)) and len(v) == 2:
                return complex(v[0], v[1])
            return complex(v)

        result = []
        for k in range(full_len):
            s = complex(0, 0)
            for j in range(nv_l):
                ai = k - nv_l + 1 + j
                if 0 <= ai < na_l:
                    s += _to_cplx(a_list[ai]) * _to_cplx(v_list[j]).conjugate()
            result.append(s)
        if mode == 'valid':
            _bmin = __import__("builtins").min
            _bmax = __import__("builtins").max
            start = _bmin(na_l, nv_l) - 1
            end = _bmax(na_l, nv_l)
            result = result[start:end]
        elif mode == 'same':
            start = (full_len - na_l) // 2
            result = result[start:start + na_l]
        return _ObjectArray(result, "complex128")
    # Reverse v for correlation (correlation = convolution with reversed kernel)
    v_rev = array([v[nv - 1 - i] for i in range(nv)])
    return convolve(a, v_rev, mode=mode)
