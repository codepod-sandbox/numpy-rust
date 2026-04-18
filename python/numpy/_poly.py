"""Polynomial utilities, convolution, correlation."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray
from ._helpers import _ObjectArray, _builtin_min, _builtin_max, _flat_arraylike_data
from ._creation import array, asarray

__all__ = [
    'poly',
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


def poly(seq_of_zeros):
    """Return polynomial coefficients given sequence of roots, or characteristic polynomial of matrix."""
    import numpy as _np
    if not isinstance(seq_of_zeros, ndarray):
        seq_of_zeros = array(seq_of_zeros)
    if seq_of_zeros.ndim == 2:
        # Matrix case: characteristic polynomial via eigenvalues
        if seq_of_zeros.shape[0] == 0:
            return array([1.0])
        seq_of_zeros = _np.linalg.eigvals(seq_of_zeros)
    # Build polynomial from roots: prod(x - r for r in roots)
    roots_raw = _flat_arraylike_data(seq_of_zeros.flatten())

    def _to_complex(v):
        if isinstance(v, complex):
            return v
        if isinstance(v, tuple) and len(v) == 2:
            return complex(v[0], v[1])
        try:
            return complex(float(v))
        except (TypeError, ValueError):
            return complex(v)

    roots = [_to_complex(r) for r in roots_raw]
    coeffs = [1.0 + 0j]
    for root in roots:
        new_coeffs = [0.0 + 0j] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            new_coeffs[i] += c
            new_coeffs[i + 1] -= c * root
        coeffs = new_coeffs
    # If all imaginary parts are negligible, return real
    max_imag = max(abs(c.imag) for c in coeffs) if coeffs else 0.0
    max_real = max(abs(c.real) for c in coeffs) if coeffs else 1.0
    if max_imag < 1e-10 * max_real or max_imag == 0:
        return array([c.real for c in coeffs])
    return array(coeffs)


def polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False):
    if not isinstance(x, ndarray):
        x = array(x)
    if not isinstance(y, ndarray):
        y = array(y)
    if cov is not False:
        N = x.size
        if N <= int(deg) + 1:
            raise ValueError(
                "the number of data points must exceed order + 1 for the variance estimate"
            )
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
    # Check for NaN in coefficients - matches NumPy's behavior of raising LinAlgError
    import math as _math_mod
    for c in coeffs:
        try:
            if _math_mod.isnan(float(c)):
                from numpy import LinAlgError
                raise LinAlgError("Array must not contain infs or NaNs")
        except (TypeError, ValueError):
            pass
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


def _to_scalar(v):
    """Convert a value to a proper scalar (handle tuple complex from Rust)."""
    if isinstance(v, tuple) and len(v) == 2:
        return complex(v[0], v[1])
    return v

def polyint(p, m=1, k=0):
    """Return the integral of a polynomial."""
    if isinstance(p, poly1d):
        return p.integ(m, k)
    if isinstance(p, ndarray):
        coeffs = [_to_scalar(p[i]) for i in range(p.size)]
    else:
        coeffs = [_to_scalar(c) for c in p]
    _is_complex = any(isinstance(c, complex) for c in coeffs)
    for _ in range(m):
        n = len(coeffs)
        new_coeffs = []
        for i in range(n):
            divisor = complex(n - i) if _is_complex else float(n - i)
            new_coeffs.append(coeffs[i] / divisor)
        new_coeffs.append(_to_scalar(k))
        coeffs = new_coeffs
    return array(coeffs)


class poly1d:
    """A one-dimensional polynomial class."""
    def __init__(self, c_or_r, r=False, variable=None):
        self._dtype = None
        self._arr = None
        if isinstance(c_or_r, poly1d):
            self._coeffs = list(c_or_r._coeffs)
            self._dtype = c_or_r._dtype
        elif r:
            # c_or_r are roots, convert to coefficients
            self._coeffs = [1.0]
            if isinstance(c_or_r, ndarray):
                self._dtype = c_or_r.dtype
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
                self._dtype = c_or_r.dtype
                self._coeffs = [_to_scalar(c_or_r[i]) for i in range(c_or_r.size)]
            elif isinstance(c_or_r, _ObjectArray):
                self._dtype = c_or_r.dtype
                self._coeffs = list(c_or_r._data)
            else:
                self._coeffs = []
                for c in c_or_r:
                    try:
                        self._coeffs.append(complex(c) if isinstance(c, complex) else float(c))
                    except TypeError:
                        self._coeffs.append(c)
        # Strip leading zeros (but keep at least one coefficient)
        while len(self._coeffs) > 1 and self._coeffs[0] == 0:
            self._coeffs.pop(0)
        self._variable = variable or 'x'

    def __array__(self, dtype=None):
        dt = dtype or self._dtype
        if dt is not None:
            return array(self._coeffs, dtype=dt)
        return array(self._coeffs)

    def _make_arr(self):
        """Build and cache a numpy array from _coeffs (the persistent coeffs array)."""
        if self._dtype is not None:
            self._arr = array(self._coeffs, dtype=self._dtype)
        else:
            self._arr = array(self._coeffs)
        return self._arr

    @property
    def coeffs(self):
        # Return the persistent array; rebuild if _coeffs list changed
        if not hasattr(self, '_arr') or self._arr is None:
            self._make_arr()
        # Check if list changed vs cached array
        if len(self._coeffs) != len(self._arr):
            self._make_arr()
        return self._arr

    @coeffs.setter
    def coeffs(self, value):
        if isinstance(value, ndarray):
            if value.ndim == 0:
                raise AttributeError("Cannot set coeffs to a 0-d array")
            self._dtype = value.dtype
            self._arr = value
            self._coeffs = [_to_scalar(value[i]) for i in range(value.size)]
        elif isinstance(value, (list, tuple)):
            self._coeffs = list(value)
            self._arr = None  # invalidate cache
        else:
            raise AttributeError("Cannot set coeffs to scalar")

    @property
    def c(self):
        return self.coeffs

    @c.setter
    def c(self, value):
        self.coeffs = value

    @property
    def order(self):
        return len(self._coeffs) - 1

    @property
    def roots(self):
        return _poly1d_roots(self._coeffs)

    @property
    def r(self):
        return self.roots

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
        if isinstance(other, ndarray):
            oc = [_to_scalar(other[i]) for i in range(other.size)]
            return poly1d(polymul(oc, self._coeffs))
        return self.__mul__(other)

    def __neg__(self):
        return poly1d([-c for c in self._coeffs])

    def __eq__(self, other):
        if isinstance(other, poly1d):
            if len(self._coeffs) != len(other._coeffs):
                return False
            return all(a == b for a, b in zip(self._coeffs, other._coeffs))
        return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        return id(self)

    def __len__(self):
        return self.order

    def __getitem__(self, idx):
        # poly1d[i] returns coefficient of x^i (reverse indexing)
        if idx < 0 or idx > self.order:
            val = 0.0
        else:
            val = self._coeffs[self.order - idx]
        if self._dtype is not None:
            return array(val, dtype=self._dtype)
        return val

    def __setitem__(self, idx, val):
        # poly1d[i] = val sets coefficient of x^i
        idx = int(idx)
        if idx < 0:
            return  # out of range, no-op for negative index
        # Expand _coeffs if needed
        while self.order < idx:
            self._coeffs.insert(0, 0.0)
        self._coeffs[self.order - idx] = val
        self._arr = None  # invalidate cached array

    def size(self):
        return len(self._coeffs)

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
        if isinstance(k, (list, tuple)):
            k_vals = list(k)
        else:
            k_vals = [k] * m
        while len(k_vals) < m:
            k_vals.append(0)
        for step in range(m):
            n = len(coeffs)
            new_coeffs = []
            for i in range(n):
                new_coeffs.append(coeffs[i] / (n - i))
            try:
                new_coeffs.append(float(k_vals[step]))
            except TypeError:
                new_coeffs.append(k_vals[step])
            coeffs = new_coeffs
        return poly1d(coeffs)

    def __truediv__(self, other):
        q, r = polydiv(self, other)
        return q, r

    def __rtruediv__(self, other):
        q, r = polydiv(other, self)
        return q, r

    def __pow__(self, exp):
        if not isinstance(exp, int) or exp < 0:
            raise ValueError("Polynomial can only be raised to non-negative integer powers")
        result = poly1d([1.0])
        for _ in range(exp):
            result = result * self
        return result

    def __repr__(self):
        coeffs = self.coeffs
        parts = []
        for i in range(len(self._coeffs)):
            v = coeffs[i]
            # Format floats like numpy: 1.0 → '1.', 1.5 → '1.5'
            try:
                fv = float(v)
                if fv == int(fv) and not isinstance(v, complex):
                    parts.append(repr(int(fv)) + '.')
                else:
                    s = repr(fv)
                    parts.append(s)
            except Exception:
                parts.append(repr(v))
        return "poly1d([" + ', '.join(parts) + "])"

    def __str__(self):
        """Format polynomial as multi-line string with exponent notation."""
        var = self._variable

        def _fmt_real(f, sig):
            if f == int(f):
                return str(int(f))
            return f'{f:.{sig}g}'

        def _fmt_num(c):
            """Format a number for display, returns (str, is_negative)."""
            try:
                if isinstance(c, complex):
                    re, im = c.real, c.imag
                    re_s = _fmt_real(re, 4)
                    im_s = _fmt_real(abs(im), 4)
                    if re == 0:
                        return f'{im_s}j', im < 0
                    sign = '+' if im >= 0 else '-'
                    return f'({re_s} {sign} {im_s}j)', False
                fv = float(c)
                return _fmt_real(abs(fv), 4), fv < 0
            except Exception:
                return str(c), False

        # Skip leading zero coefficients for display
        coeffs = list(self._coeffs)
        while len(coeffs) > 1 and coeffs[0] == 0:
            coeffs = coeffs[1:]
        n = len(coeffs) - 1

        if n == 0:
            s, neg = _fmt_num(coeffs[0])
            body = ('-' + s) if neg else s
            return ' ' * len(body) + '\n' + body

        body_parts = []
        exp_positions = []  # (global_position, power)
        pos = 0

        for i, c in enumerate(coeffs):
            power = n - i
            coeff_str, is_neg = _fmt_num(c)
            if not isinstance(c, complex):
                try:
                    abs_c = abs(float(c))
                    coeff_str = _fmt_real(abs_c, 4)
                except Exception:
                    pass

            if power == 0:
                term = coeff_str
            else:
                term = f'{coeff_str} {var}'

            if i == 0:
                prefix = '-' if is_neg else ''
            else:
                prefix = ' - ' if is_neg else ' + '

            full_term = prefix + term

            if power >= 2:
                var_idx = full_term.rindex(var)
                exp_positions.append((pos + var_idx + len(var), power))

            body_parts.append(full_term)
            pos += len(full_term)

        term_line = ''.join(body_parts)

        if not exp_positions:
            return term_line

        # Build header line from global exponent positions
        header_len = max(p + len(str(pw)) for p, pw in exp_positions)
        exp_chars = [' '] * header_len
        for p, pw in exp_positions:
            s = str(pw)
            for j, ch in enumerate(s):
                if p + j < len(exp_chars):
                    exp_chars[p + j] = ch
        exp_line = ''.join(exp_chars).rstrip()
        return exp_line + '\n' + term_line


# Alias so poly1d.roots can call without name clash with the module-level roots()
_poly1d_roots = roots


def polydiv(u, v):
    """Polynomial division: returns (quotient, remainder) as poly1d if inputs are poly1d."""
    _u_is_poly = isinstance(u, poly1d)
    _v_is_poly = isinstance(v, poly1d)
    if _u_is_poly:
        u_list = list(u._coeffs)
    elif isinstance(u, ndarray):
        u_list = [_to_scalar(u[i]) for i in range(u.size)]
    else:
        u_list = [_to_scalar(c) for c in u]
    if _v_is_poly:
        v_list = list(v._coeffs)
    elif isinstance(v, ndarray):
        v_list = [_to_scalar(v[i]) for i in range(v.size)]
    else:
        v_list = [_to_scalar(c) for c in v]
    n = len(u_list)
    nv = len(v_list)
    # Determine output type
    _is_complex = any(isinstance(c, complex) for c in u_list) or any(isinstance(c, complex) for c in v_list)
    _zero = complex(0) if _is_complex else 0.0
    if nv > n:
        q_out = [_zero]
        r_out = u_list if u_list else [_zero]
    else:
        q_out = [_zero] * (n - nv + 1)
        r_list = list(u_list)
        for i in range(n - nv + 1):
            q_out[i] = r_list[i] / v_list[0]
            for j in range(nv):
                r_list[i + j] -= q_out[i] * v_list[j]
        r_out = r_list[n - nv + 1:]
        if not r_out:
            r_out = [_zero]
    if _u_is_poly or _v_is_poly:
        return poly1d(q_out), poly1d(r_out)
    return array(q_out), array(r_out)


def convolve(a, v, mode='full'):
    """Discrete, linear convolution of two one-dimensional sequences."""
    a = asarray(a).flatten()
    v = asarray(v).flatten()
    # Validate empty inputs
    if len(a) == 0:
        raise ValueError("a cannot be empty")
    if len(v) == 0:
        raise ValueError("v cannot be empty")
    # Normalize mode: support integer modes
    if isinstance(mode, int):
        _mode_map = {0: 'valid', 1: 'same', 2: 'full'}
        if mode not in _mode_map:
            raise ValueError("mode must be 0, 1, or 2")
        mode = _mode_map[mode]
    elif isinstance(mode, str):
        if mode not in ('full', 'same', 'valid'):
            raise ValueError("mode must be 'full', 'same', or 'valid', got '" + str(mode) + "'")
    elif mode is None:
        raise TypeError("mode must not be None")
    na = len(a)
    nv = len(v)
    n_full = na + nv - 1
    a_list = _flat_arraylike_data(a)
    v_list = _flat_arraylike_data(v)
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
        if mode not in ('full', 'same', 'valid'):
            raise ValueError("mode must be 'full', 'same', or 'valid', got '" + str(mode) + "'")
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
            _d = _flat_arraylike_data(a.flatten()) if hasattr(a, 'flatten') else list(a)
            if len(_d) > 0 and isinstance(_d[0], complex):
                _has_complex = True
        except Exception:
            pass
    if _has_complex:
        # Pure Python complex correlation
        a_list = _flat_arraylike_data(a.flatten())
        v_list = _flat_arraylike_data(v.flatten())
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
