"""numpy.polynomial - polynomial module."""
import numpy as np
from numpy._helpers import _flat_arraylike_data
from numpy.polynomial import polyutils as pu


def _last_axis_idx(ndim, i):
    """Build index tuple equivalent to [..., i] for given ndim."""
    return tuple([slice(None)] * (ndim - 1) + [i])

def _set_last(arr, i, val):
    """arr[..., i] = val  without using Ellipsis."""
    arr[_last_axis_idx(arr.ndim, i)] = val

def _get_last(arr, i):
    """arr[..., i] without using Ellipsis."""
    return arr[_last_axis_idx(arr.ndim, i)]


def _flat_list(values):
    return _flat_arraylike_data(np.asarray(values).flatten())


def _coef_list(values):
    return list(values) if isinstance(values, list) else _flat_list(values)


def _flat_int_list(values):
    return [int(v) for v in _flat_list(values)]

# ---------------------------------------------------------------------------
# Default print style
# ---------------------------------------------------------------------------
_default_printstyle = 'ascii'

def set_default_printstyle(style):
    global _default_printstyle
    _default_printstyle = style

# ---------------------------------------------------------------------------
# Helper: generic trim
# ---------------------------------------------------------------------------

def _trimcoef(c, tol=0):
    if tol < 0:
        raise ValueError("tol must be non-negative")
    c = _flat_list(c)
    while len(c) > 1 and abs(c[-1]) <= tol:
        c.pop()
    if len(c) == 1 and abs(c[0]) <= tol:
        c = [0.0]
    return np.array(c)

# ---------------------------------------------------------------------------
# ABCPolyBase  -- shared base for all polynomial series classes
# ---------------------------------------------------------------------------

class ABCPolyBase:
    # Subclasses MUST set these:
    # _mul_func, _add_func, _sub_func, _val_func, _int_func, _der_func,
    # _fit_func, _fromroots_func, _roots_func, _vander_func, _div_func,
    # _pow_func
    # Also: domain, window, basis_name, nickname

    domain = np.array([-1., 1.])
    window = np.array([-1., 1.])
    basis_name = None
    nickname = None

    # ---- construction ----

    def __init__(self, coef, domain=None, window=None, symbol='x'):
        if symbol is None or not isinstance(symbol, str):
            raise TypeError("symbol must be a string")
        if symbol == '' or symbol[0].isdigit():
            raise ValueError("symbol must be a non-empty string not starting with a digit")
        self._coef = np.asarray(coef).flatten() * 1.0  # ensure float copy
        # try to preserve object dtype
        _cc = np.asarray(coef)
        if hasattr(_cc, 'dtype') and _cc.dtype == np.dtype('O'):
            self._coef = np.asarray(list(_cc.flatten()), dtype=object)
        elif hasattr(_cc, 'dtype') and _cc.dtype == np.float32:
            self._coef = np.asarray(list(_cc.flatten()), dtype=np.float32)
        else:
            self._coef = np.asarray(_flat_list(coef), dtype='float64')
        self._domain = np.array(domain if domain is not None else self.__class__.domain, dtype='float64')
        self._window = np.array(window if window is not None else self.__class__.window, dtype='float64')
        self._symbol = symbol

    @property
    def coef(self):
        return self._coef

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = np.asarray(value, dtype='float64')

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, value):
        self._window = np.asarray(value, dtype='float64')

    @property
    def symbol(self):
        return self._symbol

    def degree(self):
        return len(self._coef) - 1

    def __len__(self):
        return len(self._coef)

    def __hash__(self):
        return id(self)

    # ---- mapping helpers ----

    def mapparms(self):
        """Return (offset, scale) for mapping domain -> window."""
        return pu.mapparms(self._domain, self._window)

    def _map_x(self, x):
        """Map x from domain to window."""
        off, scl = self.mapparms()
        return np.asarray(x) * scl + off

    # ---- evaluation ----

    def __call__(self, x):
        if isinstance(x, ABCPolyBase):
            # Polynomial composition: p(q) returns a new polynomial
            # Map x through self's domain/window mapping first
            x_mapped = pu.mapdomain(x, self._domain, self._window)
            # Evaluate self at x_mapped using Horner-like composition
            c = _flat_arraylike_data(self._coef.flatten()) if hasattr(self._coef, 'flatten') else list(self._coef)
            if len(c) == 0:
                return x.__class__([0], domain=x._domain, window=x._window, symbol=x._symbol)
            result = x.__class__([c[-1]], domain=x._domain, window=x._window, symbol=x._symbol)
            for i in range(len(c) - 2, -1, -1):
                result = result * x_mapped + c[i]
            return result
        x_mapped = self._map_x(x)
        return self.__class__._val_func(x_mapped, self._coef)

    # ---- arithmetic helpers ----

    def _check_compatible(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError("Incompatible types: {} and {}".format(
                type(self).__name__, type(other).__name__))
        if not np.array_equal(self._domain, other._domain):
            raise TypeError("Domains differ")
        if not np.array_equal(self._window, other._window):
            raise TypeError("Windows differ")
        if self._symbol != other._symbol:
            raise ValueError("Symbols differ")

    def _coerce_other(self, other):
        """Try to coerce other to coefficient array."""
        if isinstance(other, ABCPolyBase):
            if not isinstance(other, self.__class__):
                raise TypeError("Incompatible types")
            self._check_compatible(other)
            return other._coef
        if isinstance(other, (list, tuple, np.ndarray)):
            return np.asarray(other, dtype='float64').flatten()
        if isinstance(other, (int, float, complex)):
            return np.array([float(other)])
        return NotImplemented

    # ---- arithmetic ----

    def __add__(self, other):
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        rc = self.__class__._add_func(self._coef, c)
        return self.__class__(rc, domain=self._domain, window=self._window, symbol=self._symbol)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        rc = self.__class__._sub_func(self._coef, c)
        return self.__class__(rc, domain=self._domain, window=self._window, symbol=self._symbol)

    def __rsub__(self, other):
        p = self.__sub__(other)
        if p is NotImplemented:
            return NotImplemented
        return -p

    def __mul__(self, other):
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        rc = self.__class__._mul_func(self._coef, c)
        return self.__class__(rc, domain=self._domain, window=self._window, symbol=self._symbol)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__class__(-self._coef, domain=self._domain, window=self._window, symbol=self._symbol)

    def __pos__(self):
        return self.__class__(self._coef.copy(), domain=self._domain, window=self._window, symbol=self._symbol)

    def __pow__(self, n):
        n_int = int(n)
        if n_int != n or n_int < 0:
            raise ValueError("Power must be a non-negative integer")
        rc = self.__class__._pow_func(self._coef, n_int)
        return self.__class__(rc, domain=self._domain, window=self._window, symbol=self._symbol)

    def __floordiv__(self, other):
        if isinstance(other, (int, float, complex)):
            # scalar division
            rc = self._coef / float(other)
            return self.__class__(rc, domain=self._domain, window=self._window, symbol=self._symbol)
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        quo, rem = self.__class__._div_func(self._coef, c)
        return self.__class__(quo, domain=self._domain, window=self._window, symbol=self._symbol)

    def __rfloordiv__(self, other):
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        quo, rem = self.__class__._div_func(c, self._coef)
        return self.__class__(quo, domain=self._domain, window=self._window, symbol=self._symbol)

    def __mod__(self, other):
        if isinstance(other, (int, float, complex)):
            return self.__class__([0.0], domain=self._domain, window=self._window, symbol=self._symbol)
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        quo, rem = self.__class__._div_func(self._coef, c)
        return self.__class__(rem, domain=self._domain, window=self._window, symbol=self._symbol)

    def __rmod__(self, other):
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        quo, rem = self.__class__._div_func(c, self._coef)
        return self.__class__(rem, domain=self._domain, window=self._window, symbol=self._symbol)

    def __divmod__(self, other):
        if isinstance(other, (int, float, complex)):
            quo = self.__class__(self._coef / float(other), domain=self._domain,
                                  window=self._window, symbol=self._symbol)
            rem = self.__class__([0.0], domain=self._domain, window=self._window, symbol=self._symbol)
            return quo, rem
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        q, r = self.__class__._div_func(self._coef, c)
        return (self.__class__(q, domain=self._domain, window=self._window, symbol=self._symbol),
                self.__class__(r, domain=self._domain, window=self._window, symbol=self._symbol))

    def __rdivmod__(self, other):
        c = self._coerce_other(other)
        if c is NotImplemented:
            return NotImplemented
        q, r = self.__class__._div_func(c, self._coef)
        return (self.__class__(q, domain=self._domain, window=self._window, symbol=self._symbol),
                self.__class__(r, domain=self._domain, window=self._window, symbol=self._symbol))

    def __truediv__(self, other):
        from numbers import Number
        if isinstance(other, bool) or not isinstance(other, Number):
            raise TypeError("unsupported operand type for /")
        return self.__class__(self._coef / complex(other), domain=self._domain,
                              window=self._window, symbol=self._symbol)

    def __rtruediv__(self, other):
        raise TypeError("unsupported operand type for /")

    # ---- comparison ----

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (self._symbol == other._symbol and
                np.array_equal(self._domain, other._domain) and
                np.array_equal(self._window, other._window) and
                np.array_equal(self._coef, other._coef))

    def __ne__(self, other):
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return eq
        return not eq

    # ---- ufunc override ----

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        raise TypeError("unsupported operand type for ufunc")

    # ---- copy / pickle ----

    def copy(self):
        return self.__class__(self._coef.copy(), domain=self._domain.copy(),
                              window=self._window.copy(), symbol=self._symbol)

    def __deepcopy__(self, memo):
        return self.copy()

    def __getstate__(self):
        return {'coef': self._coef, 'domain': self._domain,
                'window': self._window, 'symbol': self._symbol}

    def __setstate__(self, state):
        self._coef = state['coef']
        self._domain = state['domain']
        self._window = state['window']
        self._symbol = state.get('symbol', 'x')

    def __reduce__(self):
        return (self.__class__, (self._coef, self._domain, self._window, self._symbol))

    # ---- derivative / integral ----

    def deriv(self, m=1):
        c = self.__class__._der_func(self._coef, m)
        return self.__class__(c, domain=self._domain, window=self._window, symbol=self._symbol)

    def integ(self, m=1, k=[], lbnd=None):
        if lbnd is None:
            lbnd = 0
        c = self.__class__._int_func(self._coef, m, k=k, lbnd=lbnd)
        return self.__class__(c, domain=self._domain, window=self._window, symbol=self._symbol)

    # ---- trim / truncate / cutdeg ----

    def trim(self, tol=0):
        c = _trimcoef(self._coef, tol)
        return self.__class__(c, domain=self._domain, window=self._window, symbol=self._symbol)

    def truncate(self, size):
        if size != int(size) or size < 1:
            raise ValueError("size must be a positive integer")
        size = int(size)
        if size >= len(self._coef):
            return self.copy()
        c = self._coef[:size]
        return self.__class__(c, domain=self._domain, window=self._window, symbol=self._symbol)

    def cutdeg(self, deg):
        if deg != int(deg) or deg < 0:
            raise ValueError("deg must be a non-negative integer")
        deg = int(deg)
        return self.truncate(deg + 1)

    # ---- roots ----

    def roots(self):
        r = self.__class__._roots_func(self._coef)
        # roots are in the "window" coordinate system, need to map to domain
        off, scl = self.mapparms()
        if scl != 0:
            return (np.asarray(r) - off) / scl
        return np.asarray(r)

    # ---- linspace ----

    def linspace(self, n=100, domain=None):
        if domain is None:
            domain = self._domain
        x = np.linspace(domain[0], domain[1], n)
        y = self(x)
        return x, y

    # ---- convert / cast ----

    def convert(self, domain=None, kind=None, window=None):
        if kind is None:
            kind = self.__class__
        if domain is None:
            domain = kind.domain
        if window is None:
            window = kind.window
        deg = self.degree()
        n = max(deg + 1, 10)
        x = np.linspace(domain[0], domain[1], n * 3)
        y = self(x)
        result = kind.fit(x, y, deg, domain=domain, window=window)
        result._symbol = self._symbol
        return result

    @classmethod
    def cast(cls, series, domain=None, window=None):
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        return series.convert(domain=domain, kind=cls, window=window)

    # ---- class construction methods ----

    @classmethod
    def fit(cls, x, y, deg, domain=None, window=None, w=None, symbol='x'):
        x = np.asarray(x, dtype='float64')
        y = np.asarray(y, dtype='float64')

        if domain is None:
            domain = pu.getdomain(x)
        elif isinstance(domain, (list, tuple)) and len(domain) == 0:
            domain = cls.domain
        else:
            domain = np.asarray(domain, dtype='float64')

        if window is None:
            window = cls.window

        # Map x to window
        off, scl = pu.mapparms(domain, window)
        xw = x * scl + off

        c = cls._fit_func(xw, y, deg, w=w)
        return cls(c, domain=domain, window=window, symbol=symbol)

    @classmethod
    def fromroots(cls, roots, domain=None, window=None, symbol='x'):
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        # Map roots from domain to window
        off, scl = pu.mapparms(domain, window)
        r = np.asarray(roots, dtype='float64')
        rw = r * scl + off
        c = cls._fromroots_func(rw)
        return cls(c, domain=domain, window=window, symbol=symbol)

    @classmethod
    def identity(cls, domain=None, window=None, symbol='x'):
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        n = 20
        x = np.linspace(domain[0], domain[1], n)
        return cls.fit(x, x, 1, domain=domain, window=window, symbol=symbol)

    @classmethod
    def basis(cls, deg, domain=None, window=None, symbol='x'):
        if domain is None:
            domain = cls.domain
        if window is None:
            window = cls.window
        c = [0] * deg + [1]
        return cls(c, domain=domain, window=window, symbol=symbol)

    # ---- repr / str ----

    def __repr__(self):
        return "{}({}, domain={}, window={}, symbol='{}')".format(
            self.__class__.__name__,
            list(self._coef.tolist()) if hasattr(self._coef, 'tolist') else list(self._coef),
            list(self._domain.tolist()) if hasattr(self._domain, 'tolist') else list(self._domain),
            list(self._window.tolist()) if hasattr(self._window, 'tolist') else list(self._window),
            self._symbol)

    def _str_term(self, i, c_str):
        """Format a single term for __str__."""
        bn = self.basis_name
        sym = self._symbol
        if bn is None:
            if i == 0:
                return c_str
            elif i == 1:
                return c_str + '\u00b7' + sym
            else:
                return c_str + '\u00b7' + sym + _superscript(i)
        else:
            if i == 0:
                return c_str
            else:
                return c_str + '\u00b7' + bn + _subscript(i) + '(' + sym + ')'

    def __str__(self):
        coefs = self._coef
        parts = []
        for i in range(len(coefs)):
            c = coefs[i] if not hasattr(coefs[i], 'item') else coefs[i].item()
            c_val = float(c)
            if i == 0:
                c_str = str(c_val)
                parts.append(self._str_term(i, c_str))
            else:
                if c_val < 0:
                    c_str = str(abs(c_val))
                    parts.append('- ' + self._str_term(i, c_str))
                else:
                    c_str = str(c_val)
                    parts.append('+ ' + self._str_term(i, c_str))

        result = ' '.join(parts)
        # line wrapping at 79 chars
        if len(result) > 79:
            result = _wrap_line(result, 79)
        return result


def _superscript(n):
    """Convert integer to Unicode superscript."""
    sup = str.maketrans('0123456789', '\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079')
    return str(n).translate(sup)


def _subscript(n):
    """Convert integer to Unicode subscript."""
    sub = str.maketrans('0123456789', '\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089')
    return str(n).translate(sub)


def _wrap_line(s, width):
    """Wrap at ' + ' or ' - ' boundaries near width."""
    parts = []
    current = ''
    tokens = s.split(' ')
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if current and len(current) + 1 + len(token) > width and (token == '+' or token == '-'):
            parts.append(current)
            current = token
        else:
            if current:
                current += ' ' + token
            else:
                current = token
        i += 1
    if current:
        parts.append(current)
    return '\n'.join(parts)


# ===========================================================================
#  POLYNOMIAL (power series) module-level functions and class
# ===========================================================================

# --- Constants ---
_polydomain = np.array([-1., 1.])
_polyzero = np.array([0.])
_polyone = np.array([1.])
_polyx = np.array([0., 1.])

# --- Functions ---

def _polytrim(c, tol=0):
    return _trimcoef(c, tol)

def _polyline(off, scl):
    if scl == 0:
        return np.array([off])
    return np.array([off, scl])

def _polyadd(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n:
        c1.append(0.0)
    while len(c2) < n:
        c2.append(0.0)
    return np.array([c1[i] + c2[i] for i in range(n)])

def _polysub(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n:
        c1.append(0.0)
    while len(c2) < n:
        c2.append(0.0)
    return np.array([c1[i] - c2[i] for i in range(n)])

def _polymulx(c):
    c = _flat_list(c)
    result = [0.0] + c
    # trim trailing zeros
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return np.array(result)

def _polymul(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = len(c1) + len(c2) - 1
    result = [0.0] * n
    for i in range(len(c1)):
        for j in range(len(c2)):
            result[i + j] += c1[i] * c2[j]
    return np.array(result)

def _polydiv(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    # trim trailing zeros
    while len(c2) > 1 and c2[-1] == 0:
        c2.pop()
    if len(c2) == 0 or c2[-1] == 0:
        raise ZeroDivisionError("polynomial division by zero")
    if len(c1) < len(c2):
        return np.array([0.0]), np.array(c1)
    rem = list(c1)
    quo = [0.0] * (len(c1) - len(c2) + 1)
    for i in range(len(quo) - 1, -1, -1):
        q = rem[i + len(c2) - 1] / c2[-1]
        quo[i] = q
        for j in range(len(c2)):
            rem[i + j] -= q * c2[j]
    # trim remainder
    while len(rem) > 1 and abs(rem[-1]) < 1e-15:
        rem.pop()
    while len(quo) > 1 and abs(quo[-1]) < 1e-15:
        quo.pop()
    return np.array(quo), np.array(rem)

def _polypow(c, n, maxpower=None):
    c = np.asarray(c)
    if maxpower is not None and n > maxpower:
        raise ValueError("Power exceeds maxpower")
    result = np.array([1.0])
    for _ in range(n):
        result = _polymul(result, c)
    return result

def _validate_fit_args(x, y, w=None):
    """Validate arguments for polynomial fitting functions."""
    if x.ndim != 1:
        raise TypeError("expected 1D vector for x")
    if y.ndim > 2:
        raise TypeError("expected 1D or 2D array for y")
    if x.size == 0:
        raise TypeError("expected non-empty vector for x")
    if len(x) != (y.shape[0] if y.ndim > 0 else 1):
        raise TypeError("expected x and y to have the same length")
    if w is not None:
        w_arr = np.asarray(w, dtype='float64')
        if w_arr.ndim != 1:
            raise TypeError("expected 1D vector for w")
        if len(w_arr) != len(x):
            raise TypeError("expected w and x to have the same length")

def _polyval(x, c):
    x = np.asarray(x)
    c_list = _flat_list(c)
    if len(c_list) == 0:
        return np.zeros_like(x)
    # Horner's method (ascending order) - works for any shape including 0-d
    result = np.full(x.shape, c_list[-1]) + np.zeros_like(x)
    for i in range(len(c_list) - 2, -1, -1):
        result = result * x + c_list[i]
    return result

def _polyval2d(x, y, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y have incompatible shapes")
    result = np.zeros(x.shape)
    for i in range(c.shape[0]):
        result = result + _polyval(y, c[i]) * x**i
    return result

def _polyval3d(x, y, z, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z have incompatible shapes")
    result = np.zeros(x.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            result = result + _polyval(z, c[i, j]) * x**i * y**j
    return result

def _polygrid2d(x, y, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    shape = x.shape + y.shape
    result = np.zeros(shape)
    for i in range(c.shape[0]):
        yvals = _polyval(y, c[i])
        xi = x**i
        result = result + np.multiply.outer(xi, yvals)
    return result

def _polygrid3d(x, y, z, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    shape = x.shape + y.shape + z.shape
    result = np.zeros(shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            zvals = _polyval(z, c[i, j])
            xi = x**i
            yj = y**j
            result = result + _outer3(xi, yj, zvals)
    return result

def _outer3(a, b, c):
    """Triple outer product."""
    return np.multiply.outer(np.multiply.outer(a, b), c)

def _polyder(c, m=1, scl=1, axis=0):
    if isinstance(m, float):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0:
        raise ValueError("m must be non-negative")
    c = np.asarray(c)
    if c.ndim > 1:
        return _polyder_nd(c, m, scl, axis)
    c = _flat_arraylike_data(c.flatten())
    for _ in range(m):
        c = [c[i] * i * scl for i in range(1, len(c))]
        if not c:
            c = [0.0]
    return np.array(c)

def _polyder_nd(c, m, scl, axis):
    c = np.asarray(c)
    c = np.moveaxis(c, axis, 0)
    for _ in range(m):
        n = c.shape[0]
        if n <= 1:
            c = np.zeros((1,) + c.shape[1:])
            break
        dc = np.zeros((n - 1,) + c.shape[1:])
        for i in range(1, n):
            dc[i - 1] = c[i] * i * scl
        c = dc
    c = np.moveaxis(c, 0, axis)
    return c

def _polyint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    if isinstance(m, float):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0:
        raise ValueError("m must be non-negative")
    if isinstance(axis, float):
        raise TypeError("axis must be an integer")
    c = np.asarray(c)
    if isinstance(k, (int, float)):
        k = [k]
    if isinstance(lbnd, (list, tuple, np.ndarray)):
        raise ValueError("lbnd must be a scalar")
    if isinstance(scl, (list, tuple, np.ndarray)):
        raise ValueError("scl must be a scalar")
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if c.ndim > 1:
        return _polyint_nd(c, m, k, lbnd, scl, axis)
    c = _flat_arraylike_data(c.flatten())
    k_list = list(k) if k else []
    while len(k_list) < m:
        k_list.append(0)
    for step in range(m):
        n = len(c)
        ic = [0.0] * (n + 1)
        for i in range(n):
            ic[i + 1] = c[i] * scl / (i + 1)
        # Apply integration constant: adjust so that polyval(lbnd, ic) = k_list[step]
        val_at_lbnd = 0.0
        for i in range(1, len(ic)):
            val_at_lbnd += ic[i] * (lbnd ** i)
        ic[0] = k_list[step] - val_at_lbnd
        c = ic
    return np.array(c)

def _polyint_nd(c, m, k, lbnd, scl, axis):
    c = np.asarray(c)
    c = np.moveaxis(c, axis, 0)
    k_list = list(k) if k else []
    while len(k_list) < m:
        k_list.append(0)
    for step in range(m):
        n = c.shape[0]
        ic = np.zeros((n + 1,) + c.shape[1:])
        for i in range(n):
            ic[i + 1] = c[i] * scl / (i + 1)
        # Apply integration constant
        val_at_lbnd = np.zeros(c.shape[1:])
        for i in range(1, n + 1):
            val_at_lbnd = val_at_lbnd + ic[i] * (lbnd ** i)
        ic[0] = k_list[step] - val_at_lbnd
        c = ic
    c = np.moveaxis(c, 0, axis)
    return c

def _polyroots(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    if len(c) <= 1:
        return np.array([])
    if len(c) == 2:
        return np.array([-c[0] / c[1]])
    # Build companion matrix
    comp = _polycompanion(c)
    roots = np.linalg.eigvals(comp)
    roots = np.sort(roots.real) if all(abs(r.imag) < 1e-10 for r in _flat_arraylike_data(roots)) else np.sort(roots)
    return roots

def _polyvander(x, deg):
    x = np.asarray(x)
    deg = int(deg)
    if deg < 0:
        raise ValueError("deg must be non-negative")
    shape = x.shape + (deg + 1,)
    v = np.zeros(shape)
    _set_last(v, 0, np.ones(x.shape) if len(x.shape) > 0 else 1.0)
    if deg >= 1:
        _set_last(v, 1, x)
    for i in range(2, deg + 1):
        _set_last(v, i, x * _get_last(v, i - 1))
    return v

def _polyvander2d(x, y, deg):
    x = np.asarray(x)
    y = np.asarray(y)
    degx, degy = deg
    vx = _polyvander(x, degx)
    vy = _polyvander(y, degy)
    shape = x.shape + ((degx + 1) * (degy + 1),)
    v = np.zeros(shape)
    k = 0
    for i in range(degx + 1):
        for j in range(degy + 1):
            _set_last(v, k, _get_last(vx, i) * _get_last(vy, j))
            k += 1
    return v

def _polyvander3d(x, y, z, deg):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    degx, degy, degz = deg
    vx = _polyvander(x, degx)
    vy = _polyvander(y, degy)
    vz = _polyvander(z, degz)
    shape = x.shape + ((degx + 1) * (degy + 1) * (degz + 1),)
    v = np.zeros(shape)
    k = 0
    for i in range(degx + 1):
        for j in range(degy + 1):
            for l in range(degz + 1):
                _set_last(v, k, _get_last(vx, i) * _get_last(vy, j) * _get_last(vz, l))
                k += 1
    return v

def _polycompanion(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    if len(c) < 2:
        raise ValueError("Series must have maximum degree >= 1")
    n = len(c) - 1
    comp = np.zeros((n, n))
    for i in range(n - 1):
        comp[i + 1, i] = 1.0
    for i in range(n):
        comp[i, n - 1] = -c[i] / c[n]
    return comp

def _polyfromroots(roots):
    roots = _flat_list(roots)
    if len(roots) == 0:
        return np.array([1.0])
    c = [1.0]
    for r in roots:
        new_c = [0.0] * (len(c) + 1)
        for i in range(len(c)):
            new_c[i] -= r * c[i]
            new_c[i + 1] += c[i]
        c = new_c
    return np.array(c)

def _polyvalfromroots(x, r, tensor=True):
    x = np.asarray(x)
    r = np.asarray(r)
    if r.ndim == 1:
        if tensor is False and r.ndim == 1:
            raise ValueError("x and r must have at least 2 dimensions for tensor=False")
        result = np.ones(x.shape)
        for ri in _flat_arraylike_data(r.flatten()):
            result = result * (x - ri)
        return result
    # r has multiple dimensions
    if tensor is False:
        if x.shape != r.shape[1:]:
            result = np.empty(r.shape[1:])
            for ii in range(result.size):
                xi = x.flatten()[ii] if x.size > ii else x
                ri = r[:, ii] if r.ndim == 2 else r[:, ii]
                v = 1.0
                for rr in _flat_arraylike_data(ri.flatten()):
                    v *= (float(xi) - rr)
                result.flat[ii] = v
            return result
        result = np.ones(r.shape[1:])
        for i in range(r.shape[0]):
            result = result * (x - r[i])
        return result
    else:
        shape = r.shape[1:] + x.shape
        result = np.ones(shape)
        for ii in range(r.shape[1]):
            for jj in range(x.shape[0] if x.ndim >= 1 else 1):
                xslice = x[jj] if x.ndim >= 1 else x
                rslice = r[:, ii] if r.ndim == 2 else r[:, ii]
                v = np.ones(xslice.shape if hasattr(xslice, 'shape') and len(xslice.shape) > 0 else ())
                for rr in _flat_arraylike_data(rslice.flatten()):
                    v = v * (xslice - rr)
                if x.ndim == 1:
                    result[ii] = v
                elif x.ndim == 2:
                    result[ii, jj] = v
        return result

def _polyfit(x, y, deg, w=None):
    """Fit polynomial (power series) of given degree."""
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    _validate_fit_args(x, y, w)
    if isinstance(deg, (list, tuple, np.ndarray)):
        deg_list = _flat_int_list(deg)
        if any(d < 0 for d in deg_list):
            raise ValueError("deg must be non-negative")
        if len(deg_list) == 0:
            raise TypeError("deg must be non-empty")
        max_deg = max(deg_list)
        V_full = _polyvander(x, max_deg)
        V = np.zeros((len(x), max_deg + 1))
        for d in sorted(set(deg_list)):
            V[:, d] = V_full[:, d]
        if w is not None:
            w = np.asarray(w, dtype='float64')
            V = V * w[:, None]
            if y.ndim == 1:
                y = y * w
            else:
                y = y * w[:, None]
        c = np.linalg.lstsq(V, y)[0]
        return c
    else:
        deg = int(deg)
        if deg < 0:
            raise ValueError("deg must be non-negative")
        V = _polyvander(x, deg)
        if w is not None:
            w = np.asarray(w, dtype='float64')
            V = V * w[:, None]
            if y.ndim == 1:
                y = y * w
            else:
                y = y * w[:, None]
        c = np.linalg.lstsq(V, y)[0]
        return c


class Polynomial(ABCPolyBase):
    domain = np.array([-1., 1.])
    window = np.array([-1., 1.])
    basis_name = None
    nickname = 'poly'

    _add_func = staticmethod(_polyadd)
    _sub_func = staticmethod(_polysub)
    _mul_func = staticmethod(_polymul)
    _div_func = staticmethod(_polydiv)
    _pow_func = staticmethod(_polypow)
    _val_func = staticmethod(_polyval)
    _int_func = staticmethod(_polyint)
    _der_func = staticmethod(_polyder)
    _fit_func = staticmethod(_polyfit)
    _fromroots_func = staticmethod(_polyfromroots)
    _roots_func = staticmethod(_polyroots)
    _vander_func = staticmethod(_polyvander)


# Expose the polynomial module-level functions via a class namespace
class polynomial:
    polydomain = _polydomain
    polyzero = _polyzero
    polyone = _polyone
    polyx = _polyx
    Polynomial = Polynomial
    polytrim = staticmethod(_polytrim)
    polyline = staticmethod(_polyline)
    polyadd = staticmethod(_polyadd)
    polysub = staticmethod(_polysub)
    polymulx = staticmethod(_polymulx)
    polymul = staticmethod(_polymul)
    polydiv = staticmethod(_polydiv)
    polypow = staticmethod(_polypow)
    polyval = staticmethod(_polyval)
    polyval2d = staticmethod(_polyval2d)
    polyval3d = staticmethod(_polyval3d)
    polygrid2d = staticmethod(_polygrid2d)
    polygrid3d = staticmethod(_polygrid3d)
    polyder = staticmethod(_polyder)
    polyint = staticmethod(_polyint)
    polyroots = staticmethod(_polyroots)
    polyvander = staticmethod(_polyvander)
    polyvander2d = staticmethod(_polyvander2d)
    polyvander3d = staticmethod(_polyvander3d)
    polycompanion = staticmethod(_polycompanion)
    polyfromroots = staticmethod(_polyfromroots)
    polyvalfromroots = staticmethod(_polyvalfromroots)
    polyfit = staticmethod(_polyfit)


# ===========================================================================
#  CHEBYSHEV
# ===========================================================================

def _chebtrim(c, tol=0):
    return _trimcoef(c, tol)

def _chebline(off, scl):
    return np.array([off, scl])

def _chebadd(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] + c2[i] for i in range(n)])

def _chebsub(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] - c2[i] for i in range(n)])

def _chebmulx(c):
    c = _flat_list(c)
    if len(c) == 1 and c[0] == 0:
        return np.array([0.0])
    n = len(c)
    result = [0.0] * (n + 1)
    for i in range(n):
        if i == 0:
            result[1] += c[0]
        else:
            result[i - 1] += 0.5 * c[i]
            result[i + 1] += 0.5 * c[i]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return np.array(result)

def _chebmul(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n1 = len(c1)
    n2 = len(c2)
    out_len = n1 + n2 - 1
    result = [0.0] * out_len
    for i in range(n1):
        for j in range(n2):
            v = c1[i] * c2[j]
            idx_sum = i + j
            idx_diff = abs(i - j)
            if i == 0 and j == 0:
                result[0] += v
            elif i == 0 or j == 0:
                result[idx_sum] += v
            else:
                result[idx_sum] += v * 0.5
                result[idx_diff] += v * 0.5
    return np.array(result)

def _chebdiv(c1, c2):
    """Divide Chebyshev series c1 by c2."""
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    while len(c2) > 1 and c2[-1] == 0:
        c2.pop()
    if len(c2) == 0 or c2[-1] == 0:
        raise ZeroDivisionError("division by zero")
    if len(c1) < len(c2):
        return np.array([0.0]), np.array(c1)
    # Use iterative subtraction
    rem = list(c1)
    deg1 = len(c1) - 1
    deg2 = len(c2) - 1
    quo_len = deg1 - deg2 + 1
    quo = [0.0] * quo_len
    for i in range(quo_len - 1, -1, -1):
        if len(rem) <= deg2 + i:
            continue
        q = rem[deg2 + i] / c2[-1]
        quo[i] = q
        sub = _chebmul([0.0] * i + [q], c2)
        sub = _flat_list(sub)
        for j in range(len(sub)):
            if j < len(rem):
                rem[j] -= sub[j]
    while len(rem) > 1 and abs(rem[-1]) < 1e-15:
        rem.pop()
    while len(quo) > 1 and abs(quo[-1]) < 1e-15:
        quo.pop()
    return np.array(quo), np.array(rem)

def _chebpow(c, n, maxpower=None):
    if maxpower is not None and n > maxpower:
        raise ValueError("Power exceeds maxpower")
    result = np.array([1.0])
    for _ in range(n):
        result = _chebmul(result, c)
    return result

def _chebval(x, c):
    """Clenshaw recurrence for Chebyshev series."""
    x = np.asarray(x)
    c_list = _flat_list(c)
    if len(c_list) == 0:
        return np.zeros_like(x)
    if len(c_list) == 1:
        return np.full(x.shape, c_list[0]) + np.zeros_like(x)
    nd = len(c_list)
    c0 = np.asarray(c_list[-2]) + np.zeros_like(x)
    c1 = np.asarray(c_list[-1]) + np.zeros_like(x)
    x2 = x * 2.0
    for i in range(3, nd + 1):
        tmp = c0
        c0 = float(c_list[-i]) - c1
        c1 = tmp + c1 * x2
    return c0 + c1 * x

def _chebval2d(x, y, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x and y have incompatible shapes")
    result = np.zeros(x.shape)
    for i in range(c.shape[0]):
        result = result + _chebval(y, c[i]) * np.asarray(_chebval(x, [0.0]*i + [1.0]))
    return result

def _chebval3d(x, y, z, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z have incompatible shapes")
    result = np.zeros(x.shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ti = _chebval(x, [0.0]*i + [1.0])
            tj = _chebval(y, [0.0]*j + [1.0])
            result = result + _chebval(z, c[i, j]) * np.asarray(ti) * np.asarray(tj)
    return result

def _chebgrid2d(x, y, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    shape = x.shape + y.shape
    result = np.zeros(shape)
    for i in range(c.shape[0]):
        yvals = _chebval(y, c[i])
        ti = _chebval(x, [0.0]*i + [1.0])
        result = result + np.multiply.outer(np.asarray(ti), np.asarray(yvals))
    return result

def _chebgrid3d(x, y, z, c):
    c = np.asarray(c)
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    shape = x.shape + y.shape + z.shape
    result = np.zeros(shape)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            ti = _chebval(x, [0.0]*i + [1.0])
            tj = _chebval(y, [0.0]*j + [1.0])
            zvals = _chebval(z, c[i, j])
            result = result + _outer3(np.asarray(ti), np.asarray(tj), np.asarray(zvals))
    return result

def _chebder(c, m=1, scl=1, axis=0):
    if isinstance(m, float):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0:
        raise ValueError("m must be non-negative")
    c = np.asarray(c)
    if c.ndim > 1:
        return _chebder_nd(c, m, scl, axis)
    c = _flat_arraylike_data(c.flatten())
    for _ in range(m):
        n = len(c)
        if n <= 1:
            c = [0.0]
            continue
        dc = [0.0] * (n - 1)
        dc[n - 2] = 2.0 * (n - 1) * c[n - 1] * scl
        if n - 3 >= 0:
            dc[n - 3] = 2.0 * (n - 2) * c[n - 2] * scl
        for k in range(n - 4, -1, -1):
            dc[k] = dc[k + 2] + 2.0 * (k + 1) * c[k + 1] * scl
        dc[0] *= 0.5
        c = dc
    return np.array(c)

def _chebder_nd(c, m, scl, axis):
    c = np.moveaxis(c, axis, 0)
    for _ in range(m):
        n = c.shape[0]
        if n <= 1:
            c = np.zeros((1,) + c.shape[1:])
            break
        dc = np.zeros((n - 1,) + c.shape[1:])
        dc[n - 2] = 2.0 * (n - 1) * c[n - 1] * scl
        if n - 3 >= 0:
            dc[n - 3] = 2.0 * (n - 2) * c[n - 2] * scl
        for k in range(n - 4, -1, -1):
            dc[k] = dc[k + 2] + 2.0 * (k + 1) * c[k + 1] * scl
        dc[0] = dc[0] * 0.5
        c = dc
    c = np.moveaxis(c, 0, axis)
    return c

def _chebval_scalar(x, c):
    """Value Chebyshev series at scalar x."""
    c_list = _coef_list(c)
    if len(c_list) == 0:
        return 0.0
    if len(c_list) == 1:
        return float(c_list[0])
    nd = len(c_list)
    c0 = float(c_list[-2])
    c1 = float(c_list[-1])
    x2 = x * 2.0
    for i in range(3, nd + 1):
        tmp = c0
        c0 = float(c_list[-i]) - c1
        c1 = tmp + c1 * x2
    return c0 + c1 * x

def _chebint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    if isinstance(m, float):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0:
        raise ValueError("m must be non-negative")
    if isinstance(axis, float):
        raise TypeError("axis must be an integer")
    c = np.asarray(c)
    if isinstance(k, (int, float)):
        k = [k]
    if isinstance(lbnd, (list, tuple, np.ndarray)):
        raise ValueError("lbnd must be a scalar")
    if isinstance(scl, (list, tuple, np.ndarray)):
        raise ValueError("scl must be a scalar")
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if c.ndim > 1:
        return _chebint_nd(c, m, k, lbnd, scl, axis)
    c = _flat_arraylike_data(c.flatten())
    k_list = list(k) if k else []
    while len(k_list) < m:
        k_list.append(0)
    for step in range(m):
        n = len(c)
        if n == 0:
            c = [0.0]
            continue
        ic = [0.0] * (n + 1)
        for j in range(n):
            if j == 0:
                ic[j + 1] += c[j] * scl
            elif j == 1:
                ic[j + 1] += c[j] * scl / (2.0 * (j + 1))
            else:
                ic[j + 1] += c[j] * scl / (2.0 * (j + 1))
                ic[j - 1] -= c[j] * scl / (2.0 * (j - 1))
        # Set constant: ic[0] = k - chebval(lbnd, ic_without_c0)
        val_at_lbnd = _chebval_scalar(lbnd, [0.0] + ic[1:])
        ic[0] = k_list[step] - val_at_lbnd
        c = ic
    return np.array(c)

def _chebint_nd(c, m, k, lbnd, scl, axis):
    c = np.moveaxis(c, axis, 0)
    k_list = list(k) if k else []
    while len(k_list) < m:
        k_list.append(0)
    for step in range(m):
        n = c.shape[0]
        ic = np.zeros((n + 1,) + c.shape[1:])
        for j in range(n):
            if j == 0:
                ic[j + 1] = ic[j + 1] + c[j] * scl
            elif j == 1:
                ic[j + 1] = ic[j + 1] + c[j] * scl / (2.0 * (j + 1))
            else:
                ic[j + 1] = ic[j + 1] + c[j] * scl / (2.0 * (j + 1))
                ic[j - 1] = ic[j - 1] - c[j] * scl / (2.0 * (j - 1))
        val_at_lbnd = np.zeros(c.shape[1:])
        for j in range(1, n + 1):
            tj = _chebval_scalar(lbnd, [0.0]*j + [1.0])
            val_at_lbnd = val_at_lbnd + ic[j] * tj
        ic[0] = k_list[step] - val_at_lbnd
        c = ic
    c = np.moveaxis(c, 0, axis)
    return c

def _cheb2poly(c):
    """Convert Chebyshev series to polynomial (power series) coefficients."""
    c = _flat_list(c)
    n = len(c)
    if n == 0:
        return np.array([0.0])
    t_prev = [1.0]
    if n == 1:
        return np.array([c[0] * t_prev[0]])
    t_curr = [0.0, 1.0]
    result = [0.0] * (n)
    for i in range(len(t_prev)):
        if i < len(result):
            result[i] += c[0] * t_prev[i]
    for i in range(len(t_curr)):
        if i < len(result):
            result[i] += c[1] * t_curr[i]
    for kk in range(2, n):
        t_next = [0.0] * (len(t_curr) + 1)
        for i in range(len(t_curr)):
            t_next[i + 1] += 2.0 * t_curr[i]
        for i in range(len(t_prev)):
            t_next[i] -= t_prev[i]
        while len(result) < len(t_next):
            result.append(0.0)
        for i in range(len(t_next)):
            result[i] += c[kk] * t_next[i]
        t_prev = t_curr
        t_curr = t_next
    return np.array(result)

def _poly2cheb(c):
    """Convert polynomial (power series) to Chebyshev series."""
    c = _flat_list(c)
    n = len(c)
    if n == 0:
        return np.array([0.0])
    result = [0.0] * n
    x_cheb = [[0.0] * n for _ in range(n)]
    x_cheb[0][0] = 1.0
    if n > 1:
        x_cheb[1][1] = 1.0
    for kk in range(2, n):
        prev = x_cheb[kk - 1]
        cur = [0.0] * n
        for j in range(n):
            if prev[j] == 0:
                continue
            if j == 0:
                if 1 < n:
                    cur[1] += prev[0]
            else:
                if j + 1 < n:
                    cur[j + 1] += prev[j] * 0.5
                if j - 1 >= 0:
                    cur[j - 1] += prev[j] * 0.5
        x_cheb[kk] = cur
    for kk in range(n):
        for j in range(n):
            result[j] += c[kk] * x_cheb[kk][j]
    return np.array(result)

def _chebfromroots(roots):
    roots = _flat_list(roots)
    if len(roots) == 0:
        return np.array([1.0])
    c = np.array([-roots[0], 1.0])
    for i in range(1, len(roots)):
        c = _chebmul(c, [-roots[i], 1.0])
    return c

def _chebroots(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    if len(c) <= 1:
        return np.array([])
    if len(c) == 2:
        return np.array([-c[0] / c[1]])
    m = _chebcompanion(c)
    r = np.linalg.eigvals(m)
    r_list = _flat_arraylike_data(r.flatten())
    r_list.sort(key=lambda x: (x.real if isinstance(x, complex) else x, x.imag if isinstance(x, complex) else 0))
    return np.array(r_list)

def _chebcompanion(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0:
        c.pop()
    if len(c) < 2:
        raise ValueError("Series must have maximum degree >= 1")
    n = len(c) - 1
    mat = np.zeros((n, n))
    if n == 1:
        mat[0, 0] = -c[0] / c[1]
        return mat
    import math
    sqrt_half = math.sqrt(0.5)
    # Build symmetric Chebyshev companion matrix (scaled form)
    scl = [1.0] + [sqrt_half] * (n - 1)
    # Super-diagonal: [sqrt(0.5), 0.5, 0.5, ...]
    # Sub-diagonal: same (symmetric)
    mat[0, 1] = sqrt_half
    mat[1, 0] = sqrt_half
    for i in range(1, n - 1):
        mat[i, i + 1] = 0.5
        mat[i + 1, i] = 0.5
    # Modify last column
    for j in range(n):
        mat[j, n - 1] -= (c[j] / c[n]) * (scl[j] / scl[n - 1]) * 0.5
    return mat

def _chebvander(x, deg):
    x = np.asarray(x)
    deg = int(deg)
    if deg < 0:
        raise ValueError("deg must be non-negative")
    shape = x.shape + (deg + 1,)
    # Use same dtype as x (important for complex inputs)
    _dt = str(x.dtype) if hasattr(x, 'dtype') else None
    if _dt and 'complex' in _dt:
        v = np.zeros(shape, dtype=x.dtype)
    else:
        v = np.zeros(shape)
    _set_last(v, 0, np.ones(x.shape) if len(x.shape) > 0 else 1.0)
    if deg >= 1:
        _set_last(v, 1, x)
    for i in range(2, deg + 1):
        _set_last(v, i, 2.0 * x * _get_last(v, i - 1) - _get_last(v, i - 2))
    return v

def _chebvander2d(x, y, deg):
    x = np.asarray(x)
    y = np.asarray(y)
    degx, degy = deg
    vx = _chebvander(x, degx)
    vy = _chebvander(y, degy)
    shape = x.shape + ((degx + 1) * (degy + 1),)
    v = np.zeros(shape)
    kk = 0
    for i in range(degx + 1):
        for j in range(degy + 1):
            _set_last(v, kk, _get_last(vx, i) * _get_last(vy, j))
            kk += 1
    return v

def _chebvander3d(x, y, z, deg):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    degx, degy, degz = deg
    vx = _chebvander(x, degx)
    vy = _chebvander(y, degy)
    vz = _chebvander(z, degz)
    shape = x.shape + ((degx + 1) * (degy + 1) * (degz + 1),)
    v = np.zeros(shape)
    kk = 0
    for i in range(degx + 1):
        for j in range(degy + 1):
            for ll in range(degz + 1):
                _set_last(v, kk, _get_last(vx, i) * _get_last(vy, j) * _get_last(vz, ll))
                kk += 1
    return v

def _chebfit(x, y, deg, w=None):
    x = np.asarray(x)
    y = np.asarray(y)
    # Only cast to float64 if not complex
    if not (hasattr(x, 'dtype') and 'complex' in str(x.dtype)):
        x = np.asarray(x, dtype='float64')
    if not (hasattr(y, 'dtype') and 'complex' in str(y.dtype)):
        y = np.asarray(y, dtype='float64')
    _validate_fit_args(x, y, w)
    if isinstance(deg, (list, tuple, np.ndarray)):
        deg_list = _flat_int_list(deg)
        if any(d < 0 for d in deg_list):
            raise ValueError("deg must be non-negative")
        if len(deg_list) == 0:
            raise TypeError("deg must be non-empty")
        max_deg = max(deg_list)
        V_full = _chebvander(x, max_deg)
        V = np.zeros((len(x), max_deg + 1))
        for d in sorted(set(deg_list)):
            V[:, d] = V_full[:, d]
        if w is not None:
            w = np.asarray(w, dtype='float64')
            V = V * w[:, None]
            if y.ndim == 1: y = y * w
            else: y = y * w[:, None]
        c = np.linalg.lstsq(V, y)[0]
        return c
    else:
        deg = int(deg)
        if deg < 0:
            raise ValueError("deg must be non-negative")
        V = _chebvander(x, deg)
        if w is not None:
            w = np.asarray(w, dtype='float64')
            V = V * w[:, None]
            if y.ndim == 1: y = y * w
            else: y = y * w[:, None]
        c = np.linalg.lstsq(V, y)[0]
        return c

def _chebweight(x):
    x = np.asarray(x)
    return 1.0 / (np.sqrt(1.0 + x) * np.sqrt(1.0 - x))

def _chebpts1(n):
    if n != int(n) or n < 1:
        raise ValueError("n must be a positive integer")
    n = int(n)
    import math
    return np.array([math.cos(math.pi * (2 * kk + 1) / (2 * n)) for kk in range(n - 1, -1, -1)])

def _chebpts2(n):
    if n != int(n) or n < 2:
        raise ValueError("n must be an integer >= 2")
    n = int(n)
    import math
    return np.array([math.cos(math.pi * kk / (n - 1)) for kk in range(n - 1, -1, -1)])

def _chebgauss(n):
    """Chebyshev-Gauss quadrature."""
    import math
    n = int(n)
    x = np.array([math.cos(math.pi * (2 * kk + 1) / (2 * n)) for kk in range(n - 1, -1, -1)])
    w = np.full(n, math.pi / n)
    return x, w

def _chebinterpolate(func, deg, args=()):
    """Interpolate function at Chebyshev points."""
    if deg != int(deg):
        raise TypeError("deg must be an integer")
    deg = int(deg)
    if deg < 0:
        raise ValueError("deg must be non-negative")
    import math
    n = deg + 1
    x = np.array([math.cos(math.pi * (2 * kk + 1) / (2 * n)) for kk in range(n)])
    y = np.array([func(xi, *args) for xi in _flat_arraylike_data(x)])
    c = np.zeros(n)
    for j in range(n):
        s = 0.0
        for kk in range(n):
            s += y[kk] * math.cos(math.pi * j * (2 * kk + 1) / (2 * n))
        c[j] = 2.0 * s / n
    c[0] /= 2.0
    return c

def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series."""
    c = np.asarray(c, dtype=np.float64)
    n = len(c)
    zs = np.zeros(2 * n - 1, dtype=np.float64)
    zs[n - 1] = c[0]
    for i in range(1, n):
        zs[n - 1 + i] = c[i] / 2.0
        zs[n - 1 - i] = c[i] / 2.0
    return zs

def _zseries_to_cseries(zs):
    """Convert z-series to Chebyshev series."""
    zs = np.asarray(zs, dtype=np.float64)
    n = (len(zs) + 1) // 2
    c = np.zeros(n, dtype=np.float64)
    c[0] = zs[n - 1]
    for i in range(1, n):
        c[i] = zs[n - 1 + i] + zs[n - 1 - i]
    return c


class Chebyshev(ABCPolyBase):
    domain = np.array([-1., 1.])
    window = np.array([-1., 1.])
    basis_name = 'T'
    nickname = 'cheb'

    _add_func = staticmethod(_chebadd)
    _sub_func = staticmethod(_chebsub)
    _mul_func = staticmethod(_chebmul)
    _div_func = staticmethod(_chebdiv)
    _pow_func = staticmethod(_chebpow)
    _val_func = staticmethod(_chebval)
    _int_func = staticmethod(_chebint)
    _der_func = staticmethod(_chebder)
    _fit_func = staticmethod(_chebfit)
    _fromroots_func = staticmethod(_chebfromroots)
    _roots_func = staticmethod(_chebroots)
    _vander_func = staticmethod(_chebvander)

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=(), symbol='x'):
        if deg != int(deg):
            raise TypeError("deg must be an integer")
        if int(deg) < 0:
            raise ValueError("deg must be non-negative")
        if domain is None:
            domain = cls.domain
        import math
        n = int(deg) + 1
        x_cheb = np.array([math.cos(math.pi * (2 * kk + 1) / (2 * n)) for kk in range(n)])
        off, scl = pu.mapparms([-1, 1], domain)
        x_domain = x_cheb * scl + off
        y = np.array([func(xi, *args) for xi in _flat_arraylike_data(x_domain)])
        c = np.zeros(n)
        for j in range(n):
            s = 0.0
            for kk in range(n):
                s += y[kk] * math.cos(math.pi * j * (2 * kk + 1) / (2 * n))
            c[j] = 2.0 * s / n
        c[0] /= 2.0
        return cls(c, domain=domain, symbol=symbol)


class chebyshev:
    chebdomain = np.array([-1., 1.])
    chebzero = np.array([0.])
    chebone = np.array([1.])
    chebx = np.array([0., 1.])
    Chebyshev = Chebyshev
    chebtrim = staticmethod(_chebtrim)
    chebline = staticmethod(_chebline)
    chebadd = staticmethod(_chebadd)
    chebsub = staticmethod(_chebsub)
    chebmulx = staticmethod(_chebmulx)
    chebmul = staticmethod(_chebmul)
    chebdiv = staticmethod(_chebdiv)
    chebpow = staticmethod(_chebpow)
    chebval = staticmethod(_chebval)
    chebval2d = staticmethod(_chebval2d)
    chebval3d = staticmethod(_chebval3d)
    chebgrid2d = staticmethod(_chebgrid2d)
    chebgrid3d = staticmethod(_chebgrid3d)
    chebder = staticmethod(_chebder)
    chebint = staticmethod(_chebint)
    chebroots = staticmethod(_chebroots)
    chebvander = staticmethod(_chebvander)
    chebvander2d = staticmethod(_chebvander2d)
    chebvander3d = staticmethod(_chebvander3d)
    chebcompanion = staticmethod(_chebcompanion)
    chebfromroots = staticmethod(_chebfromroots)
    chebfit = staticmethod(_chebfit)
    chebweight = staticmethod(_chebweight)
    chebpts1 = staticmethod(_chebpts1)
    chebpts2 = staticmethod(_chebpts2)
    chebgauss = staticmethod(_chebgauss)
    chebinterpolate = staticmethod(_chebinterpolate)
    cheb2poly = staticmethod(_cheb2poly)
    poly2cheb = staticmethod(_poly2cheb)
    _cseries_to_zseries = staticmethod(_cseries_to_zseries)
    _zseries_to_cseries = staticmethod(_zseries_to_cseries)


# ===========================================================================
#  LEGENDRE
# ===========================================================================

def _legtrim(c, tol=0):
    return _trimcoef(c, tol)

def _legline(off, scl):
    if scl == 0:
        return np.array([off])
    return np.array([off, scl])

def _legadd(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] + c2[i] for i in range(n)])

def _legsub(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] - c2[i] for i in range(n)])

def _legmulx(c):
    c = _flat_list(c)
    if len(c) == 1 and c[0] == 0:
        return np.array([0.0])
    n = len(c)
    result = [0.0] * (n + 1)
    for i in range(n):
        if i == 0:
            result[1] += c[0]
        else:
            result[i + 1] += (i + 1) * c[i] / (2 * i + 1)
            result[i - 1] += i * c[i] / (2 * i + 1)
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return np.array(result)

def _legmul(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n1, n2 = len(c1), len(c2)
    out_deg = n1 + n2 - 2
    n_pts = out_deg + 1
    import math
    pts = [math.cos(math.pi * (2 * kk + 1) / (2 * n_pts)) for kk in range(n_pts)]
    vals = []
    for xi in pts:
        v1 = _legval_scalar(xi, c1)
        v2 = _legval_scalar(xi, c2)
        vals.append(v1 * v2)
    x_arr = np.array(pts)
    y_arr = np.array(vals)
    return _legfit(x_arr, y_arr, out_deg)

def _legdiv(c1, c2):
    p1 = _leg2poly(c1)
    p2 = _leg2poly(c2)
    q, r = _polydiv(p1, p2)
    return _poly2leg(q), _poly2leg(r)

def _legpow(c, n, maxpower=None):
    if maxpower is not None and n > maxpower:
        raise ValueError("Power exceeds maxpower")
    result = np.array([1.0])
    for _ in range(n):
        result = _legmul(result, c)
    return result

def _legval_scalar(x, c):
    c = _coef_list(c)
    if len(c) == 0: return 0.0
    if len(c) == 1: return float(c[0])
    p0, p1 = 1.0, float(x)
    result = c[0] * p0 + c[1] * p1
    for i in range(2, len(c)):
        p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / float(i)
        result += c[i] * p2
        p0, p1 = p1, p2
    return result

def _legval(x, c):
    x = np.asarray(x)
    c_list = _flat_list(c)
    if len(c_list) == 0:
        return np.zeros_like(x)
    if len(c_list) == 1:
        return np.full(x.shape, c_list[0]) + np.zeros_like(x)
    nd = len(c_list)
    p0 = np.ones(x.shape) + np.zeros_like(x)
    p1 = x * 1.0
    result = c_list[0] * p0 + c_list[1] * p1
    for i in range(2, nd):
        p2 = ((2.0 * i - 1.0) * x * p1 - (i - 1.0) * p0) / float(i)
        result = result + c_list[i] * p2
        p0 = p1
        p1 = p2
    return result

def _legder(c, m=1, scl=1, axis=0):
    if isinstance(m, float):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0:
        raise ValueError("m must be non-negative")
    c = np.asarray(c)
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        for _ in range(m):
            n = c.shape[0]
            if n <= 1:
                c = np.zeros((1,) + c.shape[1:])
                break
            dc = np.zeros((n - 1,) + c.shape[1:])
            c_work = c.copy()
            for j in range(n - 1, 0, -1):
                dc[j - 1] = dc[j - 1] + (2 * j - 1) * c_work[j] * scl
                if j - 2 >= 0:
                    c_work[j - 2] = c_work[j - 2] + c_work[j]
            c = dc
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    for _ in range(m):
        n = len(c)
        if n <= 1:
            c = [0.0]
            continue
        dc = [0.0] * (n - 1)
        for j in range(n - 1, 0, -1):
            dc[j - 1] += (2 * j - 1) * c[j] * scl
            if j - 2 >= 0:
                c[j - 2] += c[j]
        c = dc
    return np.array(c)

def _legint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    if isinstance(m, float) or (hasattr(m, 'is_integer') and not isinstance(m, int)):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0:
        raise ValueError("m must be non-negative")
    if isinstance(axis, float) or (hasattr(axis, 'is_integer') and not isinstance(axis, int)):
        raise TypeError("axis must be an integer")
    c = np.asarray(c)
    if isinstance(k, (int, float)): k = [k]
    if isinstance(lbnd, (list, tuple, np.ndarray)):
        raise ValueError("lbnd must be a scalar")
    if isinstance(scl, (list, tuple, np.ndarray)):
        raise ValueError("scl must be a scalar")
    if len(k) > m:
        raise ValueError("Too many integration constants")
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        k_list = list(k) if k else []
        while len(k_list) < m: k_list.append(0)
        for step in range(m):
            n = c.shape[0]
            ic = np.zeros((n + 1,) + c.shape[1:])
            for j in range(n):
                if j == 0:
                    ic[1] = ic[1] + c[0] * scl
                else:
                    ic[j + 1] = ic[j + 1] + c[j] * scl / (2.0 * j + 1.0)
                    if j - 1 >= 0:
                        ic[j - 1] = ic[j - 1] - c[j] * scl / (2.0 * j + 1.0)
            val_at_lbnd = np.zeros(c.shape[1:])
            for j in range(1, n + 1):
                tj = _legval_scalar(lbnd, [0.0]*j + [1.0])
                val_at_lbnd = val_at_lbnd + ic[j] * tj
            ic[0] = k_list[step] - val_at_lbnd
            c = ic
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    k_list = list(k) if k else []
    while len(k_list) < m: k_list.append(0)
    for step in range(m):
        n = len(c)
        if n == 0:
            c = [0.0]
            continue
        ic = [0.0] * (n + 1)
        for j in range(n):
            if j == 0:
                ic[1] += c[0] * scl
            else:
                ic[j + 1] += c[j] * scl / (2.0 * j + 1.0)
                if j - 1 >= 0:
                    ic[j - 1] -= c[j] * scl / (2.0 * j + 1.0)
        val_at_lbnd = _legval_scalar(lbnd, [0.0] + ic[1:])
        ic[0] = k_list[step] - val_at_lbnd
        c = ic
    return np.array(c)

def _legfromroots(roots):
    roots = _flat_list(roots)
    if len(roots) == 0:
        return np.array([1.0])
    c = _poly2leg(np.array([-roots[0], 1.0]))
    for i in range(1, len(roots)):
        c = _legmul(c, _poly2leg(np.array([-roots[i], 1.0])))
    return c

def _legroots(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0: c.pop()
    if len(c) <= 1: return np.array([])
    if len(c) == 2: return np.array([-c[0] / c[1]])
    poly_c = _leg2poly(c)
    return _polyroots(poly_c)

def _legcompanion(c):
    poly_c = _leg2poly(c)
    return _polycompanion(_flat_arraylike_data(poly_c.flatten()))

def _legvander(x, deg):
    x = np.asarray(x)
    deg = int(deg)
    if deg < 0: raise ValueError("deg must be non-negative")
    shape = x.shape + (deg + 1,)
    v = np.zeros(shape)
    _set_last(v, 0, np.ones(x.shape) if len(x.shape) > 0 else 1.0)
    if deg >= 1:
        _set_last(v, 1, x)
    for i in range(2, deg + 1):
        _set_last(v, i, ((2.0 * i - 1.0) * x * _get_last(v, i - 1) - (i - 1.0) * _get_last(v, i - 2)) / float(i))
    return v

def _legfit(x, y, deg, w=None):
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    _validate_fit_args(x, y, w)
    if isinstance(deg, (list, tuple, np.ndarray)):
        deg_list = _flat_int_list(deg)
        if any(d < 0 for d in deg_list): raise ValueError("deg must be non-negative")
        if len(deg_list) == 0: raise TypeError("deg must be non-empty")
        max_deg = max(deg_list)
        V_full = _legvander(x, max_deg)
        V = np.zeros((len(x), max_deg + 1))
        for d in sorted(set(deg_list)): V[:, d] = V_full[:, d]
        if w is not None:
            w = np.asarray(w, dtype='float64')
            V = V * w[:, None]
            if y.ndim == 1: y = y * w
            else: y = y * w[:, None]
        return np.linalg.lstsq(V, y)[0]
    else:
        deg = int(deg)
        if deg < 0: raise ValueError("deg must be non-negative")
        V = _legvander(x, deg)
        if w is not None:
            w = np.asarray(w, dtype='float64')
            V = V * w[:, None]
            if y.ndim == 1: y = y * w
            else: y = y * w[:, None]
        return np.linalg.lstsq(V, y)[0]

def _leg2poly(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    p_prev = [1.0]
    if n == 1: return np.array([c[0]])
    p_curr = [0.0, 1.0]
    result = [0.0] * n
    for i in range(len(p_prev)):
        result[i] += c[0] * p_prev[i]
    for i in range(len(p_curr)):
        if i < len(result): result[i] += c[1] * p_curr[i]
    for kk in range(2, n):
        p_next = [0.0] * (len(p_curr) + 1)
        for i in range(len(p_curr)):
            p_next[i + 1] += (2 * kk - 1) * p_curr[i] / kk
        for i in range(len(p_prev)):
            p_next[i] -= (kk - 1) * p_prev[i] / kk
        while len(result) < len(p_next):
            result.append(0.0)
        for i in range(len(p_next)):
            result[i] += c[kk] * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return np.array(result)

def _poly2leg(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    result = [0.0] * n
    x_leg = [[0.0] * n for _ in range(n)]
    x_leg[0][0] = 1.0
    if n > 1:
        x_leg[1][1] = 1.0
    for kk in range(2, n):
        prev = x_leg[kk - 1]
        cur = [0.0] * n
        for j in range(n):
            if prev[j] == 0: continue
            if j + 1 < n:
                cur[j + 1] += prev[j] * (j + 1) / (2 * j + 1)
            if j - 1 >= 0:
                cur[j - 1] += prev[j] * j / (2 * j + 1)
        x_leg[kk] = cur
    for kk in range(n):
        for j in range(n):
            result[j] += c[kk] * x_leg[kk][j]
    return np.array(result)


class Legendre(ABCPolyBase):
    domain = np.array([-1., 1.])
    window = np.array([-1., 1.])
    basis_name = 'P'
    nickname = 'leg'

    _add_func = staticmethod(_legadd)
    _sub_func = staticmethod(_legsub)
    _mul_func = staticmethod(_legmul)
    _div_func = staticmethod(_legdiv)
    _pow_func = staticmethod(_legpow)
    _val_func = staticmethod(_legval)
    _int_func = staticmethod(_legint)
    _der_func = staticmethod(_legder)
    _fit_func = staticmethod(_legfit)
    _fromroots_func = staticmethod(_legfromroots)
    _roots_func = staticmethod(_legroots)
    _vander_func = staticmethod(_legvander)


def _make_val2d(valfunc):
    def val2d(x, y, c):
        c = np.asarray(c)
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("x and y have incompatible shapes")
        result = np.zeros(x.shape)
        for i in range(c.shape[0]):
            ti = valfunc(x, [0.0]*i + [1.0])
            result = result + valfunc(y, c[i]) * np.asarray(ti)
        return result
    return val2d

def _make_val3d(valfunc):
    def val3d(x, y, z, c):
        c = np.asarray(c)
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        if x.shape != y.shape or x.shape != z.shape:
            raise ValueError("x, y, and z have incompatible shapes")
        result = np.zeros(x.shape)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ti = valfunc(x, [0.0]*i + [1.0])
                tj = valfunc(y, [0.0]*j + [1.0])
                result = result + valfunc(z, c[i, j]) * np.asarray(ti) * np.asarray(tj)
        return result
    return val3d

def _make_grid2d(valfunc):
    def grid2d(x, y, c):
        c = np.asarray(c)
        x, y = np.asarray(x), np.asarray(y)
        shape = x.shape + y.shape
        result = np.zeros(shape)
        for i in range(c.shape[0]):
            yvals = valfunc(y, c[i])
            ti = valfunc(x, [0.0]*i + [1.0])
            result = result + np.multiply.outer(np.asarray(ti), np.asarray(yvals))
        return result
    return grid2d

def _make_grid3d(valfunc):
    def grid3d(x, y, z, c):
        c = np.asarray(c)
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        shape = x.shape + y.shape + z.shape
        result = np.zeros(shape)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                ti = valfunc(x, [0.0]*i + [1.0])
                tj = valfunc(y, [0.0]*j + [1.0])
                zvals = valfunc(z, c[i, j])
                result = result + _outer3(np.asarray(ti), np.asarray(tj), np.asarray(zvals))
        return result
    return grid3d

def _make_vander2d(vanderfunc):
    def vander2d(x, y, deg):
        x, y = np.asarray(x), np.asarray(y)
        degx, degy = deg
        vx = vanderfunc(x, degx)
        vy = vanderfunc(y, degy)
        shape = x.shape + ((degx + 1) * (degy + 1),)
        v = np.zeros(shape)
        kk = 0
        for i in range(degx + 1):
            for j in range(degy + 1):
                _set_last(v, kk, _get_last(vx, i) * _get_last(vy, j))
                kk += 1
        return v
    return vander2d

def _make_vander3d(vanderfunc):
    def vander3d(x, y, z, deg):
        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
        degx, degy, degz = deg
        vx = vanderfunc(x, degx)
        vy = vanderfunc(y, degy)
        vz = vanderfunc(z, degz)
        shape = x.shape + ((degx + 1) * (degy + 1) * (degz + 1),)
        v = np.zeros(shape)
        kk = 0
        for i in range(degx + 1):
            for j in range(degy + 1):
                for ll in range(degz + 1):
                    _set_last(v, kk, _get_last(vx, i) * _get_last(vy, j) * _get_last(vz, ll))
                    kk += 1
        return v
    return vander3d

def _leggauss(n):
    """Legendre-Gauss quadrature."""
    import math
    n = int(n)
    x = np.zeros(n)
    w = np.zeros(n)
    for i in range(n):
        z = math.cos(math.pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, z
            for j in range(1, n):
                p2 = ((2 * j + 1) * z * p1 - j * p0) / (j + 1)
                p0, p1 = p1, p2
            pp = n * (z * p1 - p0) / (z * z - 1)
            z_new = z - p1 / pp
            if abs(z_new - z) < 1e-15:
                break
            z = z_new
        x[i] = z
        w[i] = 2.0 / ((1 - z * z) * pp * pp)
    idx = np.argsort(x)
    return x[idx], w[idx]

_legval2d = _make_val2d(_legval)
_legval3d = _make_val3d(_legval)
_leggrid2d = _make_grid2d(_legval)
_leggrid3d = _make_grid3d(_legval)
_legvander2d = _make_vander2d(_legvander)
_legvander3d = _make_vander3d(_legvander)


class legendre:
    legdomain = np.array([-1., 1.])
    legzero = np.array([0.])
    legone = np.array([1.])
    legx = np.array([0., 1.])
    Legendre = Legendre
    legtrim = staticmethod(_legtrim)
    legline = staticmethod(_legline)
    legadd = staticmethod(_legadd)
    legsub = staticmethod(_legsub)
    legmulx = staticmethod(_legmulx)
    legmul = staticmethod(_legmul)
    legdiv = staticmethod(_legdiv)
    legpow = staticmethod(_legpow)
    legval = staticmethod(_legval)
    legval2d = staticmethod(_legval2d)
    legval3d = staticmethod(_legval3d)
    leggrid2d = staticmethod(_leggrid2d)
    leggrid3d = staticmethod(_leggrid3d)
    legder = staticmethod(_legder)
    legint = staticmethod(_legint)
    legroots = staticmethod(_legroots)
    legvander = staticmethod(_legvander)
    legvander2d = staticmethod(_legvander2d)
    legvander3d = staticmethod(_legvander3d)
    legcompanion = staticmethod(_legcompanion)
    legfromroots = staticmethod(_legfromroots)
    legfit = staticmethod(_legfit)
    legweight = staticmethod(lambda x: np.ones_like(np.asarray(x)))
    leggauss = staticmethod(lambda n: _leggauss(n))
    leg2poly = staticmethod(_leg2poly)
    poly2leg = staticmethod(_poly2leg)
    legline = staticmethod(_legline)


# ===========================================================================
#  HERMITE (physicist's)
# ===========================================================================

def _hermval_scalar(x, c):
    c = _coef_list(c)
    if len(c) == 0: return 0.0
    if len(c) == 1: return float(c[0])
    h0, h1 = 1.0, 2.0 * x
    result = c[0] * h0 + c[1] * h1
    for i in range(2, len(c)):
        h2 = 2.0 * x * h1 - 2.0 * (i - 1) * h0
        result += c[i] * h2
        h0, h1 = h1, h2
    return result

def _hermval(x, c):
    x = np.asarray(x)
    c_list = _flat_list(c)
    if len(c_list) == 0: return np.zeros_like(x)
    if len(c_list) == 1:
        return np.full(x.shape, c_list[0]) + np.zeros_like(x)
    h0 = np.ones(x.shape) + np.zeros_like(x)
    h1 = x * 2.0
    result = c_list[0] * h0 + c_list[1] * h1
    for i in range(2, len(c_list)):
        h2 = 2.0 * x * h1 - 2.0 * (i - 1) * h0
        result = result + c_list[i] * h2
        h0, h1 = h1, h2
    return result

def _hermadd(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] + c2[i] for i in range(n)])

def _hermsub(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] - c2[i] for i in range(n)])

def _hermmul(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n1, n2 = len(c1), len(c2)
    out_deg = n1 + n2 - 2
    n_pts = out_deg + 1
    pts = [2.0 * kk / max(n_pts - 1, 1) - 1.0 for kk in range(n_pts)]
    vals = []
    for xi in pts:
        vals.append(_hermval_scalar(xi, c1) * _hermval_scalar(xi, c2))
    return _hermfit(np.array(pts), np.array(vals), out_deg)

def _hermdiv(c1, c2):
    p1 = _herm2poly(c1)
    p2 = _herm2poly(c2)
    q, r = _polydiv(p1, p2)
    return _poly2herm(q), _poly2herm(r)

def _hermpow(c, n, maxpower=None):
    if maxpower is not None and n > maxpower:
        raise ValueError("Power exceeds maxpower")
    result = np.array([1.0])
    for _ in range(n):
        result = _hermmul(result, c)
    return result

def _hermder(c, m=1, scl=1, axis=0):
    if isinstance(m, float): raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0: raise ValueError("m must be non-negative")
    c = np.asarray(c)
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        for _ in range(m):
            n = c.shape[0]
            if n <= 1:
                c = np.zeros((1,) + c.shape[1:])
                break
            dc = np.zeros((n - 1,) + c.shape[1:])
            for j in range(1, n):
                dc[j - 1] = 2.0 * j * c[j] * scl
            c = dc
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    for _ in range(m):
        n = len(c)
        if n <= 1: c = [0.0]; continue
        dc = [2.0 * j * c[j] * scl for j in range(1, n)]
        c = dc
    return np.array(c)

def _hermint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    if isinstance(m, float) or (hasattr(m, 'is_integer') and not isinstance(m, int)):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0: raise ValueError("m must be non-negative")
    if isinstance(axis, float) or (hasattr(axis, 'is_integer') and not isinstance(axis, int)):
        raise TypeError("axis must be an integer")
    c = np.asarray(c)
    if isinstance(k, (int, float)): k = [k]
    if isinstance(lbnd, (list, tuple, np.ndarray)): raise ValueError("lbnd must be a scalar")
    if isinstance(scl, (list, tuple, np.ndarray)): raise ValueError("scl must be a scalar")
    if len(k) > m: raise ValueError("Too many integration constants")
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        k_list = list(k) if k else []
        while len(k_list) < m: k_list.append(0)
        for step in range(m):
            n = c.shape[0]
            ic = np.zeros((n + 1,) + c.shape[1:])
            for j in range(n):
                ic[j + 1] = c[j] * scl / (2.0 * (j + 1))
            val_at_lbnd = np.zeros(c.shape[1:])
            for j in range(1, n + 1):
                val_at_lbnd = val_at_lbnd + ic[j] * _hermval_scalar(lbnd, [0.0]*j + [1.0])
            ic[0] = k_list[step] - val_at_lbnd
            c = ic
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    k_list = list(k) if k else []
    while len(k_list) < m: k_list.append(0)
    for step in range(m):
        n = len(c)
        ic = [0.0] * (n + 1)
        for j in range(n):
            ic[j + 1] = c[j] * scl / (2.0 * (j + 1))
        val_at_lbnd = _hermval_scalar(lbnd, [0.0] + ic[1:])
        ic[0] = k_list[step] - val_at_lbnd
        c = ic
    return np.array(c)

def _hermvander(x, deg):
    x = np.asarray(x)
    deg = int(deg)
    if deg < 0: raise ValueError("deg must be non-negative")
    shape = x.shape + (deg + 1,)
    v = np.zeros(shape)
    _set_last(v, 0, np.ones(x.shape) if len(x.shape) > 0 else 1.0)
    if deg >= 1: _set_last(v, 1, x * 2.0)
    for i in range(2, deg + 1):
        _set_last(v, i, 2.0 * x * _get_last(v, i - 1) - 2.0 * (i - 1) * _get_last(v, i - 2))
    return v

def _hermfit(x, y, deg, w=None):
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    _validate_fit_args(x, y, w)
    if isinstance(deg, (list, tuple, np.ndarray)):
        deg_list = _flat_int_list(deg)
        if any(d < 0 for d in deg_list): raise ValueError("deg must be non-negative")
        if len(deg_list) == 0: raise TypeError("deg must be non-empty")
        max_deg = max(deg_list)
        V_full = _hermvander(x, max_deg)
        V = np.zeros((len(x), max_deg + 1))
        for d in sorted(set(deg_list)): V[:, d] = V_full[:, d]
    else:
        deg = int(deg)
        if deg < 0: raise ValueError("deg must be non-negative")
        V = _hermvander(x, deg)
    if w is not None:
        w = np.asarray(w, dtype='float64')
        V = V * w[:, None]
        if y.ndim == 1: y = y * w
        else: y = y * w[:, None]
    return np.linalg.lstsq(V, y)[0]

def _hermfromroots(roots):
    roots = _flat_list(roots)
    if len(roots) == 0: return np.array([1.0])
    c = _poly2herm(np.array([-roots[0], 1.0]))
    for i in range(1, len(roots)):
        c = _hermmul(c, _poly2herm(np.array([-roots[i], 1.0])))
    return c

def _hermroots(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0: c.pop()
    if len(c) <= 1: return np.array([])
    if len(c) == 2: return np.array([-c[0] / c[1]])
    poly_c = _herm2poly(c)
    return _polyroots(poly_c)

def _hermcompanion(c):
    poly_c = _herm2poly(c)
    return _polycompanion(_flat_arraylike_data(poly_c.flatten()))

def _herm2poly(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    h_prev = [1.0]
    if n == 1: return np.array([c[0]])
    h_curr = [0.0, 2.0]
    result = [0.0] * (2 * n)
    for i in range(len(h_prev)): result[i] += c[0] * h_prev[i]
    for i in range(len(h_curr)):
        if i < len(result): result[i] += c[1] * h_curr[i]
    for kk in range(2, n):
        h_next = [0.0] * (len(h_curr) + 1)
        for i in range(len(h_curr)): h_next[i + 1] += 2.0 * h_curr[i]
        for i in range(len(h_prev)): h_next[i] -= 2.0 * (kk - 1) * h_prev[i]
        while len(result) < len(h_next): result.append(0.0)
        for i in range(len(h_next)): result[i] += c[kk] * h_next[i]
        h_prev = h_curr
        h_curr = h_next
    while len(result) > 1 and abs(result[-1]) < 1e-15: result.pop()
    return np.array(result)

def _poly2herm(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    result = [0.0] * n
    x_herm = [[0.0] * n for _ in range(n)]
    x_herm[0][0] = 1.0
    if n > 1: x_herm[1][1] = 0.5  # x = H_1/2
    for kk in range(2, n):
        prev = x_herm[kk - 1]
        cur = [0.0] * n
        for j in range(n):
            if prev[j] == 0: continue
            if j + 1 < n: cur[j + 1] += prev[j] * 0.5
            if j > 0: cur[j - 1] += prev[j] * j
        x_herm[kk] = cur
    for kk in range(n):
        for j in range(n):
            result[j] += c[kk] * x_herm[kk][j]
    return np.array(result)


class Hermite(ABCPolyBase):
    domain = np.array([-1., 1.])
    window = np.array([-1., 1.])
    basis_name = 'H'
    nickname = 'herm'
    _add_func = staticmethod(_hermadd)
    _sub_func = staticmethod(_hermsub)
    _mul_func = staticmethod(_hermmul)
    _div_func = staticmethod(_hermdiv)
    _pow_func = staticmethod(_hermpow)
    _val_func = staticmethod(_hermval)
    _int_func = staticmethod(_hermint)
    _der_func = staticmethod(_hermder)
    _fit_func = staticmethod(_hermfit)
    _fromroots_func = staticmethod(_hermfromroots)
    _roots_func = staticmethod(_hermroots)
    _vander_func = staticmethod(_hermvander)


def _hermgauss(n):
    """Hermite-Gauss quadrature (physicist's)."""
    import math
    n = int(n)
    # Build symmetric tridiagonal matrix
    comp = np.zeros((n, n))
    for i in range(n - 1):
        comp[i, i + 1] = math.sqrt((i + 1) / 2.0)
        comp[i + 1, i] = math.sqrt((i + 1) / 2.0)
    x = np.linalg.eigvalsh(comp)
    x = np.sort(x)
    w = np.zeros(n)
    for i in range(n):
        hn1 = _hermval_scalar(float(x[i]), [0.0]*(n-1) + [1.0])
        if abs(hn1) > 1e-300:
            w[i] = math.sqrt(math.pi) * 2.0**(n-1) * math.factorial(n) / (float(n) * hn1)**2
    return x, w

def _hermmulx(c):
    c = _flat_list(c)
    if len(c) == 1 and c[0] == 0:
        return np.array([0.0])
    n = len(c)
    result = [0.0] * (n + 1)
    for i in range(n):
        # x*H_i = 0.5*H_{i+1} + i*H_{i-1}
        result[i + 1] += 0.5 * c[i]
        if i > 0:
            result[i - 1] += i * c[i]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return np.array(result)

_hermval2d = _make_val2d(_hermval)
_hermval3d = _make_val3d(_hermval)
_hermgrid2d = _make_grid2d(_hermval)
_hermgrid3d = _make_grid3d(_hermval)
_hermvander2d = _make_vander2d(_hermvander)
_hermvander3d = _make_vander3d(_hermvander)


class hermite:
    hermdomain = np.array([-1., 1.])
    hermzero = np.array([0.])
    hermone = np.array([1.])
    hermx = np.array([0., 0.5])
    Hermite = Hermite
    hermadd = staticmethod(_hermadd)
    hermsub = staticmethod(_hermsub)
    hermmul = staticmethod(_hermmul)
    hermdiv = staticmethod(_hermdiv)
    hermpow = staticmethod(_hermpow)
    hermval = staticmethod(_hermval)
    hermval2d = staticmethod(_hermval2d)
    hermval3d = staticmethod(_hermval3d)
    hermgrid2d = staticmethod(_hermgrid2d)
    hermgrid3d = staticmethod(_hermgrid3d)
    hermder = staticmethod(_hermder)
    hermint = staticmethod(_hermint)
    hermroots = staticmethod(_hermroots)
    hermvander = staticmethod(_hermvander)
    hermvander2d = staticmethod(_hermvander2d)
    hermvander3d = staticmethod(_hermvander3d)
    hermcompanion = staticmethod(_hermcompanion)
    hermfromroots = staticmethod(_hermfromroots)
    hermfit = staticmethod(_hermfit)
    hermtrim = staticmethod(_trimcoef)
    hermline = staticmethod(lambda off, scl: np.array([off, scl / 2.0]))
    hermmulx = staticmethod(lambda c: _hermmulx(c))
    hermweight = staticmethod(lambda x: np.exp(-np.asarray(x)**2))
    hermgauss = staticmethod(lambda n: _hermgauss(n))
    herm2poly = staticmethod(_herm2poly)
    poly2herm = staticmethod(_poly2herm)


# ===========================================================================
#  HERMITE_E (probabilist's)
# ===========================================================================

def _hermeval_scalar(x, c):
    c = _coef_list(c)
    if len(c) == 0: return 0.0
    if len(c) == 1: return float(c[0])
    h0, h1 = 1.0, float(x)
    result = c[0] * h0 + c[1] * h1
    for i in range(2, len(c)):
        h2 = x * h1 - (i - 1) * h0
        result += c[i] * h2
        h0, h1 = h1, h2
    return result

def _hermeval(x, c):
    x = np.asarray(x)
    c_list = _flat_list(c)
    if len(c_list) == 0: return np.zeros_like(x)
    if len(c_list) == 1:
        return np.full(x.shape, c_list[0]) + np.zeros_like(x)
    h0 = np.ones(x.shape) + np.zeros_like(x)
    h1 = x * 1.0
    result = c_list[0] * h0 + c_list[1] * h1
    for i in range(2, len(c_list)):
        h2 = x * h1 - (i - 1) * h0
        result = result + c_list[i] * h2
        h0, h1 = h1, h2
    return result

def _hermeadd(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] + c2[i] for i in range(n)])

def _hermesub(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] - c2[i] for i in range(n)])

def _hermemul(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n1, n2 = len(c1), len(c2)
    out_deg = n1 + n2 - 2
    n_pts = out_deg + 1
    pts = [2.0 * kk / max(n_pts - 1, 1) - 1.0 for kk in range(n_pts)]
    vals = []
    for xi in pts:
        vals.append(_hermeval_scalar(xi, c1) * _hermeval_scalar(xi, c2))
    return _hermefit(np.array(pts), np.array(vals), out_deg)

def _hermediv(c1, c2):
    p1 = _herme2poly(c1)
    p2 = _herme2poly(c2)
    q, r = _polydiv(p1, p2)
    return _poly2herme(q), _poly2herme(r)

def _hermepow(c, n, maxpower=None):
    if maxpower is not None and n > maxpower:
        raise ValueError("Power exceeds maxpower")
    result = np.array([1.0])
    for _ in range(n):
        result = _hermemul(result, c)
    return result

def _hermeder(c, m=1, scl=1, axis=0):
    if isinstance(m, float): raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0: raise ValueError("m must be non-negative")
    c = np.asarray(c)
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        for _ in range(m):
            n = c.shape[0]
            if n <= 1: c = np.zeros((1,) + c.shape[1:]); break
            dc = np.zeros((n - 1,) + c.shape[1:])
            for j in range(1, n): dc[j - 1] = float(j) * c[j] * scl
            c = dc
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    for _ in range(m):
        n = len(c)
        if n <= 1: c = [0.0]; continue
        dc = [float(j) * c[j] * scl for j in range(1, n)]
        c = dc
    return np.array(c)

def _hermeint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    if isinstance(m, float) or (hasattr(m, 'is_integer') and not isinstance(m, int)):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0: raise ValueError("m must be non-negative")
    if isinstance(axis, float) or (hasattr(axis, 'is_integer') and not isinstance(axis, int)):
        raise TypeError("axis must be an integer")
    c = np.asarray(c)
    if isinstance(k, (int, float)): k = [k]
    if isinstance(lbnd, (list, tuple, np.ndarray)): raise ValueError("lbnd must be a scalar")
    if isinstance(scl, (list, tuple, np.ndarray)): raise ValueError("scl must be a scalar")
    if len(k) > m: raise ValueError("Too many integration constants")
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        k_list = list(k) if k else []
        while len(k_list) < m: k_list.append(0)
        for step in range(m):
            n = c.shape[0]
            ic = np.zeros((n + 1,) + c.shape[1:])
            for j in range(n): ic[j + 1] = c[j] * scl / float(j + 1)
            val_at_lbnd = np.zeros(c.shape[1:])
            for j in range(1, n + 1):
                val_at_lbnd = val_at_lbnd + ic[j] * _hermeval_scalar(lbnd, [0.0]*j + [1.0])
            ic[0] = k_list[step] - val_at_lbnd
            c = ic
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    k_list = list(k) if k else []
    while len(k_list) < m: k_list.append(0)
    for step in range(m):
        n = len(c)
        ic = [0.0] * (n + 1)
        for j in range(n): ic[j + 1] = c[j] * scl / float(j + 1)
        val_at_lbnd = _hermeval_scalar(lbnd, [0.0] + ic[1:])
        ic[0] = k_list[step] - val_at_lbnd
        c = ic
    return np.array(c)

def _hermevander(x, deg):
    x = np.asarray(x)
    deg = int(deg)
    if deg < 0: raise ValueError("deg must be non-negative")
    shape = x.shape + (deg + 1,)
    v = np.zeros(shape)
    _set_last(v, 0, np.ones(x.shape) if len(x.shape) > 0 else 1.0)
    if deg >= 1: _set_last(v, 1, x)
    for i in range(2, deg + 1):
        _set_last(v, i, x * _get_last(v, i - 1) - (i - 1) * _get_last(v, i - 2))
    return v

def _hermefit(x, y, deg, w=None):
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    _validate_fit_args(x, y, w)
    if isinstance(deg, (list, tuple, np.ndarray)):
        deg_list = _flat_int_list(deg)
        if any(d < 0 for d in deg_list): raise ValueError("deg must be non-negative")
        if len(deg_list) == 0: raise TypeError("deg must be non-empty")
        max_deg = max(deg_list)
        V_full = _hermevander(x, max_deg)
        V = np.zeros((len(x), max_deg + 1))
        for d in sorted(set(deg_list)): V[:, d] = V_full[:, d]
    else:
        deg = int(deg)
        if deg < 0: raise ValueError("deg must be non-negative")
        V = _hermevander(x, deg)
    if w is not None:
        w = np.asarray(w, dtype='float64')
        V = V * w[:, None]
        if y.ndim == 1: y = y * w
        else: y = y * w[:, None]
    return np.linalg.lstsq(V, y)[0]

def _hermefromroots(roots):
    roots = _flat_list(roots)
    if len(roots) == 0: return np.array([1.0])
    c = _poly2herme(np.array([-roots[0], 1.0]))
    for i in range(1, len(roots)):
        c = _hermemul(c, _poly2herme(np.array([-roots[i], 1.0])))
    return c

def _hermeroots(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0: c.pop()
    if len(c) <= 1: return np.array([])
    if len(c) == 2: return np.array([-c[0] / c[1]])
    poly_c = _herme2poly(c)
    return _polyroots(poly_c)

def _hermecompanion(c):
    poly_c = _herme2poly(c)
    return _polycompanion(_flat_arraylike_data(poly_c.flatten()))

def _herme2poly(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    h_prev = [1.0]
    if n == 1: return np.array([c[0]])
    h_curr = [0.0, 1.0]
    result = [0.0] * (2 * n)
    for i in range(len(h_prev)): result[i] += c[0] * h_prev[i]
    for i in range(len(h_curr)):
        if i < len(result): result[i] += c[1] * h_curr[i]
    for kk in range(2, n):
        h_next = [0.0] * (len(h_curr) + 1)
        for i in range(len(h_curr)): h_next[i + 1] += h_curr[i]
        for i in range(len(h_prev)): h_next[i] -= (kk - 1) * h_prev[i]
        while len(result) < len(h_next): result.append(0.0)
        for i in range(len(h_next)): result[i] += c[kk] * h_next[i]
        h_prev = h_curr
        h_curr = h_next
    while len(result) > 1 and abs(result[-1]) < 1e-15: result.pop()
    return np.array(result)

def _poly2herme(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    result = [0.0] * n
    x_herme = [[0.0] * n for _ in range(n)]
    x_herme[0][0] = 1.0
    if n > 1: x_herme[1][1] = 1.0
    for kk in range(2, n):
        prev = x_herme[kk - 1]
        cur = [0.0] * n
        for j in range(n):
            if prev[j] == 0: continue
            if j + 1 < n: cur[j + 1] += prev[j]
            if j > 0: cur[j - 1] += prev[j] * j
        x_herme[kk] = cur
    for kk in range(n):
        for j in range(n):
            result[j] += c[kk] * x_herme[kk][j]
    return np.array(result)


class HermiteE(ABCPolyBase):
    domain = np.array([-1., 1.])
    window = np.array([-1., 1.])
    basis_name = 'He'
    nickname = 'herme'
    _add_func = staticmethod(_hermeadd)
    _sub_func = staticmethod(_hermesub)
    _mul_func = staticmethod(_hermemul)
    _div_func = staticmethod(_hermediv)
    _pow_func = staticmethod(_hermepow)
    _val_func = staticmethod(_hermeval)
    _int_func = staticmethod(_hermeint)
    _der_func = staticmethod(_hermeder)
    _fit_func = staticmethod(_hermefit)
    _fromroots_func = staticmethod(_hermefromroots)
    _roots_func = staticmethod(_hermeroots)
    _vander_func = staticmethod(_hermevander)


def _hermemulx(c):
    c = _flat_list(c)
    if len(c) == 1 and c[0] == 0:
        return np.array([0.0])
    n = len(c)
    result = [0.0] * (n + 1)
    for i in range(n):
        # x*He_i = He_{i+1} + i*He_{i-1}
        result[i + 1] += c[i]
        if i > 0:
            result[i - 1] += i * c[i]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return np.array(result)

_hermeval2d = _make_val2d(_hermeval)
_hermeval3d = _make_val3d(_hermeval)
_hermegrid2d = _make_grid2d(_hermeval)
_hermegrid3d = _make_grid3d(_hermeval)
_hermevander2d = _make_vander2d(_hermevander)
_hermevander3d = _make_vander3d(_hermevander)

def _hermegauss(n):
    """Hermite-Gauss quadrature (probabilist's)."""
    import math
    n = int(n)
    # Build symmetric tridiagonal matrix for He polynomials
    # He_n has recurrence: x*He_i = He_{i+1} + i*He_{i-1}
    # So the companion tridiagonal has off-diagonal sqrt(i+1)
    comp = np.zeros((n, n))
    for i in range(n - 1):
        comp[i, i + 1] = math.sqrt(i + 1)
        comp[i + 1, i] = math.sqrt(i + 1)
    x = np.linalg.eigvalsh(comp)
    x = np.sort(x)
    w = np.zeros(n)
    for i in range(n):
        hen1 = _hermeval_scalar(float(x[i]), [0.0] * (n - 1) + [1.0])
        if abs(hen1) > 1e-300:
            w[i] = math.sqrt(2.0 * math.pi) * math.factorial(n) / (float(n) * hen1) ** 2
    return x, w


class hermite_e:
    hermedomain = np.array([-1., 1.])
    hermezero = np.array([0.])
    hermeone = np.array([1.])
    hermex = np.array([0., 1.])
    HermiteE = HermiteE
    hermeadd = staticmethod(_hermeadd)
    hermesub = staticmethod(_hermesub)
    hermemul = staticmethod(_hermemul)
    hermediv = staticmethod(_hermediv)
    hermepow = staticmethod(_hermepow)
    hermeval = staticmethod(_hermeval)
    hermeval2d = staticmethod(_hermeval2d)
    hermeval3d = staticmethod(_hermeval3d)
    hermegrid2d = staticmethod(_hermegrid2d)
    hermegrid3d = staticmethod(_hermegrid3d)
    hermeder = staticmethod(_hermeder)
    hermeint = staticmethod(_hermeint)
    hermeroots = staticmethod(_hermeroots)
    hermevander = staticmethod(_hermevander)
    hermevander2d = staticmethod(_hermevander2d)
    hermevander3d = staticmethod(_hermevander3d)
    hermecompanion = staticmethod(_hermecompanion)
    hermefromroots = staticmethod(_hermefromroots)
    hermefit = staticmethod(_hermefit)
    hermetrim = staticmethod(_trimcoef)
    hermeline = staticmethod(lambda off, scl: np.array([off, scl]))
    hermemulx = staticmethod(lambda c: _hermemulx(c))
    hermeweight = staticmethod(lambda x: np.exp(-np.asarray(x)**2 / 2.0))
    hermegauss = staticmethod(lambda n: _hermegauss(n))
    herme2poly = staticmethod(_herme2poly)
    poly2herme = staticmethod(_poly2herme)


# ===========================================================================
#  LAGUERRE
# ===========================================================================

def _lagval_scalar(x, c):
    c = _coef_list(c)
    if len(c) == 0: return 0.0
    if len(c) == 1: return float(c[0])
    l0, l1 = 1.0, 1.0 - float(x)
    result = c[0] * l0 + c[1] * l1
    for i in range(2, len(c)):
        l2 = ((2.0 * i - 1.0 - x) * l1 - (i - 1.0) * l0) / float(i)
        result += c[i] * l2
        l0, l1 = l1, l2
    return result

def _lagval(x, c):
    x = np.asarray(x)
    c_list = _flat_list(c)
    if len(c_list) == 0: return np.zeros_like(x)
    if len(c_list) == 1:
        return np.full(x.shape, c_list[0]) + np.zeros_like(x)
    l0 = np.ones(x.shape) + np.zeros_like(x)
    l1 = (np.ones(x.shape) if hasattr(x, 'shape') else 1.0) - x
    result = c_list[0] * l0 + c_list[1] * l1
    for i in range(2, len(c_list)):
        l2 = ((2.0 * i - 1.0 - x) * l1 - (i - 1.0) * l0) / float(i)
        result = result + c_list[i] * l2
        l0 = l1
        l1 = l2
    return result

def _lagadd(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] + c2[i] for i in range(n)])

def _lagsub(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n = max(len(c1), len(c2))
    while len(c1) < n: c1.append(0.0)
    while len(c2) < n: c2.append(0.0)
    return np.array([c1[i] - c2[i] for i in range(n)])

def _lagmul(c1, c2):
    c1 = _flat_list(c1)
    c2 = _flat_list(c2)
    n1, n2 = len(c1), len(c2)
    out_deg = n1 + n2 - 2
    n_pts = out_deg + 1
    pts = [float(kk) * 10.0 / max(n_pts - 1, 1) for kk in range(n_pts)]
    vals = []
    for xi in pts:
        vals.append(_lagval_scalar(xi, c1) * _lagval_scalar(xi, c2))
    return _lagfit(np.array(pts), np.array(vals), out_deg)

def _lagdiv(c1, c2):
    p1 = _lag2poly(c1)
    p2 = _lag2poly(c2)
    q, r = _polydiv(p1, p2)
    return _poly2lag(q), _poly2lag(r)

def _lagpow(c, n, maxpower=None):
    if maxpower is not None and n > maxpower:
        raise ValueError("Power exceeds maxpower")
    result = np.array([1.0])
    for _ in range(n):
        result = _lagmul(result, c)
    return result

def _lagder(c, m=1, scl=1, axis=0):
    if isinstance(m, float): raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0: raise ValueError("m must be non-negative")
    c = np.asarray(c)
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        for _ in range(m):
            n = c.shape[0]
            if n <= 1: c = np.zeros((1,) + c.shape[1:]); break
            dc = np.zeros((n - 1,) + c.shape[1:])
            for kk in range(n - 1):
                s = np.zeros(c.shape[1:])
                for j in range(kk + 1, n): s = s - c[j]
                dc[kk] = s * scl
            c = dc
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    for _ in range(m):
        n = len(c)
        if n <= 1: c = [0.0]; continue
        dc = [0.0] * (n - 1)
        for kk in range(n - 1):
            s = 0.0
            for j in range(kk + 1, n): s -= c[j]
            dc[kk] = s * scl
        c = dc
    return np.array(c)

def _lagint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    if isinstance(m, float) or (hasattr(m, 'is_integer') and not isinstance(m, int)):
        raise TypeError("m must be an integer, not float")
    m = int(m)
    if m < 0: raise ValueError("m must be non-negative")
    if isinstance(axis, float) or (hasattr(axis, 'is_integer') and not isinstance(axis, int)):
        raise TypeError("axis must be an integer")
    c = np.asarray(c)
    if isinstance(k, (int, float)): k = [k]
    if isinstance(lbnd, (list, tuple, np.ndarray)): raise ValueError("lbnd must be a scalar")
    if isinstance(scl, (list, tuple, np.ndarray)): raise ValueError("scl must be a scalar")
    if len(k) > m: raise ValueError("Too many integration constants")
    if c.ndim > 1:
        c = np.moveaxis(c, axis, 0)
        k_list = list(k) if k else []
        while len(k_list) < m: k_list.append(0)
        for step in range(m):
            n = c.shape[0]
            ic = np.zeros((n + 1,) + c.shape[1:])
            for j in range(n):
                ic[j] = ic[j] + c[j] * scl
                ic[j + 1] = ic[j + 1] - c[j] * scl
            val_at_lbnd = np.zeros(c.shape[1:])
            for j in range(n + 1):
                val_at_lbnd = val_at_lbnd + ic[j] * _lagval_scalar(lbnd, [0.0]*j + [1.0])
            ic[0] = ic[0] + k_list[step] - val_at_lbnd
            c = ic
        return np.moveaxis(c, 0, axis)
    c = _flat_arraylike_data(c.flatten())
    k_list = list(k) if k else []
    while len(k_list) < m: k_list.append(0)
    for step in range(m):
        n = len(c)
        if n == 0: c = [0.0]; continue
        ic = [0.0] * (n + 1)
        for j in range(n):
            ic[j] += c[j] * scl
            ic[j + 1] -= c[j] * scl
        val_at_lbnd = _lagval_scalar(lbnd, ic)
        ic[0] += k_list[step] - val_at_lbnd
        c = ic
    return np.array(c)

def _lagvander(x, deg):
    x = np.asarray(x)
    deg = int(deg)
    if deg < 0: raise ValueError("deg must be non-negative")
    shape = x.shape + (deg + 1,)
    v = np.zeros(shape)
    _set_last(v, 0, np.ones(x.shape) if len(x.shape) > 0 else 1.0)
    if deg >= 1: _set_last(v, 1, 1.0 - x)
    for i in range(2, deg + 1):
        _set_last(v, i, ((2.0 * i - 1.0 - x) * _get_last(v, i - 1) - (i - 1.0) * _get_last(v, i - 2)) / float(i))
    return v

def _lagfit(x, y, deg, w=None):
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    _validate_fit_args(x, y, w)
    if isinstance(deg, (list, tuple, np.ndarray)):
        deg_list = _flat_int_list(deg)
        if any(d < 0 for d in deg_list): raise ValueError("deg must be non-negative")
        if len(deg_list) == 0: raise TypeError("deg must be non-empty")
        max_deg = max(deg_list)
        V_full = _lagvander(x, max_deg)
        V = np.zeros((len(x), max_deg + 1))
        for d in sorted(set(deg_list)): V[:, d] = V_full[:, d]
    else:
        deg = int(deg)
        if deg < 0: raise ValueError("deg must be non-negative")
        V = _lagvander(x, deg)
    if w is not None:
        w = np.asarray(w, dtype='float64')
        V = V * w[:, None]
        if y.ndim == 1: y = y * w
        else: y = y * w[:, None]
    return np.linalg.lstsq(V, y)[0]

def _lagfromroots(roots):
    roots = _flat_list(roots)
    if len(roots) == 0: return np.array([1.0])
    c = _poly2lag(np.array([-roots[0], 1.0]))
    for i in range(1, len(roots)):
        c = _lagmul(c, _poly2lag(np.array([-roots[i], 1.0])))
    return c

def _lagroots(c):
    c = _flat_list(c)
    while len(c) > 1 and c[-1] == 0: c.pop()
    if len(c) <= 1: return np.array([])
    if len(c) == 2: return np.array([-c[0] / c[1]])
    poly_c = _lag2poly(c)
    return _polyroots(poly_c)

def _lagcompanion(c):
    poly_c = _lag2poly(c)
    return _polycompanion(_flat_arraylike_data(poly_c.flatten()))

def _lag2poly(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    l_prev = [1.0]
    if n == 1: return np.array([c[0]])
    l_curr = [1.0, -1.0]
    result = [0.0] * n
    for i in range(len(l_prev)): result[i] += c[0] * l_prev[i]
    for i in range(len(l_curr)):
        if i < len(result): result[i] += c[1] * l_curr[i]
    for kk in range(2, n):
        l_next = [0.0] * (len(l_curr) + 1)
        for i in range(len(l_curr)):
            l_next[i] += (2 * kk - 1) * l_curr[i] / kk
            l_next[i + 1] -= l_curr[i] / kk
        for i in range(len(l_prev)):
            l_next[i] -= (kk - 1) * l_prev[i] / kk
        while len(result) < len(l_next): result.append(0.0)
        for i in range(len(l_next)): result[i] += c[kk] * l_next[i]
        l_prev = l_curr
        l_curr = l_next
    return np.array(result)

def _poly2lag(c):
    c = _flat_list(c)
    n = len(c)
    if n == 0: return np.array([0.0])
    result = [0.0] * n
    x_lag = [[0.0] * n for _ in range(n)]
    x_lag[0][0] = 1.0
    if n > 1:
        x_lag[1][0] = 1.0
        x_lag[1][1] = -1.0
    for kk in range(2, n):
        prev = x_lag[kk - 1]
        cur = [0.0] * n
        for j in range(n):
            if prev[j] == 0: continue
            if j + 1 < n: cur[j + 1] -= (j + 1) * prev[j]
            cur[j] += (2 * j + 1) * prev[j]
            if j > 0: cur[j - 1] -= j * prev[j]
        x_lag[kk] = cur
    for kk in range(n):
        for j in range(n):
            result[j] += c[kk] * x_lag[kk][j]
    return np.array(result)


class Laguerre(ABCPolyBase):
    domain = np.array([0., 1.])
    window = np.array([0., 1.])
    basis_name = 'L'
    nickname = 'lag'
    _add_func = staticmethod(_lagadd)
    _sub_func = staticmethod(_lagsub)
    _mul_func = staticmethod(_lagmul)
    _div_func = staticmethod(_lagdiv)
    _pow_func = staticmethod(_lagpow)
    _val_func = staticmethod(_lagval)
    _int_func = staticmethod(_lagint)
    _der_func = staticmethod(_lagder)
    _fit_func = staticmethod(_lagfit)
    _fromroots_func = staticmethod(_lagfromroots)
    _roots_func = staticmethod(_lagroots)
    _vander_func = staticmethod(_lagvander)


def _laggauss(n):
    """Laguerre-Gauss quadrature."""
    import math
    n = int(n)
    # Build tridiagonal matrix for Laguerre: alpha_i = 2*i+1, beta_i = -(i+1)
    comp = np.zeros((n, n))
    for i in range(n):
        comp[i, i] = 2.0 * i + 1.0
        if i + 1 < n:
            comp[i, i + 1] = -(i + 1.0)
            comp[i + 1, i] = -(i + 1.0)
    x = np.linalg.eigvalsh(comp)
    x = np.sort(x)
    # Compute weights
    w = np.zeros(n)
    for i in range(n):
        xi = float(x[i])
        l_n1 = _lagval_scalar(xi, [0.0]*n + [1.0])
        if abs(l_n1) > 1e-300:
            w[i] = xi / ((n + 1.0)**2 * l_n1**2)
    return x, w

def _lagmulx(c):
    c = _flat_list(c)
    if len(c) == 1 and c[0] == 0:
        return np.array([0.0])
    n = len(c)
    result = [0.0] * (n + 1)
    for i in range(n):
        # x*L_i = -(i+1)*L_{i+1} + (2*i+1)*L_i - i*L_{i-1}
        result[i + 1] -= (i + 1) * c[i]
        result[i] += (2 * i + 1) * c[i]
        if i > 0:
            result[i - 1] -= i * c[i]
    while len(result) > 1 and abs(result[-1]) < 1e-15:
        result.pop()
    return np.array(result)

_lagval2d = _make_val2d(_lagval)
_lagval3d = _make_val3d(_lagval)
_laggrid2d = _make_grid2d(_lagval)
_laggrid3d = _make_grid3d(_lagval)
_lagvander2d = _make_vander2d(_lagvander)
_lagvander3d = _make_vander3d(_lagvander)


class laguerre:
    lagdomain = np.array([0., 1.])
    lagzero = np.array([0.])
    lagone = np.array([1.])
    lagx = np.array([1., -1.])
    Laguerre = Laguerre
    lagadd = staticmethod(_lagadd)
    lagsub = staticmethod(_lagsub)
    lagmul = staticmethod(_lagmul)
    lagdiv = staticmethod(_lagdiv)
    lagpow = staticmethod(_lagpow)
    lagval = staticmethod(_lagval)
    lagval2d = staticmethod(_lagval2d)
    lagval3d = staticmethod(_lagval3d)
    laggrid2d = staticmethod(_laggrid2d)
    laggrid3d = staticmethod(_laggrid3d)
    lagder = staticmethod(_lagder)
    lagint = staticmethod(_lagint)
    lagroots = staticmethod(_lagroots)
    lagvander = staticmethod(_lagvander)
    lagvander2d = staticmethod(_lagvander2d)
    lagvander3d = staticmethod(_lagvander3d)
    lagcompanion = staticmethod(_lagcompanion)
    lagfromroots = staticmethod(_lagfromroots)
    lagfit = staticmethod(_lagfit)
    lagtrim = staticmethod(_trimcoef)
    lagline = staticmethod(lambda off, scl: np.array([off + scl, -scl]))
    lagmulx = staticmethod(lambda c: _lagmulx(c))
    lagweight = staticmethod(lambda x: np.exp(-np.asarray(x)))
    laggauss = staticmethod(lambda n: _laggauss(n))
    lag2poly = staticmethod(_lag2poly)
    poly2lag = staticmethod(_poly2lag)


# ===========================================================================
#  polyutils additions
# ===========================================================================

def _pu_div(mul_func, c1, c2):
    c2 = _flat_list(c2)
    while len(c2) > 1 and c2[-1] == 0: c2.pop()
    if len(c2) == 0 or c2[-1] == 0:
        raise ZeroDivisionError("polynomial division by zero")
    return np.array([0.0]), np.array(c1)

def _pu_pow(mul_func, c, n, maxpower=None):
    if maxpower is not None and n > maxpower:
        raise ValueError("Power is too large")
    result = np.array([1.0])
    for _ in range(n):
        result = mul_func(result, c)
    return result

def _pu_vander_nd(vander_funcs, points, degrees):
    if len(vander_funcs) == 0 or len(degrees) == 0:
        raise ValueError("n_dims must be positive")
    if len(vander_funcs) != len(points):
        raise ValueError("n_dims != len(points)")
    if len(vander_funcs) != len(degrees):
        raise ValueError("n_dims != len(degrees)")
    return None

pu._div = _pu_div
pu._pow = _pu_pow
pu._vander_nd = _pu_vander_nd
