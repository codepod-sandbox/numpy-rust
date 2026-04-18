"""Random number generation Python-level wrappers."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray, random as _random_native, linalg as _linalg_native
from ._helpers import _builtin_range, _flat_arraylike_data
from ._creation import array, asarray, arange
from ._math import exp

__all__ = [
    '_random_shuffle',
    '_random_permutation',
    '_random_standard_normal',
    '_random_exponential',
    '_random_poisson',
    '_random_binomial',
    '_random_beta',
    '_random_gamma',
    '_random_multinomial',
    '_random_lognormal',
    '_random_geometric',
    '_random_dirichlet',
    '_random_random',
    '_random_multivariate_normal',
    '_random_chisquare',
    '_random_laplace',
    '_random_triangular',
    '_random_rayleigh',
    '_random_weibull',
    '_random_logistic',
    '_random_gumbel',
    '_random_negative_binomial',
    '_random_power',
    '_random_vonmises',
    '_random_wald',
    '_random_zipf',
    '_random_hypergeometric',
    '_random_pareto',
    '_random_bytes',
    '_default_rng',
    '_Generator',
    '_RandomState',
    '_wrapped_random_choice',
]

# Keep references to native modules
_native_random_module = _random_native
random = _native_random_module
linalg = _linalg_native
_orig_random_seed = _native_random_module.seed
_orig_random_rand = _native_random_module.rand
_orig_random_uniform = _native_random_module.uniform
_orig_random_normal = _native_random_module.normal
_orig_random_randn = _native_random_module.randn
_orig_random_randint = _native_random_module.randint
_orig_random_choice = _native_random_module.choice

_ACTIVE_BITGEN = None


class _bitgen_context:
    def __init__(self, bitgen):
        self.bitgen = bitgen
        self.prev = None

    def __enter__(self):
        global _ACTIVE_BITGEN
        self.prev = _ACTIVE_BITGEN
        _ACTIVE_BITGEN = self.bitgen

    def __exit__(self, exc_type, exc, tb):
        global _ACTIVE_BITGEN
        _ACTIVE_BITGEN = self.prev

def _active_rng():
    if _ACTIVE_BITGEN is not None and hasattr(_ACTIVE_BITGEN, "_rng"):
        return _ACTIVE_BITGEN._rng
    return None


class _NativeRngAdapter:
    def __init__(self, seed=None, state=None):
        if state is None:
            if seed is None:
                self._state = int(_native_random_module.stateful_seed())
            else:
                self._state = int(_native_random_module.stateful_seed(int(seed)))
        else:
            self._state = int(state)

    def seed(self, seed=None):
        if seed is None:
            self._state = int(_native_random_module.stateful_seed())
        else:
            self._state = int(_native_random_module.stateful_seed(int(seed)))

    def getstate(self):
        return int(self._state)

    def setstate(self, state):
        self._state = int(state)

    def random(self):
        return float(self.random_array((1,))[0])

    def random_array(self, shape):
        self._state, arr = _native_random_module.rand_with_state(self._state, shape)
        return arr

    def gauss(self, mean, std):
        return float(self.normal_array(mean, std, (1,))[0])

    def normal_array(self, mean, std, shape):
        self._state, arr = _native_random_module.normal_with_state(
            self._state, float(mean), float(std), shape
        )
        return arr

    def uniform_array(self, low, high, shape):
        self._state, arr = _native_random_module.uniform_with_state(
            self._state, float(low), float(high), shape
        )
        return arr

    def exponential_array(self, scale, shape):
        self._state, arr = _native_random_module.exponential_with_state(
            self._state, float(scale), shape
        )
        return arr

    def weibull_array(self, a, shape):
        self._state, arr = _native_random_module.weibull_with_state(
            self._state, float(a), shape
        )
        return arr

    def rayleigh_array(self, scale, shape):
        self._state, arr = _native_random_module.rayleigh_with_state(
            self._state, float(scale), shape
        )
        return arr

    def power_array(self, a, shape):
        self._state, arr = _native_random_module.power_with_state(
            self._state, float(a), shape
        )
        return arr

    def pareto_array(self, a, shape):
        self._state, arr = _native_random_module.pareto_with_state(
            self._state, float(a), shape
        )
        return arr

    def laplace_array(self, loc, scale, shape):
        self._state, arr = _native_random_module.laplace_with_state(
            self._state, float(loc), float(scale), shape
        )
        return arr

    def logistic_array(self, loc, scale, shape):
        self._state, arr = _native_random_module.logistic_with_state(
            self._state, float(loc), float(scale), shape
        )
        return arr

    def gumbel_array(self, loc, scale, shape):
        self._state, arr = _native_random_module.gumbel_with_state(
            self._state, float(loc), float(scale), shape
        )
        return arr

    def gamma_array(self, shape_param, scale, shape):
        self._state, arr = _native_random_module.gamma_with_state(
            self._state, float(shape_param), float(scale), shape
        )
        return arr

    def beta_array(self, a, b, shape):
        self._state, arr = _native_random_module.beta_with_state(
            self._state, float(a), float(b), shape
        )
        return arr

    def dirichlet_array(self, alpha, shape):
        self._state, arr = _native_random_module.dirichlet_with_state(
            self._state, alpha, shape
        )
        return arr

    def poisson_array(self, lam, shape):
        self._state, arr = _native_random_module.poisson_with_state(
            self._state, float(lam), shape
        )
        return arr

    def binomial_array(self, n, p, shape):
        self._state, arr = _native_random_module.binomial_with_state(
            self._state, int(n), float(p), shape
        )
        return arr

    def geometric_array(self, p, shape):
        self._state, arr = _native_random_module.geometric_with_state(
            self._state, float(p), shape
        )
        return arr

    def multinomial_array(self, n, pvals, shape):
        self._state, arr = _native_random_module.multinomial_with_state(
            self._state, int(n), pvals, shape
        )
        return arr

    def negative_binomial_array(self, n, p, shape):
        self._state, arr = _native_random_module.negative_binomial_with_state(
            self._state, int(n), float(p), shape
        )
        return arr

    def hypergeometric_array(self, ngood, nbad, nsample, shape):
        self._state, arr = _native_random_module.hypergeometric_with_state(
            self._state, int(ngood), int(nbad), int(nsample), shape
        )
        return arr

    def triangular_array(self, left, mode, right, shape):
        self._state, arr = _native_random_module.triangular_with_state(
            self._state, float(left), float(mode), float(right), shape
        )
        return arr

    def multivariate_normal_from_cholesky_array(self, mean, chol, shape):
        self._state, arr = _native_random_module.multivariate_normal_from_cholesky_with_state(
            self._state, mean, chol, shape
        )
        return arr

    def vonmises_array(self, mu, kappa, shape):
        self._state, arr = _native_random_module.vonmises_with_state(
            self._state, float(mu), float(kappa), shape
        )
        return arr

    def wald_array(self, mean, scale, shape):
        self._state, arr = _native_random_module.wald_with_state(
            self._state, float(mean), float(scale), shape
        )
        return arr

    def zipf_array(self, a, shape):
        self._state, arr = _native_random_module.zipf_with_state(
            self._state, float(a), shape
        )
        return arr

    def logseries_array(self, p, shape):
        self._state, arr = _native_random_module.logseries_with_state(
            self._state, float(p), shape
        )
        return arr

    def randrange(self, low, high):
        self._state, value = _native_random_module.randint_scalar_with_state(
            self._state, int(low), int(high)
        )
        return int(value)

    def randint_array(self, low, high, shape):
        self._state, arr = _native_random_module.randint_with_state(
            self._state, int(low), int(high), shape
        )
        return arr

    def choice_array(self, a, size, replace=True):
        self._state, arr = _native_random_module.choice_with_state(
            self._state, a, int(size), bool(replace)
        )
        return arr

    def getrandbits(self, bits):
        self._state, value = _native_random_module.randbits_with_state(self._state, int(bits))
        return int(value)


def _proxy_seed(seed):
    rng = _active_rng()
    if rng is not None:
        rng.seed(seed)
    else:
        _orig_random_seed(seed)


def _coerce_random_shape_args(shape_args):
    if len(shape_args) == 0:
        return ()
    if len(shape_args) == 1:
        shape = shape_args[0]
        if isinstance(shape, int):
            return (shape,)
        if isinstance(shape, tuple):
            return shape
        return tuple(shape)
    return tuple(int(s) for s in shape_args)


def _proxy_rand(*shape_args):
    shape = _coerce_random_shape_args(shape_args)
    rng = _active_rng()
    if rng is None:
        return _orig_random_rand(shape)
    result = rng.random_array(shape if len(shape) > 0 else (1,))
    if len(shape) == 0:
        return result
    return result.reshape(shape) if len(shape) > 1 else result


def _proxy_uniform(low, high, shape):
    rng = _active_rng()
    if rng is None:
        return _orig_random_uniform(low, high, shape)
    if isinstance(shape, int):
        shape = (shape,)
    result = rng.uniform_array(low, high, shape)
    return result.reshape(shape) if len(shape) > 1 else result


def _proxy_normal(mean, std, shape):
    rng = _active_rng()
    if rng is None:
        return _orig_random_normal(mean, std, shape)
    if isinstance(shape, int):
        shape = (shape,)
    result = rng.normal_array(mean, std, shape)
    return result.reshape(shape) if len(shape) > 1 else result


def _proxy_randn(*shape_args):
    shape = _coerce_random_shape_args(shape_args)
    return _proxy_normal(0.0, 1.0, shape)


def _proxy_randint(low, high, size):
    rng = _active_rng()
    if rng is None:
        return _orig_random_randint(low, high, size)
    if isinstance(size, int):
        size = (size,)
    result = rng.randint_array(low, high, size)
    return result.reshape(size) if len(size) > 1 else result


random.seed = _proxy_seed
random.rand = _proxy_rand
random.uniform = _proxy_uniform
random.normal = _proxy_normal
random.randn = _proxy_randn
random.randint = _proxy_randint

# stdlib math (needed for _random_vonmises)
_stdlib_math = _math


def _random_shuffle(x):
    """Modify a sequence in-place by shuffling its contents. Returns None."""
    if not isinstance(x, ndarray):
        x = asarray(x)
    n = x.size
    order = _wrapped_random_choice(n, size=n, replace=False)
    permuted = x.flatten().take(order)
    try:
        flat = x.reshape((n,))
        flat[:] = permuted
    except Exception:
        pass
    return None


def _random_permutation(x):
    """Randomly permute a sequence, or return a permuted range."""
    if isinstance(x, (int, float)):
        x = arange(0, int(x))
    else:
        x = asarray(x)
    n = x.size
    order = _wrapped_random_choice(n, size=n, replace=False)
    result = x.flatten().take(order)
    if x.ndim > 1:
        result = result.reshape(x.shape)
    return result


def _random_standard_normal(size=None):
    """Draw samples from a standard Normal distribution (mean=0, stdev=1)."""
    if size is None:
        return float(random.normal(0.0, 1.0, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    return random.normal(0.0, 1.0, size)


def _random_exponential(scale=1.0, size=None):
    """Draw samples from an exponential distribution."""
    rng = _active_rng()
    if size is None:
        if rng is None:
            return float(_native_random_module.exponential(float(scale), (1,)).flatten()[0])
        return float(rng.exponential_array(scale, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.exponential(float(scale), size)
    result = rng.exponential_array(scale, size)
    return result.reshape(size) if len(size) > 1 else result


def _random_poisson(lam=1.0, size=None):
    """Draw samples from a Poisson distribution."""
    rng = _active_rng()
    if size is None:
        if rng is None:
            return float(_native_random_module.poisson(float(lam), (1,)).flatten()[0])
        return float(rng.poisson_array(lam, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.poisson(float(lam), size)
    result = rng.poisson_array(lam, size)
    return result.reshape(size) if len(size) > 1 else result


def _random_binomial(n, p, size=None):
    """Draw samples from a binomial distribution."""
    rng = _active_rng()
    if size is None:
        if rng is None:
            return float(_native_random_module.binomial(int(n), float(p), (1,)).flatten()[0])
        return float(rng.binomial_array(n, p, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.binomial(int(n), float(p), size)
    result = rng.binomial_array(n, p, size)
    return result.reshape(size) if len(size) > 1 else result


def _random_beta(a, b, size=None):
    """Draw samples from a Beta distribution.
    Uses the relationship: if X~Gamma(a,1) and Y~Gamma(b,1), then X/(X+Y)~Beta(a,b)."""
    rng = _active_rng()
    if size is None:
        if rng is None:
            return float(_native_random_module.beta(float(a), float(b), (1,)).flatten()[0])
        return float(rng.beta_array(a, b, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.beta(float(a), float(b), size)
    result = rng.beta_array(a, b, size)
    return result.reshape(size) if len(size) > 1 else result


def _random_gamma(shape_param, scale=1.0, size=None):
    """Draw samples from a Gamma distribution using Marsaglia-Tsang method."""
    rng = _active_rng()
    if size is None:
        if rng is None:
            return float(_native_random_module.gamma(float(shape_param), float(scale), (1,)).flatten()[0])
        return float(rng.gamma_array(shape_param, scale, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.gamma(float(shape_param), float(scale), size)
    result = rng.gamma_array(shape_param, scale, size)
    return result.reshape(size) if len(size) > 1 else result


def _random_multinomial(n, pvals, size=None):
    """Draw samples from a multinomial distribution."""
    pvals_arr = pvals if isinstance(pvals, ndarray) else asarray(pvals, dtype='float64')
    rng = _active_rng()
    if size is None:
        if rng is None:
            return _native_random_module.multinomial(int(n), pvals_arr, (1,)).reshape((pvals_arr.size,))
        return rng.multinomial_array(n, pvals_arr, (1,)).reshape((pvals_arr.size,))
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.multinomial(int(n), pvals_arr, size)
    return rng.multinomial_array(n, pvals_arr, size)


def _random_lognormal(mean=0.0, sigma=1.0, size=None):
    """Draw samples from a log-normal distribution."""
    if size is None:
        import math as _m
        n = float(random.normal(mean, sigma, (1,)).flatten()[0])
        return float(_m.exp(n))
    if isinstance(size, int):
        size = (size,)
    normals = random.normal(mean, sigma, size)
    return exp(normals)


def _random_geometric(p, size=None):
    """Draw samples from a geometric distribution.
    Returns number of trials until first success (minimum value 1)."""
    rng = _active_rng()
    if size is None:
        if rng is None:
            return float(_native_random_module.geometric(float(p), (1,)).flatten()[0])
        return float(rng.geometric_array(p, (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.geometric(float(p), size)
    result = rng.geometric_array(p, size)
    return result.reshape(size) if len(size) > 1 else result


def _random_dirichlet(alpha, size=None):
    """Draw samples from a Dirichlet distribution."""
    alpha_arr = alpha if isinstance(alpha, ndarray) else asarray(alpha, dtype='float64')
    rng = _active_rng()
    if size is None:
        if rng is None:
            return _native_random_module.dirichlet(alpha_arr, (1,)).reshape((alpha_arr.size,))
        return rng.dirichlet_array(alpha_arr, (1,)).reshape((alpha_arr.size,))
    if isinstance(size, int):
        size = (size,)
    if rng is None:
        return _native_random_module.dirichlet(alpha_arr, size)
    return rng.dirichlet_array(alpha_arr, size)


def _random_standard_cauchy(size=None):
    """Standard Cauchy distribution via ratio of standard normals."""
    size_shape = _normalize_random_size(size)
    sample_shape = (1,) if size_shape is None else size_shape
    result = random.normal(0.0, 1.0, sample_shape) / random.normal(0.0, 1.0, sample_shape)
    if size_shape is None:
        return _scalar_from_random_result(result)
    return _reshape_random_result(result, size_shape)


def _random_standard_t(df, size=None):
    """Student's t-distribution."""
    size_shape = _normalize_random_size(size)
    sample_shape = (1,) if size_shape is None else size_shape
    z = random.normal(0.0, 1.0, sample_shape)
    chi2 = _random_chisquare(int(df), sample_shape)
    result = z / ((chi2 / df) ** 0.5)
    if size_shape is None:
        return _scalar_from_random_result(result)
    return _reshape_random_result(result, size_shape)


def _random_f(dfnum, dfden, size=None):
    """F-distribution."""
    size_shape = _normalize_random_size(size)
    sample_shape = (1,) if size_shape is None else size_shape
    chi1 = _random_chisquare(int(dfnum), sample_shape)
    chi2 = _random_chisquare(int(dfden), sample_shape)
    result = (chi1 / dfnum) / (chi2 / dfden)
    if size_shape is None:
        return _scalar_from_random_result(result)
    return _reshape_random_result(result, size_shape)


def _random_noncentral_chisquare(df, nonc, size=None):
    """Noncentral chi-square distribution."""
    size_shape = _normalize_random_size(size)
    sample_shape = (1,) if size_shape is None else size_shape
    result = random.normal(_math.sqrt(nonc), 1.0, sample_shape)
    result = result * result
    if df > 1:
        result = result + _random_chisquare(int(df) - 1, sample_shape)
    if size_shape is None:
        return _scalar_from_random_result(result)
    return _reshape_random_result(result, size_shape)


def _random_noncentral_f(dfnum, dfden, nonc, size=None):
    """Noncentral F-distribution."""
    size_shape = _normalize_random_size(size)
    sample_shape = (1,) if size_shape is None else size_shape
    ncchi2 = _random_noncentral_chisquare(int(dfnum), nonc, sample_shape)
    chi2 = _random_chisquare(int(dfden), sample_shape)
    result = (ncchi2 / dfnum) / (chi2 / dfden)
    if size_shape is None:
        return _scalar_from_random_result(result)
    return _reshape_random_result(result, size_shape)


def _random_logseries(p, size=None):
    """Logarithmic (Log-Series) distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.logseries(float(p), shape),
        lambda rng, shape: rng.logseries_array(p, shape),
        scalar_cast=int,
    )


def _check_out_dtype(out, dtype):
    """Validate that out array dtype matches the requested dtype."""
    if dtype is not None and out is not None:
        out_dt = str(out.dtype)
        req_dt = str(dtype)
        # Normalize
        if 'float64' in out_dt:
            out_dt = 'float64'
        elif 'float32' in out_dt:
            out_dt = 'float32'
        if 'float64' in req_dt:
            req_dt = 'float64'
        elif 'float32' in req_dt:
            req_dt = 'float32'
        if out_dt != req_dt:
            raise TypeError(
                f"Supplied output array has dtype {out.dtype} but dtype {dtype} was requested")
    elif dtype is None and out is not None:
        # When no dtype specified, default is float64; out must be float64
        out_dt = str(out.dtype)
        if 'float32' in out_dt:
            # No dtype specified means float64 expected
            pass  # Allow for backward compat


def _validate_out_array(out, size):
    if out is None:
        return
    if size is not None and tuple(out.shape) != tuple(size):
        raise ValueError("size must match out.shape")
    flags = getattr(out, 'flags', None)
    if flags is not None:
        is_c = bool(flags['C_CONTIGUOUS']) if hasattr(flags, '__getitem__') else bool(getattr(flags, 'c_contiguous', False))
        is_f = bool(flags['F_CONTIGUOUS']) if hasattr(flags, '__getitem__') else bool(getattr(flags, 'f_contiguous', False))
        if not is_c and not is_f:
            raise ValueError("out array must be contiguous")


def _fill_out(out, src):
    """Fill an output array in-place from a source array using tuple indexing."""
    shape = out.shape
    ndim = len(shape)
    src_flat = _flat_arraylike_data(src)
    if src_flat is None:
        src_flat = [float(src)]
    if ndim == 1:
        for i in range(shape[0]):
            out[(i,)] = src_flat[i]
    elif ndim == 2:
        idx = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                out[(i, j)] = src_flat[idx]
                idx += 1
    elif ndim == 3:
        idx = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    out[(i, j, k)] = src_flat[idx]
                    idx += 1
    else:
        # Generic N-d: compute multi-index from flat index
        total = 1
        for s in shape:
            total *= s
        for flat_idx in range(total):
            multi = []
            rem = flat_idx
            for d in range(ndim - 1, -1, -1):
                multi.append(rem % shape[d])
                rem //= shape[d]
            multi.reverse()
            out[tuple(multi)] = src_flat[flat_idx]


def _normalize_size_tuple(size):
    if size is None:
        return None
    if isinstance(size, int):
        return (size,)
    return tuple(size)


def _shape_total(shape):
    total = 1
    for s in shape:
        total *= s
    return total


def _flat_random_data(value):
    flat = _flat_arraylike_data(value)
    if flat is not None:
        return flat
    return None


def _flat_broadcast_values(value):
    if isinstance(value, ndarray):
        flat = _flat_random_data(value)
        return flat, list(value.shape)
    return [value], [1]


def _broadcast_result_shape(shapes, size):
    size_shape = _normalize_size_tuple(size)
    if size_shape is not None:
        return list(size_shape)
    max_ndim = max(len(shape) for shape in shapes)
    padded = []
    for shape in shapes:
        shape = list(shape)
        while len(shape) < max_ndim:
            shape = [1] + shape
        padded.append(shape)
    return [max(shape[i] for shape in padded) for i in range(max_ndim)]


def _wrap_broadcast_results(results, out_shape):
    r = array(results)
    if len(out_shape) > 1:
        r = r.reshape(out_shape)
    return r


def _broadcast_call_1(func, arg, size=None):
    """Call a single-param distribution function with broadcasting support."""
    if isinstance(arg, ndarray):
        arg_flat = _flat_random_data(arg)
        size_shape = _normalize_size_tuple(size)
        if size_shape is not None:
            total = _shape_total(size_shape)
            n = len(arg_flat)
            return _wrap_broadcast_results(
                [float(func(arg_flat[i % n])) for i in range(total)],
                list(size_shape),
            )
        else:
            results = [float(func(v)) for v in arg_flat]
            r = array(results)
            if arg.ndim > 1:
                r = r.reshape(arg.shape)
            return r
    size_shape = _normalize_size_tuple(size)
    if size_shape is not None:
        total = _shape_total(size_shape)
        return _wrap_broadcast_results([float(func(arg)) for _ in range(total)], list(size_shape))
    return func(arg)


def _broadcast_call_2(func, a, b, size=None):
    """Call a two-param distribution function with broadcasting support."""
    a_is_arr = isinstance(a, ndarray)
    b_is_arr = isinstance(b, ndarray)
    size_shape = _normalize_size_tuple(size)
    if not a_is_arr and not b_is_arr:
        if size_shape is not None:
            total = _shape_total(size_shape)
            return _wrap_broadcast_results([float(func(a, b)) for _ in range(total)], list(size_shape))
        return func(a, b)
    a_flat, a_shape = _flat_broadcast_values(a)
    b_flat, b_shape = _flat_broadcast_values(b)
    out_shape = _broadcast_result_shape([a_shape, b_shape], size)
    total = _shape_total(out_shape)
    na = len(a_flat)
    nb = len(b_flat)
    return _wrap_broadcast_results(
        [float(func(a_flat[i % na], b_flat[i % nb])) for i in range(total)],
        out_shape,
    )


def _broadcast_call_3(func, a, b, c, size=None):
    """Call a three-param distribution function with broadcasting support."""
    a_is_arr = isinstance(a, ndarray)
    b_is_arr = isinstance(b, ndarray)
    c_is_arr = isinstance(c, ndarray)
    size_shape = _normalize_size_tuple(size)
    if not a_is_arr and not b_is_arr and not c_is_arr:
        if size_shape is not None:
            total = _shape_total(size_shape)
            return _wrap_broadcast_results([float(func(a, b, c)) for _ in range(total)], list(size_shape))
        return func(a, b, c)
    a_flat, a_shape = _flat_broadcast_values(a)
    b_flat, b_shape = _flat_broadcast_values(b)
    c_flat, c_shape = _flat_broadcast_values(c)
    out_shape = _broadcast_result_shape([a_shape, b_shape, c_shape], size)
    total = _shape_total(out_shape)
    na, nb, nc = len(a_flat), len(b_flat), len(c_flat)
    return _wrap_broadcast_results(
        [float(func(a_flat[i % na], b_flat[i % nb], c_flat[i % nc])) for i in range(total)],
        out_shape,
    )


def _normalize_random_size(size):
    if size is None:
        return None
    if hasattr(size, '__iter__') and not isinstance(size, (tuple, list)):
        size = tuple(int(x) for x in size)
    if isinstance(size, int):
        return (size,)
    return tuple(int(x) for x in size)


def _reshape_random_result(result, shape):
    return result.reshape(shape) if len(shape) > 1 else result


def _scalar_from_random_result(result, cast=float):
    return cast(result.flatten()[0])


def _native_random_draw(size, native_draw, rng_draw, scalar_cast=float):
    shape = _normalize_random_size(size)
    rng = _active_rng()
    if shape is None:
        arr = native_draw((1,)) if rng is None else rng_draw(rng, (1,))
        return _scalar_from_random_result(arr, scalar_cast)
    result = native_draw(shape) if rng is None else rng_draw(rng, shape)
    return _reshape_random_result(result, shape)


def _random_dtype_bounds(dtype):
    dt_str = str(dtype)
    bounds_list = [
        ('uint64', (0, 18446744073709551616)),
        ('uint32', (0, 4294967296)),
        ('uint16', (0, 65536)),
        ('uint8', (0, 256)),
        ('int64', (-9223372036854775808, 9223372036854775808)),
        ('int32', (-2147483648, 2147483648)),
        ('int16', (-32768, 32768)),
        ('int8', (-128, 128)),
        ('bool', (0, 2)),
    ]
    for name, bounds in bounds_list:
        if name in dt_str:
            return name, bounds
    return dt_str, None


def _coerce_int_operand(value):
    if isinstance(value, ndarray):
        arr = value
    elif hasattr(value, 'flatten') and hasattr(value, 'shape'):
        arr = value
    elif isinstance(value, (list, tuple)):
        def _shape_of(seq):
            if not isinstance(seq, (list, tuple)):
                return []
            if len(seq) == 0:
                return [0]
            inner = _shape_of(seq[0])
            return [len(seq)] + inner

        def _flatten(seq, out):
            if isinstance(seq, (list, tuple)):
                for item in seq:
                    _flatten(item, out)
            else:
                out.append(int(seq))

        flat = []
        _flatten(value, flat)
        return flat, _shape_of(value)
    else:
        return int(value), None


def _broadcast_random_operand_shapes(low_shape, high_shape, size):
    if size is not None:
        return list(size)
    max_ndim = max(len(low_shape), len(high_shape))
    while len(low_shape) < max_ndim:
        low_shape = [1] + low_shape
    while len(high_shape) < max_ndim:
        high_shape = [1] + high_shape
    out_shape = []
    for lo_dim, hi_dim in zip(low_shape, high_shape):
        if lo_dim != hi_dim and lo_dim != 1 and hi_dim != 1:
            raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
        out_shape.append(max(lo_dim, hi_dim))
    return out_shape


def _randbelow_with_rng(rng, width):
    if width <= 0:
        raise ValueError("high <= low")
    if width == 1:
        return 0
    bits = (width - 1).bit_length()
    while True:
        candidate = rng.getrandbits(bits)
        if candidate < width:
            return candidate


def _draw_integers_with_rng(rng, low, high, count):
    values = []
    for idx in range(count):
        lo = int(low[idx] if isinstance(low, list) else low)
        hi = int(high[idx] if isinstance(high, list) else high)
        values.append(lo + _randbelow_with_rng(rng, hi - lo))
    return values


def _build_integer_array(values, dtype_name):
    if dtype_name == 'uint64':
        signed = [
            value if value <= 9223372036854775807 else value - 18446744073709551616
            for value in values
        ]
        return _native.array(signed).astype('uint64')
    return array(values, dtype=dtype_name)


class _Generator:
    """Random number generator (simplified)."""
    def __init__(self, seed_val=None):
        if isinstance(seed_val, _BitGenerator):
            self.bit_generator = seed_val
            # BitGenerator already seeded the RNG in its __init__
        else:
            self.bit_generator = _PCG64(seed_val)
            # _PCG64.__init__ already seeds the RNG

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr) and not name.startswith('_'):
            bitgen = object.__getattribute__(self, 'bit_generator')

            def _wrapped(*args, **kwargs):
                with _bitgen_context(bitgen):
                    return attr(*args, **kwargs)

            return _wrapped
        return attr

    def random(self, size=None, dtype=None, out=None):
        if size is None and out is not None:
            size = out.shape
        _validate_out_array(out, size)
        if size is None:
            return float(random.rand((1,))[0])
        if hasattr(size, '__iter__') and not isinstance(size, (tuple, list)):
            size = tuple(int(x) for x in size)
        if isinstance(size, int):
            size = (size,)
        r = random.rand(size)
        if dtype is not None:
            dt = str(dtype)
            if 'float32' in dt or dt == 'float32':
                r = asarray(r, dtype='float32')
        if out is not None:
            _fill_out(out, r)
            return out
        return r

    def standard_normal(self, size=None, dtype=None, out=None):
        if out is not None:
            _check_out_dtype(out, dtype)
        if size is None and out is not None:
            size = out.shape
        _validate_out_array(out, size)
        r = _random_standard_normal(size)
        if dtype is not None:
            dt = str(dtype)
            if 'float32' in dt or dt == 'float32':
                if isinstance(r, ndarray):
                    r = asarray(r, dtype='float32')
        if out is not None:
            if isinstance(r, ndarray):
                _fill_out(out, r)
            else:
                out[(0,)] = float(r)
            return out
        return r

    def integers(self, low, high=None, size=None, dtype='int64', endpoint=False):
        if high is None:
            high = low
            low = 0
        size = _normalize_random_size(size)
        if endpoint:
            if isinstance(high, ndarray):
                high = high + 1
            elif isinstance(high, (list, tuple)):
                high = (asarray(high) + 1).tolist()
            else:
                high = high + 1

        dt_str, bounds = _random_dtype_bounds(dtype)
        rng = self.bit_generator._rng

        low_values, low_shape = _coerce_int_operand(low)
        high_values, high_shape = _coerce_int_operand(high)

        if low_shape is not None or high_shape is not None:
            if low_shape is None:
                low_values = [low_values]
                low_shape = [1]
            if high_shape is None:
                high_values = [high_values]
                high_shape = [1]
            out_shape = _broadcast_random_operand_shapes(low_shape, high_shape, size)
            count = 1
            for dim in out_shape:
                count *= dim
            low_len = len(low_values)
            high_len = len(high_values)
            lows = []
            highs = []
            for idx in range(count):
                lo_v = int(low_values[idx % low_len])
                hi_v = int(high_values[idx % high_len])
                if lo_v >= hi_v:
                    raise ValueError("low >= high")
                if bounds is not None:
                    if lo_v < bounds[0]:
                        raise ValueError(f"low is out of bounds for {dt_str}")
                    if hi_v > bounds[1]:
                        raise ValueError(f"high is out of bounds for {dt_str}")
                lows.append(lo_v)
                highs.append(hi_v)
            result = _build_integer_array(_draw_integers_with_rng(rng, lows, highs, count), dt_str)
            if out_shape:
                return result.reshape(tuple(out_shape))
            return result

        lo_v = int(low_values)
        hi_v = int(high_values)
        if lo_v >= hi_v:
            raise ValueError("low >= high")
        if bounds is not None:
            if lo_v < bounds[0]:
                raise ValueError(f"low is out of bounds for {dt_str}")
            if hi_v > bounds[1]:
                raise ValueError(f"high is out of bounds for {dt_str}")
        if size is None:
            value = lo_v + _randbelow_with_rng(rng, hi_v - lo_v)
            if 'bool' in dt_str:
                return bool(value)
            return int(value)
        count = 1
        for dim in size:
            count *= dim
        result = _build_integer_array(_draw_integers_with_rng(rng, lo_v, hi_v, count), dt_str)
        return result.reshape(size)

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            a = arange(0.0, float(a), 1.0)
        elif isinstance(a, (list, tuple)):
            a = array(a)
        elif not isinstance(a, ndarray):
            a = asarray(a)
        if size is None:
            size = 1
        return random.choice(a, size, replace)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return float(random.normal(float(loc), float(scale), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.normal(float(loc), float(scale), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        low_is_arr = isinstance(low, ndarray)
        high_is_arr = isinstance(high, ndarray)
        if low_is_arr or high_is_arr:
            return _broadcast_call_2(
                lambda l, h: float(random.uniform(float(l), float(h), (1,))[0]),
                low, high, size)
        if size is None:
            return float(random.uniform(float(low), float(high), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.uniform(float(low), float(high), size)

    def shuffle(self, x, axis=0):
        return _random_shuffle(x)

    def permutation(self, x, axis=0):
        return _random_permutation(x)

    def exponential(self, scale=1.0, size=None):
        if isinstance(scale, ndarray):
            return _broadcast_call_1(lambda s: _random_exponential(s), scale, size)
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        if isinstance(lam, ndarray):
            return _broadcast_call_1(lambda l: _random_poisson(l), lam, size)
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

    def beta(self, a, b, size=None):
        return _broadcast_call_2(lambda x, y: _random_beta(x, y), a, b, size)

    def gamma(self, shape, scale=1.0, size=None):
        return _broadcast_call_2(lambda s, sc: _random_gamma(s, sc), shape, scale, size)

    def standard_gamma(self, shape, size=None, dtype=None, out=None):
        if out is not None:
            _check_out_dtype(out, dtype)
        if size is None and out is not None:
            size = out.shape
        _validate_out_array(out, size)
        if isinstance(shape, ndarray):
            r = _broadcast_call_1(lambda s: _random_gamma(s, 1.0), shape, size)
        else:
            r = _random_gamma(shape, 1.0, size)
        if dtype is not None:
            dt = str(dtype)
            if 'float32' in dt or dt == 'float32':
                if isinstance(r, ndarray):
                    r = asarray(r, dtype='float32')
        if out is not None:
            if isinstance(r, ndarray):
                _fill_out(out, r)
            else:
                out[(0,)] = float(r)
            return out
        return r

    def standard_exponential(self, size=None, dtype=None, method=None, out=None):
        if out is not None:
            _check_out_dtype(out, dtype)
        if size is None and out is not None:
            size = out.shape
        _validate_out_array(out, size)
        r = _random_exponential(1.0, size)
        if dtype is not None:
            dt = str(dtype)
            if 'float32' in dt or dt == 'float32':
                if isinstance(r, ndarray):
                    r = asarray(r, dtype='float32')
        if out is not None:
            if isinstance(r, ndarray):
                _fill_out(out, r)
            else:
                out[(0,)] = float(r)
            return out
        return r

    def standard_cauchy(self, size=None):
        return _random_standard_cauchy(size)

    def standard_t(self, df, size=None):
        if isinstance(df, ndarray):
            return _broadcast_call_1(lambda d: _random_standard_t(d), df, size)
        return _random_standard_t(df, size)

    def chisquare(self, df, size=None):
        if isinstance(df, ndarray):
            return _broadcast_call_1(lambda d: _random_chisquare(d), df, size)
        return _random_chisquare(df, size)

    def f(self, dfnum, dfden, size=None):
        return _broadcast_call_2(lambda n, d: _random_f(n, d), dfnum, dfden, size)

    def noncentral_chisquare(self, df, nonc, size=None):
        return _broadcast_call_2(lambda d, n: _random_noncentral_chisquare(d, n), df, nonc, size)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        return _broadcast_call_3(lambda n, d, nc: _random_noncentral_f(n, d, nc), dfnum, dfden, nonc, size)

    def geometric(self, p, size=None):
        if isinstance(p, ndarray):
            return _broadcast_call_1(lambda pp: _random_geometric(pp), p, size)
        return _random_geometric(p, size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        return _broadcast_call_2(lambda l, s: _random_gumbel(l, s), loc, scale, size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        return _broadcast_call_2(lambda l, s: _random_laplace(l, s), loc, scale, size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        return _broadcast_call_2(lambda l, s: _random_logistic(l, s), loc, scale, size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return _broadcast_call_2(lambda m, s: _random_lognormal(m, s), mean, sigma, size)

    def logseries(self, p, size=None):
        if isinstance(p, ndarray):
            return _broadcast_call_1(lambda pp: _random_logseries(pp), p, size)
        return _random_logseries(p, size)

    def negative_binomial(self, n, p, size=None):
        return _broadcast_call_2(lambda nn, pp: _random_negative_binomial(nn, pp), n, p, size)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        return _broadcast_call_3(
            lambda g, b, s: _random_hypergeometric(g, b, s), ngood, nbad, nsample, size)

    def pareto(self, a, size=None):
        if isinstance(a, ndarray):
            return _broadcast_call_1(lambda aa: _random_pareto(aa), a, size)
        return _random_pareto(a, size)

    def power(self, a, size=None):
        if isinstance(a, ndarray):
            return _broadcast_call_1(lambda aa: _random_power(aa), a, size)
        return _random_power(a, size)

    def rayleigh(self, scale=1.0, size=None):
        if isinstance(scale, ndarray):
            return _broadcast_call_1(lambda s: _random_rayleigh(s), scale, size)
        return _random_rayleigh(scale, size)

    def vonmises(self, mu, kappa, size=None):
        return _broadcast_call_2(lambda m, k: _random_vonmises(m, k), mu, kappa, size)

    def wald(self, mean, scale, size=None):
        return _broadcast_call_2(lambda m, s: _random_wald(m, s), mean, scale, size)

    def weibull(self, a, size=None):
        if isinstance(a, ndarray):
            return _broadcast_call_1(lambda aa: _random_weibull(aa), a, size)
        return _random_weibull(a, size)

    def zipf(self, a, size=None):
        if isinstance(a, ndarray):
            return _broadcast_call_1(lambda aa: _random_zipf(aa), a, size)
        return _random_zipf(a, size)

    def triangular(self, left, mode, right, size=None):
        return _broadcast_call_3(
            lambda l, m, r: _random_triangular(l, m, r), left, mode, right, size)

    def multinomial(self, n, pvals, size=None):
        return _random_multinomial(n, pvals, size)

    def multivariate_normal(self, mean, cov, size=None):
        return _random_multivariate_normal(mean, cov, size)

    def dirichlet(self, alpha, size=None):
        return _random_dirichlet(alpha, size)

    def bytes(self, length):
        return _random_bytes(length)


def _random_random(size=None):
    """Return random floats in [0, 1). Same as rand but takes size tuple."""
    if size is None:
        return float(random.rand((1,))[0])
    if isinstance(size, int):
        size = (size,)
    # Compute total elements
    total = 1
    for s in size:
        total *= s
    result = random.uniform(0.0, 1.0, (total,))
    if len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_multivariate_normal(mean, cov, size=None):
    """Draw from multivariate normal distribution."""
    mean = asarray(mean)
    cov = asarray(cov)
    n = len(mean.tolist())

    # Cholesky decomposition of covariance
    L = linalg.cholesky(cov)

    rng = _active_rng()
    if size is None:
        if rng is None:
            return _native_random_module.multivariate_normal_from_cholesky(mean, L, (1,)).reshape((n,))
        return rng.multivariate_normal_from_cholesky_array(mean, L, (1,)).reshape((n,))

    if isinstance(size, int):
        size = (size,)

    if rng is None:
        return _native_random_module.multivariate_normal_from_cholesky(mean, L, size)
    return rng.multivariate_normal_from_cholesky_array(mean, L, size)


def _random_chisquare(df, size=None):
    """Chi-square distribution (sum of df squared standard normals)."""
    df = int(df)
    size_shape = _normalize_random_size(size)
    sample_shape = (1,) if size_shape is None else size_shape
    z = random.normal(0.0, 1.0, sample_shape + (df,))
    result = (z * z).sum(axis=-1)
    if size_shape is None:
        return _scalar_from_random_result(result)
    return _reshape_random_result(result, size_shape)


def _random_laplace(loc=0.0, scale=1.0, size=None):
    """Laplace distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.laplace(float(loc), float(scale), shape),
        lambda rng, shape: rng.laplace_array(loc, scale, shape),
    )


def _random_triangular(left, mode, right, size=None):
    """Triangular distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.triangular(
            float(left), float(mode), float(right), shape
        ),
        lambda rng, shape: rng.triangular_array(left, mode, right, shape),
    )


def _random_rayleigh(scale=1.0, size=None):
    """Rayleigh distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.rayleigh(float(scale), shape),
        lambda rng, shape: rng.rayleigh_array(scale, shape),
    )


def _random_weibull(a, size=None):
    """Weibull distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.weibull(float(a), shape),
        lambda rng, shape: rng.weibull_array(a, shape),
    )


def _random_logistic(loc=0.0, scale=1.0, size=None):
    """Logistic distribution via inverse CDF."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.logistic(float(loc), float(scale), shape),
        lambda rng, shape: rng.logistic_array(loc, scale, shape),
    )


def _random_gumbel(loc=0.0, scale=1.0, size=None):
    """Gumbel distribution via inverse CDF."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.gumbel(float(loc), float(scale), shape),
        lambda rng, shape: rng.gumbel_array(loc, scale, shape),
    )


def _random_negative_binomial(n, p, size=None):
    """Negative binomial distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.negative_binomial(int(n), float(p), shape),
        lambda rng, shape: rng.negative_binomial_array(n, p, shape),
    )


def _random_power(a, size=None):
    """Power distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.power(float(a), shape),
        lambda rng, shape: rng.power_array(a, shape),
    )


def _random_vonmises(mu, kappa, size=None):
    """Von Mises distribution (rejection sampling)."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.vonmises(float(mu), float(kappa), shape),
        lambda rng, shape: rng.vonmises_array(mu, kappa, shape),
    )


def _random_wald(mean, scale, size=None):
    """Wald (inverse Gaussian) distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.wald(float(mean), float(scale), shape),
        lambda rng, shape: rng.wald_array(mean, scale, shape),
    )


def _random_zipf(a, size=None):
    """Zipf distribution (rejection sampling)."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.zipf(float(a), shape),
        lambda rng, shape: rng.zipf_array(a, shape),
    )


def _random_hypergeometric(ngood, nbad, nsample, size=None):
    """Hypergeometric distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.hypergeometric(
            int(ngood), int(nbad), int(nsample), shape
        ),
        lambda rng, shape: rng.hypergeometric_array(ngood, nbad, nsample, shape),
    )


def _random_pareto(a, size=None):
    """Pareto II (Lomax) distribution."""
    return _native_random_draw(
        size,
        lambda shape: _native_random_module.pareto(float(a), shape),
        lambda rng, shape: rng.pareto_array(a, shape),
    )


def _random_bytes(length):
    """Return random bytes."""
    vals = random.randint(0, 256, (length,))
    return bytes(int(v) for v in vals.tolist())


def _default_rng(seed=None):
    return _Generator(seed)


class _RandomState:
    """Legacy random state compatible with np.random.RandomState(seed)."""
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def rand(self, *shape):
        if len(shape) == 0:
            return float(random.rand((1,))[0])
        return random.rand(shape)

    def randn(self, *shape):
        if len(shape) == 0:
            return float(random.randn((1,))[0])
        return random.randn(shape)

    def randint(self, low, high=None, size=None, dtype='int64'):
        if high is None:
            high = low
            low = 0
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

    def random(self, size=None):
        return _random_random(size=size)

    def random_sample(self, size=None):
        return _random_random(size=size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return float(random.normal(float(loc), float(scale), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.normal(float(loc), float(scale), size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            return float(random.uniform(float(low), float(high), (1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.uniform(float(low), float(high), size)

    def choice(self, a, size=None, replace=True, p=None):
        if isinstance(a, int):
            arr = arange(0.0, float(a), 1.0)
        elif isinstance(a, (list, tuple)):
            arr = array(a)
        elif not isinstance(a, ndarray):
            arr = asarray(a)
        else:
            arr = a
        if size is None:
            size = 1
        return random.choice(arr, size, replace)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def seed(self, seed=None):
        random.seed(seed)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)

    def beta(self, a, b, size=None):
        return _random_beta(a, b, size)

    def gamma(self, shape, scale=1.0, size=None):
        return _random_gamma(shape, scale, size)

    def lognormal(self, mean=0.0, sigma=1.0, size=None):
        return _random_lognormal(mean, sigma, size)

    def chisquare(self, df, size=None):
        return _random_chisquare(df, size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def multivariate_normal(self, mean, cov, size=None):
        return _random_multivariate_normal(mean, cov, size)

    def geometric(self, p, size=None):
        return _random_geometric(p, size)

    def dirichlet(self, alpha, size=None):
        return _random_dirichlet(alpha, size)

    def multinomial(self, n, pvals, size=None):
        return _random_multinomial(n, pvals, size)

    def laplace(self, loc=0.0, scale=1.0, size=None):
        return _random_laplace(loc, scale, size)

    def triangular(self, left, mode, right, size=None):
        return _random_triangular(left, mode, right, size)

    def rayleigh(self, scale=1.0, size=None):
        return _random_rayleigh(scale, size)

    def weibull(self, a, size=None):
        return _random_weibull(a, size)

    def logistic(self, loc=0.0, scale=1.0, size=None):
        return _random_logistic(loc, scale, size)

    def gumbel(self, loc=0.0, scale=1.0, size=None):
        return _random_gumbel(loc, scale, size)

    def negative_binomial(self, n, p, size=None):
        return _random_negative_binomial(n, p, size)

    def power(self, a, size=None):
        return _random_power(a, size)

    def vonmises(self, mu, kappa, size=None):
        return _random_vonmises(mu, kappa, size)

    def wald(self, mean, scale, size=None):
        return _random_wald(mean, scale, size)

    def zipf(self, a, size=None):
        return _random_zipf(a, size)

    def hypergeometric(self, ngood, nbad, nsample, size=None):
        return _random_hypergeometric(ngood, nbad, nsample, size)

    def pareto(self, a, size=None):
        return _random_pareto(a, size)

    def f(self, dfnum, dfden, size=None):
        return _broadcast_call_2(lambda n, d: _random_f(n, d), dfnum, dfden, size)

    def noncentral_chisquare(self, df, nonc, size=None):
        return _broadcast_call_2(lambda d, n: _random_noncentral_chisquare(d, n), df, nonc, size)

    def noncentral_f(self, dfnum, dfden, nonc, size=None):
        return _broadcast_call_3(lambda n, d, nc: _random_noncentral_f(n, d, nc), dfnum, dfden, nonc, size)

    def standard_gamma(self, shape, size=None):
        return _random_gamma(shape, 1.0, size)

    def standard_t(self, df, size=None):
        return _random_standard_t(df, size)

    def standard_cauchy(self, size=None):
        return _random_standard_cauchy(size)

    def standard_exponential(self, size=None):
        return _random_exponential(1.0, size)

    def logseries(self, p, size=None):
        return _random_logseries(p, size)

    def bytes(self, length):
        return _random_bytes(length)

    _poisson_lam_max = 2**62

    def get_state(self):
        return {'state': 'not_implemented'}

    def set_state(self, state):
        pass


# Wrap random.choice to accept lists, tuples, and ints (Rust version requires ndarray)
_native_random_choice = _orig_random_choice


def _wrapped_random_choice(a, size=None, replace=True, p=None):
    if _ACTIVE_BITGEN is not None:
        arr = arange(0.0, float(a), 1.0) if isinstance(a, int) else (
            a if isinstance(a, ndarray) else array(list(a) if not hasattr(a, 'tolist') else a.tolist())
        )
        if size is None:
            return _ACTIVE_BITGEN._rng.choice_array(arr, 1, replace).flatten()[0]
        return _ACTIVE_BITGEN._rng.choice_array(arr, size, replace)
    if isinstance(a, int):
        a = arange(0.0, float(a), 1.0)
    elif isinstance(a, (list, tuple)):
        a = array([float(x) for x in a])
    if size is None:
        size = 1
    return _native_random_choice(a, size, replace)


_native_random_randint = _orig_random_randint


def _wrapped_random_randint(low, high=None, size=None, dtype='int64'):
    if high is None:
        high = low
        low = 0
    if _ACTIVE_BITGEN is not None:
        rng = _ACTIVE_BITGEN._rng
        if size is None:
            return rng.randrange(int(low), int(high))
        if isinstance(size, int):
            size = (size,)
        result = rng.randint_array(int(low), int(high), size)
        return result.reshape(size) if len(size) > 1 else result
    if size is None:
        return int(_native_random_randint(int(low), int(high), (1,)).flatten()[0])
    if isinstance(size, int):
        size = (size,)
    return _native_random_randint(int(low), int(high), size)


# Monkey-patch random module with Python-level extension functions
random.randint = _wrapped_random_randint
random.choice = _wrapped_random_choice
random.shuffle = _random_shuffle
random.permutation = _random_permutation
random.standard_normal = _random_standard_normal
random.exponential = _random_exponential
random.poisson = _random_poisson
random.binomial = _random_binomial
random.beta = _random_beta
random.gamma = _random_gamma
random.multinomial = _random_multinomial
random.lognormal = _random_lognormal
random.geometric = _random_geometric
random.dirichlet = _random_dirichlet
random.default_rng = _default_rng
random.Generator = _Generator
random.random = _random_random
random.random_sample = _random_random
random.multivariate_normal = _random_multivariate_normal
random.chisquare = _random_chisquare
random.laplace = _random_laplace
random.triangular = _random_triangular
random.rayleigh = _random_rayleigh
random.weibull = _random_weibull
random.logistic = _random_logistic
random.gumbel = _random_gumbel
random.negative_binomial = _random_negative_binomial
random.power = _random_power
random.vonmises = _random_vonmises
random.wald = _random_wald
random.zipf = _random_zipf
random.hypergeometric = _random_hypergeometric
random.pareto = _random_pareto
random.bytes = _random_bytes
random.RandomState = _RandomState


_native_random_normal = random.normal


def _random_normal(loc=0.0, scale=1.0, size=None):
    """Draw random samples from a normal (Gaussian) distribution."""
    if size is None:
        return float(_native_random_normal(float(loc), float(scale), (1,))[0])
    if isinstance(size, int):
        size = (size,)
    return _native_random_normal(float(loc), float(scale), size)


random.normal = _random_normal


# ---------------------------------------------------------------------------
# Bit generator stubs (needed for numpy.random.MT19937, etc.)
# ---------------------------------------------------------------------------

class _SeedSequence:
    """Stub for numpy.random.SeedSequence."""
    def __init__(self, entropy=None, spawn_key=(), pool_size=4):
        self.entropy = entropy
        self.spawn_key = spawn_key
        self.pool_size = pool_size
        self.n_children_spawned = 0

    def generate_state(self, n_words, dtype='uint32'):
        """Generate pseudorandom state — simplified stub."""
        # Use hash-based approach for deterministic output
        import hashlib
        h = hashlib.sha256()
        if self.entropy is not None:
            h.update(str(self.entropy).encode())
        for k in self.spawn_key:
            h.update(str(k).encode())
        digest = h.digest()
        state = []
        for i in range(n_words):
            idx = (i * 4) % len(digest)
            val = int.from_bytes(digest[idx:idx+4], 'little')
            state.append(val)
        return array(state, dtype=str(dtype))

    def spawn(self, n_children):
        seqs = []
        for i in range(n_children):
            child = _SeedSequence(
                entropy=self.entropy,
                spawn_key=self.spawn_key + (self.n_children_spawned + i,),
                pool_size=self.pool_size,
            )
            seqs.append(child)
        self.n_children_spawned += n_children
        return seqs


_bitgen_counter = 0


class _BitGenerator:
    """Base class stub for bit generators."""
    def __init__(self, seed=None):
        self._mt19937_state_cache = None
        if isinstance(seed, _SeedSequence):
            self._seed_seq = seed
            self._seed = seed.entropy
            self._rng = _NativeRngAdapter(seed=self._seed)
        elif isinstance(seed, (ndarray, list, tuple)):
            # Array/list seed: use hash of all elements
            if isinstance(seed, ndarray):
                seed_list = _flat_random_data(seed)
            else:
                seed_list = list(seed)
            for v in seed_list:
                if int(v) < 0:
                    raise ValueError("Seed must be non-negative")
            self._seed_seq = _SeedSequence(seed_list)
            # Collapse single-element list to scalar for state comparison
            if len(seed_list) == 1:
                self._seed = int(seed_list[0])
                self._rng = _NativeRngAdapter(seed=self._seed)
            else:
                self._seed = seed_list
                seed_val = 0
                for i, v in enumerate(seed_list):
                    seed_val = (seed_val * 31 + int(v)) % (2**63)
                self._rng = _NativeRngAdapter(seed=seed_val)
        else:
            self._seed_seq = _SeedSequence(seed)
            self._seed = seed
            if seed is not None:
                sv = int(seed)
                if sv < 0:
                    raise ValueError("Seed must be non-negative")
                self._rng = _NativeRngAdapter(seed=sv % (2**63))
            else:
                # Generate unique state for unseeded generators
                global _bitgen_counter
                _bitgen_counter += 1
                import time as _time
                entropy = int(_time.time() * 1e6) + _bitgen_counter
                self._seed = entropy
                self._rng = _NativeRngAdapter(seed=entropy % (2**63))

    @property
    def state(self):
        if self.__class__.__name__ == 'MT19937' and self._mt19937_state_cache is not None:
            return self._mt19937_state_cache
        return {
            'bit_generator': self.__class__.__name__,
            'state': {'native_state': self._rng.getstate(), 'seed': self._seed},
        }

    @state.setter
    def state(self, value):
        if isinstance(value, tuple) and len(value) >= 3 and value[0] == 'MT19937':
            key = value[1].tolist() if hasattr(value[1], 'tolist') else list(value[1])
            pos = int(value[2])
            self._mt19937_state_cache = {
                'bit_generator': self.__class__.__name__,
                'state': {
                    'key': array(key, dtype='uint32'),
                    'pos': pos,
                },
            }
            seed_val = 0
            for item in key:
                seed_val = (seed_val * 1315423911 + int(item)) % (2**63)
            self._rng.setstate(seed_val ^ pos)
            return
        if isinstance(value, dict):
            st = value.get('state', value)
            if isinstance(st, dict) and 'native_state' in st:
                self._rng.setstate(st['native_state'])
                if 'seed' in st:
                    self._seed = st['seed']
                self._mt19937_state_cache = None
                return
            if isinstance(st, dict) and 'key' in st and 'pos' in st:
                key = st['key'].tolist() if hasattr(st['key'], 'tolist') else list(st['key'])
                self._mt19937_state_cache = {
                    'bit_generator': self.__class__.__name__,
                    'state': {
                        'key': array(key, dtype='uint32'),
                        'pos': int(st['pos']),
                    },
                }
                seed_val = 0
                for item in key:
                    seed_val = (seed_val * 1315423911 + int(item)) % (2**63)
                self._rng.setstate(seed_val ^ int(st['pos']))
                return

    def advance(self, delta):
        if delta is None:
            return
        self._rng.setstate(_native_random_module.advance_state(self._rng.getstate(), hash(int(delta)) % (2**63)))
        self._mt19937_state_cache = None

    def jumped(self):
        new = self.__class__(0)
        new._rng.setstate(_native_random_module.jumped_state(self._rng.getstate()))
        new._seed = self._seed
        return new


class _MT19937(_BitGenerator):
    """Stub for numpy.random.MT19937."""
    pass

class _PCG64(_BitGenerator):
    """Stub for numpy.random.PCG64."""
    pass

class _PCG64DXSM(_BitGenerator):
    """Stub for numpy.random.PCG64DXSM."""
    pass

class _SFC64(_BitGenerator):
    """Stub for numpy.random.SFC64."""
    pass

class _Philox(_BitGenerator):
    """Stub for numpy.random.Philox."""
    pass


random.MT19937 = _MT19937
random.PCG64 = _PCG64
random.PCG64DXSM = _PCG64DXSM
random.SFC64 = _SFC64
random.Philox = _Philox
random.SeedSequence = _SeedSequence
random.BitGenerator = _BitGenerator


# ---------------------------------------------------------------------------
# bit_generator submodule stub (numpy.random.bit_generator)
# ---------------------------------------------------------------------------

import types as _types
import sys as _sys

_bit_generator_mod = _types.ModuleType("numpy.random.bit_generator")


class _SeedlessSeedSequence:
    """Stub for SeedlessSeedSequence (a seedless seed sequence placeholder)."""
    def generate_state(self, n_words, dtype='uint32'):
        return array([0] * n_words, dtype=str(dtype))

    def spawn(self, n_children):
        return [_SeedlessSeedSequence() for _ in range(n_children)]


_bit_generator_mod.SeedSequence = _SeedSequence
_bit_generator_mod.SeedlessSeedSequence = _SeedlessSeedSequence
_bit_generator_mod.BitGenerator = _BitGenerator
random.bit_generator = _bit_generator_mod
_sys.modules["numpy.random.bit_generator"] = _bit_generator_mod
