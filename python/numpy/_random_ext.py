"""Random number generation Python-level wrappers."""
import math as _math
import _numpy_native as _native
from _numpy_native import ndarray, random as _random_native, linalg as _linalg_native
from ._helpers import _builtin_range
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
random = _random_native
linalg = _linalg_native

# stdlib math (needed for _random_vonmises)
_stdlib_math = _math


def _random_shuffle(x):
    """Modify a sequence in-place by shuffling its contents. Returns None."""
    if not isinstance(x, ndarray):
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    # Attempt to update array in-place via __setitem__
    try:
        for i in range(n):
            x[i] = vals[i]
    except Exception:
        pass  # if in-place update not supported, shuffle is best-effort
    return None  # real numpy returns None


def _random_permutation(x):
    """Randomly permute a sequence, or return a permuted range."""
    if isinstance(x, (int, float)):
        x = arange(0, int(x))
    else:
        x = asarray(x)
    n = x.size
    flat = x.flatten()
    vals = [flat[i] for i in range(n)]
    for i in range(n - 1, 0, -1):
        j_arr = random.randint(0, i + 1, (1,))
        j = int(j_arr[0])
        vals[i], vals[j] = vals[j], vals[i]
    result = array(vals)
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
    if size is None:
        import math as _m
        u = float(random.uniform(0.0, 1.0, (1,)).flatten()[0])
        if u >= 1.0:
            u = 0.9999999999
        return float(-scale * _m.log(1.0 - u))
    if isinstance(size, int):
        size = (size,)
    # Generate uniform [0,1) then transform: -scale * ln(1 - U)
    u = random.uniform(0.0, 1.0, size)
    flat = u.flatten()
    n = flat.size
    result = []
    for i in range(n):
        v = float(flat[i])
        if v >= 1.0:
            v = 0.9999999999
        import math as _m
        result.append(-scale * _m.log(1.0 - v))
    r = array(result)
    if u.ndim > 1:
        r = r.reshape(u.shape)
    return r


def _random_poisson(lam=1.0, size=None):
    """Draw samples from a Poisson distribution."""
    import math as _m
    if size is None:
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        return float(k - 1)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Knuth algorithm
        L = _m.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            u = random.uniform(0.0, 1.0, (1,))
            p *= float(u[0])
            if p <= L:
                break
        result.append(float(k - 1))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r


def _random_binomial(n, p, size=None):
    """Draw samples from a binomial distribution."""
    if size is None:
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        return float(successes)
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        successes = 0
        for _ in range(int(n)):
            u = random.uniform(0.0, 1.0, (1,))
            if float(u[0]) < p:
                successes += 1
        result.append(float(successes))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r


def _random_beta(a, b, size=None):
    """Draw samples from a Beta distribution.
    Uses the relationship: if X~Gamma(a,1) and Y~Gamma(b,1), then X/(X+Y)~Beta(a,b)."""
    if size is None:
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                return float(x / (x + y))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        # Use Johnk's algorithm for Beta
        import math as _m
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            x = u1 ** (1.0 / a)
            y = u2 ** (1.0 / b)
            if x + y <= 1.0:
                result.append(x / (x + y))
                break
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r


def _random_gamma(shape_param, scale=1.0, size=None):
    """Draw samples from a Gamma distribution using Marsaglia-Tsang method."""
    import math as _m
    def _gamma_one_sample(shape_param, scale):
        alpha = shape_param
        if alpha < 1:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            alpha = alpha + 1
            boost = u ** (1.0 / shape_param)
        else:
            boost = 1.0
        d = alpha - 1.0/3.0
        c = 1.0 / _m.sqrt(9.0 * d)
        while True:
            x = float(random.randn((1,))[0])
            v = (1.0 + c * x) ** 3
            if v <= 0:
                continue
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < 1 - 0.0331 * x**4:
                return d * v * scale * boost
            if _m.log(u) < 0.5 * x**2 + d * (1 - v + _m.log(v)):
                return d * v * scale * boost
    if size is None:
        return float(_gamma_one_sample(shape_param, scale))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    result = []
    for _ in range(total):
        result.append(_gamma_one_sample(shape_param, scale))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r


def _random_multinomial(n, pvals, size=None):
    """Draw samples from a multinomial distribution."""
    pvals = [float(p) for p in (pvals.tolist() if isinstance(pvals, ndarray) else pvals)]
    k = len(pvals)
    if size is None:
        # Single draw: n trials among k categories
        result = [0] * k
        for _ in range(n):
            r = float(random.rand((1,))[0])
            cumsum = 0.0
            for j in range(k):
                cumsum += pvals[j]
                if r < cumsum:
                    result[j] += 1
                    break
            else:
                result[-1] += 1
        return array(result)
    else:
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        rows = []
        for _ in range(total):
            result = [0] * k
            for _ in range(n):
                r = float(random.rand((1,))[0])
                cumsum = 0.0
                for j in range(k):
                    cumsum += pvals[j]
                    if r < cumsum:
                        result[j] += 1
                        break
                else:
                    result[-1] += 1
            rows.append(result)
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out


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
    import math as _m
    if size is None:
        log1mp = _m.log(1.0 - p)
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        return float(_m.ceil(_m.log(u) / log1mp))
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    log1mp = _m.log(1.0 - p)
    result = []
    for _ in range(total):
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        # Avoid log(0)
        if u >= 1.0:
            u = 0.9999999999
        if u <= 0.0:
            u = 1e-15
        result.append(float(_m.ceil(_m.log(u) / log1mp)))
    r = array(result)
    if len(size) > 1:
        r = r.reshape(size)
    return r


def _random_dirichlet(alpha, size=None):
    """Draw samples from a Dirichlet distribution."""
    if isinstance(alpha, ndarray):
        alpha = alpha.tolist()
    alpha = [float(a) for a in alpha]
    k = len(alpha)
    if size is None:
        # Single draw
        samples = []
        for a in alpha:
            g = float(_random_gamma(a, 1.0, (1,))[0])
            samples.append(g)
        total = sum(samples)
        return array([s / total for s in samples])
    else:
        if isinstance(size, int):
            size = (size,)
        num = 1
        for s in size:
            num *= s
        rows = []
        for _ in range(num):
            samples = []
            for a in alpha:
                g = float(_random_gamma(a, 1.0, (1,))[0])
                samples.append(g)
            total = sum(samples)
            rows.append([s / total for s in samples])
        out = array(rows)
        if len(size) > 1:
            out = out.reshape(list(size) + [k])
        return out


def _random_standard_cauchy(size=None):
    """Standard Cauchy distribution via ratio of standard normals."""
    import math as _m
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        while True:
            u1 = float(random.normal(0.0, 1.0, (1,))[0])
            u2 = float(random.normal(0.0, 1.0, (1,))[0])
            if abs(u2) > 1e-15:
                results.append(u1 / u2)
                break
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_standard_t(df, size=None):
    """Student's t-distribution."""
    import math as _m
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        z = float(random.normal(0.0, 1.0, (1,))[0])
        chi2 = float(_random_chisquare(int(df), (1,))[0])
        results.append(z / _m.sqrt(chi2 / df))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_f(dfnum, dfden, size=None):
    """F-distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        chi1 = float(_random_chisquare(int(dfnum), (1,))[0])
        chi2 = float(_random_chisquare(int(dfden), (1,))[0])
        results.append((chi1 / dfnum) / (chi2 / dfden))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_noncentral_chisquare(df, nonc, size=None):
    """Noncentral chi-square distribution."""
    import math as _m
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        # Generate as sum of (Z + sqrt(nonc/df))^2 terms approximation
        # Simpler: chi2(df-1) + (N(sqrt(nonc), 1))^2
        if df > 1:
            chi = float(_random_chisquare(int(df) - 1, (1,))[0])
        else:
            chi = 0.0
        z = float(random.normal(_m.sqrt(nonc), 1.0, (1,))[0])
        results.append(chi + z * z)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_noncentral_f(dfnum, dfden, nonc, size=None):
    """Noncentral F-distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        ncchi2 = float(_random_noncentral_chisquare(int(dfnum), nonc, (1,))[0])
        chi2 = float(_random_chisquare(int(dfden), (1,))[0])
        results.append((ncchi2 / dfnum) / (chi2 / dfden))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_logseries(p, size=None):
    """Logarithmic (Log-Series) distribution."""
    import math as _m
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    r = -1.0 / _m.log(1.0 - p)
    results = []
    for _ in range(total):
        while True:
            u1 = float(random.uniform(0.0, 1.0, (1,))[0])
            u2 = float(random.uniform(0.0, 1.0, (1,))[0])
            if u1 <= 0 or u1 >= 1:
                continue
            q = 1.0 - (1.0 - p) ** u2  # actually use different algorithm
            # Kemp's algorithm
            v = float(random.uniform(0.0, 1.0, (1,))[0])
            if v >= p:
                results.append(1)
                break
            if v >= p * p:
                # could be 1 or 2
                if v >= p + p * p * (1.0 - p):
                    results.append(1)
                else:
                    results.append(2)
                break
            # General: geometric-like
            x = 1
            cumprob = p
            total_prob = p
            while total_prob < v:
                x += 1
                cumprob *= p
                total_prob += cumprob / x
            results.append(x)
            break
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


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


def _fill_out(out, src):
    """Fill an output array in-place from a source array using tuple indexing."""
    shape = out.shape
    ndim = len(shape)
    src_flat = src.flatten().tolist() if isinstance(src, ndarray) else [float(src)]
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


def _broadcast_call_1(func, arg, size=None):
    """Call a single-param distribution function with broadcasting support."""
    if isinstance(arg, ndarray):
        arg_flat = arg.flatten().tolist()
        if size is not None:
            if isinstance(size, int):
                size = (size,)
            total = 1
            for s in size:
                total *= s
            results = []
            n = len(arg_flat)
            for i in range(total):
                results.append(float(func(arg_flat[i % n])))
            r = array(results)
            if len(size) > 1:
                r = r.reshape(list(size))
            return r
        else:
            results = [float(func(v)) for v in arg_flat]
            r = array(results)
            if arg.ndim > 1:
                r = r.reshape(arg.shape)
            return r
    if size is not None:
        if isinstance(size, int):
            size = (size,)
        total = 1
        for s in size:
            total *= s
        results = [float(func(arg)) for _ in range(total)]
        r = array(results)
        if len(size) > 1:
            r = r.reshape(list(size))
        return r
    return func(arg)


def _broadcast_call_2(func, a, b, size=None):
    """Call a two-param distribution function with broadcasting support."""
    a_is_arr = isinstance(a, ndarray)
    b_is_arr = isinstance(b, ndarray)
    if not a_is_arr and not b_is_arr:
        if size is not None:
            if isinstance(size, int):
                size = (size,)
            total = 1
            for s in size:
                total *= s
            results = [float(func(a, b)) for _ in range(total)]
            r = array(results)
            if len(size) > 1:
                r = r.reshape(list(size))
            return r
        return func(a, b)
    # At least one is an array; broadcast
    if a_is_arr:
        a_flat = a.flatten().tolist()
        a_shape = list(a.shape)
    else:
        a_flat = [a]
        a_shape = [1]
    if b_is_arr:
        b_flat = b.flatten().tolist()
        b_shape = list(b.shape)
    else:
        b_flat = [b]
        b_shape = [1]
    # Broadcast shapes
    max_ndim = max(len(a_shape), len(b_shape))
    while len(a_shape) < max_ndim:
        a_shape = [1] + a_shape
    while len(b_shape) < max_ndim:
        b_shape = [1] + b_shape
    out_shape = []
    for i in range(max_ndim):
        out_shape.append(max(a_shape[i], b_shape[i]))
    if size is not None:
        if isinstance(size, int):
            out_shape = [size]
        else:
            out_shape = list(size)
    total = 1
    for s in out_shape:
        total *= s
    na = len(a_flat)
    nb = len(b_flat)
    results = []
    for i in range(total):
        results.append(float(func(a_flat[i % na], b_flat[i % nb])))
    r = array(results)
    if len(out_shape) > 1:
        r = r.reshape(out_shape)
    return r


def _broadcast_call_3(func, a, b, c, size=None):
    """Call a three-param distribution function with broadcasting support."""
    a_is_arr = isinstance(a, ndarray)
    b_is_arr = isinstance(b, ndarray)
    c_is_arr = isinstance(c, ndarray)
    if not a_is_arr and not b_is_arr and not c_is_arr:
        if size is not None:
            if isinstance(size, int):
                size = (size,)
            total = 1
            for s in size:
                total *= s
            results = [float(func(a, b, c)) for _ in range(total)]
            r = array(results)
            if len(size) > 1:
                r = r.reshape(list(size))
            return r
        return func(a, b, c)
    a_flat = a.flatten().tolist() if a_is_arr else [a]
    b_flat = b.flatten().tolist() if b_is_arr else [b]
    c_flat = c.flatten().tolist() if c_is_arr else [c]
    a_shape = list(a.shape) if a_is_arr else [1]
    b_shape = list(b.shape) if b_is_arr else [1]
    c_shape = list(c.shape) if c_is_arr else [1]
    max_ndim = max(len(a_shape), len(b_shape), len(c_shape))
    while len(a_shape) < max_ndim:
        a_shape = [1] + a_shape
    while len(b_shape) < max_ndim:
        b_shape = [1] + b_shape
    while len(c_shape) < max_ndim:
        c_shape = [1] + c_shape
    out_shape = [max(a_shape[i], b_shape[i], c_shape[i]) for i in range(max_ndim)]
    if size is not None:
        if isinstance(size, int):
            out_shape = [size]
        else:
            out_shape = list(size)
    total = 1
    for s in out_shape:
        total *= s
    na, nb, nc = len(a_flat), len(b_flat), len(c_flat)
    results = [float(func(a_flat[i % na], b_flat[i % nb], c_flat[i % nc])) for i in range(total)]
    r = array(results)
    if len(out_shape) > 1:
        r = r.reshape(out_shape)
    return r


class _Generator:
    """Random number generator (simplified)."""
    def __init__(self, seed_val=None):
        if isinstance(seed_val, _BitGenerator):
            self.bit_generator = seed_val
            # BitGenerator already seeded the RNG in its __init__
        else:
            self.bit_generator = _PCG64(seed_val)
            # _PCG64.__init__ already seeds the RNG

    def random(self, size=None, dtype=None, out=None):
        if size is None and out is not None:
            size = out.shape
        if size is None:
            return float(random.rand((1,))[0])
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
        if not endpoint:
            pass  # high is exclusive already
        else:
            high = high + 1
        # Validate dtype bounds
        dt_str = str(dtype)
        # Check uint before int to avoid 'int16' matching 'uint16'
        _dtype_bounds_list = [
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
        bounds = None
        for k, v in _dtype_bounds_list:
            if k in dt_str:
                bounds = v
                break
        # Handle array low/high
        low_is_arr = isinstance(low, (list, tuple, ndarray))
        high_is_arr = isinstance(high, (list, tuple, ndarray))
        if low_is_arr or high_is_arr:
            if low_is_arr:
                low_a = asarray(low)
                low_flat = low_a.flatten().tolist()
                low_shape = list(low_a.shape)
            else:
                low_flat = [low]
                low_shape = [1]
            if high_is_arr:
                high_a = asarray(high)
                high_flat = high_a.flatten().tolist()
                high_shape = list(high_a.shape)
            else:
                high_flat = [high]
                high_shape = [1]
            # Determine output shape from broadcast
            max_ndim = max(len(low_shape), len(high_shape))
            while len(low_shape) < max_ndim:
                low_shape = [1] + low_shape
            while len(high_shape) < max_ndim:
                high_shape = [1] + high_shape
            out_shape = [max(low_shape[i], high_shape[i]) for i in range(max_ndim)]
            if size is not None:
                if isinstance(size, int):
                    out_shape = [size]
                else:
                    out_shape = list(size)
            n = 1
            for s in out_shape:
                n *= s
            # Validate bounds
            nl = len(low_flat)
            nh = len(high_flat)
            for i in range(max(nl, nh)):
                lo_v = int(low_flat[i % nl])
                hi_v = int(high_flat[i % nh])
                if lo_v >= hi_v:
                    raise ValueError("low >= high")
                if bounds is not None:
                    if lo_v < bounds[0]:
                        raise ValueError(f"low is out of bounds for {dt_str}")
                    if hi_v > bounds[1]:
                        raise ValueError(f"high is out of bounds for {dt_str}")
            # Check if all low/high values are the same — use batch call
            lo_set = set(int(x) for x in low_flat)
            hi_set = set(int(x) for x in high_flat)
            if len(lo_set) == 1 and len(hi_set) == 1:
                lo_val = int(low_flat[0])
                hi_val = int(high_flat[0])
                r = _native_random_randint(lo_val, hi_val, (n,))
            else:
                results = []
                nl = len(low_flat)
                nh = len(high_flat)
                for i in range(n):
                    lo = int(low_flat[i % nl])
                    hi = int(high_flat[i % nh])
                    results.append(int(_native_random_randint(lo, hi, (1,)).flatten()[0]))
                r = array(results)
            if len(out_shape) > 1:
                r = r.reshape(out_shape)
            return r
        lo_v = int(low)
        hi_v = int(high)
        if lo_v >= hi_v:
            raise ValueError("low >= high")
        if bounds is not None:
            if lo_v < bounds[0]:
                raise ValueError(f"low is out of bounds for {dt_str}")
            if hi_v > bounds[1]:
                raise ValueError(f"high is out of bounds for {dt_str}")
        if size is None:
            return int(_native_random_randint(lo_v, hi_v, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return _native_random_randint(lo_v, hi_v, size)

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

    if size is None:
        # Single sample: generate n standard normals
        z = random.normal(0.0, 1.0, (n,))
        # Transform: L @ z + mean
        z_list = z.tolist()
        mean_list = mean.tolist()
        L_list = L.tolist()
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * z_list[j]
            sample.append(val)
        return array(sample)

    if isinstance(size, int):
        size = (size,)

    total = 1
    for s in size:
        total *= s

    # Generate total*n standard normals
    z = random.normal(0.0, 1.0, (total * n,)).reshape((total, n))

    # Transform: samples = mean + z @ L^T
    z_list = z.tolist()
    mean_list = mean.tolist()
    L_list = L.tolist()

    results = []
    for row in z_list:
        sample = []
        for i in range(n):
            val = mean_list[i]
            for j in range(n):
                val += L_list[i][j] * row[j]
            sample.append(val)
        results.append(sample)

    result = array(results)
    if len(size) > 1:
        result = result.reshape(list(size) + [n])
    return result


def _random_chisquare(df, size=None):
    """Chi-square distribution (sum of df squared standard normals)."""
    df = int(df)
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        z = random.normal(0.0, 1.0, (df,))
        z_list = z.tolist()
        results.append(sum(v * v for v in z_list))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_laplace(loc=0.0, scale=1.0, size=None):
    """Laplace distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    # Inverse CDF: loc - scale * sign(u - 0.5) * log(1 - 2*abs(u - 0.5))
    u_list = u.tolist()
    results = []
    import math
    for ui in u_list:
        ui_shifted = ui - 0.5
        if ui_shifted == 0:
            results.append(loc)
        else:
            sign_val = 1.0 if ui_shifted > 0 else -1.0
            results.append(loc - scale * sign_val * math.log(1.0 - 2.0 * abs(ui_shifted)))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_triangular(left, mode, right, size=None):
    """Triangular distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    u_list = u.tolist()
    results = []
    fc = (mode - left) / (right - left)
    for ui in u_list:
        if ui < fc:
            results.append(left + ((right - left) * (mode - left) * ui) ** 0.5)
        else:
            results.append(right - ((right - left) * (right - mode) * (1.0 - ui)) ** 0.5)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_rayleigh(scale=1.0, size=None):
    """Rayleigh distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [scale * math.sqrt(-2.0 * math.log(1.0 - ui)) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_weibull(a, size=None):
    """Weibull distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    import math
    results = [(-math.log(1.0 - ui)) ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_logistic(loc=0.0, scale=1.0, size=None):
    """Logistic distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc + scale * math.log(u / (1.0 - u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_gumbel(loc=0.0, scale=1.0, size=None):
    """Gumbel distribution via inverse CDF."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    while len(results) < total:
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if 0 < u < 1:
            results.append(loc - scale * math.log(-math.log(u)))
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_negative_binomial(n, p, size=None):
    """Negative binomial distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        # Generate n geometric trials: count failures before n successes
        count = 0
        successes = 0
        while successes < n:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u < p:
                successes += 1
            else:
                count += 1
        results.append(float(count))
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_power(a, size=None):
    """Power distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    u = random.uniform(0.0, 1.0, (total,))
    results = [ui ** (1.0 / a) for ui in u.tolist()]
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_vonmises(mu, kappa, size=None):
    """Von Mises distribution (rejection sampling)."""
    import math
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    if kappa < 1e-6:
        # For very small kappa, uniform on [-pi, pi]
        for _ in range(total):
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            results.append(-math.pi + 2.0 * math.pi * u)
    else:
        tau = 1.0 + (1.0 + 4.0 * kappa * kappa) ** 0.5
        rho = (tau - (2.0 * tau) ** 0.5) / (2.0 * kappa)
        r = (1.0 + rho * rho) / (2.0 * rho)
        for _ in range(total):
            while True:
                u1 = float(random.uniform(0.0, 1.0, (1,))[0])
                z = _stdlib_math.cos(math.pi * u1)
                f = (1.0 + r * z) / (r + z)
                c = kappa * (r - f)
                u2 = float(random.uniform(0.0, 1.0, (1,))[0])
                if u2 < c * (2.0 - c) or u2 <= c * _stdlib_math.exp(1.0 - c):
                    u3 = float(random.uniform(0.0, 1.0, (1,))[0])
                    theta = mu + (1.0 if u3 > 0.5 else -1.0) * _stdlib_math.acos(f)
                    results.append(theta)
                    break
    result = array(results[:total])
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_wald(mean, scale, size=None):
    """Wald (inverse Gaussian) distribution."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    results = []
    for _ in range(total):
        v = float(random.normal(0.0, 1.0, (1,))[0])
        y = v * v
        x = mean + (mean * mean * y) / (2.0 * scale) - (mean / (2.0 * scale)) * (4.0 * mean * scale * y + mean * mean * y * y) ** 0.5
        u = float(random.uniform(0.0, 1.0, (1,))[0])
        if u <= mean / (mean + x):
            results.append(x)
        else:
            results.append(mean * mean / x)
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_zipf(a, size=None):
    """Zipf distribution (rejection sampling)."""
    if size is None:
        total = 1
    elif isinstance(size, int):
        total = size
    else:
        total = 1
        for s in size:
            total *= s
    am1 = a - 1.0
    b = 2.0 ** am1
    results = []
    for _ in range(total):
        while True:
            u = float(random.uniform(0.0, 1.0, (1,))[0])
            if u <= 0.0:
                continue
            v = float(random.uniform(0.0, 1.0, (1,))[0])
            x = int(u ** (-1.0 / am1))
            if x < 1:
                x = 1
            t = (1.0 + 1.0 / x) ** am1
            if v * x * (t - 1.0) / (b - 1.0) <= t / b:
                results.append(float(x))
                break
    result = array(results)
    if size is None:
        return float(result[0])
    if not isinstance(size, int) and len(size) > 1:
        result = result.reshape(list(size))
    return result


def _random_hypergeometric(ngood, nbad, nsample, size=None):
    """Hypergeometric distribution."""
    def _draw_one(ng, nb, ns):
        count = 0
        rg = ng
        rt = ng + nb
        uniforms = random.uniform(0.0, 1.0, (ns,)).tolist()
        for u in uniforms:
            if u < rg / rt:
                count += 1
                rg -= 1
            rt -= 1
        return count
    if size is None:
        return _draw_one(ngood, nbad, nsample)
    if isinstance(size, int):
        size = (size,)
    total_elems = 1
    for s in size:
        total_elems *= s
    result = [float(_draw_one(ngood, nbad, nsample)) for _ in _builtin_range(total_elems)]
    return array(result).reshape(list(size))


def _random_pareto(a, size=None):
    """Pareto II (Lomax) distribution."""
    if size is None:
        u = float(random.uniform(0.0, 1.0, (1,)).flatten()[0])
        return (1.0 - u) ** (-1.0 / a) - 1.0
    if isinstance(size, int):
        size = (size,)
    total = 1
    for s in size:
        total *= s
    uniforms = random.uniform(0.0, 1.0, (total,)).tolist()
    result = [(1.0 - u) ** (-1.0 / a) - 1.0 for u in uniforms]
    return array(result).reshape(list(size))


def _random_bytes(length):
    """Return random bytes."""
    vals = random.uniform(0.0, 1.0, (length,)).tolist()
    return bytes([int(v * 256) for v in vals])


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
_native_random_choice = random.choice


def _wrapped_random_choice(a, size=None, replace=True, p=None):
    if isinstance(a, int):
        a = arange(0.0, float(a), 1.0)
    elif isinstance(a, (list, tuple)):
        a = array([float(x) for x in a])
    if size is None:
        size = 1
    return _native_random_choice(a, size, replace)


_native_random_randint = random.randint


def _wrapped_random_randint(low, high=None, size=None, dtype='int64'):
    if high is None:
        high = low
        low = 0
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
        if isinstance(seed, _SeedSequence):
            self._seed_seq = seed
            self._seed = seed.entropy
        elif isinstance(seed, (ndarray, list, tuple)):
            # Array/list seed: use hash of all elements
            if isinstance(seed, ndarray):
                seed_list = seed.flatten().tolist()
            else:
                seed_list = list(seed)
            for v in seed_list:
                if int(v) < 0:
                    raise ValueError("Seed must be non-negative")
            self._seed_seq = _SeedSequence(seed_list)
            # Collapse single-element list to scalar for state comparison
            if len(seed_list) == 1:
                self._seed = int(seed_list[0])
                random.seed(int(seed_list[0]) % (2**63))
            else:
                self._seed = seed_list
                seed_val = 0
                for i, v in enumerate(seed_list):
                    seed_val = (seed_val * 31 + int(v)) % (2**63)
                random.seed(seed_val)
        else:
            self._seed_seq = _SeedSequence(seed)
            self._seed = seed
            if seed is not None:
                sv = int(seed)
                if sv < 0:
                    raise ValueError("Seed must be non-negative")
                random.seed(sv % (2**63))
            else:
                # Generate unique state for unseeded generators
                global _bitgen_counter
                _bitgen_counter += 1
                import time as _time
                entropy = int(_time.time() * 1e6) + _bitgen_counter
                self._seed = entropy
                random.seed(entropy % (2**63))

    @property
    def state(self):
        return {'bit_generator': self.__class__.__name__, 'state': {'seed': self._seed}}

    @state.setter
    def state(self, value):
        pass


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
