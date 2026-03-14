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


class _Generator:
    """Random number generator (simplified)."""
    def __init__(self, seed_val=None):
        if seed_val is not None:
            random.seed(int(seed_val))

    def random(self, size=None):
        if size is None:
            return float(random.rand((1,))[0])
        if isinstance(size, int):
            size = (size,)
        return random.rand(size)

    def standard_normal(self, size=None):
        return _random_standard_normal(size)

    def integers(self, low, high=None, size=None, dtype='int64', endpoint=False):
        if high is None:
            high = low
            low = 0
        if not endpoint:
            pass  # high is exclusive already
        else:
            high = high + 1
        if size is None:
            return int(random.randint(low, high, (1,)).flatten()[0])
        if isinstance(size, int):
            size = (size,)
        return random.randint(low, high, size)

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
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.normal(loc, scale, size)

    def uniform(self, low=0.0, high=1.0, size=None):
        if size is None:
            size = (1,)
        if isinstance(size, int):
            size = (size,)
        return random.uniform(low, high, size)

    def shuffle(self, x):
        return _random_shuffle(x)

    def permutation(self, x):
        return _random_permutation(x)

    def exponential(self, scale=1.0, size=None):
        return _random_exponential(scale, size)

    def poisson(self, lam=1.0, size=None):
        return _random_poisson(lam, size)

    def binomial(self, n, p, size=None):
        return _random_binomial(n, p, size)


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


# Monkey-patch random module with Python-level extension functions
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
