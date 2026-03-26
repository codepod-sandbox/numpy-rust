"""Signal window functions."""
import math as _math
from ._creation import array, arange

__all__ = ['bartlett', 'blackman', 'hamming', 'hanning', 'kaiser']

_pi = _math.pi


def _window_int(M):
    """Convert M to a Python int, handling ndarray/ObjectArray scalars."""
    if hasattr(M, 'item'):
        return int(M.item())
    return int(M)


def bartlett(M):
    """Return the Bartlett window."""
    M = _window_int(M)
    if M < 1:
        return array([], dtype='float64')
    if M == 1:
        return array([1.0])
    # Use min(i, M-1-i) to guarantee exact symmetry
    vals = [0.0] * M
    for i in range((M + 1) // 2):
        v = 2.0 * i / (M - 1)
        vals[i] = v
        vals[M - 1 - i] = v
    return array(vals)


def blackman(M):
    """Return the Blackman window."""
    M = _window_int(M)
    if M < 1:
        return array([], dtype='float64')
    if M == 1:
        return array([1.0])
    vals = [0.0] * M
    for i in range((M + 1) // 2):
        v = 0.42 - 0.5 * _math.cos(2.0 * _pi * i / (M - 1)) + 0.08 * _math.cos(4.0 * _pi * i / (M - 1))
        vals[i] = v
        vals[M - 1 - i] = v
    return array(vals)


def hamming(M):
    """Return the Hamming window."""
    M = _window_int(M)
    if M < 1:
        return array([], dtype='float64')
    if M == 1:
        return array([1.0])
    vals = [0.0] * M
    for i in range((M + 1) // 2):
        v = 0.54 - 0.46 * _math.cos(2.0 * _pi * i / (M - 1))
        vals[i] = v
        vals[M - 1 - i] = v
    return array(vals)


def hanning(M):
    """Return the Hanning window."""
    M = _window_int(M)
    if M < 1:
        return array([], dtype='float64')
    if M == 1:
        return array([1.0])
    vals = [0.0] * M
    for i in range((M + 1) // 2):
        v = 0.5 - 0.5 * _math.cos(2.0 * _pi * i / (M - 1))
        vals[i] = v
        vals[M - 1 - i] = v
    return array(vals)


def kaiser(M, beta):
    """Return the Kaiser window."""
    M = _window_int(M)
    if M < 1:
        return array([], dtype='float64')
    if M == 1:
        return array([1.0])
    # I0 is modified Bessel function of first kind, order 0
    # Use series approximation
    def _i0(x):
        """Modified Bessel function I0 via series."""
        val = 1.0
        term = 1.0
        for k in range(1, 25):
            term *= (x / 2.0) ** 2 / (k * k)
            val += term
        return val

    alpha = (M - 1) / 2.0
    i0_beta = _i0(float(beta))
    vals = [0.0] * M
    for i in range((M + 1) // 2):
        arg = beta * _math.sqrt(1.0 - ((i - alpha) / alpha) ** 2)
        v = _i0(float(arg)) / i0_beta
        vals[i] = v
        vals[M - 1 - i] = v
    return array(vals)
