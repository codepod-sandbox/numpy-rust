"""numpy.polynomial.chebyshev - Chebyshev polynomial functions.

Provides module-level access to Chebyshev polynomial functions for imports like:
    from numpy.polynomial.chebyshev import chebval
"""
from numpy.polynomial import chebyshev as _cheb

chebval = _cheb.chebval
chebfit = _cheb.chebfit
chebadd = _cheb.chebadd
chebsub = _cheb.chebsub
chebmul = _cheb.chebmul
chebder = _cheb.chebder
chebint = _cheb.chebint
