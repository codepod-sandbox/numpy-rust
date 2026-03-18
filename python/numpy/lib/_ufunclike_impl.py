"""numpy.lib._ufunclike_impl - ufunc-like proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._ufunclike_impl' has no attribute {name!r}")
