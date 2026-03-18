"""numpy.lib._polynomial_impl - polynomial proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._polynomial_impl' has no attribute {name!r}")
