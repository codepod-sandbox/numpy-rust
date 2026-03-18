"""numpy.lib._arraysetops_impl - array set operations proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._arraysetops_impl' has no attribute {name!r}")
