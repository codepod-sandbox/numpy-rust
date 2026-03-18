"""numpy.lib._histograms_impl - histogram proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._histograms_impl' has no attribute {name!r}")
