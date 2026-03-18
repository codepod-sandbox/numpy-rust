"""numpy.lib._twodim_base_impl - 2D base proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._twodim_base_impl' has no attribute {name!r}")
