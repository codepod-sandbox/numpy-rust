"""numpy.lib._shape_base_impl - shape base proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._shape_base_impl' has no attribute {name!r}")
