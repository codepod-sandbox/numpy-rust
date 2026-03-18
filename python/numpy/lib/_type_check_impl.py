"""numpy.lib._type_check_impl - type check proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._type_check_impl' has no attribute {name!r}")
