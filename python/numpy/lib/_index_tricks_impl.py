"""numpy.lib._index_tricks_impl - index tricks proxy."""


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._index_tricks_impl' has no attribute {name!r}")
