"""numpy.lib._function_base_impl - function base implementation."""


def add_newdoc(place, obj, doc, warn_on_python=True):
    """Stub for add_newdoc - no-op in this implementation."""
    pass


def __getattr__(name):
    import numpy
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module 'numpy.lib._function_base_impl' has no attribute {name!r}")
