"""numpy._core.function_base - linspace, logspace, add_newdoc."""
import numpy

linspace = numpy.linspace
logspace = numpy.logspace


def add_newdoc(place, obj, doc, warn_on_python=True):
    """Stub for add_newdoc (no-op in numpy-rust)."""
    pass


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
