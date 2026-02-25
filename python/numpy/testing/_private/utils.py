"""numpy.testing._private.utils - deep import target."""
from numpy.testing._utils import *  # noqa: F401,F403

def requires_memory(size):
    """Decorator stub - skip memory-intensive tests."""
    def decorator(fn):
        return fn
    return decorator

def _no_tracing(fn):
    """Decorator stub - no-op."""
    return fn
