"""numpy.testing._private.utils - deep import target."""
from numpy.testing._utils import *  # noqa: F401,F403

def requires_memory(free_bytes):
    """Decorator stub - skip memory-intensive tests."""
    def dec(fn):
        fn._skip = True
        fn._skip_reason = "requires_memory"
        return fn
    return dec

def _no_tracing(fn):
    """Decorator stub - no-op."""
    return fn

def _glibc_older_than(version):
    return False

def run_threaded(workers, target, *args):
    """Run target function (no real threading in RustPython)."""
    target(*args)

def _gen_alignment_data(dtype="float64", type="binary", max_size=24):
    import numpy as np
    for sz in [1, 2, 4, 8, max_size]:
        if type == "unary":
            yield np.zeros(sz, dtype=dtype), np.zeros(sz, dtype=dtype)
        else:
            yield np.zeros(sz, dtype=dtype), np.zeros(sz, dtype=dtype), np.zeros(sz, dtype=dtype)
