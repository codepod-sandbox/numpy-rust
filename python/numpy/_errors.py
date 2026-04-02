"""Error and warning state management for numpy-rust."""

__all__ = [
    '_err_state', 'seterr', 'geterr', 'errstate',
    'seterrcall', 'geterrcall',
]

# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

_err_state = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
_UNSET_CALL = object()  # sentinel for errstate call= parameter

def seterr(**kwargs):
    """Set floating point error handling."""
    global _err_state
    old = dict(_err_state)
    for k, v in kwargs.items():
        if k == "all":
            for key in _err_state:
                _err_state[key] = v
            continue
        if k not in _err_state:
            raise ValueError("invalid key: %r" % k)
        _err_state[k] = v
    return old

def geterr():
    return dict(_err_state)

class errstate:
    """Context manager for floating point error handling."""
    def __init__(self, *, call=_UNSET_CALL, **kwargs):
        self._kwargs = kwargs
        self._call = call
        self._old = None
        self._old_call = None
        self._entered = False
    def __enter__(self):
        if self._entered:
            raise TypeError("Cannot enter `np.errstate` twice")
        self._entered = True
        self._old = seterr(**self._kwargs)
        if self._call is not _UNSET_CALL:
            self._old_call = seterrcall(self._call)
        return self
    def __exit__(self, *args):
        seterr(**self._old)
        if self._call is not _UNSET_CALL:
            seterrcall(self._old_call)
    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with type(self)(**self._kwargs, call=self._call):
                return func(*args, **kwargs)
        return wrapper


# ---------------------------------------------------------------------------
# Error callback
# ---------------------------------------------------------------------------

_errcall_func = None

def seterrcall(func):
    """Set callback for floating-point error handler."""
    global _errcall_func
    old = _errcall_func
    _errcall_func = func
    return old

def geterrcall():
    """Get callback for floating-point error handler."""
    return _errcall_func
