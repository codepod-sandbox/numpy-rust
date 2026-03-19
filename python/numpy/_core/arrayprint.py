"""numpy._core.arrayprint - array printing utilities."""
import numpy

# Set of types that should be printed without type prefix
_typelessdata = {
    numpy.bool_,
    numpy.int64,
    numpy.float64,
    numpy.complex128,
}

# Global print options state
_print_options = {
    'precision': 8,
    'threshold': 1000,
    'edgeitems': 3,
    'linewidth': 75,
    'suppress': False,
    'nanstr': 'nan',
    'infstr': 'inf',
    'sign': '-',
    'formatter': None,
    'floatmode': 'maxprec',
    'legacy': False,
    'override_repr': None,
}


def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None, nanstr=None,
                     infstr=None, formatter=None, sign=None,
                     floatmode=None, legacy=None, override_repr=None):
    """Set printing options."""
    if precision is not None:
        _print_options['precision'] = precision
    if threshold is not None:
        _print_options['threshold'] = threshold
    if edgeitems is not None:
        _print_options['edgeitems'] = edgeitems
    if linewidth is not None:
        _print_options['linewidth'] = linewidth
    if suppress is not None:
        _print_options['suppress'] = suppress
    if nanstr is not None:
        _print_options['nanstr'] = nanstr
    if infstr is not None:
        _print_options['infstr'] = infstr
    if formatter is not None:
        _print_options['formatter'] = formatter
    if sign is not None:
        if sign not in ('-', '+', ' '):
            raise ValueError("sign must be one of '-', '+', or ' '")
        _print_options['sign'] = sign
    if floatmode is not None:
        _print_options['floatmode'] = floatmode
    if legacy is not None:
        _print_options['legacy'] = legacy
    if override_repr is not None:
        _print_options['override_repr'] = override_repr


def get_printoptions():
    """Return the current print options."""
    return dict(_print_options)


def printoptions(**kwargs):
    """Context manager for temporarily setting print options."""
    class _PrintoptionsCtx:
        def __init__(self, **kw):
            self._kw = kw
            self._old = None
        def __enter__(self):
            self._old = get_printoptions()
            set_printoptions(**self._kw)
            return self
        def __exit__(self, *args):
            set_printoptions(**self._old)
    return _PrintoptionsCtx(**kwargs)


def array2string(a, max_line_width=None, precision=None, suppress_small=None,
                 separator=' ', prefix='', style=None, formatter=None,
                 threshold=None, edgeitems=None, sign=None, floatmode=None,
                 suffix='', legacy=None):
    """Return a string representation of an array."""
    # For now, delegate to str(a) with basic formatting
    return str(a)


def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """Return the string representation of an array."""
    return repr(arr)


def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """Return a string representation of an array."""
    return str(a)


def format_float_positional(x, precision=None, unique=True, fractional=True,
                             trim='k', sign=False, pad_left=None,
                             pad_right=None, min_digits=None):
    """Format a floating-point scalar as a decimal string in positional notation."""
    if precision is None:
        precision = _print_options['precision']
    return format(float(x), f'.{precision}f')


def format_float_scientific(x, precision=None, unique=True, trim='k',
                             sign=False, pad_left=None, exp_digits=None,
                             min_digits=None):
    """Format a floating-point scalar as a decimal string in scientific notation."""
    if precision is None:
        precision = _print_options['precision']
    return format(float(x), f'.{precision}e')


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
