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


def _get_float_precision_digits(x):
    """Get the number of significant digits for unique representation based on dtype."""
    import struct, math
    v = float(x)
    if not math.isfinite(v) or v == 0.0:
        return None, v  # special values
    dtype = getattr(x, 'dtype', None)
    if dtype is not None:
        dname = str(dtype)
        if dname == 'float16':
            # float16: ~3.31 decimal digits, need up to 5 significant digits
            # Find shortest repr that round-trips through float16
            for ndig in range(1, 12):
                s = f'{v:.{ndig}g}'
                sv = float(s)
                # Check round-trip through float16
                b1 = struct.pack('e', v)
                try:
                    b2 = struct.pack('e', sv)
                except (OverflowError, struct.error):
                    continue
                if b1 == b2:
                    return ndig, v
            return 5, v
        elif dname == 'float32':
            # float32: ~7.22 decimal digits
            for ndig in range(1, 20):
                s = f'{v:.{ndig}g}'
                sv = float(s)
                b1 = struct.pack('f', v)
                try:
                    b2 = struct.pack('f', sv)
                except (OverflowError, struct.error):
                    continue
                if b1 == b2:
                    return ndig, v
            return 9, v
    # float64: use Python's repr which already gives shortest unique
    s = repr(v)
    # Count significant digits
    s2 = s.lstrip('-')
    if 'e' in s2:
        mantissa = s2.split('e')[0]
    else:
        mantissa = s2
    mantissa = mantissa.replace('.', '')
    mantissa = mantissa.lstrip('0')
    return len(mantissa) if mantissa else 1, v


def _format_positional_raw(v, precision, unique, fractional, min_digits, dtype_name=None):
    """Core positional formatting logic."""
    import math, struct
    if not math.isfinite(v):
        if math.isnan(v):
            return 'nan'
        return 'inf' if v > 0 else '-inf'

    if unique and precision is None and min_digits is None:
        # Unique mode, no precision constraint - find shortest representation
        # that round-trips
        if dtype_name == 'float16':
            for ndig in range(0, 12):
                s = f'{v:.{ndig}f}'
                sv = float(s)
                try:
                    b1 = struct.pack('e', v)
                    b2 = struct.pack('e', sv)
                    if b1 == b2:
                        if '.' not in s:
                            s += '.'
                        return s
                except (OverflowError, struct.error):
                    pass
            return f'{v:.5f}'
        elif dtype_name == 'float32':
            for ndig in range(0, 20):
                s = f'{v:.{ndig}f}'
                sv = float(s)
                try:
                    b1 = struct.pack('f', v)
                    b2 = struct.pack('f', sv)
                    if b1 == b2:
                        if '.' not in s:
                            s += '.'
                        return s
                except (OverflowError, struct.error):
                    pass
            return f'{v:.9f}'
        else:
            # float64 - use Python repr, convert to positional
            return _repr_to_positional(v)

    if unique:
        # Unique mode with precision constraint
        if dtype_name == 'float16':
            base = _unique_positional(v, 'float16')
        elif dtype_name == 'float32':
            base = _unique_positional(v, 'float32')
        else:
            base = _repr_to_positional(v)

        if precision is not None:
            if fractional:
                # Precision limits fractional digits
                # Round the base value to that many fractional digits
                rounded = f'{v:.{precision}f}'
                # But use the uniquely-rounded value
                # First get the unique digits
                parts = base.split('.')
                int_part = parts[0]
                frac_part = parts[1] if len(parts) > 1 else ''
                if len(frac_part) > precision:
                    # Need to round
                    rounded_val = round(v, precision)
                    base = f'{rounded_val:.{precision}f}'
                    # But strip trailing zeros to match unique
                    parts2 = base.split('.')
                    frac2 = parts2[1] if len(parts2) > 1 else ''
                    # Keep unique precision
                    base = parts2[0] + '.' + frac2
            else:
                # precision counts total significant digits
                if precision == 0:
                    raise ValueError(
                        "unique mode with fractional=False requires precision > 0")
                # Round to given significant digits
                if v != 0:
                    import math as _m
                    mag = _m.floor(_m.log10(abs(v))) + 1
                    nfrac = max(0, precision - int(mag))
                    base = f'{v:.{nfrac}f}'
                else:
                    base = '0.'

        if min_digits is not None:
            # Extend with zeros if needed
            parts = base.split('.')
            int_part = parts[0]
            frac_part = parts[1] if len(parts) > 1 else ''
            if not fractional or fractional:
                # min_digits applies to fractional part for fractional=True
                # and significant digits for fractional=False
                if fractional:
                    while len(frac_part) < min_digits:
                        frac_part += '0'
                else:
                    # Count sig digits
                    sig = int_part.lstrip('-').lstrip('0') + frac_part
                    total_sig = len(sig.rstrip('0')) if sig else 0
                    if not sig:
                        total_sig = 0
                    # Need to extend to min_digits significant digits
                    needed = min_digits - total_sig
                    if needed > 0:
                        # Extend fractional part or integer significant digits
                        while len(int_part.lstrip('-').lstrip('0') + frac_part) < min_digits:
                            frac_part += '0'
            base = int_part + '.' + frac_part

        return base

    else:
        # Non-unique mode: format with exact precision
        if fractional:
            return f'{v:.{precision}f}'
        else:
            # Precision counts significant digits
            if v == 0:
                return '0.' + '0' * max(0, precision - 1)
            import math as _m
            mag = _m.floor(_m.log10(abs(v))) + 1
            nfrac = max(0, precision - int(mag))
            return f'{v:.{nfrac}f}'


def _unique_positional(v, dtype_name):
    """Get unique positional representation for a given float type."""
    import struct
    if dtype_name == 'float16':
        for ndig in range(0, 12):
            s = f'{v:.{ndig}f}'
            sv = float(s)
            try:
                b1 = struct.pack('e', v)
                b2 = struct.pack('e', sv)
                if b1 == b2:
                    if '.' not in s:
                        s += '.'
                    return s
            except (OverflowError, struct.error):
                pass
        return f'{v:.5f}'
    elif dtype_name == 'float32':
        for ndig in range(0, 20):
            s = f'{v:.{ndig}f}'
            sv = float(s)
            try:
                b1 = struct.pack('f', v)
                b2 = struct.pack('f', sv)
                if b1 == b2:
                    if '.' not in s:
                        s += '.'
                    return s
            except (OverflowError, struct.error):
                pass
        return f'{v:.9f}'
    else:
        return _repr_to_positional(v)


def _repr_to_positional(v):
    """Convert a float64 to positional notation using repr for unique digits."""
    import math
    if v == 0.0:
        if math.copysign(1.0, v) < 0:
            return '-0.'
        return '0.'

    s = repr(v)
    neg = s.startswith('-')
    s2 = s.lstrip('-')

    if 'e' in s2:
        # Scientific notation from repr
        mantissa, exp_str = s2.split('e')
        exp = int(exp_str)
        parts = mantissa.split('.')
        int_digits = parts[0]
        frac_digits = parts[1] if len(parts) > 1 else ''
        all_digits = int_digits + frac_digits
        # Position of decimal point: after int_digits[0], shifted by exp
        decimal_pos = 1 + exp  # number of digits before decimal point
        if decimal_pos <= 0:
            # Need leading zeros: 0.000...digits
            result = '0.' + '0' * (-decimal_pos) + all_digits
        elif decimal_pos >= len(all_digits):
            # All digits before decimal, pad with zeros
            result = all_digits + '0' * (decimal_pos - len(all_digits)) + '.'
        else:
            result = all_digits[:decimal_pos] + '.' + all_digits[decimal_pos:]
    else:
        # Already in positional notation
        if '.' not in s2:
            result = s2 + '.'
        else:
            result = s2

    if neg:
        result = '-' + result
    return result


def format_float_positional(x, precision=None, unique=True, fractional=True,
                             trim='k', sign=False, pad_left=None,
                             pad_right=None, min_digits=None):
    """Format a floating-point scalar as a decimal string in positional notation."""
    import math

    v = float(x)

    # Detect dtype
    dtype = getattr(x, 'dtype', None)
    dtype_name = str(dtype) if dtype is not None else 'float64'
    if dtype_name not in ('float16', 'float32', 'float64'):
        dtype_name = 'float64'

    # Handle special values
    if math.isnan(v):
        result = 'nan'
    elif math.isinf(v):
        result = 'inf' if v > 0 else '-inf'
    else:
        # Check for overflow-causing precision/padding
        if not unique and precision is not None and precision > 10000:
            raise RuntimeError("Float formatting result too large")
        if pad_left is not None and pad_left > 10000:
            raise RuntimeError("Float formatting result too large")
        if pad_right is not None and pad_right > 10000:
            raise RuntimeError("Float formatting result too large")

        result = _format_positional_raw(v, precision, unique, fractional,
                                         min_digits, dtype_name)

    # Apply trimming
    if result not in ('nan', 'inf', '-inf'):
        result = _apply_trim(result, trim)

    # Apply sign
    if sign and not result.startswith('-'):
        result = '+' + result

    # Apply padding
    if result not in ('nan', 'inf', '-inf', '+nan', '+inf'):
        result = _apply_padding(result, pad_left, pad_right)

    return result


def _apply_trim(s, trim):
    """Apply trimming mode to a formatted float string."""
    if trim == 'k':
        # Keep trailing zeros (default)
        return s
    neg = s.startswith('-')
    if neg:
        core = s[1:]
    else:
        core = s

    if '.' not in core:
        if neg:
            return '-' + core
        return core

    int_part, frac_part = core.split('.', 1)

    if trim == '.':
        # Remove trailing zeros, keep dot only if frac non-empty after strip
        frac_part = frac_part.rstrip('0')
        if frac_part:
            result = int_part + '.' + frac_part
        else:
            result = int_part + '.'
    elif trim == '0':
        # Remove trailing zeros but keep at least one digit after dot
        frac_part = frac_part.rstrip('0')
        if not frac_part:
            frac_part = '0'
        result = int_part + '.' + frac_part
    elif trim == '-':
        # Remove trailing zeros and dot
        frac_part = frac_part.rstrip('0')
        if frac_part:
            result = int_part + '.' + frac_part
        else:
            result = int_part
    else:
        result = int_part + '.' + frac_part

    if neg:
        return '-' + result
    return result


def _apply_padding(s, pad_left, pad_right):
    """Apply left/right padding to a formatted float string."""
    if pad_left is None and pad_right is None:
        return s

    neg = s.startswith('-')
    if neg:
        core = s[1:]
    else:
        core = s

    if '.' in core:
        int_part, frac_part = core.split('.', 1)
    else:
        int_part = core
        frac_part = ''

    # Build the left side (sign + int_part)
    left = ('-' + int_part) if neg else int_part

    if pad_left is not None:
        while len(left) < pad_left:
            left = ' ' + left

    result = left + '.' + frac_part if frac_part or '.' in core else left

    if pad_right is not None and '.' in result:
        dot_pos = result.index('.')
        frac = result[dot_pos+1:]
        while len(frac) < pad_right:
            frac += ' '
        result = result[:dot_pos+1] + frac

    return result


def format_float_scientific(x, precision=None, unique=True, trim='k',
                             sign=False, pad_left=None, exp_digits=None,
                             min_digits=None):
    """Format a floating-point scalar as a decimal string in scientific notation."""
    import math, struct

    v = float(x)

    # Detect dtype
    dtype = getattr(x, 'dtype', None)
    dtype_name = str(dtype) if dtype is not None else 'float64'
    if dtype_name not in ('float16', 'float32', 'float64'):
        dtype_name = 'float64'

    # Handle special values
    if math.isnan(v):
        result = 'nan'
    elif math.isinf(v):
        result = 'inf' if v > 0 else '-inf'
    else:
        if unique and precision is None:
            # Get unique significant digits count for this dtype
            ndig, _ = _get_float_precision_digits(x)
            # Format with that many digits in scientific notation
            frac_digits = max(0, ndig - 1)
            result = f'{v:.{frac_digits}e}'
        elif unique and precision is not None:
            # Unique with precision limit
            ndig, _ = _get_float_precision_digits(x)
            frac_digits = max(0, ndig - 1)
            if precision < frac_digits:
                # Round to fewer digits
                result = f'{v:.{precision}e}'
            else:
                result = f'{v:.{frac_digits}e}'
        else:
            # Non-unique: use exact precision
            if precision is None:
                precision = _print_options['precision']
            result = f'{v:.{precision}e}'

        # Ensure mantissa always has a dot
        if 'e' in result:
            _neg = result.startswith('-')
            _core = result[1:] if _neg else result
            _mant, _exp = _core.split('e', 1)
            if '.' not in _mant:
                _mant += '.'
                _core = _mant + 'e' + _exp
                result = ('-' + _core) if _neg else _core

        if min_digits is not None and result not in ('nan', 'inf', '-inf'):
            # Ensure minimum number of fractional digits in mantissa
            neg = result.startswith('-')
            if neg:
                core = result[1:]
            else:
                core = result
            if 'e' in core:
                mantissa, exp_part = core.split('e', 1)
                if '.' in mantissa:
                    m_int, m_frac = mantissa.split('.', 1)
                else:
                    m_int = mantissa
                    m_frac = ''
                while len(m_frac) < min_digits:
                    m_frac += '0'
                mantissa = m_int + '.' + m_frac
                core = mantissa + 'e' + exp_part
            if neg:
                result = '-' + core
            else:
                result = core

    # Apply trimming
    if result not in ('nan', 'inf', '-inf'):
        result = _apply_sci_trim(result, trim)

    # Apply exp_digits
    if exp_digits is not None and result not in ('nan', 'inf', '-inf'):
        result = _apply_exp_digits(result, exp_digits)

    # Apply sign
    if sign and not result.startswith('-'):
        result = '+' + result

    return result


def _apply_sci_trim(s, trim):
    """Apply trimming to scientific notation string."""
    if trim == 'k':
        return s
    neg = s.startswith('-')
    if neg:
        core = s[1:]
    else:
        core = s

    if 'e' not in core:
        if neg:
            return '-' + core
        return core

    mantissa, exp_part = core.split('e', 1)

    if '.' not in mantissa:
        if neg:
            return '-' + mantissa + 'e' + exp_part
        return mantissa + 'e' + exp_part

    int_part, frac_part = mantissa.split('.', 1)

    if trim == '.':
        frac_part = frac_part.rstrip('0')
        if frac_part:
            mantissa = int_part + '.' + frac_part
        else:
            mantissa = int_part + '.'
    elif trim == '0':
        frac_part = frac_part.rstrip('0')
        if not frac_part:
            frac_part = '0'
        mantissa = int_part + '.' + frac_part
    elif trim == '-':
        frac_part = frac_part.rstrip('0')
        if frac_part:
            mantissa = int_part + '.' + frac_part
        else:
            mantissa = int_part

    result = mantissa + 'e' + exp_part
    if neg:
        return '-' + result
    return result


def _apply_exp_digits(s, exp_digits):
    """Ensure minimum exponent digits in scientific notation."""
    if 'e' not in s:
        return s
    neg = s.startswith('-')
    if neg:
        core = s[1:]
    else:
        core = s

    mantissa, exp_part = core.split('e', 1)
    exp_sign = exp_part[0] if exp_part[0] in '+-' else '+'
    exp_num = exp_part.lstrip('+-')
    while len(exp_num) < exp_digits:
        exp_num = '0' + exp_num
    result = mantissa + 'e' + exp_sign + exp_num
    if neg:
        return '-' + result
    return result


def __getattr__(name):
    if hasattr(numpy, name):
        return getattr(numpy, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
