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
        return _unique_positional(v, dtype_name)

    if unique:
        # Unique mode: find shortest representation that round-trips
        # precision can REDUCE digits (show min(unique, precision))
        # min_digits can ADD digits (show max(unique, min_digits))
        #
        # For fractional=True: precision/min_digits count fractional digits
        # For fractional=False: precision/min_digits count significant digits

        # Get unique significant digit count for this dtype
        ndig_unique, _ = _get_float_precision_digits_for_pos(v, dtype_name)

        if fractional:
            # fractional mode: precision/min_digits count digits after decimal point
            if precision is not None and min_digits is not None:
                # Both specified: use max(min_digits, min(unique_frac, precision))
                # But precision limits, min_digits extends
                # Result: clamp to [min_digits, precision] around unique
                base = _format_unique_fractional(v, dtype_name, precision, min_digits)
            elif precision is not None:
                # Precision limits fractional digits
                base = _unique_positional(v, dtype_name)
                parts = base.split('.')
                frac = parts[1] if len(parts) > 1 else ''
                if len(frac) > precision:
                    rounded = round(v, precision)
                    base = f'{rounded:.{precision}f}'
                    if '.' not in base:
                        base += '.'
                    # Strip trailing zeros for unique
                    base = base.rstrip('0')
                    if base.endswith('.'):
                        pass  # keep dot
            elif min_digits is not None:
                # min_digits extends: show exact integer digits + at least min_digits fractional
                full = f'{v:.{max(min_digits, 0)}f}'
                if '.' not in full:
                    full += '.'
                base = full
            else:
                base = _unique_positional(v, dtype_name)
        else:
            # non-fractional mode: precision/min_digits count significant digits
            if precision is not None and precision == 0:
                raise ValueError(
                    "unique mode with fractional=False requires precision > 0")

            # Determine target significant digits
            if precision is not None and min_digits is not None:
                target = max(min_digits, min(ndig_unique, precision))
            elif precision is not None:
                target = min(ndig_unique, precision)
            elif min_digits is not None:
                target = max(ndig_unique, min_digits)
            else:
                target = ndig_unique

            base = _round_to_sig_digits_positional(v, target)

        return base

    else:
        # Non-unique mode: format with exact precision
        if fractional:
            result = f'{v:.{precision}f}'
        else:
            # Precision counts significant digits
            if v == 0:
                return '0.' + '0' * max(0, precision - 1)
            result = _round_to_sig_digits_positional(v, precision)
        # Ensure decimal point is always present
        if '.' not in result:
            result += '.'
        return result


def _get_float_precision_digits_for_pos(v, dtype_name):
    """Get unique significant digit count for positional formatting."""
    import struct, math
    if not math.isfinite(v) or v == 0:
        return 1, v
    if dtype_name == 'float16':
        pack_fmt = 'e'
    elif dtype_name == 'float32':
        pack_fmt = 'f'
    else:
        # float64: use repr
        s = repr(v).lstrip('-')
        if 'e' in s:
            mantissa = s.split('e')[0]
        else:
            mantissa = s
        mantissa = mantissa.replace('.', '').lstrip('0')
        return len(mantissa) if mantissa else 1, v

    try:
        b1 = struct.pack(pack_fmt, v)
    except (OverflowError, struct.error):
        return 17, v  # fallback to max

    for ndig in range(1, 20):
        s = f'{v:.{ndig}g}'
        sv = float(s)
        try:
            b2 = struct.pack(pack_fmt, sv)
            if b1 == b2:
                return ndig, v
        except (OverflowError, struct.error):
            continue
    return 17, v


def _format_unique_fractional(v, dtype_name, precision, min_digits):
    """Format with both precision (limit) and min_digits (extend) for fractional mode.
    When min_digits is specified, use full exact integer digits."""
    # Use full exact representation for integer part (since min_digits extends)
    target_frac = max(min_digits, 0)
    full = f'{v:.{target_frac}f}'
    if '.' not in full:
        full += '.'
    parts = full.split('.')
    int_part = parts[0]
    frac = parts[1] if len(parts) > 1 else ''

    # precision limits fractional digits (but min_digits guarantees a minimum)
    # precision can only REDUCE, min_digits can only EXTEND
    # But when both specified: show max(min_digits, clamp_to_precision) fractional digits
    # In practice: just show min_digits fractional digits (precision doesn't reduce below min_digits)

    # Ensure min_digits fractional digits
    while len(frac) < min_digits:
        frac += '0'

    if frac:
        return int_part + '.' + frac
    return int_part + '.'


def _round_to_sig_digits_positional(v, precision):
    """Round v to precision significant digits and format as positional string."""
    import math
    if v == 0:
        return '0.'
    neg = v < 0
    av = abs(v)
    mag = math.floor(math.log10(av)) + 1  # number of integer digits
    nfrac = max(0, precision - mag)

    if nfrac > 0:
        # Has fractional digits
        result = f'{av:.{nfrac}f}'
    else:
        # All integer digits; round and zero-pad
        # Use the g format to get the right significant digits, then reconstruct
        s = f'{av:.{precision}g}'
        # Parse the result
        if 'e' in s or 'E' in s:
            # Scientific notation - convert to positional with zero padding
            parts = s.lower().split('e')
            mantissa = parts[0]
            exp = int(parts[1])
            m_parts = mantissa.split('.')
            int_d = m_parts[0]
            frac_d = m_parts[1] if len(m_parts) > 1 else ''
            all_d = int_d + frac_d
            dpos = 1 + exp  # digits before decimal
            if dpos >= len(all_d):
                result = all_d + '0' * (dpos - len(all_d)) + '.'
            else:
                result = all_d[:dpos] + '.' + all_d[dpos:]
        else:
            if '.' not in s:
                s += '.'
            result = s

    if '.' not in result:
        result += '.'
    if neg:
        result = '-' + result
    return result


def _unique_positional(v, dtype_name):
    """Get unique positional representation for a given float type."""
    import struct, math
    if dtype_name == 'float64':
        return _repr_to_positional(v)

    # For float16/float32, find shortest significant digits
    pack_fmt = 'e' if dtype_name == 'float16' else 'f'
    try:
        b1 = struct.pack(pack_fmt, v)
    except (OverflowError, struct.error):
        return _repr_to_positional(v)

    # Find minimum significant digits using g format
    for ndig in range(1, 20):
        s = f'{v:.{ndig}g}'
        sv = float(s)
        try:
            b2 = struct.pack(pack_fmt, sv)
            if b1 == b2:
                # Convert the g-format result to positional notation
                # Parse the value and format as positional
                if 'e' in s or 'E' in s:
                    # Scientific notation - convert to positional
                    parts = s.lower().split('e')
                    mantissa = parts[0]
                    exp = int(parts[1])
                    neg = mantissa.startswith('-')
                    if neg:
                        mantissa = mantissa[1:]
                    m_parts = mantissa.split('.')
                    int_d = m_parts[0]
                    frac_d = m_parts[1] if len(m_parts) > 1 else ''
                    all_d = int_d + frac_d
                    dpos = 1 + exp  # digits before decimal
                    if dpos <= 0:
                        result = '0.' + '0' * (-dpos) + all_d
                    elif dpos >= len(all_d):
                        result = all_d + '0' * (dpos - len(all_d)) + '.'
                    else:
                        result = all_d[:dpos] + '.' + all_d[dpos:]
                    if neg:
                        result = '-' + result
                else:
                    # Already positional
                    if '.' not in s:
                        s += '.'
                    result = s
                return result
        except (OverflowError, struct.error):
            continue
    # Fallback
    max_frac = 5 if dtype_name == 'float16' else 9
    return f'{v:.{max_frac}f}'


def _repr_to_positional(v):
    """Convert a float64 to positional notation using repr for unique digits.
    Strips trailing zeros from fractional part (Dragon4-style)."""
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

    # Strip trailing zeros from fractional part (Dragon4 style)
    if '.' in result:
        result = result.rstrip('0')
        if result.endswith('.'):
            pass  # keep the trailing dot
        elif '.' not in result:
            result += '.'

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


def _unique_scientific(v, dtype_name):
    """Get unique scientific representation for a given float type."""
    import struct, math

    if v == 0.0:
        sign = '-' if math.copysign(1.0, v) < 0 else ''
        return f'{sign}0.e+00'

    if dtype_name == 'float64':
        # Use Python repr which gives unique shortest for float64
        s = repr(v)
        neg = s.startswith('-')
        s2 = s.lstrip('-')
        if 'e' in s2:
            # Already scientific - normalize
            mant, exp_s = s2.split('e')
            exp = int(exp_s)
        else:
            # Positional - convert to scientific
            if '.' in s2:
                int_d, frac_d = s2.split('.')
            else:
                int_d, frac_d = s2, ''
            all_d = int_d + frac_d
            # Find first non-zero
            first_nz = 0
            for i, c in enumerate(all_d):
                if c != '0':
                    first_nz = i
                    break
            if int_d == '0':
                # 0.00xyz... → x.yz...e-N
                exp = -(len(frac_d) - len(frac_d.lstrip('0'))) - 1 + len(int_d) - 1
                # Actually, easier to compute from the value
                exp = math.floor(math.log10(abs(v)))
                all_d = all_d.lstrip('0')
            else:
                exp = len(int_d) - 1
                all_d = all_d  # already correct

            # Strip trailing zeros from unique digits
            all_d = all_d.rstrip('0') if len(all_d) > 1 else all_d
            if len(all_d) == 0:
                all_d = '0'

            mant = all_d[0] + ('.' + all_d[1:] if len(all_d) > 1 else '.')

        exp_sign = '+' if exp >= 0 else '-'
        exp_str = f'{abs(exp):02d}'
        result = f'{mant}e{exp_sign}{exp_str}'
        if neg:
            result = '-' + result
        return result

    # For float16/float32: find shortest significant digits
    pack_fmt = 'e' if dtype_name == 'float16' else 'f'
    try:
        b1 = struct.pack(pack_fmt, v)
    except (OverflowError, struct.error):
        return _unique_scientific(v, 'float64')

    for ndig in range(1, 20):
        s = f'{v:.{ndig}g}'
        sv = float(s)
        try:
            b2 = struct.pack(pack_fmt, sv)
            if b1 == b2:
                # Convert to scientific notation
                frac_digits = max(0, ndig - 1)
                return f'{v:.{frac_digits}e}'
        except (OverflowError, struct.error):
            continue

    # Fallback
    return f'{v:.6e}'


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
        if unique:
            # Get unique scientific representation
            result = _unique_scientific(v, dtype_name)

            if precision is not None:
                # precision can reduce digits from unique
                neg = result.startswith('-')
                core = result[1:] if neg else result
                if 'e' in core:
                    mant, exp_part = core.split('e', 1)
                    if '.' in mant:
                        m_int, m_frac = mant.split('.', 1)
                    else:
                        m_int, m_frac = mant, ''
                    if len(m_frac) > precision:
                        # Re-format with limited precision
                        result = f'{v:.{precision}e}'
                    # else: keep unique result (precision doesn't add digits)
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
            core = result[1:] if neg else result
            if 'e' in core:
                mantissa, exp_part = core.split('e', 1)
                if '.' in mantissa:
                    m_int, m_frac = mantissa.split('.', 1)
                else:
                    m_int, m_frac = mantissa, ''
                if len(m_frac) < min_digits:
                    # Need more digits - compute from the actual value
                    extended = f'{v:.{min_digits}e}'
                    ext_neg = extended.startswith('-')
                    ext_core = extended[1:] if ext_neg else extended
                    if 'e' in ext_core:
                        ext_mant, ext_exp = ext_core.split('e', 1)
                        # Use extended mantissa digits but keep original exponent
                        result = ext_mant + 'e' + exp_part
                        if neg:
                            result = '-' + result
                    else:
                        pass  # shouldn't happen
                # else: unique has enough digits, keep as is

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
