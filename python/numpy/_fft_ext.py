"""FFT Python-level wrappers."""
import _numpy_native as _native
from _numpy_native import ndarray, fft as _fft_native
from ._helpers import _builtin_range, _flat_arraylike_data
from ._creation import array, asarray
from ._manipulation import stack, roll, apply_along_axis
from ._math import conj

__all__ = [
    '_fft_rfftfreq',
    '_fft_fftfreq',
    '_fft_fftshift',
    '_fft_ifftshift',
    '_fft_complex_column_fft',
    '_fft_fft2',
    '_fft_ifft2',
    '_fft_fftn',
    '_fft_ifftn',
    '_fft_rfft',
    '_fft_irfft',
    '_fft_rfft2',
    '_fft_irfft2',
    '_fft_rfftn',
    '_fft_irfftn',
    '_fft_hfft',
    '_fft_ihfft',
]

# Keep a reference to the native fft module (same object as numpy.fft)
fft = _fft_native


def _fft_rfftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies for rfft."""
    val = 1.0 / (n * d)
    N = n // 2 + 1
    results = []
    for i in range(N):
        results.append(float(i) * val)
    return array(results)


def _fft_fftfreq(n, d=1.0):
    """Return the Discrete Fourier Transform sample frequencies."""
    results = []
    half = (n - 1) // 2 + 1
    for i in range(half):
        results.append(float(i) / (n * d))
    for i in range(-(n // 2), 0):
        results.append(float(i) / (n * d))
    return array(results)


def _fft_fftshift(x, axes=None):
    """Shift the zero-frequency component to the center of the spectrum."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = n // 2
        result = roll(result, shift, axis=ax)
    return result


def _fft_ifftshift(x, axes=None):
    """The inverse of fftshift."""
    x = asarray(x)
    if axes is None:
        axes = list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    result = x
    for ax in axes:
        n = result.shape[ax]
        shift = -(n // 2)
        result = roll(result, shift, axis=ax)
    return result


def _fft_complex_column_fft(row_ffts, rows, cols, inverse=False):
    """Apply FFT/IFFT along columns of a complex (rows, cols, 2) representation.

    row_ffts is a list of (cols, 2) arrays from fft.fft applied to each row.
    Returns a (rows, cols, 2) shaped array representing the 2D FFT result.
    The complex representation uses [real, imag] pairs.
    """
    fft_fn = fft.ifft if inverse else fft.fft
    # For each column j, extract real and imaginary parts across all rows,
    # apply FFT to each separately, then combine using:
    #   DFT(xr + j*xi) = DFT(xr) + j*DFT(xi)
    #   result_real = DFT(xr)_real - DFT(xi)_imag
    #   result_imag = DFT(xr)_imag + DFT(xi)_real
    col_results = []  # col_results[j] is a list of (real, imag) for each row i
    for j in range(cols):
        col_real = array([row_ffts[i][j][0] for i in range(rows)])
        col_imag = array([row_ffts[i][j][1] for i in range(rows)])
        if inverse:
            # ifft requires complex format (n, 2) - convert real arrays to complex with zero imag
            col_real_c = array([[float(col_real[i]), 0.0] for i in range(rows)])
            col_imag_c = array([[float(col_imag[i]), 0.0] for i in range(rows)])
            fft_of_real = fft_fn(col_real_c)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag_c)   # (rows, 2)
        else:
            fft_of_real = fft_fn(col_real)   # (rows, 2)
            fft_of_imag = fft_fn(col_imag)   # (rows, 2)
        # Combine: for each row i
        col_ri = []
        for i in range(rows):
            r = fft_of_real[i][0] - fft_of_imag[i][1]
            im = fft_of_real[i][1] + fft_of_imag[i][0]
            col_ri.append((r, im))
        col_results.append(col_ri)
    # Reconstruct as (rows, cols, 2) using stack
    final_rows = []
    for i in range(rows):
        row_data = []
        for j in range(cols):
            row_data.append([col_results[j][i][0], col_results[j][i][1]])
        final_rows.append(array(row_data))
    return stack(final_rows)


def _fft_fft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("fft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # FFT each row -> list of (cols, 2) complex arrays
    row_ffts = [fft.fft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_ffts, rows, cols, inverse=False)


def _fft_ifft2(a, s=None, axes=(-2, -1), norm=None):
    """Compute the 2-D inverse discrete Fourier Transform."""
    a = asarray(a)
    if a.ndim != 2:
        raise ValueError("ifft2 requires a 2-D array")
    rows = a.shape[0]
    cols = a.shape[1]
    # IFFT each row -> list of (cols, 2) complex arrays
    row_iffts = [fft.ifft(a[i]) for i in range(rows)]
    return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)


def _fft_fftn(a, s=None, axes=None):
    """N-dimensional FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.fft(a)
    elif a.ndim == 2:
        return fft.fft2(a, s=s)
    else:
        # For higher dimensions, apply fft2 on last two axes as approximation
        # This is a simplified implementation
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            # Apply 1D FFT along each axis using apply_along_axis
            result = apply_along_axis(lambda x: array(_flat_arraylike_data(fft.fft(array(x)))), ax, result)
        return result


def _fft_ifftn(a, s=None, axes=None):
    """N-dimensional inverse FFT."""
    a = asarray(a)
    if a.ndim == 1:
        return fft.ifft(a)
    elif a.ndim == 2:
        return fft.ifft2(a, s=s)
    elif a.ndim == 3 and a.shape[-1] == 2:
        # Complex representation from fft2/fftn: shape (rows, cols, 2)
        # This is a complex array; apply ifft2 logic
        rows = a.shape[0]
        cols = a.shape[1]
        # Extract row-wise complex data and apply ifft
        row_iffts = []
        for i in _builtin_range(rows):
            row_iffts.append(fft.ifft(a[i]))
        return _fft_complex_column_fft(row_iffts, rows, cols, inverse=True)
    else:
        result = a
        ax_list = axes if axes is not None else list(range(a.ndim))
        for ax in reversed(ax_list):
            result = apply_along_axis(lambda x: array(_flat_arraylike_data(fft.ifft(array(x)))), ax, result)
        return result


def _rfft_1d_list(data, n=None, norm=None):
    """Apply 1D rfft to a list of floats. Returns list of [real, imag] pairs."""
    import cmath as _cmath
    N = n if n is not None else len(data)
    if len(data) < N:
        data = list(data) + [0.0] * (N - len(data))
    elif len(data) > N:
        data = list(data)[:N]
    out_len = N // 2 + 1
    result = []
    for k in _builtin_range(out_len):
        s = 0.0 + 0.0j
        for n_idx in _builtin_range(N):
            angle = -2.0 * 3.141592653589793 * k * n_idx / N
            s += data[n_idx] * _cmath.exp(1j * angle)
        if norm == "ortho":
            s /= N ** 0.5
        result.append([s.real, s.imag])
    return result


def _irfft_1d_list(re_parts, im_parts, n=None, norm=None):
    """Apply 1D irfft given separate real/imag lists of rfft output. Returns real list."""
    import math as _math_mod
    m = len(re_parts)
    N = n if n is not None else 2 * (m - 1)
    # Reconstruct full Hermitian-symmetric spectrum
    full_r = [0.0] * N
    full_i = [0.0] * N
    for k in _builtin_range(min(m, N)):
        full_r[k] = re_parts[k]
        full_i[k] = im_parts[k]
    # Fill Hermitian conjugate: X[N-k] = conj(X[k]) for k=1..m-1
    for k in _builtin_range(1, m):
        conj_k = N - k
        if 0 < conj_k < N and conj_k >= m:
            full_r[conj_k] = re_parts[k]
            full_i[conj_k] = -im_parts[k]
    # IDFT
    result = []
    for nn in _builtin_range(N):
        s = 0.0
        for k in _builtin_range(N):
            angle = 2.0 * 3.141592653589793 * k * nn / N
            s += full_r[k] * _math_mod.cos(angle) - full_i[k] * _math_mod.sin(angle)
        s /= N ** 0.5 if norm == "ortho" else N
        result.append(s)
    return result


def _fft_rfft(a, n=None, axis=-1, norm=None):
    """Real FFT - only positive frequencies. Supports 1D and 2D input."""
    a = asarray(a).astype("float64")
    if a.ndim == 0:
        a = a.reshape([1])
    if a.ndim == 1:
        return array(_rfft_1d_list(_flat_arraylike_data(a), n, norm))
    elif a.ndim == 2:
        ax = axis % 2 if axis >= 0 else 2 + axis  # normalize to 0 or 1
        a_list = [_flat_arraylike_data(a[i]) for i in _builtin_range(a.shape[0])]
        rows, cols = a.shape
        if ax == 1:
            # rfft along each row
            results = [_rfft_1d_list(a_list[i], n, norm) for i in _builtin_range(rows)]
            return array(results)  # (rows, out_len, 2)
        else:
            # rfft along each column (axis=0)
            N_ax = n if n is not None else rows
            out_len = N_ax // 2 + 1
            col_results = [_rfft_1d_list([a_list[i][j] for i in _builtin_range(rows)], n, norm)
                           for j in _builtin_range(cols)]
            # Reshape to (out_len, cols, 2)
            out = [[col_results[j][k] for j in _builtin_range(cols)] for k in _builtin_range(out_len)]
            return array(out)  # (out_len, cols, 2)
    else:
        raise NotImplementedError("rfft only supports 1D and 2D arrays")


def _fft_irfft(a, n=None, axis=-1, norm=None):
    """Inverse real FFT. Supports 1D, (M,2) complex, and (A,B,2) multi-D complex."""
    a = asarray(a)
    if a.ndim == 1:
        re_parts = _flat_arraylike_data(a)
        return array(_irfft_1d_list(re_parts, [0.0] * len(re_parts), n, norm))
    elif a.ndim == 2 and a.shape[-1] == 2:
        # 1D complex: (M, 2) from rfft
        data_list = [_flat_arraylike_data(a[i]) for i in _builtin_range(a.shape[0])]
        re_parts = [row[0] for row in data_list]
        im_parts = [row[1] for row in data_list]
        return array(_irfft_1d_list(re_parts, im_parts, n, norm))
    elif a.ndim == 3 and a.shape[-1] == 2:
        # Multi-D complex (A, B, 2): axis refers to the logical 2D complex axis
        logical_ndim = 2
        ax = axis if axis >= 0 else axis + logical_ndim
        a_list = [[_flat_arraylike_data(a[i, j]) for j in _builtin_range(a.shape[1])] for i in _builtin_range(a.shape[0])]
        dim0, dim1, _ = a.shape
        if ax == 1:
            # irfft along axis 1: apply to each a[i] of shape (dim1, 2)
            results = []
            for i in _builtin_range(dim0):
                re_parts = [a_list[i][k][0] for k in _builtin_range(dim1)]
                im_parts = [a_list[i][k][1] for k in _builtin_range(dim1)]
                results.append(_irfft_1d_list(re_parts, im_parts, n, norm))
            return array(results)  # (dim0, N)
        else:
            # irfft along axis 0: apply to each column a[:, j, :] of shape (dim0, 2)
            N_out = n if n is not None else 2 * (dim0 - 1)
            col_results = []
            for j in _builtin_range(dim1):
                re_parts = [a_list[k][j][0] for k in _builtin_range(dim0)]
                im_parts = [a_list[k][j][1] for k in _builtin_range(dim0)]
                col_results.append(_irfft_1d_list(re_parts, im_parts, n, norm))
            # Reassemble as (N_out, dim1)
            out = [[col_results[j][i] for j in _builtin_range(dim1)] for i in _builtin_range(N_out)]
            return array(out)  # (N_out, dim1)
    else:
        # Fallback for other shapes
        data_list = _flat_arraylike_data(a)
        if isinstance(data_list, list) and len(data_list) > 0 and isinstance(data_list[0], list):
            re_parts = [row[0] if isinstance(row, list) else row for row in data_list]
            im_parts = [row[1] if isinstance(row, list) else 0.0 for row in data_list]
        else:
            re_parts = data_list if isinstance(data_list, list) else [data_list]
            im_parts = [0.0] * len(re_parts)
        return array(_irfft_1d_list(re_parts, im_parts, n, norm))


def _fft_rfft2(a, s=None, axes=(-2, -1), norm=None):
    """2D real FFT — rfft along last axis for each row."""
    a = asarray(a)
    if a.ndim < 2:
        return fft.rfft(a)
    rows = [_flat_arraylike_data(a[i]) for i in _builtin_range(a.shape[0])]
    rfft_rows = []
    for row in rows:
        r = fft.rfft(array(row))
        rfft_rows.append(r)
    # Stack results: each r is an ndarray
    return stack(rfft_rows)


def _fft_irfft2(a, s=None, axes=(-2, -1), norm=None):
    """2D inverse real FFT — irfft along last axis for each row."""
    a = asarray(a)
    if a.ndim < 2:
        return fft.irfft(a)
    n_val = s[-1] if s else None
    result_rows = []
    for i in range(a.shape[0]):
        row = a[i]
        r = fft.irfft(row, n=n_val)
        result_rows.append(r)
    return stack(result_rows)


def _fft_rfftn(a, s=None, axes=None, norm=None):
    """N-D real FFT."""
    return _fft_rfft2(a, s=s, norm=norm)


def _fft_irfftn(a, s=None, axes=None, norm=None):
    """N-D inverse real FFT."""
    a = asarray(a)
    if axes is not None:
        ax_list = list(axes) if hasattr(axes, '__iter__') else [axes]
        is_complex = str(a.dtype).startswith('complex')
        if is_complex:
            # For complex input, delegate to ifftn which handles complex-to-complex
            result = a
            for ax in reversed(ax_list):
                result = _fft_ifftn(result, s=s, axes=[ax])
            return result
        else:
            # For real input, apply irfft along each axis
            return _fft_irfft2(a, s=s, norm=norm)
    return _fft_irfft2(a, s=s, norm=norm)


def _fft_hfft(a, n=None, axis=-1, norm=None):
    """Hermitian FFT - input is Hermitian symmetric, output is real."""
    a = asarray(a)
    # hfft(a) = irfft(conj(a)) * N
    conj_a = conj(a)
    N = n if n is not None else 2 * (a.shape[0] - 1) if a.ndim > 0 else 2
    result = fft.irfft(conj_a, n=N, norm=norm)
    if norm != 'ortho':
        result = result * N
    return result


def _fft_ihfft(a, n=None, axis=-1, norm=None):
    """Inverse Hermitian FFT - input is real, output is Hermitian."""
    a = asarray(a)
    # ihfft(a) = conj(rfft(a)) / N
    N = n if n is not None else (a.shape[0] if a.ndim > 0 else 1)
    result = fft.rfft(a, n=N, norm=norm)
    return conj(result) / N if norm != 'ortho' else conj(result)


# Monkey-patch fft module with Python-level extension functions
fft.rfftfreq = _fft_rfftfreq
fft.fftfreq = _fft_fftfreq
fft.fftshift = _fft_fftshift
fft.ifftshift = _fft_ifftshift
fft.fft2 = _fft_fft2
fft.ifft2 = _fft_ifft2
fft.fftn = _fft_fftn
fft.ifftn = _fft_ifftn
fft.rfft = _fft_rfft
fft.irfft = _fft_irfft
fft.rfft2 = _fft_rfft2
fft.irfft2 = _fft_irfft2
fft.rfftn = _fft_rfftn
fft.irfftn = _fft_irfftn
fft.hfft = _fft_hfft
fft.ihfft = _fft_ihfft
