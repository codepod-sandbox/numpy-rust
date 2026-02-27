# Tier 10: Numerical Utilities — Design

## Goal

Add meshgrid, pad, interp, gradient, polyfit/polyval, and linalg.lstsq — the most common numerical utilities in LLM-generated data science code.

## Features

### 1. `meshgrid(*xi, indexing='xy')` — Coordinate grids
- Takes N 1-D arrays, returns N N-D arrays
- For 2 inputs of length M, N: returns two (N, M) arrays (xy indexing) or (M, N) (ij indexing)
- Implementation: repeat/reshape each input across other dimensions

### 2. `pad(array, pad_width, mode='constant', constant_values=0)`
- Pad array along each dimension
- Support `mode='constant'` only (covers 95% of LLM usage)
- `pad_width` can be int (all sides), tuple of (before, after), or tuple of tuples

### 3. `interp(x, xp, fp)` — Linear interpolation
- 1-D piecewise linear interpolation
- xp must be sorted; values outside range clamp to fp[0]/fp[-1]
- Pure arithmetic on sorted Float64 arrays

### 4. `gradient(f, *varargs)` — Numerical gradient
- Central differences in interior, forward/backward at edges
- For 1-D: returns single array; for N-D: returns list of arrays
- `varargs` specifies spacing (default 1.0)

### 5. `polyfit(x, y, deg)` / `polyval(p, x)` — Polynomial fitting
- `polyfit`: Least-squares polynomial fit, returns coefficients [highest...lowest degree]
- `polyval`: Evaluate polynomial at given points using Horner's method
- `polyfit` uses Vandermonde matrix + least-squares solve

### 6. `linalg.lstsq(a, b)` — Least squares
- Solve `||Ax - b||` via SVD-based approach (already have SVD)
- Returns (solution, residuals, rank, singular_values)

## Architecture

- `meshgrid`, `pad`: new functions in `manipulation.rs`
- `interp`, `gradient`: new file `ops/numerical.rs`
- `polyfit`/`polyval`: new file `ops/polynomial.rs`
- `lstsq`: add to `linalg.rs`
- All get Python bindings in `lib.rs` and wrappers in `__init__.py`

## Testing

Rust unit tests + Python integration tests for each function.
