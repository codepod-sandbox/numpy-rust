# Design: Migrate Python Math Loops to Rust

**Date:** 2026-03-14
**Scope:** Full sweep — `_math.py`, `_reductions.py`, `_stubs.py` (`_ScimathModule`)
**Goal:** Eliminate all Python-level loops and composites from math/reduction code; achieve a complete, fast NumPy implementation suitable for WASM deployment.

---

## Background

After splitting the monolithic `__init__.py` into 18 submodules, the Python math code is clearly organized but still contains:

- **Python element-wise loops** in `_math.py`: `copysign`, `frexp`, `ldexp`, `nextafter`, `spacing`, `nan_to_num`, `gamma`/`lgamma`, `erf`/`erfc`, `j0`/`j1`/`y0`/`y1`, `i0`
- **Python composites** (chain multiple Rust ops): `cbrt`, `hypot`, `fmod`, `modf`, `maximum`/`minimum`, `fmax`/`fmin`, `logaddexp`/`logaddexp2`
- **Python loops in `_reductions.py`**: `trapz`, `cumulative_trapezoid`, `gradient`
- **Python loops in `_ScimathModule`** (`_stubs.py`): complex-safe `sqrt`, `log`, `log2`, `log10`, `power`, `arcsin`, `arccos`, `arctanh`

In WASM there is no JIT — every Python loop is expensive. This work pushes all element-wise computation into Rust.

---

## New Dependency

Add `libm = "0.2"` to `crates/numpy-rust-core/Cargo.toml`.

`libm` is a pure-Rust port of the C math library, explicitly targeting `no_std` and `wasm32-unknown-unknown`. It provides `cbrt`, `copysign`, `erf`, `erfc`, `frexp`, `hypot`, `j0`, `j1`, `ldexp`, `lgamma`, `modf`, `nextafter`, `tgamma`, `y0`, `y1` and more at float32 and float64 precision.

`i0` (modified Bessel I0) is not in `libm` — we port the existing series expansion to Rust.

---

## Architecture

The three-layer architecture is unchanged. This work only moves the computation boundary down:

```
Python layer  (_math.py / _reductions.py / _stubs.py)
  thin wrappers: asarray() → _native.func() → handle out= param
      ↓
Binding layer  (crates/numpy-rust-python/src/lib.rs)
  #[pyfunction]: obj_to_ndarray() → call core method → PyNdArray::from_core()
      ↓
Core layer  (crates/numpy-rust-core/src/ops/math.rs)
  libm calls via macros + manual impls for special cases
```

---

## Components

### `crates/numpy-rust-core/src/ops/math.rs`

All new Rust implementations land here, grouped by pattern:

**Group 1 — `libm_unary!` macro** (new macro, float32/float64 only, calls libm):
- `cbrt` → `libm::cbrt` / `libm::cbrtf`
- `gamma` → `libm::tgamma` / `libm::tgammaf`
- `lgamma` → `libm::lgamma` / `libm::lgammaf`
- `erf` → `libm::erf` / `libm::erff`
- `erfc` → `libm::erfc` / `libm::erfcf`
- `j0` → `libm::j0` / `libm::j0f`
- `j1` → `libm::j1` / `libm::j1f`
- `y0` → `libm::y0` / `libm::y0f`
- `y1` → `libm::y1` / `libm::y1f`

**Group 2 — `math_binary!` macro** (new macro, mirrors `prepare_binary` → broadcast → match dtype):
- `copysign` → `libm::copysign` / `libm::copysignf`
- `hypot` → `libm::hypot` / `libm::hypotf`
- `fmod` → `libm::fmod` / `libm::fmodf`
- `ldexp` → `libm::ldexp` / `libm::ldexpf`
- `nextafter` → `libm::nextafter` / `libm::nextafterf`
- `logaddexp` → `(a.max(b) + (-(a - b).abs()).exp().ln_1p())`
- `logaddexp2` → logaddexp in base 2
- `maximum` → `f64::max(a, b)` (NaN-propagating)
- `minimum` → `f64::min(a, b)` (NaN-propagating)
- `fmax` → `a.max(b)` where NaN is ignored (returns other operand)
- `fmin` → `a.min(b)` where NaN is ignored

**Group 3 — Tuple-return ops** (manual `impl NdArray`):
- `frexp(x) → (NdArray, NdArray)`: mantissa (Float64) + exponent (Int32) arrays
- `modf(x) → (NdArray, NdArray)`: fractional + integer parts, both Float64

**Group 4 — Multi-param ops** (manual `impl NdArray`):
- `nan_to_num(x, nan: f64, posinf: f64, neginf: f64) → NdArray`: replaces NaN/±Inf in-place on a copy
- `spacing(x) → NdArray`: `nextafter(|x|, +∞) - |x|` element-wise
- `i0(x) → NdArray`: Rust port of series expansion `Σ((x/2)^k / k!)²`

**Group 5 — Complex-safe scimath ops** (new `scimath_unary!` / `scimath_binary!` macros):
- Input scanned for out-of-domain values; if found, entire array cast to Complex128 before compute
- `scimath_sqrt`, `scimath_log`, `scimath_log2`, `scimath_log10`: negative → complex
- `scimath_arcsin`, `scimath_arccos`: |x| > 1 → complex
- `scimath_arctanh`: |x| > 1 → complex
- `scimath_power(x, p)`: negative base → complex via `exp(p * ln(x))`

**Group 6 — Reductions** (new functions in `ops/numerical.rs` or `ops/reduction.rs`):
- `trapz(y, x_or_dx, axis) → NdArray`: trapezoidal integration
- `cumulative_trapezoid(y, x_or_dx, axis) → NdArray`: cumulative trapezoidal
- `gradient(f, spacing, axis) → NdArray`: finite differences (central/forward/backward)

### `crates/numpy-rust-python/src/lib.rs`

~35 new `#[pyfunction]` entries. Tuple-return functions (`frexp`, `modf`) return `PyResult<(PyNdArray, PyNdArray)>` — RustPython handles tuple packing automatically.

`nan_to_num` keyword args (`nan`, `posinf`, `neginf`) are handled in the Python wrapper; the Rust binding takes positional `f64` args with sentinel values (e.g. `f64::NAN` = use default).

### `python/numpy/_math.py`

Each Python-loop function becomes a 3-line wrapper:
```python
def gamma(x):
    return _native.gamma(asarray(x))
```

`frexp` / `modf` unpack the tuple:
```python
def frexp(x):
    return _native.frexp(asarray(x))  # already returns (ndarray, ndarray)
```

`_ScimathModule` methods delegate to `_native.scimath_*` functions.

### `python/numpy/_reductions.py`

`trapz`, `cumulative_trapezoid`, `gradient` become thin wrappers around `_native.*`.

---

## Data Flow

**Standard unary** (e.g. `cbrt`):
```
np.cbrt(x)
  → _native.cbrt(asarray(x))
  → obj_to_ndarray → x.cbrt()
  → ensure_float → mapv(|v: f64| libm::cbrt(v))
```

**Binary with broadcast** (e.g. `hypot`):
```
np.hypot(x1, x2)
  → _native.hypot(asarray(x1), asarray(x2))
  → obj_to_ndarray × 2 → x.hypot(&y)
  → prepare_binary (promote + broadcast) → match (F32,F32)|(F64,F64)
  → Zip::map_collect(|a,b| libm::hypot(a,b))
```

**Tuple return** (`frexp`):
```
np.frexp(x)
  → _native.frexp(asarray(x)) → PyResult<(PyNdArray, PyNdArray)>
  → x.frexp() → (NdArray<f64>, NdArray<i32>)
```

**scimath** (e.g. `scimath.sqrt(-4.0)`):
```
np.lib.scimath.sqrt(x)
  → _native.scimath_sqrt(asarray(x))
  → scan for negatives → upcast to Complex128 if any found
  → mapv(|z: Complex<f64>| z.sqrt())
```

---

## Error Handling

- **Domain errors** (log of negative, sqrt of negative for real ops): libm returns `NaN` naturally — no special handling needed.
- **scimath ops**: out-of-domain triggers full upcast to Complex128 before computation, matching NumPy `lib.scimath` contract.
- **Division by zero in `fmod`**: follows C `fmod` semantics (returns `NaN`).
- **`nan_to_num`**: `posinf=None` / `neginf=None` in Python wrapper maps to `f64::MAX` / `f64::MIN`.
- No panics in any new code.

---

## Testing

**Rust unit tests** (`#[cfg(test)]` in `ops/math.rs`):
- Each new function: one normal value, one edge case (NaN/Inf/zero/negative), float32 vs float64 consistency.

**Vendored Python tests**:
- New `tests/python/test_math_special.py`: covers all libm-backed functions.
- New `tests/python/test_scimath.py`: covers complex-safe variants.
- Extend `tests/python/test_numeric.py` for `trapz`, `gradient`, `cumulative_trapezoid`.

**Compat tests**:
- Full suite run after — goal is zero new xfails.
- Any scimath or special-function compat test currently failing should pass.

**WASM**: guaranteed by `libm`'s `no_std`/WASM-native design; native test suite passing is sufficient signal.

---

## Implementation Order

1. Add `libm` dependency
2. Add `libm_unary!` macro + Group 1 functions (pure unary, lowest risk)
3. Add `math_binary!` macro + Group 2 functions
4. Add Group 3 tuple-return functions (`frexp`, `modf`)
5. Add Group 4 multi-param functions (`nan_to_num`, `spacing`, `i0`)
6. Add Group 5 scimath functions
7. Add Group 6 reductions (`trapz`, `gradient`, `cumulative_trapezoid`)
8. Wire all in `lib.rs`
9. Simplify Python wrappers
10. Write tests, run full suite
