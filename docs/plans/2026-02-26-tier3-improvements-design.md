# Tier 3: Core Correctness + Feature Expansion — Design

## Goal

Address the gaps identified in the project review: make existing features correct and robust (Phase A), then expand the API surface with complex numbers, einsum, string ops, and missing utilities (Phase B).

## Architecture

Two sequential phases. Phase A fixes correctness issues in existing operations without changing the type system. Phase B extends `DType` and `ArrayData` with complex number variants and adds new operation modules. Both phases follow the established pattern: core logic in `numpy-rust-core`, Python bindings in `numpy-rust-python`, stubs updated in `python/numpy/__init__.py`.

## Phase A: Core Correctness

### A1. `keepdims` parameter for reductions

Add `keepdims: bool` to `sum`, `mean`, `min`, `max`, `std`, `var` in `reduction.rs`. When true and an axis is specified, insert a size-1 dimension at the reduced axis position using ndarray's `insert_axis(Axis(ax))`.

**Files:** `reduction.rs`, `py_array.rs`, `__init__.py`

### A2. `ddof` parameter for `std`/`var`

Add `ddof: usize` (default 0) to `var` and `std`. Change variance formula to divide by `(N - ddof)` instead of `N`. When `ddof >= N`, return NaN (matching NumPy).

**Files:** `reduction.rs`, `py_array.rs`, `__init__.py`

### A3. `expand_dims` and `squeeze`

- `expand_dims(axis: usize)` — insert size-1 dim at given axis via reshape
- `squeeze(axis: Option<usize>)` — remove all size-1 dims, or specific axis if provided (error if that axis isn't size 1)

**Files:** `manipulation.rs`, `__init__.py`

### A4. Replace `unwrap()` with proper error handling

Replace panicking `unwrap()` in:
- `manipulation.rs:20-30` (reshape's `into_shape_with_order`) — these can't actually fail since size is pre-validated, but use `.expect("size pre-validated")` for clarity
- `creation.rs:24,40` (arange/linspace) — same treatment
- `sorting.rs:32,76` — same treatment
- `manipulation.rs:49` (flatten) — propagate Result
- `manipulation.rs:182` (stack's reshape) — propagate Result

### A5. `eye(n, M, k)` improvements

Extend `eye(n, dtype)` to `eye(n, m, k, dtype)`:
- `m: Option<usize>` — number of columns (default = n)
- `k: isize` — diagonal offset (0 = main diagonal, positive = superdiagonal, negative = subdiagonal)

**Files:** `creation.rs`, Python bindings, `__init__.py`

### A6. `argmin`/`argmax` with axis parameter

Change signatures from `fn argmin(&self) -> Result<usize>` to `fn argmin(&self, axis: Option<usize>) -> Result<NdArray>`:
- `axis=None` — flatten, return scalar Int64 NdArray (preserving old behavior as NdArray)
- `axis=Some(ax)` — return Int64 array of indices along that axis, using the `lanes` pattern from `argsort`

**Files:** `reduction.rs`, `py_array.rs`

### A7. Edge case tests

Add Python tests for:
- Empty array operations (`np.sum(np.array([]))`, `np.mean(np.array([]))`)
- NaN propagation (`np.sum(np.array([1.0, float('nan')]))`)
- Inf handling (`np.max(np.array([1.0, float('inf')]))`)
- dtype preservation (`np.array([1, 2, 3], dtype='int32') + np.array([1, 2, 3], dtype='int32')` stays int32)
- keepdims shape verification
- ddof edge cases (ddof=1, ddof=N)

---

## Phase B: Feature Expansion

### B1. Complex number support

**DType:** Add `Complex64` and `Complex128` variants. Use `num_complex::Complex<f32>` and `num_complex::Complex<f64>`.

**Promotion rules:**
- `Float64 + Complex64 → Complex128`
- `Float32 + Complex64 → Complex64`
- `Int* + Complex* → Complex128`
- `Complex64 + Complex128 → Complex128`

**ArrayData:** Add `Complex64(ArrayD<Complex<f32>>)` and `Complex128(ArrayD<Complex<f64>>)` variants.

**Operations supported:**
- Arithmetic: `+`, `-`, `*`, `/`, `**` — via `num_complex` trait impls
- Math: `abs` (magnitude, returns real), `exp`, `log`, `sqrt`
- New accessors: `real()`, `imag()`, `conj()`, `angle()`
- Reductions: `sum`, `mean` — work on complex
- Reductions NOT supported: `min`, `max`, `sort`, `argsort` — error ("complex not orderable")
- Comparison: only `==`, `!=`

**Python bindings:**
- Parse Python `complex` objects → Complex128
- `repr` format: `(a+bj)`
- `np.complex64`, `np.complex128` dtype aliases

**Scope exclusions:** No complex support in sort/argsort, searchsorted, bitwise ops.

### B2. `einsum`

Parse explicit subscript form (`"ij,jk->ik"`). Algorithm:
1. Tokenize subscripts: split on `,`, extract output after `->`
2. Build index-to-axis mapping for each operand
3. Identify contracted indices (in inputs but not output)
4. Iterate over all output index combinations, sum over contracted indices

Support up to 3 input operands. Cast all inputs to Float64 (or Complex128 if any complex).

**Files:** New `ops/einsum.rs`, `lib.rs` (pyfunction), `__init__.py`

### B3. String operations

Add `ops/string_ops.rs` with element-wise string methods:
- `str_upper`, `str_lower`, `str_capitalize`, `str_strip`
- `str_len` (returns Int64 array)
- `str_startswith(suffix)`, `str_endswith(suffix)` (return Bool array)
- `str_replace(old, new)`

All operate on `ArrayData::Str` via `mapv`. Non-string arrays → TypeError.

**Python:** Expose as `np.char.upper(a)`, etc. in `__init__.py`.

### B4. `searchsorted`, `choose`, `compress`

- `searchsorted(sorted_arr, values, side)` — binary search returning Int64 insertion indices. Side = "left" (bisect_left) or "right" (bisect_right).
- `choose(index_arr, choices)` — select from choice arrays based on index values
- `compress(condition, arr, axis)` — select slices where bool condition is true

**Files:** New `ops/selection.rs` or add to `indexing.rs`

### B5. Creation function improvements

- `arange(start, stop, step, dtype)` — cast result to requested dtype
- `linspace(start, stop, num, retstep)` — when `retstep=True`, return `(array, step_value)` tuple
- `eye(n, M, k, dtype)` — already in A5

**Files:** `creation.rs`, Python bindings

---

## Testing Strategy

Each feature gets Rust unit tests in the relevant module AND Python integration tests in `tests/python/test_numeric.py` (or new test files for complex/string/einsum). Tests written before implementation where possible.

## Verification

After each phase:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `./tests/python/run_tests.sh target/debug/numpy-python`
