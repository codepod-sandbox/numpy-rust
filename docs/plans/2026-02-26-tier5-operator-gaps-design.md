# Tier 5: Operator Gaps & Element-wise Rust Migrations — Design

## Goal

Fill remaining operator protocol gaps (bitwise shifts, xor, abs, bool, divmod) and migrate 8 pure-Python loop-based element-wise functions to Rust core with proper Python bindings.

## Architecture

Same pattern as previous tiers: core logic in `numpy-rust-core` with Rust unit tests, Python bindings in `numpy-rust-python`, and thin wrappers in `python/numpy/__init__.py`.

## Features

### 1. Bitwise xor, left shift, right shift

- `bitwise_xor` — Add to `ops/logical.rs` alongside existing `bitwise_and`/`bitwise_or`
- `left_shift`, `right_shift` — New methods in `ops/logical.rs`
- Integer types only; error on float/complex
- Wire `xor`, `lshift`, `rshift` + `inplace_xor`, `inplace_lshift`, `inplace_rshift` in AS_NUMBER

### 2. abs/bool protocol slots

- `absolute` slot in AS_NUMBER — call existing `abs()` method
- `bool` slot — scalar arrays: truthiness of single element; multi-element: error (ambiguous)

### 3. isnan, isinf, isfinite

- `isnan` and `isfinite` already exist in Rust core (`utility.rs`)
- Add `isinf` to core
- Add pymethods and pyfunctions for all three
- Replace Python loop implementations with native calls

### 4. around (round) and signbit

- `around(decimals)` — new method in `ops/math.rs`, rounds to given decimal places
- `signbit` — new method in `ops/math.rs`, returns Bool array (true for negative)
- Add pyfunctions and replace Python loops

### 5. logical_not and power module function

- `logical_not` — new method in `ops/logical.rs`, element-wise boolean NOT (truthy/falsy)
- `power` pyfun — delegates to existing `pow()` core method
- Replace Python loop implementations

### 6. nonzero and count_nonzero

- `nonzero` — return tuple of Int64 index arrays, one per axis (NumPy convention)
- `count_nonzero(axis)` — count non-zero elements, optionally along axis
- Add to `utility.rs` alongside existing `argwhere`
- Add pyfunctions and replace Python loops

## Testing

Each feature gets:
- Rust unit tests in the relevant core module
- Python integration tests in `tests/python/test_numeric.py`

## Verification

After each task:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `./tests/python/run_tests.sh target/debug/numpy-python`
