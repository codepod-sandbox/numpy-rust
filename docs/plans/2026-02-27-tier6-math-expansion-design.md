# Tier 6: Math Function Expansion — Design

## Goal

Add 14 commonly-used math functions that follow the existing `float_unary!` / `float_only_unary!` macro patterns in `ops/math.rs`.

## Architecture

All functions use the established macro pattern — core implementation is a one-liner per function. Python bindings follow the existing sqrt/exp/log pattern (pymethod + pyfunction + __init__.py wrapper).

## Features

### Logarithmic: `log10`, `log2`, `log1p`
- `log10` — base-10 logarithm, supports complex
- `log2` — base-2 logarithm, supports complex
- `log1p` — `ln(1 + x)`, accurate for small x, float-only (no complex log1p in std)

### Exponential: `expm1`
- `expm1` — `exp(x) - 1`, accurate for small x, float-only

### Sign: `sign`
- Returns -1, 0, or 1. Float-only (complex sign is different in NumPy).

### Angle conversion: `deg2rad`, `rad2deg`
- `deg2rad` = `x * pi / 180`, float-only
- `rad2deg` = `x * 180 / pi`, float-only
- Also aliased as `radians` / `degrees`

### Hyperbolic: `sinh`, `cosh`, `tanh`
- All support complex via `num_complex` methods

### Inverse trigonometric: `arcsin`, `arccos`, `arctan`
- All support complex via `num_complex` methods

## Testing

- Rust unit tests for each new function
- Python integration tests verifying correct output values

## Verification

After each task:
1. `cargo fmt --all -- --check`
2. `cargo clippy --workspace --all-features -- -D warnings`
3. `cargo test --workspace --all-features`
4. `./tests/python/run_tests.sh target/debug/numpy-python`
