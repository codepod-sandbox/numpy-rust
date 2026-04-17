# Random Native Runtime Finish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the remaining high-value Python-owned random distribution execution paths by moving them into the native Rust runtime.

**Architecture:** Keep Python responsible for public API shaping, broadcasting wrappers, and output handling, but move the actual sampling kernels into `crates/numpy-rust-core/src/random.rs` and expose them through `crates/numpy-rust-python/src/py_random.rs`. Prefer grouped kernel batches with focused upstream verification after each batch, then a final full random smoke pass.

**Tech Stack:** Rust core (`numpy-rust-core`), RustPython bindings (`numpy-rust-python`), Python wrappers in `python/numpy/_random_ext.py`, pytest upstream compatibility suite.

---

### Task 1: Finish Plan And Checkpoint Scope

**Files:**
- Create: `docs/superpowers/plans/2026-04-17-random-native-runtime-finish.md`
- Modify: `crates/numpy-rust-core/src/random.rs`
- Modify: `crates/numpy-rust-python/src/py_random.rs`
- Modify: `python/numpy/_random_ext.py`

- [ ] **Step 1: Confirm the remaining Python-owned random kernels**

Run:

```bash
rg -n "def _random_(multivariate_normal|vonmises|wald|zipf|negative_binomial|hypergeometric|triangular)" python/numpy/_random_ext.py
```

Expected: the remaining kernels are all still present in `_random_ext.py`.

- [ ] **Step 2: Confirm the current random smoke baseline**

Run:

```bash
python3 -m pytest tests/numpy_compat/upstream/random_test_smoke.py -q
```

Expected: all current random smoke tests pass before the next batch.

### Task 2: Native `multivariate_normal`

**Files:**
- Modify: `crates/numpy-rust-core/src/random.rs`
- Modify: `crates/numpy-rust-python/src/py_random.rs`
- Modify: `python/numpy/_random_ext.py`
- Test: `tests/numpy_compat/upstream/random_test_smoke.py`

- [ ] **Step 1: Add a native multivariate normal kernel in Rust**

Implement a kernel that:
- accepts flattened `mean`, lower-triangular `chol`, and sample shape
- generates `standard_normal` vectors natively
- multiplies by the Cholesky factor and adds the mean
- returns an `NdArray` shaped as `sample_shape + [n]`

- [ ] **Step 2: Expose the kernel through `py_random.rs`**

Add Python bindings that:
- parse `mean` and `chol` as ndarray-like inputs
- flatten/convert them to `float64`
- call the new Rust kernel for both stateless and stateful paths

- [ ] **Step 3: Replace Python execution in `_random_multivariate_normal`**

Keep:
- input normalization
- Cholesky decomposition in Python if no native factorization helper is introduced

Remove:
- Python-side loop over normal samples
- Python-side matrix multiply loop per sample

- [ ] **Step 4: Verify the focused multivariate tests**

Run:

```bash
python3 -m pytest tests/numpy_compat/upstream/random_test_smoke.py -k "multivariate_normal" -q
```

Expected: PASS.

### Task 3: Native Rejection-Sampling Family

**Files:**
- Modify: `crates/numpy-rust-core/src/random.rs`
- Modify: `crates/numpy-rust-python/src/py_random.rs`
- Modify: `python/numpy/_random_ext.py`
- Test: `tests/numpy_compat/upstream/random_test_smoke.py`

- [ ] **Step 1: Add native `vonmises`, `wald`, and `zipf` kernels**

Implement the current algorithms in Rust first, preserving existing semantics:
- `vonmises`: Best-Fisher rejection path
- `wald`: inverse-Gaussian transform using native normal/uniform draws
- `zipf`: current rejection sampler

- [ ] **Step 2: Expose stateful and stateless bindings**

Add RustPython entrypoints for:
- `vonmises`
- `wald`
- `zipf`

Each should support both global and per-generator stateful execution.

- [ ] **Step 3: Replace Python execution in `_random_ext.py`**

Keep:
- broadcasting wrappers where inputs can be arrays

Remove:
- scalar draw loops
- repeated `random.uniform` / `random.normal` calls in Python

- [ ] **Step 4: Verify the focused rejection family**

Run:

```bash
python3 -m pytest tests/numpy_compat/upstream/random_test_smoke.py -k "vonmises or wald or zipf" -q
```

Expected: PASS.

### Task 4: Remaining Discrete/Derived Audit

**Files:**
- Modify: `python/numpy/_random_ext.py`
- Modify: `crates/numpy-rust-core/src/random.rs`
- Modify: `crates/numpy-rust-python/src/py_random.rs`
- Test: `tests/numpy_compat/upstream/random_test_smoke.py`

- [ ] **Step 1: Re-scan `_random_ext.py` for remaining runtime-heavy loops**

Run:

```bash
rg -n "for _ in range\\(|while True:|flatten\\(\\)\\.tolist\\(|tolist\\(" python/numpy/_random_ext.py
```

Expected: the remaining hits should mainly be:
- broadcast helper shells
- legacy state helpers
- any still-unmoved random kernels

- [ ] **Step 2: Move any remaining obvious kernel if still execution-heavy**

If one kernel still clearly owns sampling logic rather than API shaping, move it in this task.
Otherwise, leave the wrapper in place and note why it remains.

- [ ] **Step 3: Re-run targeted random smoke slices if another kernel moved**

Run the narrow `-k` slice matching the kernel you touched.

Expected: PASS.

### Task 5: Final Random Verification

**Files:**
- Modify: `crates/numpy-rust-core/src/random.rs`
- Modify: `crates/numpy-rust-python/src/py_random.rs`
- Modify: `python/numpy/_random_ext.py`
- Test: `tests/numpy_compat/upstream/random_test_smoke.py`

- [ ] **Step 1: Build both Rust crates**

Run:

```bash
cargo test -p numpy-rust-core --no-run
cargo test -p numpy-rust-python --no-run
```

Expected: both commands succeed.

- [ ] **Step 2: Run the full random smoke suite**

Run:

```bash
python3 -m pytest tests/numpy_compat/upstream/random_test_smoke.py -q
```

Expected: full suite passes.

- [ ] **Step 3: Summarize remaining random debt**

Document which random surfaces are still Python-owned and why:
- API/broadcasting shell only
- legacy compatibility/state surface
- not worth moving yet vs still an obvious next target
