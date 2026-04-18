# Python Runtime Debt Paydown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the remaining Python-owned execution shells in reductions, indexing, and manipulation without changing public behavior.

**Architecture:** Keep Python responsible for API normalization and compatibility branching, but push repeated data-walking and matrix/index construction onto existing native kernels or array-level operations. Prefer deleting duplicated fallback code over adding new wrappers, and verify each cut against the narrow upstream slices that already cover the behavior.

**Tech Stack:** Python wrappers in `python/numpy`, Rust core/runtime bindings in `crates/numpy-rust-core` and `crates/numpy-rust-python`, pytest upstream compat suite, cargo build/test.

---

### Task 1: Thin Remaining Reduction Helpers

**Files:**
- Modify: `python/numpy/_reductions.py`
- Test: `tests/numpy_compat/upstream/lib_test_function_base.py`

- [ ] **Step 1: Centralize duplicated scalar/list fallback fragments in reductions**

Code targets:
```python
def _stack_quantile_results(...): ...
def _warn_all_nan_slices(...): ...
def _validate_weight_vector(...): ...
```

- [ ] **Step 2: Verify the reductions-focused slices still pass**

Run: `python3 -m pytest tests/numpy_compat/upstream/lib_test_function_base.py -k "TestPercentile or TestQuantile or nanmedian or nanquantile or nanpercentile or TestMedian" -q`
Expected: PASS with only the existing unknown-mark warning

- [ ] **Step 3: Commit**

```bash
git add python/numpy/_reductions.py
git commit -m "refactor: share reduction warning and temporal helpers"
```

### Task 2: Keep Histogram Paths Columnar

**Files:**
- Modify: `python/numpy/_indexing.py`
- Test: `tests/numpy_compat/upstream/lib_test_histograms.py`
- Test: `tests/numpy_compat/upstream/lib_test_twodim_base.py`

- [ ] **Step 1: Route `histogram2d` and `histogramdd` through shared columnar helpers**

Code targets:
```python
def histogram2d(...): ...
def histogramdd(...): ...
```

- [ ] **Step 2: Verify histogram compatibility slices**

Run: `python3 -m pytest tests/numpy_compat/upstream/lib_test_histograms.py -k "histogramdd or histogram2d" -q`
Expected: PASS

Run: `python3 -m pytest tests/numpy_compat/upstream/lib_test_twodim_base.py -k "histogram2d" -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add python/numpy/_indexing.py
git commit -m "refactor: share 2d histogram columnar path"
```

### Task 3: Remove Small Numeric Loop Nests

**Files:**
- Modify: `python/numpy/_manipulation.py`
- Modify: `python/numpy/_indexing.py`
- Test: `tests/numpy_compat/upstream/poly_test_hermite.py`
- Test: `tests/numpy_compat/upstream/lib_test_twodim_base.py`

- [ ] **Step 1: Replace numeric loop nests with broadcasted array operations**

Code targets:
```python
def vander(...): ...
def tri(...): ...
```

- [ ] **Step 2: Verify targeted manipulation/indexing slices**

Run: `python3 -m pytest tests/numpy_compat/upstream/poly_test_hermite.py -k "vander" -q`
Expected: PASS

Run: `python3 -m pytest tests/numpy_compat/upstream/lib_test_twodim_base.py -k "tri or tril or triu" -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add python/numpy/_manipulation.py python/numpy/_indexing.py
git commit -m "refactor: thin manipulation numeric execution paths"
```

### Task 4: Finish the Remaining Random Outlier

**Files:**
- Modify: `crates/numpy-rust-core/src/random.rs`
- Modify: `crates/numpy-rust-python/src/py_random.rs`
- Modify: `python/numpy/_random_ext.py`
- Test: `tests/numpy_compat/upstream/random_test_smoke.py`

- [ ] **Step 1: Move the remaining Python-owned `logseries` sampler into native runtime**

Code targets:
```rust
pub fn logseries(...)
fn logseries_with_state(...)
```

```python
def _random_logseries(...): ...
```

- [ ] **Step 2: Verify focused and full random smoke**

Run: `cargo test -p numpy-rust-python --no-run`
Expected: PASS

Run: `python3 -m pytest tests/numpy_compat/upstream/random_test_smoke.py -k "logseries" -q`
Expected: PASS

Run: `python3 -m pytest tests/numpy_compat/upstream/random_test_smoke.py -q`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/numpy-rust-core/src/random.rs crates/numpy-rust-python/src/py_random.rs python/numpy/_random_ext.py
git commit -m "refactor: move logseries sampling into native runtime"
```

### Task 5: Final Audit

**Files:**
- Modify: `docs/superpowers/plans/2026-04-17-python-runtime-debt-paydown.md`

- [ ] **Step 1: Re-run a narrow architecture audit**

Run:
```bash
python3 - <<'PY'
from pathlib import Path
for rel in [
    'python/numpy/_reductions.py',
    'python/numpy/_indexing.py',
    'python/numpy/_helpers.py',
    'python/numpy/_manipulation.py',
    'python/numpy/_random_ext.py',
]:
    text = Path(rel).read_text()
    print(rel, text.count('for '), text.count('flatten().tolist('))
PY
```
Expected: reductions still largest, random noticeably thinner than before

- [ ] **Step 2: Report remaining real seams**

Write a concise summary of what still remains architecture-wise and which file is the next best target.
