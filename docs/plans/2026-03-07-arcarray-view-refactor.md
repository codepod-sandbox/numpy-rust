# ArcArray View Refactor Plan

## Goal

Replace `ArrayD<T>` with `ArcArray<T, IxDyn>` throughout the Rust core so that slicing and `view()` return shared references instead of copies. This fixes the last major architectural limitation and makes numpy-rust virtually feature-complete.

## What this fixes

- `ndarray.view()` returns a true view (shared memory) instead of a copy
- Slicing (`a[1:3]`) returns O(1) views instead of O(n) copies
- `shares_memory()` / `may_share_memory()` work correctly for all cases (not just reshape)
- `clip(out=)` memory overlap xfail — views share buffers, so in-place ops see mutations
- Removes all 4 "stubs & intentional deviations" bullet points from README

## After this refactor

- Compat xfails drop from 3 → 2 (only NEP50 isclose + custom C dtype remain)
- Both remaining xfails are unfixable edge cases no real code depends on

---

## Architecture

### Current: owned data
```
ArrayData::Float64(ArrayD<f64>)  // owns Vec<f64>
    .slice() → ArrayView → .to_owned() → new ArrayD<f64>  // always copies
```

### Target: shared data
```
ArrayData::Float64(ArcArray<f64, IxDyn>)  // Arc<OwnedRepr<f64>>
    .slice() → ArcArray<f64, IxDyn>  // shared ref, O(1)
    mutation → automatic Copy-on-Write if refcount > 1
```

### Key semantic changes

| Operation | Before | After |
|-----------|--------|-------|
| `a.clone()` | Deep copy | Arc refcount++ (cheap) |
| `a[1:3]` | O(n) copy | O(1) shared view |
| `a[i] = v` | Direct write | CoW check, then write |
| `view()` | Copy | Shared reference |
| `shares_memory(a, b)` | Memory tag heuristic | Arc pointer equality |
| `copy()` | `.clone()` (was deep) | Must explicitly deep-copy |

---

## Implementation Steps

### Phase 1: Core type change (array_data.rs, array.rs)

**Files**: `array_data.rs` (78 lines), `array.rs` (301 lines)

1. Change `ArrayData` enum variants from `ArrayD<T>` to `ArcArray<T, IxDyn>`:
   ```rust
   use ndarray::ArcArray;

   pub enum ArrayData {
       Bool(ArcArray<bool, IxDyn>),
       Int32(ArcArray<i32, IxDyn>),
       Int64(ArcArray<i64, IxDyn>),
       Float32(ArcArray<f32, IxDyn>),
       Float64(ArcArray<f64, IxDyn>),
       Complex64(ArcArray<Complex<f32>, IxDyn>),
       Complex128(ArcArray<Complex<f64>, IxDyn>),
       Str(ArcArray<String, IxDyn>),
   }
   ```

2. Add helper methods to `ArrayData`:
   ```rust
   impl ArrayData {
       /// Deep copy (not just Arc clone)
       pub fn deep_copy(&self) -> Self { ... }

       /// Check if two ArrayData share the same underlying buffer
       pub fn shares_memory_with(&other: &Self) -> bool { ... }
   }
   ```

3. Update `NdArray` constructors — creation functions produce `ArcArray` via `ArrayD::into_shared()`.

4. Add `NdArray::deep_copy()` that calls `ArrayData::deep_copy()`.

**Approach for construction**: Build `ArrayD<T>` as today, then call `.into_shared()` to convert to `ArcArray<T, IxDyn>`. This is O(1) — just wraps the existing allocation in an Arc. Minimizes changes to creation logic.

### Phase 2: Slicing returns views (indexing.rs)

**File**: `indexing.rs` (896 lines)

1. Change `slice()` to return shared `ArcArray` instead of calling `.to_owned()`:
   ```rust
   // Before:
   ArrayData::Float64(a) => ArrayData::Float64(a.slice(info).to_owned())

   // After:
   ArrayData::Float64(a) => {
       let view = a.slice(info);
       ArrayData::Float64(view.to_shared())  // or .into_shared()
   }
   ```

   Note: `ArrayView::to_shared()` creates a new `ArcArray` that shares data with the source — this is the key operation that makes views work.

   **IMPORTANT**: Check if ndarray 0.16 provides `ArrayView::to_shared()`. If not, we may need `.to_owned().into_shared()` for slices (still copies, but shared afterward). The real win would be slicing an `ArcArray` directly via `.slice_axis()` which returns another `ArcArray` sharing the same buffer. Investigate the exact API.

2. Keep mutation methods (`set()`, `set_slice()`, `mask_set()`) — these automatically trigger CoW on `ArcArray` when using `.get_mut()`, `.slice_mut()`, etc.

### Phase 3: Mutation paths (indexing.rs, creation.rs, manipulation.rs)

**Files**: `indexing.rs`, `creation.rs` (303 lines), `manipulation.rs` (1213 lines)

ArcArray handles CoW automatically for most mutation methods:
- `.get_mut(idx)` → clones if shared, then returns `&mut T`
- `.slice_mut()` → clones if shared, then returns mutable view
- `.iter_mut()` → clones if shared, then returns mutable iterator
- `.index_axis_mut()` → clones if shared

**No manual CoW logic needed** — ndarray does it internally.

Review and test:
1. `index_set()` — uses `index_axis_mut()`, should Just Work
2. `set()` — uses pattern matching + direct mutation, needs `get_mut()`
3. `set_slice()` — uses `.slice_mut()` + `.assign()`, should Just Work
4. `mask_set()` — iterates with mutation, needs CoW-safe iteration
5. `eye()` creation — direct element writes during construction (fine, not shared yet)
6. `roll()` — uses `iter_mut()`, should Just Work with CoW

### Phase 4: Read-only operations (ops/*.rs, ~6000 lines)

**Key insight**: Most operations consume array data via read-only access (`.iter()`, `.mapv()`, `.fold()`, arithmetic). These work identically on `ArcArray` as on `ArrayD`.

**Strategy**: Mechanical type substitution. The match arms stay the same, just the inner type changes. Most ops produce new owned arrays (which get `.into_shared()`).

Macro `dispatch_unary!`, `dispatch_binary!` etc. — if these exist, update once. If not, it's a search-and-replace across ~783 match arms:
- `ArrayD<T>` → `ArcArray<T, IxDyn>` in type annotations
- `.to_owned()` on results → `.into_shared()` where creating new arrays
- Most code: no change needed (operations return new owned arrays anyway)

### Phase 5: Python bindings (py_array.rs)

**File**: `py_array.rs` (3853 lines)

1. **`view()` method**: Return a new `PyNdArray` wrapping the same `ArcArray` (Arc clone, O(1)):
   ```rust
   fn view(zelf: PyRef<Self>) -> PyNdArray {
       let data = zelf.data.read().unwrap();
       let cloned = data.clone();  // Arc refcount++, NOT deep copy
       PyNdArray::from_core(cloned)
   }
   ```

2. **`__getitem__` with slices**: Return view instead of copy:
   ```rust
   // slice() now returns ArcArray sharing source buffer
   let result = data.slice(&[...]);
   PyNdArray::from_core(result)
   ```

3. **`copy()` method**: Must explicitly deep-copy:
   ```rust
   fn copy(zelf: PyRef<Self>) -> PyNdArray {
       let data = zelf.data.read().unwrap();
       let deep = data.deep_copy();  // actual data duplication
       PyNdArray::from_core(deep)
   }
   ```

4. **`__setitem__`**: Works as before — `RwLock::write()` gives `&mut NdArray`, mutations trigger CoW on ArcArray automatically.

5. **`shares_memory()`**: Replace memory_tag heuristic with Arc pointer comparison:
   ```rust
   fn shares_memory(a: &PyNdArray, b: &PyNdArray) -> bool {
       let da = a.data.read().unwrap();
       let db = b.data.read().unwrap();
       da.shares_memory_with(&db)
   }
   ```
   Remove `memory_tag` field from `PyNdArray`.

### Phase 6: Python layer (__init__.py)

**File**: `__init__.py`

Minimal changes:
1. Update `shares_memory()` / `may_share_memory()` to just call the Rust `shares_memory()` if both args are native ndarrays
2. Remove memory_tag tracking logic (no longer needed)
3. `_ObjectArray` remains copy-based (it's Python-level, not worth optimizing)

### Phase 7: Fix clip out= overlap xfail

With views sharing memory, `np.clip(a, 0, 1, out=a)` where `out` overlaps input will work correctly because mutations through `out` are visible from `a` (same buffer). Remove from `xfail.txt`.

---

## Risk Areas

### 1. ArcArray slicing API
**Risk**: ndarray's `ArcArray::slice()` may return an `ArrayView`, not another `ArcArray`. If `.to_shared()` isn't available on views, we'd need `view.to_owned().into_shared()` which still copies.

**Mitigation**: Test this first in a minimal Rust file before starting the refactor. If views can't share ArcArray buffers directly, the benefit is limited to `view()` and `clone()` only.

### 2. Copy semantics
**Risk**: Code that calls `.clone()` expecting a deep copy will get a shallow clone instead. Any mutation to the "clone" would trigger CoW, so correctness is maintained, but performance characteristics change.

**Mitigation**: Audit all `.clone()` calls. Replace with `.deep_copy()` where a true independent copy is intended (e.g., `np.copy()`).

### 3. RwLock + ArcArray interaction
**Risk**: `PyNdArray` wraps `RwLock<NdArray>`. A view's `ArcArray` shares buffer with source. If source is mutated via `RwLock::write()`, the view sees it — this is correct NumPy behavior but may expose races if Python code is multi-threaded.

**Mitigation**: Python's GIL prevents true concurrent access. RustPython also has a GIL. Non-issue in practice.

### 4. String arrays
**Risk**: `ArcArray<String, IxDyn>` requires `String: Clone`. This is fine but cloning strings is expensive. CoW on string arrays may be surprising.

**Mitigation**: String arrays are rare and already slow. No special handling needed.

---

## Testing Strategy

1. **Spike test first**: Before full refactor, write a small Rust test to verify:
   - `ArcArray` slicing returns shared views (not copies)
   - Mutation on a shared `ArcArray` triggers CoW correctly
   - `Arc::ptr_eq()` works for shares_memory detection

2. **Phase-by-phase**: After each phase, `cargo test --release` must pass (425 tests).

3. **After Phase 5**: Run full compat suite: `./target/release/numpy-python tests/numpy_compat/run_compat.py --ci`

4. **Final**: Remove `TestClip.test_clip_with_out_memory_overlap` from `xfail.txt`, verify it passes.

---

## Estimated Effort

| Phase | Scope | Effort |
|-------|-------|--------|
| 1. Core type change | 2 files, ~380 lines | Small |
| 2. Slice returns views | 1 file, ~50 lines changed | Small |
| 3. Mutation paths | 3 files, ~30 call sites | Medium |
| 4. Read-only ops | 13 files, mechanical | Medium-Large (tedious) |
| 5. Python bindings | 1 file, ~20 methods | Medium |
| 6. Python layer | 1 file, minor | Small |
| 7. Xfail fix | 1 line | Trivial |

**Total**: ~2-3 focused sessions. The bulk is Phase 4 (mechanical match-arm updates across ops).

---

## Open Questions

1. Does `ndarray 0.16` support `ArrayView::to_shared()` or equivalent? If not, how do we create an `ArcArray` view of an existing `ArcArray`'s data without copying?

2. Should we keep `ArrayD` internally for construction and convert to `ArcArray` at the boundary? Or use `ArcArray` everywhere?

3. Is there a macro to reduce the 8-arm dispatch boilerplate? If we're touching all 783 match sites anyway, this is a good time to DRY them up.
