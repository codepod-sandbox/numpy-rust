# Design: Redesign Rust Runtime Dispatch Around NumPy-Style Descriptors

**Date:** 2026-04-11
**Scope:** `numpy-rust-core` runtime architecture, plus the Rust/Python boundary in `numpy-rust-python`
**Goal:** Replace the current repeated dtype branching model with a NumPy-shaped core runtime built around array metadata, dtype descriptors, and centralized operation dispatch.

---

## Background

The current Rust core has already moved in the right direction in a few places:

- shared array/view semantics exist via `ArcArray`
- dtype identity is centralized in `DType`
- Python bindings are mostly thin wrappers around core arrays

But the actual behavior of the system is still spread across repeated `match ArrayData::...` blocks. The current design makes each subsystem rebuild part of the runtime:

- [array_data.rs](/Users/sunny/work/codepod/numpy-rust/crates/numpy-rust-core/src/array_data.rs:1) defines the runtime storage enum
- [casting.rs](/Users/sunny/work/codepod/numpy-rust/crates/numpy-rust-core/src/casting.rs:1) re-encodes dtype-specific conversion logic
- [ops/arithmetic.rs](/Users/sunny/work/codepod/numpy-rust/crates/numpy-rust-core/src/ops/arithmetic.rs:1) re-encodes binary dispatch
- [ops/reduction.rs](/Users/sunny/work/codepod/numpy-rust/crates/numpy-rust-core/src/ops/reduction.rs:1) re-encodes dtype/op selection again
- constructors in [array.rs](/Users/sunny/work/codepod/numpy-rust/crates/numpy-rust-core/src/array.rs:1) duplicate dtype-specific creation logic

This is the Rust equivalent of the Python-side `if isinstance(...)` smell: the architecture has no authoritative place for runtime type behavior, so each module patches around that missing boundary.

NumPy’s design solves this by separating responsibilities:

- arrays own shape, strides, flags, and storage references
- dtypes own dtype semantics
- operation resolution is centralized
- Python is a thin layer over the C runtime

This redesign follows that architecture within reason. We will not imitate NumPy’s C internals literally when Rust or the `ndarray` crate suggests a cleaner implementation, but the runtime boundaries and ownership model should remain recognizably NumPy-like.

---

## Design Constraints

1. The Rust core is authoritative for runtime behavior.
   Casting, promotion, dtype legality, memory semantics, and operation dispatch are core responsibilities, not Python-wrapper responsibilities.

2. The Python layer should remain thin.
   It should mostly parse Python arguments, convert Python objects to core arrays/scalars, call the core runtime, and map errors back to Python exceptions.

3. The `ndarray` crate is a backend, not the runtime model.
   We are free to use `ndarray` internally for storage, views, and iteration, but the public core architecture should be NumPy-shaped, not `ndarray`-shaped.

4. This is a replacement, not a permanent dual architecture.
   Temporary adapters are acceptable during migration, but the end state must not preserve both descriptor-based dispatch and widespread `ArrayData` matching.

5. The redesign should improve future ufunc and reduction work.
   It should become easier to add dtype support or a new operation without copying dtype matrices across modules.

---

## Target Architecture

The runtime should be reorganized around four core concepts:

- `NdArray`: array object with metadata and storage reference
- `ArrayStorage`: NumPy-shaped facade over the actual backing storage/view engine
- `DTypeDescriptor`: authoritative runtime object for dtype semantics
- `OpResolver`: central dispatcher for unary, binary, comparison, cast, and reduction operations

### Core Object Model

Target shape:

```rust
pub struct NdArray {
    storage: ArrayStorage,
    descriptor: &'static DTypeDescriptor,
    shape: SmallVec<[usize; 6]>,
    strides: SmallVec<[isize; 6]>,
    flags: ArrayFlags,
}
```

Responsibilities:

- `storage` owns the shared allocation and view relationship
- `descriptor` defines what one element means and how operations on that dtype behave
- `shape` and `strides` define array geometry
- `flags` holds array-level facts such as contiguity, alignment, and writeability

The key correction is that `NdArray` is no longer “an enum of typed arrays.” It becomes a uniform runtime object whose behavior is driven by its descriptor and the operation resolver.

---

## Array Storage

`ArrayStorage` is the abstraction layer that keeps `ndarray` useful without letting its type system dictate the architecture.

### Role

`ArrayStorage` should provide:

- shared ownership of allocation
- cheap views/slices/transposes
- mutable write paths with copy-on-write or equivalent safety
- element-addressing and strided iteration primitives
- allocation helpers for result arrays

### Backend Policy

Internally, `ArrayStorage` may continue using `ArcArray`, views, and iterators from `ndarray`. But those details should stay behind the facade.

This gives us three advantages:

- the top-level runtime looks like NumPy, not `ArcArray<T, IxDyn>`
- descriptor kernels and resolvers work against our abstractions
- if `ndarray` is awkward for a NumPy semantic, we can compensate in `ArrayStorage` instead of distorting the rest of the runtime

### Storage API Shape

At minimum, `ArrayStorage` should support:

- allocate buffer for `len * itemsize`
- create a view from parent storage with shape/strides/offset
- expose raw element bytes for descriptor kernels
- iterate in contiguous and strided forms
- copy or cast into newly allocated storage
- detect memory overlap between arrays/views

This makes storage generic over dtype at the runtime level while still permitting optimized typed access inside kernels.

---

## DType Descriptors

`DTypeDescriptor` is the central runtime object missing from the current codebase.

### Role

A descriptor should own dtype-local semantics:

- identity: name, kind, itemsize, logical dtype id
- scalar load/store from element memory
- scalar formatting and parsing/coercion
- cast support to other descriptors
- per-op kernel registration
- accumulation and reduction support where dtype-specific

Possible shape:

```rust
pub struct DTypeDescriptor {
    id: DTypeId,
    name: &'static str,
    kind: DTypeKind,
    itemsize: usize,
    flags: DTypeFlags,
    casts: &'static CastTable,
    ops: &'static OpTable,
    scalar: &'static ScalarVTable,
}
```

### What Stays Out Of Descriptors

Descriptors should not own:

- broadcasting rules
- axis reduction mechanics
- whole-array iteration policy
- Python argument parsing
- array flag computation

Those remain global runtime concerns, just as they are in NumPy.

### DType vs Descriptor

The current `DType` enum can remain, but only as a stable key or identifier:

- `DType` becomes the symbolic id
- `DTypeDescriptor` becomes the behavioral runtime object

In other words, `DType` answers “which dtype is this?” and the descriptor answers “what does this dtype do?”

---

## Central Operation Resolver

All operation dispatch should move into a single subsystem instead of being reconstructed per module.

### Role

`OpResolver` should:

- validate whether an operation is legal for the input descriptors
- choose output descriptor
- choose accumulator descriptor when relevant
- insert casts when required
- select the implementation kernel
- return an executable dispatch plan

Target usage:

```rust
let plan = resolver.resolve_binary(BinaryOp::Add, lhs.descriptor(), rhs.descriptor())?;
let result = executor.run_binary(plan, lhs, rhs)?;
```

### Dispatch Categories

The resolver should cover at least:

- unary ops
- binary ops
- comparisons
- casts
- reductions
- scan/cumulative ops
- indexed write operations where dtype coercion matters

### Why This Matters

Today, promotion, legality, and dtype selection are duplicated across `casting`, arithmetic, comparisons, and reductions. That duplication is the design bug. A central resolver gives one place to decide:

- `int32 + float64 -> float64`
- `clip` output dtype under scalar bound coercion
- which reductions preserve dtype vs widen accumulator dtype
- whether `str` supports an operation at all

This follows NumPy much more closely than encoding these choices ad hoc inside each operation module.

---

## Kernel Model

The resolver should not perform the work itself. It should choose kernels.

### Kernel Registration

Each descriptor should contribute dtype-local kernels through tables:

- unary kernels keyed by op id
- binary kernels keyed by op id
- cast kernels keyed by destination descriptor
- reduction helpers keyed by op id

The kernel interface should operate on storage/access abstractions we define, not directly on public `ndarray` types.

### Execution Split

Runtime flow:

1. parse arguments at Python boundary
2. convert to core arrays/scalars
3. resolve operation in `OpResolver`
4. allocate output storage using result descriptor and shape plan
5. execute chosen kernel over iterators/views
6. return `NdArray`

This allows modules such as arithmetic or reductions to become thin named frontends over the central runtime instead of carrying their own dtype matrix.

---

## Python Boundary

The Python layer in `numpy-rust-python` should stay thin and become thinner over time.

### Responsibilities

Bindings should:

- convert Python objects into `NdArray` or scalar inputs
- pass explicit dtype/order/casting options into the core
- construct Python-visible arrays from returned core arrays
- map core errors into Python exception classes

Bindings should not:

- decide promotion rules
- implement dtype legality checks that the core also implements
- reconstruct dispatch logic based on Python object type
- contain operation-specific fallback behavior when the core should decide it

This keeps the system aligned with NumPy’s “thin Python layer over a large native runtime” model.

---

## Error Handling

Errors should be produced at the layer that owns the violated rule.

- descriptor errors: invalid scalar coercion or unsupported cast
- resolver errors: unsupported dtype/op combination, illegal casting rule, invalid promotion
- storage errors: incompatible shapes, invalid view construction, overlap violations when relevant
- Python bindings: exception translation only

This removes today’s ambiguity where the same behavior can be partially encoded in Python wrappers, partially in Rust bindings, and partially in core ops.

---

## Migration Strategy

This should be executed as a controlled replacement, not as a permanent compatibility layer.

### Phase 1: Introduce Runtime Objects

Add new modules for:

- `descriptor`
- `storage`
- `resolver`
- `kernel`

Define the stable runtime interfaces before moving existing behavior.

### Phase 2: Rebase `NdArray`

Change `NdArray` to hold:

- `ArrayStorage`
- `&'static DTypeDescriptor`
- shape/strides/flags

At this stage, short-lived compatibility shims may exist behind the scenes, but all new work should target the new API and the shims should be deleted as soon as the migrated paths cover the old ones.

### Phase 3: Move Core Semantics

Move these concerns into descriptors/resolver:

- cast graph
- promotion logic
- scalar conversion
- binary/unary legality
- reduction dtype policy

The old operation modules should stop deciding dtype matrices directly.

### Phase 4: Rebuild Operations On Top Of Resolver

Re-implement:

- arithmetic
- logical/comparison ops
- casting
- reductions
- creation/coercion paths
- indexing writes that depend on dtype coercion

These modules should become thin entrypoints over resolver plus executor.

### Phase 5: Thin The Python Layer

Audit `numpy-rust-python` and remove duplicated semantics that now live in the core runtime.

### Phase 6: Delete Legacy Enum-Driven Dispatch

Once coverage is complete, remove or drastically narrow `ArrayData`.

The finished runtime should not require every operation module to match on all dtypes.

---

## Compatibility With `ndarray`

This redesign does not reject `ndarray`. It constrains it.

### Acceptable Uses

- allocation backend
- view and slicing backend
- strided iteration backend
- typed inner loops where convenient

### Non-Goals

We should not let the architecture become:

- `NdArray == ArcArray<T, IxDyn>`
- operation dispatch == Rust generic specialization on `T`
- public runtime semantics dictated by `ndarray` limitations

If `ndarray` gives a good primitive, use it. If it does not, wrap it or route around it. The runtime boundary should remain NumPy-shaped.

---

## Testing Strategy

The test layout should reflect the new architecture.

### Core Unit Tests

Add focused tests for:

- descriptor metadata
- cast legality and cast results
- promotion resolution
- resolver output for representative op/dtype combinations
- storage view semantics and overlap detection

### Integration Tests

Add runtime-level tests for:

- array creation and coercion via descriptors
- binary ops after promotion and broadcasting
- reductions with correct accumulator/result dtype
- indexing assignment and coercion rules

### Python Compatibility Tests

Keep the existing NumPy compatibility suite as the top-level validation layer, but it should no longer be the first place where dtype behavior is discovered.

The desired outcome is:

- core tests prove runtime correctness locally
- Python tests prove NumPy-visible behavior

---

## Success Criteria

This redesign is successful when all of the following are true:

- `NdArray` runtime behavior is descriptor-driven, not enum-driven
- dtype semantics live in descriptors and the resolver, not in scattered operation modules
- operation modules no longer duplicate dtype selection matrices
- Python bindings are thinner and stop owning runtime rules
- `ndarray` remains an internal backend detail
- adding a new dtype or operation no longer requires open-coding the full dtype matrix in multiple files

---

## Explicit Non-Goals

- Reproducing NumPy’s C internals byte-for-byte
- Eliminating `ndarray` as an implementation dependency
- Rewriting the Python API surface during this design pass
- Preserving legacy architecture for convenience after migration

The goal is architectural convergence with NumPy’s runtime model, not a literal transliteration of the C source.
