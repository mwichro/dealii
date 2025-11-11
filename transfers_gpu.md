# Porting MGTwoLevelTransfer to Portable::MatrixFree (GPU)

This document summarizes what `MGTwoLevelTransfer` currently requires from `dealii::MatrixFree`, what needs to be added or adapted in `Portable::MatrixFree` (or via a wrapper), and a concrete plan of changes to port the transfer operations to GPU.

## Summary of MatrixFree functionality used by MGTwoLevelTransfer

MGTwoLevelTransfer performs matrix-free multigrid prolongation/restriction by leveraging the `MatrixFree` infrastructure. The main pieces it relies on are:

- Initialization / metadata access
  - Access to the associated `DoFHandler`/mesh information and multigrid level.
  - Number of components and FE-related metadata.
  - Access to vector partitioning (global/local indexing) for initializing vectors and scatter/gather operations.

- Cell/batch iteration and cell identification
  - Ability to iterate over cell batches (the `cell_loop` mechanism) and, crucially, to determine the `active_cell_index()` or other unique cell identifiers for each lane/entry in a batch. This mapping is used to match fine cells to coarse cells and build transfer patterns.

- Local evaluation and I/O
  - Creation of `FEEvaluation`/`FEEvaluationNoConstraints` objects (or similar) inside the cell loop to read local DOF values from source vectors and write local contributions back to destination vectors.
  - Unconstrained read/write variants used for local operations (read_dof_values_unconstrained, distribute_local_to_global_unconstrained).

- Utility methods
  - `initialize_dof_vector()` (or similar) for creating correctly sized temporary vectors and weight vectors.

In short: MGTwoLevelTransfer needs (1) metadata and partition info, (2) deterministic mapping from batches/lanes to global cell identifiers, and (3) FE evaluation/gather-scatter primitives within the cell loop.

## Gaps in Portable::MatrixFree (and possible resolutions)

`Portable::MatrixFree` is designed for portability to GPU but currently does not expose all the features `MGTwoLevelTransfer` depends on. The key gaps are:

1. Cell identity mapping
   - Problem: Portable backend may not expose `active_cell_index()` or an equivalent mapping for each batch/lane.
   - Resolution: Record and expose a mapping from (batch_index, lane) -> global `active_cell_index()` during setup. Provide an accessor like `get_active_cell_index(batch, lane)`.

2. Metadata (DoFHandler, level, n_components)
   - Problem: Portable backend might not retain pointers to `DoFHandler` or mg-level info after setup.
   - Resolution: Either store references/pointers to the `DoFHandler` and any necessary level/FE metadata inside the portable object or provide wrapper methods that expose `n_components()`, `get_mg_level()`, and/or `get_dof_handler()`.

3. Vector partitioner and vector initialization helpers
   - Problem: MGTwoLevelTransfer relies on partitioners and `initialize_dof_vector` semantics.
   - Resolution: Implement `get_vector_partitioner()` and `initialize_dof_vector()` in the portable wrapper, or provide equivalent utilities that construct correctly sized host/device buffers and partitioner objects.

4. FE evaluation / gather-scatter API
   - Problem: The existing `FEEvaluation` classes assume CPU memory and CPU-side loops.
   - Resolution: Provide GPU-capable `PortableFEEvaluation` helpers or adapt the `cell_loop` callback interface so that lambdas receive portable-evaluation objects. Ensure they implement the operations MGTwoLevelTransfer expects:
     - `read_dof_values` / `read_dof_values_unconstrained`
     - `get_value` / `submit_value` / `distribute_local_to_global` (and unconstrained variants)

5. `cell_loop` semantics
   - Problem: The CPU `cell_loop` accepts a lambda which uses `FEEvaluation`. Portable backend should provide a similar loop abstraction or a way to launch GPU kernels with the same semantics.
   - Resolution: Extend the portable API so users can call a `cell_loop`-like method that invokes a provided functor on every batch. The functor should be able to (a) access the batch index and lane indices, (b) access the mapped `active_cell_index()` values, and (c) operate with `PortableFEEvaluation`.

## Recommended approach: wrapper + small API extensions

Two feasible approaches:

A) Wrapper type that unifies `dealii::MatrixFree` and `Portable::MatrixFree` underneath a common interface. The wrapper stores whichever concrete MatrixFree instance is in use and translates calls into the backend-specific primitives. This is low-risk and keeps the portable backend changes minimal.

B) Rebase both implementations on a common abstract base class (`AbstractMatrixFreeInterface`) implemented by `MatrixFree` and `Portable::MatrixFree`. This is cleaner long-term but requires more invasive changes.

Given development velocity and backward compatibility, I recommend approach (A) initially.

## Concrete interface additions (wrapper or Portable::MatrixFree)

Make the following available on the wrapper or add them to `Portable::MatrixFree` directly:

- Metadata accessors
  - `const DoFHandler<dim,spacedim> & get_dof_handler() const`
  - `unsigned int get_mg_level() const`
  - `unsigned int n_components() const`

- Vector utilities
  - `std::shared_ptr<const VectorPartitioner> get_vector_partitioner() const` (or equivalent)
  - `void initialize_dof_vector(VectorType &v) const` (creates sized vector on host/device)

- Cell identity mapping
  - `unsigned int active_cell_index(unsigned int batch, unsigned int lane) const`
  - Alternatively: return a pointer/reference to an internal array mapping batch/lane to active cell index.

- `cell_loop` abstraction
  - `template <typename Functor> void cell_loop(Functor f, /* other args */) const`
  - The callable `f` should receive a small context that allows constructing `PortableFEEvaluation` for the batch/lane and access to the mapped cell index.

- Portable FE evaluation helpers
  - `PortableFEEvaluation` (or a thin adapter) implementing the small subset of methods used by MGTwoLevelTransfer:
    - `read_dof_values_unconstrained(src_vector)`
    - `distribute_local_to_global_unconstrained(dst_vector)`
    - other accessors used during interpolation/restriction

## Changes to MGTwoLevelTransfer

- Template the class on the MatrixFree type (or on the wrapper interface), e.g. `template <typename MatrixFreeType>` so it can accept both `MatrixFree` and `Portable::MatrixFree` wrappers without duplicating logic.

- During `reinit`, if the backend is portable, call into a new helper that builds the fine->coarse mapping using `active_cell_index(batch,lane)`.

- Inside prolongate/restrict implementations, use a `PortableFEEvaluation` when the backend is portable. `if constexpr` or tag-dispatch can select CPU `FEEvaluation` vs `PortableFEEvaluation`.

- Keep existing CPU paths unchanged (legacy `MatrixFree`) to preserve performance and correctness.

## Edge cases and testing

Edge cases to validate:
- Nonuniform partitioning where some batches are partially empty.
- Constrained DOFs / hanging nodes: ensure unconstrained read/write variants are correctly implemented.
- Multi-component systems: ensure `n_components()` and value submission gather/scatter handle interleaved component layouts.
- Distributed runs: ensure vector partitioners and global indexing match MPI layout.

Tests to add:
- Unit test that builds the fine->coarse cell mapping from both CPU and portable backends and compares results.
- Prolongation/restriction correctness test on a small mesh using both backends (compare outputs).
- Performance micro-bench that runs prolongation on a representative patch and compares timings (optional, later).

## Implementation plan and next steps

1. Implement a thin wrapper type `MatrixFreeWrapper` that can be constructed from either `MatrixFree` or `Portable::MatrixFree` and exposes the interface listed above.
2. Add cell-mapping recording to `Portable::MatrixFree` setup (or record it from the wrapper during construction) and expose `active_cell_index(batch,lane)` through the wrapper.
3. Add `PortableFEEvaluation` helpers and a `cell_loop` variant for the portable backend so user lambdas can operate similarly to the CPU paths.
4. Template `MGTwoLevelTransfer` (or add overloads) to accept the wrapper type and branch to portable evaluation helpers when needed.
5. Add tests (mapping test + correctness test), and run through distributed mode.

Small, low-risk extras to add while here:
- Add debug checks in `reinit` that verify the cell mapping covers all expected cells.
- Add a fallback CPU emulation mode where portable backend runs on host memory for easier debugging.

## Quick contract (2–4 bullets)

- Inputs: `MatrixFree` (CPU) or `Portable::MatrixFree` (GPU/wrapped) instance, source/destination vectors, MG level info.
- Outputs: Correct prolongation/restriction results identical across backends.
- Error modes: Mismatched partitioner layouts, missing cell mapping, or unsupported FE features cause a clear assertion message.
- Success criteria: Tests compare CPU vs Portable outputs bitwise or numerically close (within tolerance for FP differences).

## Follow-up

If you want, I can:
- Draft the `MatrixFreeWrapper` header with the minimal interface described above.
- Add `active_cell_index` recording into `Portable::MatrixFree` (or show how the wrapper can capture it during construction).
- Implement `PortableFEEvaluation` stubs and update `MGTwoLevelTransfer` to accept the wrapper.

Tell me which of these you'd like me to implement next and I will proceed.
