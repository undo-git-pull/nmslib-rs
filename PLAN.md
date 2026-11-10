# Rust HNSW with Python bindings: plan

## Goal

A Rust implementation of HNSW (the core algorithm from NMSLIB) exposed to Python through maturin. The only success criterion is recall parity with NMSLIB's `hnsw` method on a 1M-vector dataset, measured at matching parameters. Speed is measured and reported but is not a gate.

Linux x86_64 comes first. Mac (Apple Silicon) is tested manually afterward, using the checklist in phase 7.

## Scope

In scope: dense `f32` vectors, L2 and cosine distance, index build, k-NN query, batch query from NumPy, `ef_search` control, saving an index and loading it memory-mapped and read-only so several processes share one copy (phase 6).

Out of scope: deletion, sparse vectors, other NMSLIB methods (VP-tree, NAPP and so on), NMSLIB API compatibility, `f16`/`i8` storage, the query server, serialization compatibility with NMSLIB indices, modifying a saved file in place.

## Dataset

SIFT1M (1,000,000 base vectors, 128 dimensions, L2, 10,000 queries with ground truth). The ann-benchmarks project distributes it as an HDF5 file (`sift-128-euclidean`) with `train`, `test` and `neighbors` arrays. Memory use is about 512 MB for the vectors plus about 128 MB for level-0 links at `M=16`.

GloVe-100 (about 1.18M vectors, angular) is a second dataset for the cosine path, run after SIFT1M passes.

Every phase below runs first on a 100k prefix of the dataset. The full 1M run happens in phase 5. Compute ground truth for the 100k subset by brute force, since the shipped ground truth covers the full set.

## Working rules for Claude Code

1. Finish each phase and pass its gate before starting the next one. Commit at each gate.
2. Read the docs for `fearless_simd` (docs.rs and the examples in the linebender/fearless_simd repository) before writing any kernel code. Check the latest crate versions and the minimum supported Rust version first.
3. Use safe Rust everywhere at first. Introduce `unsafe` only after profiling shows a specific hot spot, and record why. The one planned exception is mapping a file in phase 6, which Rust cannot make safe; keep it to one `unsafe` block with its safety condition written next to it.
4. Keep dependencies small: `fearless_simd`, `pyo3`, `numpy`, `rayon`, `rand`, plus `memmap2` and `bytemuck` from phase 6. Dev dependencies: `proptest`, `criterion`.
5. Read the NMSLIB HNSW source (under `similarity_search/src/method/`, look for `hnsw.cc`) before phase 3. Record the defaults it uses for neighbor selection, level generation and `maxM0` in `NOTES.md`, and match them.

## Layout

```
nmslib-rs/
  Cargo.toml
  pyproject.toml          # maturin build backend
  NOTES.md                # NMSLIB behaviors found while reading the source
  src/
    lib.rs                # PyO3 module
    simd.rs               # distance kernels (fearless_simd) + scalar reference
    hnsw.rs               # index: insert, search_layer, select_neighbors, knn_query
    visited.rs            # generation-stamped visited sets
    frozen.rs             # saved-index file format, read-only memory-mapped graph (phase 6)
  tests/
  bench/
    nmslib_baseline.py
    eval_recall.py
    data/
```

## Phase 0: setup

- `maturin new` with PyO3 bindings, then `maturin develop --release` to confirm an empty module imports.
- Create a Python 3.12 or 3.13 virtual environment (NMSLIB publishes wheels for 3.8 to 3.13) and install `nmslib`, `numpy`, `h5py`.
- Download the dataset into `bench/data/`.

Gate: `import nmslib_rs` works and `import nmslib` works in the same environment.

## Phase 1: fearless_simd smoke test

This is a standalone check that the SIMD approach works before any HNSW code exists. It lives in `src/simd.rs` with its own tests.

Kernels (all on `&[f32]`):

- `l2_sq(a, b) -> f32`
- `dot(a, b) -> f32`

Each has a plain scalar reference implementation (`l2_sq_scalar`, `dot_scalar`) used only for testing.

Tests:

- Dimensions 1, 3, 7, 15, 16, 17, 100, 128, 129, 768, 960, so that tail handling (lengths that are not a multiple of the vector width) gets exercised.
- Random inputs, including large and small magnitudes. Compare against the scalar version with a relative tolerance around `1e-4`. Summation order differs, so exact equality is not expected.
- A `proptest` property for random lengths between 0 and 2048.
- A zero-length input returns 0.

Benchmark (`criterion`):

- Scalar vs SIMD at dimension 128 and 960.
- A loop of 1M calls on short vectors, to measure per-call dispatch overhead. If dispatch costs a noticeable share of a dim-128 distance, restructure so the SIMD level is selected once per search or insert call and the inner loop runs inside it.
- Print which instruction set was selected on this machine, if the crate exposes that.

Gate: tests pass, and the SIMD kernels are clearly faster than scalar at dim 128 (several times faster on an AVX2 machine is the expectation; investigate if the gain is under 2x). Check the generated code or the crate's multiversioning configuration if the gain is missing.

## Phase 2: NMSLIB baseline

`bench/nmslib_baseline.py` builds and queries NMSLIB's HNSW and writes the results to JSON.

- Build: `nmslib.init(method='hnsw', space='l2')`, `addDataPointBatch`, `createIndex({'M': 16, 'efConstruction': 200, 'post': 2}, print_progress=True)`. Use a fixed `indexThreadQty`.
- Query: for `efSearch` in `[10, 20, 40, 80, 160, 320, 640]`, call `setQueryTimeParams` and `knnQueryBatch(queries, k=10, num_threads=1)`.
- Record recall@10 against ground truth, queries per second, build time and peak memory for each setting.
- Build several times with the base vectors shuffled into different insertion orders (10 builds on 100k, 2 on 1M), and record how far recall moves between builds at each `efSearch`. This gives the noise band for the parity check.

Parity rule (used by the gates in phases 3, 4 and 5): at each `efSearch`, nmslib-rs passes if `|recall - recall_mean| <= tolerance`, where `tolerance` is the larger of the measured band's half-width around the NMSLIB mean and 0.5 percentage points. The baseline JSON stores `tolerance` per `efSearch`. Two builds alone gave bands as narrow as 0.003 points on 100k, too tight to separate a real difference from noise.
- Run on the 100k prefix first, then on the full 1M set.

Gate: baseline JSON files exist for 100k and 1M, including the noise band.

## Phase 3: HNSW core in Rust, single-threaded

Work on the 100k subset.

Data layout: one flat `Vec<f32>` for vectors (`n * dim`), level-0 neighbor lists in a flat `Vec<u32>` with a fixed capacity of `2 * M` per node plus a length array, and upper-level neighbor lists stored separately with capacity `M`.

Functions:

- `random_level()`: `floor(-ln(u) * mL)` with `mL = 1 / ln(M)`.
- `search_layer(query, entry_points, ef, level)`: candidate min-heap, result max-heap, generation-stamped visited array (reused across calls, no per-call allocation).
- `select_neighbors_heuristic(candidates, m)`: the heuristic from the HNSW paper (Algorithm 4), matching the variant NMSLIB uses as recorded in `NOTES.md`.
- `insert(vector)`: greedy descent with `ef = 1` from the top level to the node's level plus one, then `search_layer` with `ef_construction` on each level at or below the node's level, neighbor selection, bidirectional linking, and pruning of any neighbor list that overflows. Update the entry point when a node reaches a new top level.
- `knn_query(query, k, ef_search)`: descent with `ef = 1` through the upper levels, then `search_layer` at level 0 with `max(ef_search, k)`, then return the `k` closest.

Cosine: normalize vectors on insert and on query, then use `1 - dot`. L2 uses squared distance internally and returns the same ordering.

Tests:

- A hand-built graph of a few nodes with known neighbors.
- Structural invariants after building 10k random vectors: no self-loops, no duplicate edges, degree at most `M` (upper levels) or `2 * M` (level 0), every node reachable from the entry point at level 0.
- Recall@10 above 0.99 against brute force on 10k vectors at a high `ef_search`.

Gate: all tests pass and recall on the 100k subset meets the parity rule (phase 2) against the NMSLIB baseline at the matching `efSearch` values. If recall is below NMSLIB, compare average level-0 degree, level distribution and neighbor-selection behavior against `NOTES.md` before touching anything else.

## Phase 4: Python binding and evaluation

Minimal API:

```python
index = nmslib_rs.Index(dim=128, space="l2", M=16, ef_construction=200, seed=0)
index.add_items(vectors)               # float32 array, shape (n, dim)
index.set_ef(80)
labels, distances = index.knn_query(queries, k=10)   # shape (nq, k)
```

Release the GIL during build and query. Validate dtype (`float32`), shape and contiguity at the boundary, and raise `ValueError` with a clear message.

`bench/eval_recall.py` loads the dataset, runs the same `efSearch` sweep as the baseline, and writes results in the same JSON format. A second script prints the two curves side by side (recall@10 vs `efSearch`, with QPS).

Gate: recall curves on 100k meet the parity rule (phase 2) across the whole sweep.

## Phase 5: parallel build and the 1M run

- Add multi-threaded insert with `rayon`: a per-node lock on neighbor lists, a lock around the entry point and max level, and a per-thread visited set. Keep the single-threaded path available for debugging.
- Add parallel batch query.
- Re-run the 100k comparison to confirm parallel build does not lower recall.
- Run the full 1M comparison with the same `M`, `ef_construction` and `efSearch` sweep as the NMSLIB baseline.

Gate (the project's success criterion): at each `efSearch` value, recall@10 on 1M meets the parity rule (phase 2): within the measured NMSLIB noise band, or within 0.5 percentage points if the measured band is narrower. Report build time, QPS and peak memory next to the NMSLIB numbers.

## Phase 6: save and memory-mapped load

Goal: several Python processes query one index while the machine holds a single copy of it. A saved index is read-only; mapping the file lets the OS page cache share its pages between every process that maps it. Today each process would build or hold its own copy (about 670 MB for SIFT1M).

File format (nmslib-rs's own, little-endian, versioned; not compatible with NMSLIB):

- Header: magic bytes, format version, `dim`, space, `M`, `maxM0`, node count, entry point, max level, and the offset and length of every section. Every section starts on a 64-byte boundary.
- Level-0 section: one fixed-stride block per node holding the vector followed by its level-0 list (count plus up to `2 * M` ids), padded to whole cache lines. This is NMSLIB's interleaved layout (NOTES.md, "Query performance"): expanding a node finds its list next to the vector already fetched for its distance, which the phase 5 profile put at about one cache miss per expansion.
- Upper-level section: per-node levels, offsets and lists.
- Cosine indexes store the normalized vectors, as in memory.

API:

```python
index.save(path)                              # writes a temp file, then renames it over `path`
ro = nmslib_rs.Index.load(path, mmap=True)        # read-only, pages shared between processes
rw = nmslib_rs.Index.load(path, mmap=False)       # private copy in memory; add_items allowed
```

- `add_items` on a memory-mapped index raises an error.
- `load` validates the header (magic, version, section sizes against the file length, alignment) and raises `ValueError` on any mismatch. It never reads out of bounds, whatever the file contains.
- `save` never writes into an existing file: it writes a temporary file in the same directory and renames it, so a process that has the old file mapped keeps a valid mapping.

Implementation:

- A read-only graph type for the saved layout. `search_layer` and the query path become generic over how neighbor lists and vectors are read, so the build graph (atomics) and the mapped graph share one search implementation. The prefetching from the phase 5 profiling applies to both.
- `memmap2` maps the file read-only and `bytemuck` turns aligned byte ranges into `&[f32]` and `&[u32]` without copying. The mapping call is `unsafe`: if another process truncates or rewrites the file while it is mapped, reads can fault or return garbage. Record that condition at the `unsafe` block and in the Python docstring for `load`.
- Optional, measured: whether the in-memory index should also use the interleaved layout after a build. Decide from the QPS numbers below, not up front. Decided yes: every `add` now ends by freezing the index into that layout in memory. Measured against the build layout: 1.25 to 1.29x QPS on SIFT 1M, 1.05 to 1.07x on GloVe 1.18M, 0.92 to 1.00x on GloVe 100k (see NOTES.md).

Tests:

- Round trip: build, save, load in both modes, and get identical labels and distances to the original index for a fixed query set. Cover L2 and cosine, with and without `post`, and an index with several upper levels.
- Malformed files: wrong magic, newer version, truncated at every section boundary, section lengths that disagree with the header, a `dim` that does not match the data. Each raises `ValueError`; none crashes.
- `load(mmap=False)` followed by `add_items` works; the same call on a mapped index raises.
- One process can load the same file twice.
- Python: 8 processes (spawn start method) map the same index and query concurrently, getting the same results as one process.

Gate:

- A loaded index returns exactly the labels and distances of the in-memory index on SIFT 1M and GloVe-100 at every `efSearch` in the sweep.
- With 8 processes mapping the SIFT 1M index and querying, the summed PSS (from `/proc/<pid>/smaps_rollup`) is at most one copy of the file plus each process's baseline Python and NumPy memory, with 10% slack. Report per-process RSS and PSS side by side, since RSS counts shared pages once per process.
- `eval_recall.py` on a mapped index meets the parity rule (phase 2) on SIFT 1M. Report its QPS next to the in-memory index and native NMSLIB.
- Report load time for SIFT 1M in both modes. A mapped load should take milliseconds; it does not read the vectors until queries touch them.

## Phase 7: Mac manual test

Done by hand on an Apple Silicon machine after phases 5 and 6 pass on Linux.

1. Install Rust and Python 3.12 or 3.13, then `maturin develop --release`.
2. Run `cargo test --release`. The phase 1 kernel tests exercise the NEON path.
3. Run the phase 1 benchmark and confirm SIMD beats scalar.
4. Install `nmslib` from its Mac wheel, run the baseline on the 100k subset, run `eval_recall.py`, and compare the curves.
5. Save an index on Linux, copy the file to the Mac, and load it with `mmap=True` and `mmap=False`. Results should match the Linux run up to floating-point differences between the AVX and NEON kernels. Then run the phase 6 round-trip and malformed-file tests on the Mac.
6. Note any difference in recall or kernel results, and which instruction set the crate reports.

## Known risks

- Recall below NMSLIB usually comes from differences in neighbor selection, entry-point handling, or `ef` handling at level 0. Check these first.
- Parallel build is non-deterministic, so compare recall curves and avoid comparing exact graphs.
- Floating-point results differ slightly between scalar, AVX2 and NEON paths. Tests use tolerances.
- Per-call SIMD dispatch overhead can matter at dimension 128 (see phase 1).
- A mapped index faults if its file is truncated or rewritten while mapped (phase 6). `save` replaces files by rename to avoid doing this to itself, but nothing stops another program.
- Under memory pressure the OS can evict a mapped index's pages, so the first queries after that run at disk speed. Measure first-query latency after a cold load as well as steady-state QPS.
