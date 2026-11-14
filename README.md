# nmslib-rs

[![PyPI](https://img.shields.io/pypi/v/nmslib-rs?label=pypi)](https://pypi.org/project/nmslib-rs/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/nmslib-rs/)
[![ci](https://img.shields.io/github/actions/workflow/status/undo-git-pull/nmslib-rs/ci.yml?branch=main&label=ci)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/actions/workflow/status/undo-git-pull/nmslib-rs/release.yml?label=release)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/release.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/undo-git-pull/nmslib-rs/blob/main/LICENSE)

| Python | Linux | macOS |
|---|---|---|
| 3.11 | [![Python 3.11 on ubuntu-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.11%20%28ubuntu-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) | [![Python 3.11 on macos-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.11%20%28macos-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) |
| 3.12 | [![Python 3.12 on ubuntu-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.12%20%28ubuntu-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) | [![Python 3.12 on macos-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.12%20%28macos-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) |
| 3.13 | [![Python 3.13 on ubuntu-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.13%20%28ubuntu-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) | [![Python 3.13 on macos-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.13%20%28macos-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) |
| 3.14 | [![Python 3.14 on ubuntu-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.14%20%28ubuntu-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) | [![Python 3.14 on macos-latest](https://img.shields.io/github/check-runs/undo-git-pull/nmslib-rs/main?nameFilter=Python%203.14%20%28macos-latest%29&label=build)](https://github.com/undo-git-pull/nmslib-rs/actions/workflows/ci.yml) |

A Rust implementation of HNSW, the graph index behind NMSLIB's `hnsw` method, with Python bindings built by maturin. It searches dense `float32` vectors under L2 or cosine distance, builds in parallel, and saves indexes to a file that several processes can memory-map and share.

The build follows NMSLIB's choices for level generation, neighbor selection, overflow pruning and the `post=2` second pass (recorded in [NOTES.md](https://github.com/undo-git-pull/nmslib-rs/blob/main/NOTES.md)), so recall at a given `ef` matches NMSLIB's. The API and file format are its own; nmslib-rs does not read NMSLIB indexes and is not a drop-in replacement for the `nmslib` module.

> [!WARNING]
> nmslib-rs has not been tested in production workloads. It implements a small part of NMSLIB: the `hnsw` and `brute_force` methods, with dense `float32` vectors under L2 and cosine distance. NMSLIB's other search methods, its other distance spaces, and its sparse and string data types are not included.

## Installing

```sh
pip install nmslib-rs
```

The package is imported as `nmslib_rs`. PyPI has wheels for CPython 3.11, 3.12, 3.13 and 3.14 on Linux (x86_64 and aarch64, glibc 2.17 or later), macOS (x86_64 and Apple Silicon) and Windows (x64). On any other platform pip builds from the source distribution, which needs Rust 1.89 or later. The only runtime dependency is NumPy 2.

To work on nmslib-rs itself, see [Development](https://github.com/undo-git-pull/nmslib-rs#development).

## Python API

```python
import numpy as np
import nmslib_rs

data = np.random.default_rng(0).standard_normal((100_000, 128), dtype=np.float32)
queries = data[:5]

index = nmslib_rs.Index(dim=128, space="l2", M=16, ef_construction=200)
index.add_items(data)
index.set_ef(80)
labels, distances = index.knn_query(queries, k=10)

index.save("sift.idx")
shared = nmslib_rs.Index.load("sift.idx", mmap=True)
```

The module exports two classes, `Index` and `BruteForceIndex`, and one function, `simd_level()`.

### `Index(dim, space="l2", M=16, ef_construction=200, seed=0, post=True)`

Creates an empty index.

- `dim`: vector dimension, positive.
- `space`: `"l2"` or `"cosine"`.
- `M`: neighbors selected per node on insert, at least 2. Upper-level lists hold up to `M` neighbors and level-0 lists up to `2 * M`.
- `ef_construction`: beam width during insert, positive.
- `seed`: seeds the random level assignment. A sequential build (`num_threads=1`) with the same seed and the same input produces the same graph.
- `post`: when true, the first `add_items` call runs a second build in reverse insertion order and merges its level-0 lists into the first, as NMSLIB does with `post=2`. This raises recall at a given `ef` and adds a second build's worth of time. It applies only to the first `add_items` call.

Invalid arguments raise `ValueError`.

### `add_items(vectors, num_threads=0)`

Adds vectors to the index. `vectors` must be a C-contiguous `numpy.ndarray` of dtype `float32` and shape `(n, dim)`; anything else raises `ValueError`, with a hint to convert using `np.ascontiguousarray(x, dtype=np.float32)`. Lists, other array types and `float64` arrays are rejected rather than converted.

Labels are insertion positions: the first vector added is label 0, and numbering continues across calls. For `space="cosine"` the index stores normalized copies of the vectors.

`num_threads=0` uses all cores. `num_threads=1` runs the sequential build, which is deterministic for a given seed. Parallel builds are not deterministic, but their recall matches the sequential build's.

When the build finishes, the index is rearranged into the cache-line-aligned layout used by saved files, which makes queries about 25% faster on SIFT1M and about 5% faster on GloVe-100. The rearrangement copies the index, so memory briefly holds two copies (about 540 MB extra on SIFT1M), and the next `add_items` call converts back before inserting.

The GIL is released during the build. Calling `add_items` on an index loaded with `mmap=True` raises `RuntimeError`.

### `set_ef(ef)`

Sets the query beam width (`efSearch` in NMSLIB). The default is 10. A query searches with `max(ef, k)`, so `ef` below `k` has no effect. Higher values raise recall and lower throughput; see the table under "Benchmark" below. `ef=0` raises `ValueError`.

### `knn_query(queries, k=10, num_threads=0, filter=None, max_candidates=0)`

Finds the `k` nearest neighbors of each row of `queries`, which has the same requirements as the `vectors` argument to `add_items`. Returns a tuple `(labels, distances)`:

- `labels`: `int64` array of shape `(n_queries, k)`.
- `distances`: `float32` array of shape `(n_queries, k)`, nearest first.

Distances are squared L2 for `space="l2"` and `1 - cosine similarity` for `space="cosine"`. If fewer than `k` results are found, the missing entries have label `-1` and distance `inf`.

`num_threads=0` spreads the queries across all cores. The GIL is released during the search, and several Python threads may query one index at the same time. `k=0` raises `ValueError`.

`filter` restricts the results to a subset of labels. It is a contiguous `numpy.ndarray` of dtype `bool` and shape `(len(index),)`, and only labels where it is `True` are returned; the same mask applies to every query. Anything else raises `ValueError`.

```python
mask = categories == "shoes"            # one bool per label
labels, distances = index.knn_query(queries, k=10, filter=mask)
```

The search still walks through nodes outside the filter, so the graph stays connected however few labels pass, and it keeps searching until it has `max(ef, k)` passing results or runs out of graph. When most labels pass this costs about the same as an unfiltered query. When few pass, a query can visit a large part of the index. On 100,000 SIFT vectors on one thread, unfiltered queries reach recall@10 0.998 at about 14,000 queries per second; with 10% of labels passing, filtered queries reach the same recall (within the filter) at about 8,000 per second, and with 1% passing at about 2,100 per second. `max_candidates` caps the number of nodes a query expands (`0`, the default, means no cap). With 1% passing, `max_candidates=2000` kept recall at 1.0 and held throughput near 1,100 per second at every `ef`, where uncapped queries at `ef=640` slowed to 67 per second. A query that hits the cap returns the passing results it has found, which may be fewer than `k`, padded with `-1` / `inf`. A negative value raises `ValueError`.

### `graph_stats()` and `self_recall(ef=64, num_threads=0)`

Debugging tools that check a built or loaded index; both are read-only and release the GIL.

`graph_stats()` walks level 0 of the graph breadth-first from the entry point and returns a dict:

- `nodes`, `reached`, `unreachable`: node counts. A search can miss an unreachable node at any `ef`, so `unreachable` should be 0. Sequential builds and builds with `post=True` have had none in testing; a parallel build without `post` can strand a few, more with more threads.
- `one_way_links`: links whose target does not link back. Pruning drops links one direction at a time, so this is normally well above zero.
- `mean_degree`, `min_degree`: level-0 neighbor list lengths.

`self_recall(ef=64)` queries every stored vector with itself and returns `{"nodes", "top1", "missing"}`: how many vectors came back as their own nearest result, and how many were absent from their own top `ef`. A vector whose exact duplicate takes first place counts as found. It costs one query per stored vector.

```python
index.graph_stats()   # {'nodes': 100000, 'reached': 100000, 'unreachable': 0, ...}
index.self_recall()   # {'nodes': 100000, 'top1': 100000, 'missing': 0}
```

### `save(path)`

Writes the index to `path`. It writes a temporary file in the same directory and renames it over `path`, so a process that has the old file memory-mapped keeps a valid mapping. The file format is little-endian, versioned and specific to nmslib-rs; `src/frozen.rs` describes it.

### `Index.load(path, mmap=True)`

A static method that loads a file written by `save`.

With `mmap=True` the file is mapped read-only. Loading takes milliseconds, since the vectors are read only when queries touch them, and every process that maps the same file shares one copy in the OS page cache. The index is read-only: `add_items` raises `RuntimeError`. The file must not be truncated or modified in place while it is mapped; doing so can crash the process or return wrong results. Replacing it with `save` is safe, because `save` renames a new file into place.

With `mmap=False` the file is read into private memory and the index can grow with `add_items`.

A file that is not a valid index for this version raises `ValueError`, and a file that cannot be read raises `OSError`. Both modes return results identical to the index that was saved.

The first queries against a mapped index whose file is not in the page cache run at disk speed. On SIFT1M the first thousand queries took about 10 times as long as the same queries warm.

### Attributes

| attribute | type | meaning |
|---|---|---|
| `dim` | `int` | vector dimension |
| `space` | `str` | `"l2"` or `"cosine"` |
| `ef` | `int` | current query beam width |
| `read_only` | `bool` | true for an index loaded with `mmap=True` |
| `len(index)` | `int` | number of vectors |

All attributes are read-only; use `set_ef` to change `ef`.

### `BruteForceIndex(dim, space="l2")`

An exact index that compares each query with every stored vector, the counterpart of NMSLIB's `brute_force` method. It uses the same distance kernels, cosine normalization and tie order as `Index`, so on the same data its results are the exact answer that `Index.knn_query` approximates. That makes it useful for measuring recall on your own data and for checking whether a bad result comes from the graph search or from the data.

```python
exact = nmslib_rs.BruteForceIndex(dim=128, space="l2")
exact.add_items(data)
true_labels, true_distances = exact.knn_query(queries, k=10)
recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, true_labels)])
```

It has `add_items(vectors)`, `knn_query(queries, k=10, num_threads=0, filter=None)`, `dim`, `space` and `len()`, with the same input checks and return conventions as `Index`. There is no `ef`, and the index cannot be saved. Each query reads every stored vector, so query time grows linearly with the index size: on an Apple M1 Max with 100,000 SIFT vectors it answers about 930 queries per second on one thread, against about 9,000 for `Index` at `ef=80` with recall@10 of 0.998 (see `bench/brute_force.py` and `bench/results/nmslib_rs_100k_mac.json`).

### `simd_level()`

Returns the name of the SIMD instruction set the distance kernels selected on this machine, one of `"AVX-512"`, `"AVX2+FMA"`, `"SSE4.2"` or `"SSE2"` on x86 and `"NEON"` on Apple Silicon. Selection happens at run time, so one build runs on any CPU of its architecture.

## Limitations

- Vectors are dense `float32`. There is no sparse or quantized storage, and no update of existing vectors.
- There is no true deletion, but a mask can hide deleted labels: keep a `bool` array with `False` for each deleted label and pass it as `filter` to every query. The vectors stay in the index and are still walked through during search, so a query slows down as the deleted fraction grows; rebuild the index once many labels are deleted.
- Labels are always insertion positions. Keep your own array to map them to external ids.
- Floating-point distances differ slightly between the AVX and NEON kernels, so a file saved on one architecture and loaded on another can return slightly different distances and, rarely, a different order among near ties.

## Development

Building from a checkout needs Rust 1.89 or later and Python 3.11 or later. Create a virtual environment at `.venv`, and install the package in editable mode with its development dependencies:

```sh
git clone https://github.com/undo-git-pull/nmslib-rs.git
cd nmslib-rs
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

`pip install -e` builds the extension once and installs the `dev` extra: maturin, pytest, h5py and `nmslib` 2.1.2 for the tests and benchmarks. After editing anything in `src/`, rebuild in place with:

```sh
maturin develop --release
```

`maturin develop` compiles the Rust code and installs it into the active virtual environment. It does not watch for changes, so run it after each edit. Without `--release` it compiles faster, but the unoptimized build queries much more slowly, so use `--release` before measuring anything.

`.cargo/config.toml` points PyO3 at `.venv/bin/python`, so `cargo test` and `cargo bench` expect the virtual environment to be at `.venv`.

### Tests

```sh
cargo test                 # Rust unit and property tests
pytest tests/              # Python binding, brute-force and persistence tests
cargo fmt --check          # formatting, enforced by CI
cargo bench                # SIMD kernels against the scalar reference
```

`cargo test` builds with optimizations (`[profile.test]` sets `opt-level = 3`) because the graph tests build 10,000-node indexes. `pytest` runs against whatever `maturin develop` last installed, so rebuild first after changing Rust code.

CI runs the formatting check, the Rust tests and the Python tests on Ubuntu and macOS with Python 3.11 to 3.14 on every push. [RELEASING.md](https://github.com/undo-git-pull/nmslib-rs/blob/main/RELEASING.md) describes how wheels are built and published to PyPI.

### Benchmark scripts

The scripts in `bench/` compare recall and speed with NMSLIB. They expect the ann-benchmarks HDF5 files (`sift-128-euclidean.hdf5`, `glove-100-angular.hdf5`) in `bench/data/`.

```sh
python bench/nmslib_baseline.py --n 100000     # NMSLIB reference curve and noise band
python bench/eval_recall.py --n 100000         # nmslib-rs through the Python binding
python bench/compare.py                        # side by side, with the parity check
python bench/compare.py --size 1M
```

Recall depends on insertion order, so the baseline script builds NMSLIB several times from shuffled orders and records how far recall moves between builds. `compare.py` passes an `efSearch` value when every nmslib-rs build is within that spread of NMSLIB's mean recall, or within 0.5 percentage points if the spread is narrower, and exits non-zero otherwise.

`bench/brute_force.py` runs exact search with `BruteForceIndex`, NMSLIB's `brute_force` method and a float64 numpy scan, then reports the queries per second and the recall of each against the float64 ground truth.

```sh
python bench/brute_force.py --n 100000 --queries 500
```

`bench/mmap_pss.py` measures the memory of several processes sharing one mapped index. It reads `/proc`, so it runs on Linux only.

## Benchmark

SIFT1M (1,000,000 vectors, 128 dimensions, L2), `M=16`, `ef_construction=200`, `post=2`, 10,000 queries on one thread, recall@10 averaged over several builds. Measured on an AMD Threadripper 7970X against NMSLIB 2.1.2 compiled with `-march=native`:

| efSearch | NMSLIB recall | nmslib-rs recall | NMSLIB QPS | nmslib-rs QPS |
|---:|---:|---:|---:|---:|
| 10 | 0.7773 | 0.7780 | 32,274 | 39,988 |
| 20 | 0.8882 | 0.8887 | 21,325 | 25,341 |
| 40 | 0.9568 | 0.9569 | 13,048 | 15,415 |
| 80 | 0.9880 | 0.9880 | 7,366 | 8,576 |
| 160 | 0.9974 | 0.9975 | 4,062 | 4,821 |
| 320 | 0.9997 | 0.9996 | 2,207 | 2,692 |
| 640 | 1.0000 | 1.0000 | 1,203 | 1,502 |

On 32 threads the build took 25.6 s against NMSLIB's 51.8 s, and the index used 731 MB against 2,094 MB. The PyPI `nmslib` wheel, which is built without native CPU flags, runs about half as fast as the native build in this table.

On an Apple M1 Max, nmslib-rs ran at 0.93x NMSLIB's QPS at efSearch 10 and 1.08 to 1.18x at efSearch 20 to 640, with the same recall. When 8 processes map one saved SIFT1M index (684 MB), their summed PSS is 909 MB; 8 private copies would take 5,312 MB. Details of each measurement, including the GloVe-100 cosine runs, are in [NOTES.md](https://github.com/undo-git-pull/nmslib-rs/blob/main/NOTES.md) and `bench/results/`.

## Acknowledgments

nmslib-rs reimplements the HNSW method from [NMSLIB](https://github.com/nmslib/nmslib), the Non-Metric Space Library by Bilegsaikhan Naidan, Leonid Boytsov, Yury Malkov, David Novak and other contributors, distributed under the Apache License 2.0. The neighbor selection, pruning, `post=2` merge, prefetching and interleaved level-0 layout follow NMSLIB's `hnsw.cc` as described in [NOTES.md](https://github.com/undo-git-pull/nmslib-rs/blob/main/NOTES.md). No NMSLIB source code is included in this repository. The `bench/` scripts use the `nmslib` Python package as the reference for recall and speed.

HNSW was introduced by Yu. A. Malkov and D. A. Yashunin in "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE Transactions on Pattern Analysis and Machine Intelligence 42(4), 2020 ([arXiv:1603.09320](https://arxiv.org/abs/1603.09320)).

## License

MIT. See [LICENSE](https://github.com/undo-git-pull/nmslib-rs/blob/main/LICENSE).
