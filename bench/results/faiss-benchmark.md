# nmslib-rs against FAISS on SIFT1M

Measured 2026-11-14 at commit 72b8a75 on an AMD Ryzen Threadripper 7970X (32 cores, 64 threads) with two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs (one used). Versions: faiss-cpu 1.16.0 and faiss-gpu 1.16.0 from PyPI (compiled with AVX2 and AVX-512, dynamic dispatch), nmslib-rs 0.2.0 built with `--release`, SIMD level AVX-512.

Raw data: `faiss_wiki_1M.json` (CPU) and `faiss_gpu_1M.json` (GPU). Scripts: `bench/faiss_wiki.py` and `bench/faiss_gpu.py`.

## Reference benchmark

FAISS's wiki page [Indexing 1M vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors) (last edited 2026-10-01) reports `IndexHNSWFlat` on SIFT1M from [`benchs/bench_hnsw.py`](https://github.com/facebookresearch/faiss/blob/main/benchs/bench_hnsw.py). That script uses M=32, efConstruction=40 and efSearch 16 to 256. It searches the 10,000 queries as one batch on 20 threads, times only the search call, and scores R@1: the fraction of queries whose top result is the true nearest neighbor, compared by id. The wiki doesn't give k, so these runs use k=10, which doesn't change the beam width because every efSearch is at least 16. The wiki's hardware for this table isn't stated; elsewhere the page names a Xeon E5-2680 v2.

## Method

- Data: `sift-128-euclidean.hdf5` from ann-benchmarks (the same 1M base vectors and 10k queries as SIFT1M), with its shipped ground truth.
- CPU: FAISS `IndexHNSWFlat` and nmslib-rs `Index`, both with M=32, efConstruction=40, and 20 threads for build and search. nmslib-rs ran with `post=False` (FAISS has no equivalent) and with `post=True`. Each configuration was built 3 times from different shuffled insertion orders, each build in a fresh process.
- GPU: `GpuIndexFlatL2` (exact search) and `GpuIndexIVFFlat` with 4,096 lists, trained on 200k base vectors. Queries and results stay in host memory, so the timings include the copies to and from the GPU.
- Timing: one warm-up, then the fastest of 3 runs. CPU tables report the median over the 3 builds.
- R@1 has a ceiling of 0.9928 on this dataset: SIFT contains duplicate vectors, and an id comparison counts a tied copy as a miss. Exact search scores 0.9928 by id and 1.0 by distance.

## CPU: FAISS HNSW against nmslib-rs

R@1 is the mean over 3 builds, with the min–max range in brackets. Time is ms per query.

| efSearch | wiki R@1 | wiki ms | FAISS R@1 | FAISS ms | nmslib-rs post=False R@1 | ms | nmslib-rs post=True R@1 | ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.8740 | 0.011 | 0.9029 (0.9011–0.9049) | 0.0044 | 0.8184 (0.8169–0.8200) | 0.0022 | 0.8987 (0.8979–0.8995) | 0.0031 |
| 32 | 0.9492 | 0.020 | 0.9543 (0.9522–0.9573) | 0.0076 | 0.9026 (0.9010–0.9034) | 0.0035 | 0.9539 (0.9529–0.9547) | 0.0050 |
| 64 | 0.9779 | 0.033 | 0.9788 (0.9778–0.9803) | 0.0130 | 0.9507 (0.9495–0.9519) | 0.0059 | 0.9792 (0.9783–0.9801) | 0.0085 |
| 128 | 0.9887 | 0.059 | 0.9887 (0.9883–0.9893) | 0.0245 | 0.9770 (0.9764–0.9775) | 0.0107 | 0.9891 (0.9887–0.9895) | 0.0148 |
| 256 | 0.9920 | 0.104 | 0.9915 (0.9913–0.9919) | 0.0440 | 0.9874 (0.9872–0.9878) | 0.0180 | 0.9919 (0.9918–0.9922) | 0.0258 |

| | Build (20 threads) | Index memory |
|---|---:|---:|
| FAISS | 9.5 s | 800 MB |
| nmslib-rs, post=False | 5.1 s | 826 MB |
| nmslib-rs, post=True | 10.0 s | 839 MB |

- FAISS 1.16.0 reproduces the wiki's R@1 to within 0.001 from efSearch 32 up. At efSearch 16 it scores 0.903 against the wiki's 0.874.
- nmslib-rs with `post=True` matches FAISS's R@1 at every efSearch and is 1.4–1.7× faster per query, with the gap growing as efSearch rises. Build time is about the same.
- With `post=False`, nmslib-rs has lower R@1 at the same efSearch but is still faster at matched recall: efSearch 64 (0.951) takes 0.0059 ms against FAISS at efSearch 32 (0.954) taking 0.0076 ms. It builds in about half FAISS's time.

## GPU: FAISS IVF against nmslib-rs on the CPU

The `faiss-gpu` wheel has no GPU HNSW, and CAGRA (FAISS's GPU graph index) is only in the conda `faiss-gpu-cuvs` build, so the GPU side uses IVF and exact search. This compares different index types at matched recall.

Exact search (`GpuIndexFlatL2`): 0.0083 ms per query (120k queries/s), recall 1.0 by distance.

GPU IVF4096 sweep (build: train 0.22 s, add 0.45 s):

| nprobe | 1 | 4 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R@1 | 0.3452 | 0.6419 | 0.8829 | 0.9469 | 0.9772 | 0.9899 | 0.9923 | 0.9927 |
| ms/query | 0.00015 | 0.00034 | 0.00102 | 0.00189 | 0.00362 | 0.00713 | 0.01433 | 0.02971 |

Fastest setting reaching each R@1, in ms per query, with one batch of 10k queries (CPU columns use 20 threads):

| R@1 ≥ | GPU IVF4096 | FAISS HNSW (CPU) | nmslib-rs post=False | nmslib-rs post=True |
|---:|---:|---:|---:|---:|
| 0.90 | 0.0019 (nprobe 32) | 0.0044 (ef 16) | 0.0035 (ef 32) | 0.0050 (ef 32) |
| 0.95 | 0.0036 (nprobe 64) | 0.0076 (ef 32) | 0.0059 (ef 64) | 0.0050 (ef 32) |
| 0.98 | 0.0071 (nprobe 128) | 0.0245 (ef 128) | 0.0180 (ef 256) | 0.0148 (ef 128) |
| 0.99 | 0.0143 (nprobe 256) | 0.0441 (ef 256) | not reached | 0.0258 (ef 256) |

### Batch size

Queries per second when the queries are sent in batches of different sizes, both sides at R@1 about 0.98: GPU IVF with nprobe 64, and nmslib-rs `post=True` with ef 64 on 20 threads. Batches of 1 and 10 were timed on the first 1,000 queries, and larger batches on all 10,000.

| Batch | GPU IVF | nmslib-rs |
|---:|---:|---:|
| 1 | 7,287 | 8,542 |
| 10 | 50,746 | 33,767 |
| 100 | 188,731 | 74,366 |
| 1,000 | 264,522 | 116,455 |
| 10,000 | 275,762 | 121,944 |

nmslib-rs used `num_threads=20` even for single queries, so its batch-1 figure includes thread-pool overhead on every call.

### nmslib-rs on all 64 threads

A separate run, not recorded in the JSON files: one `post=True` build, searched with `num_threads=64` as one batch of 10k queries.

| efSearch | R@1 | ms/query | queries/s |
|---:|---:|---:|---:|
| 32 | 0.9520 | 0.00402 | 249,001 |
| 64 | 0.9776 | 0.00687 | 145,538 |
| 128 | 0.9880 | 0.01190 | 84,005 |
| 256 | 0.9916 | 0.02078 | 48,127 |

Going from 20 to 64 threads made queries only about 1.2× faster. The cause hasn't been profiled; memory bandwidth or SMT are the likely limits.

## Summary

- Against FAISS HNSW on the same CPU and settings, nmslib-rs with `post=True` has the same recall and answers queries 1.4–1.7× faster.
- On the GPU, FAISS IVF beats nmslib-rs once queries arrive in batches of about 10 or more: at R@1 0.98 and batch 10,000 it is 2.3× faster than nmslib-rs on 20 threads and 1.9× faster than on 64. It also builds in under a second. For single queries, nmslib-rs is faster.

## Not covered

- GPU graph search (CAGRA), which needs the conda `faiss-gpu-cuvs` build.
- float16 storage and IVF-PQ on the GPU, and the second GPU.
- GloVe-100 and the README's M=16 / efConstruction=200 / single-thread setup.

## Reproducing

```sh
# CPU: the project venv, with faiss-cpu installed
python bench/faiss_wiki.py

# GPU: a separate venv, because faiss-gpu and faiss-cpu install the same module
uv venv -p 3.12 .venv-faiss-gpu
VIRTUAL_ENV=.venv-faiss-gpu uv pip install faiss-gpu numpy h5py
maturin build --release -i .venv-faiss-gpu/bin/python -o dist
VIRTUAL_ENV=.venv-faiss-gpu uv pip install dist/nmslib_rs-*.whl
.venv-faiss-gpu/bin/python bench/faiss_gpu.py
```

`faiss_gpu.py` reads the CPU curves from `faiss_wiki_1M.json`, so run `faiss_wiki.py` first.
