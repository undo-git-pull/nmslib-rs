"""nmslib_rs on SIFT through the Python binding: recall@10, QPS, build time, memory.

Mirrors nmslib_baseline.py: several builds from shuffled insertion orders,
each in a fresh process, the same efSearch sweep on one query thread, and the
same JSON layout. Also records the parity check against the NMSLIB baseline.

    python bench/eval_recall.py --n 100000
    python bench/eval_recall.py --dataset glove --n 0         # all of GloVe-100
    python bench/eval_recall.py --filter-fraction 0.01        # filtered search

With --filter-fraction, every query passes the same random mask as `filter`,
recall is measured against the exact answer within the mask, and the NMSLIB
parity check is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import sys
import time
from importlib.metadata import version

import numpy as np

import common


FILTER_SEED = 12345


def filter_mask(n: int, fraction: float) -> np.ndarray:
    """The benchmark's filter over dataset ids, the same for every build."""
    return np.random.default_rng(FILTER_SEED).random(n) < fraction


def run_build(dataset: str, n: int, seed: int, threads: int, mmap: bool,
              filter_fraction: float | None = None, max_candidates: int = 0) -> dict:
    import nmslib_rs

    ds = common.DATASETS[dataset]
    base, queries = common.load(ds, n)
    if filter_fraction is None:
        gt = common.ground_truth(base, queries, ds=ds)
    else:
        mask = filter_mask(len(base), filter_fraction)
        passing = np.flatnonzero(mask)
        gt = passing[common.brute_force_knn(base[passing], queries, common.K, ds.metric)]
    order = np.random.default_rng(seed).permutation(len(base))
    # Labels are positions in the shuffled order, so the filter is too.
    label_filter = None if filter_fraction is None else np.ascontiguousarray(mask[order])
    shuffled = np.ascontiguousarray(base[order])

    index = nmslib_rs.Index(dim=base.shape[1], space=ds.metric, M=common.M,
                        ef_construction=common.EF_CONSTRUCTION, seed=seed, post=True)
    common.reset_peak_rss()
    before = common.rss_mb()
    t0 = time.perf_counter()
    index.add_items(shuffled, num_threads=threads)
    build_time = time.perf_counter() - t0
    after_build = common.rss_mb()
    print(f"[seed {seed}] build {build_time:.1f}s", file=sys.stderr, flush=True)
    mapped = {}
    if mmap:
        # Query a memory-mapped copy instead: save, drop the built index, load.
        path = common.DATA_DIR / f".eval_recall_{os.getpid()}.nmslib_rs"
        index.save(path)
        del index
        t0 = time.perf_counter()
        index = nmslib_rs.Index.load(path, mmap=True)
        mapped = {"queried": "memory-mapped saved index", "load_ms": round((time.perf_counter() - t0) * 1e3, 2),
                  "file_mb": round(path.stat().st_size / 2**20, 1)}
        path.unlink()  # the mapping stays valid after unlinking

    sweep = []
    for ef in common.EF_SEARCH:
        index.set_ef(ef)
        common.reset_peak_rss()
        t0 = time.perf_counter()
        labels, _ = index.knn_query(queries, k=common.K, num_threads=1, filter=label_filter,
                                    max_candidates=max_candidates)
        elapsed = time.perf_counter() - t0
        found = np.where(labels >= 0, order[np.maximum(labels, 0)], -1)
        r = common.recall(base, queries, gt, found, ds.metric)
        sweep.append({
            "ef_search": ef,
            **r,
            "qps": round(len(queries) / elapsed, 1),
            "query_time_s": round(elapsed, 3),
            "peak_rss_mb": common.rss_mb()["peak_rss_mb"],
        })
        print(f"[seed {seed}] ef {ef:4d} recall {r['recall']:.4f} "
              f"qps {len(queries) / elapsed:9.1f}", file=sys.stderr, flush=True)

    return {
        "insertion_order_seed": seed,
        "build_time_s": round(build_time, 2),
        **({"mmap": mapped} if mapped else {}),
        "memory": {
            "rss_before_build_mb": before["rss_mb"],
            "peak_rss_build_mb": after_build["peak_rss_mb"],
            "rss_after_build_mb": after_build["rss_mb"],
            "index_mb": round(after_build["rss_mb"] - before["rss_mb"], 1),
        },
        "sweep": sweep,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(common.DATASETS), default="sift")
    ap.add_argument("--n", type=int, default=100_000,
                    help="base vectors: a prefix of the dataset, or 0 for all of it")
    ap.add_argument("--builds", type=int, default=3,
                    help="independent builds (insertion-order seeds 1..N)")
    ap.add_argument("--mmap", action="store_true",
                    help="query a saved, memory-mapped copy of each built index")
    ap.add_argument("--threads", type=int, default=32,
                    help="build threads (NMSLIB baseline uses indexThreadQty=32; 1 = sequential)")
    ap.add_argument("--filter-fraction", type=float,
                    help="query with a random filter passing this fraction of labels")
    ap.add_argument("--max-candidates", type=int, default=0,
                    help="expansion cap for filtered queries (0 = none)")
    ap.add_argument("--reference", help="NMSLIB JSON for the parity check "
                    "(default bench/results/nmslib_<dataset><n>.json)")
    ap.add_argument("--out", help="output JSON (default bench/results/nmslib_rs_<n>.json)")
    args = ap.parse_args()
    ds = common.DATASETS[args.dataset]
    if args.n == 0:
        args.n = ds.full_n

    filtered = args.filter_fraction is not None
    if filtered and not 0 < args.filter_fraction <= 1:
        ap.error("--filter-fraction must be in (0, 1]")
    base, queries = common.load(ds, args.n)
    if not filtered:
        common.ground_truth(base, queries, ds=ds)
    del base, queries

    import nmslib_rs

    ctx = mp.get_context("spawn")
    builds = []
    for seed in range(1, args.builds + 1):
        with ctx.Pool(1) as pool:
            builds.append(pool.apply(run_build, (args.dataset, args.n, seed, args.threads, args.mmap,
                                                 args.filter_fraction, args.max_candidates)))

    if filtered:
        reference = {"recall_metric": "distance-based recall@10 within the filter"}
        parity = []
    else:
        reference = json.loads(open(args.reference or common.result_path("nmslib", ds, args.n)).read())
        parity = common.parity(reference, builds)
    result = {
        "library": "nmslib_rs",
        "library_version": version("nmslib-rs"),
        "simd_level": nmslib_rs.simd_level(),
        "dataset": ds.name,
        "space": ds.metric,
        "n_base": args.n,
        "n_queries": 10_000,
        "k": common.K,
        "recall_metric": reference["recall_metric"],
        "query_threads": 1,
        "queried_index": "memory-mapped" if args.mmap else "in-memory",
        **({"filter": {"fraction": args.filter_fraction, "seed": FILTER_SEED,
                       "passing": int(filter_mask(args.n, args.filter_fraction).sum()),
                       "max_candidates": args.max_candidates}} if filtered else {}),
        "params": {"M": common.M, "efConstruction": common.EF_CONSTRUCTION, "post": True,
                   "build_threads": args.threads},
        "machine": common.machine_info(),
        "builds": builds,
        "noise_band": common.noise_band(builds),
        **({} if filtered else {"parity_vs_nmslib": parity,
                                "all_pass": all(p["pass"] for p in parity)}),
    }
    out = args.out or common.result_path("nmslib_rs", ds, args.n)
    if filtered and not args.out:
        out = out.with_name(f"{out.stem}_filter{args.filter_fraction:g}"
                            + (f"_cap{args.max_candidates}" if args.max_candidates else "") + out.suffix)
    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
