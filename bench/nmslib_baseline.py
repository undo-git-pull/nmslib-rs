"""NMSLIB HNSW baseline on SIFT: recall@10, QPS, build time and memory.

Builds the index twice, each from a different shuffled insertion order, and
records how far recall moves between the two builds at each efSearch. That
spread is the noise band for the rhnsw parity check.

    python bench/nmslib_baseline.py --n 100000
    python bench/nmslib_baseline.py --n 1000000

Each build runs in a fresh process so its memory numbers are not mixed with
the other build or with dataset loading.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from importlib.metadata import version

import numpy as np

import common

POST = 2
SEEDS = [1, 2]


def run_build(n: int, seed: int, threads: int) -> dict:
    import nmslib

    base, queries = common.load_sift(n)
    gt = common.ground_truth(base, queries)
    order = np.random.default_rng(seed).permutation(len(base))
    shuffled = np.ascontiguousarray(base[order])

    index = nmslib.init(method="hnsw", space="l2")
    common.reset_peak_rss()
    before = common.rss_mb()
    index.addDataPointBatch(shuffled, ids=order)
    t0 = time.perf_counter()
    index.createIndex(
        {"M": common.M, "efConstruction": common.EF_CONSTRUCTION, "post": POST,
         "indexThreadQty": threads},
        print_progress=True,
    )
    build_time = time.perf_counter() - t0
    after_build = common.rss_mb()
    print(f"[seed {seed}] build {build_time:.1f}s", file=sys.stderr, flush=True)

    sweep = []
    for ef in common.EF_SEARCH:
        index.setQueryTimeParams({"efSearch": ef})
        common.reset_peak_rss()
        t0 = time.perf_counter()
        res = index.knnQueryBatch(queries, k=common.K, num_threads=1)
        elapsed = time.perf_counter() - t0
        found = np.full((len(queries), common.K), -1, np.int64)
        for i, (ids, _) in enumerate(res):
            found[i, : len(ids)] = ids
        r = common.recall(base, queries, gt, found)
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
        "memory": {
            "rss_before_build_mb": before["rss_mb"],
            "peak_rss_build_mb": after_build["peak_rss_mb"],
            "rss_after_build_mb": after_build["rss_mb"],
            # Vectors copied into nmslib plus the graph.
            "index_mb": round(after_build["rss_mb"] - before["rss_mb"], 1),
        },
        "sweep": sweep,
    }


def noise_band(builds: list[dict]) -> list[dict]:
    band = []
    for i, ef in enumerate(common.EF_SEARCH):
        rs = [b["sweep"][i]["recall"] for b in builds]
        band.append({
            "ef_search": ef,
            "recall_min": round(min(rs), 6),
            "recall_max": round(max(rs), 6),
            "recall_mean": round(float(np.mean(rs)), 6),
            "spread": round(max(rs) - min(rs), 6),
        })
    return band


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100_000, help="base vectors (prefix of SIFT1M)")
    ap.add_argument("--threads", type=int, default=32, help="indexThreadQty for the build")
    ap.add_argument("--out", help="output JSON (default bench/results/nmslib_<n>.json)")
    args = ap.parse_args()

    # Compute and cache ground truth once, before the build processes start.
    base, queries = common.load_sift(args.n)
    common.ground_truth(base, queries)
    del base, queries

    ctx = mp.get_context("spawn")
    builds = []
    for seed in SEEDS:
        with ctx.Pool(1) as pool:
            builds.append(pool.apply(run_build, (args.n, seed, args.threads)))

    label = f"{args.n // 1000}k" if args.n < 1_000_000 else f"{args.n // 1_000_000}M"
    out = args.out or common.RESULTS_DIR / f"nmslib_{label}.json"
    result = {
        "library": "nmslib",
        "library_version": version("nmslib"),
        "dataset": "sift-128-euclidean",
        "n_base": args.n,
        "n_queries": 10_000,
        "k": common.K,
        "recall_metric": "distance-based recall@10 (ann-benchmarks style); "
                         "recall_ids is plain id overlap",
        "query_threads": 1,
        "params": {"M": common.M, "efConstruction": common.EF_CONSTRUCTION, "post": POST,
                   "indexThreadQty": args.threads},
        "machine": common.machine_info(),
        "notes": [
            "nmslib 2.1.2 PyPI wheel is compiled for baseline SSE2 only "
            "(no SSE4/AVX/AVX2); QPS is lower than a native build. Recall is unaffected.",
        ],
        "builds": builds,
        "noise_band": noise_band(builds),
    }
    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
