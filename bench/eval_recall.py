"""rhnsw on SIFT through the Python binding: recall@10, QPS, build time, memory.

Mirrors nmslib_baseline.py: several builds from shuffled insertion orders,
each in a fresh process, the same efSearch sweep on one query thread, and the
same JSON layout. Also records the parity check against the NMSLIB baseline.

    python bench/eval_recall.py --n 100000
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np

import common


def run_build(n: int, seed: int) -> dict:
    import rhnsw

    base, queries = common.load_sift(n)
    gt = common.ground_truth(base, queries)
    order = np.random.default_rng(seed).permutation(len(base))
    shuffled = np.ascontiguousarray(base[order])

    index = rhnsw.Index(dim=base.shape[1], space="l2", M=common.M,
                        ef_construction=common.EF_CONSTRUCTION, seed=seed, post=True)
    common.reset_peak_rss()
    before = common.rss_mb()
    t0 = time.perf_counter()
    index.add_items(shuffled)
    build_time = time.perf_counter() - t0
    after_build = common.rss_mb()
    print(f"[seed {seed}] build {build_time:.1f}s", file=sys.stderr, flush=True)

    sweep = []
    for ef in common.EF_SEARCH:
        index.set_ef(ef)
        common.reset_peak_rss()
        t0 = time.perf_counter()
        labels, _ = index.knn_query(queries, k=common.K)
        elapsed = time.perf_counter() - t0
        found = np.where(labels >= 0, order[np.maximum(labels, 0)], -1)
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
            "index_mb": round(after_build["rss_mb"] - before["rss_mb"], 1),
        },
        "sweep": sweep,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100_000, help="base vectors (prefix of SIFT1M)")
    ap.add_argument("--builds", type=int, default=3,
                    help="independent builds (insertion-order seeds 1..N)")
    ap.add_argument("--out", help="output JSON (default bench/results/rhnsw_<n>.json)")
    args = ap.parse_args()

    base, queries = common.load_sift(args.n)
    common.ground_truth(base, queries)
    del base, queries

    import rhnsw

    ctx = mp.get_context("spawn")
    builds = []
    for seed in range(1, args.builds + 1):
        with ctx.Pool(1) as pool:
            builds.append(pool.apply(run_build, (args.n, seed)))

    label = common.size_label(args.n)
    reference = json.loads((common.RESULTS_DIR / f"nmslib_{label}.json").read_text())
    parity = common.parity(reference, builds)
    result = {
        "library": "rhnsw",
        "library_version": "0.1.0",
        "simd_level": rhnsw.simd_level(),
        "dataset": "sift-128-euclidean",
        "n_base": args.n,
        "n_queries": 10_000,
        "k": common.K,
        "recall_metric": reference["recall_metric"],
        "query_threads": 1,
        "params": {"M": common.M, "efConstruction": common.EF_CONSTRUCTION, "post": True,
                   "build_threads": 1},
        "machine": common.machine_info(),
        "notes": ["Single-threaded build (phase 4); parallel build arrives in phase 5."],
        "builds": builds,
        "noise_band": common.noise_band(builds),
        "parity_vs_nmslib": parity,
        "all_pass": all(p["pass"] for p in parity),
    }
    out = args.out or common.RESULTS_DIR / f"rhnsw_{label}.json"
    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
