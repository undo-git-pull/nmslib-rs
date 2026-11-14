"""FAISS's wiki HNSW benchmark on SIFT1M, reproduced for FAISS and nmslib_rs.

The FAISS wiki page "Indexing 1M vectors" reports IndexHNSWFlat on SIFT1M
from benchs/bench_hnsw.py: M=32, efConstruction=40, efSearch 16..256, the
10k queries searched as one batch on 20 threads, timed around the search
call only, scored by R@1 (the top result is the true nearest neighbor, by id).
This script runs that setup for FAISS and for nmslib_rs with the same M,
ef_construction and threads, with nmslib_rs's post pass both off (FAISS has
no equivalent) and on. Each build runs in a fresh process.

    python bench/faiss_wiki.py
    python bench/faiss_wiki.py --builds 1 --n 100000
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


M = 32
EF_CONSTRUCTION = 40
EF_SEARCH = [16, 32, 64, 128, 256]
K = 10
# The wiki's HNSW Flat table: efSearch -> (ms per query, R@1), 20 threads.
WIKI = {16: (0.011, 0.8740), 32: (0.020, 0.9492), 64: (0.033, 0.9779),
        128: (0.059, 0.9887), 256: (0.104, 0.9920)}
CONFIGS = {
    "faiss": "faiss IndexHNSWFlat",
    "nmslib_rs": "nmslib_rs, post=False",
    "nmslib_rs_post": "nmslib_rs, post=True",
}


def make_index(config: str, dim: int, seed: int, threads: int):
    """An empty index, and functions to fill it, set efSearch and search it."""
    if config == "faiss":
        import faiss

        faiss.omp_set_num_threads(threads)
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = EF_CONSTRUCTION

        def set_ef(ef: int) -> None:
            index.hnsw.efSearch = ef

        return index.add, set_ef, lambda q: index.search(q, K)[1]

    import nmslib_rs

    index = nmslib_rs.Index(dim=dim, space="l2", M=M, ef_construction=EF_CONSTRUCTION,
                            seed=seed, post=config == "nmslib_rs_post")
    return (lambda x: index.add_items(x, num_threads=threads), index.set_ef,
            lambda q: index.knn_query(q, k=K, num_threads=threads)[0])


def run_build(config: str, n: int, seed: int, threads: int, repeats: int) -> dict:
    base, queries = common.load(common.SIFT, n)
    gt = common.ground_truth(base, queries)
    order = np.random.default_rng(seed).permutation(len(base))
    shuffled = np.ascontiguousarray(base[order])

    add, set_ef, search = make_index(config, base.shape[1], seed, threads)
    before = common.rss_mb()
    t0 = time.perf_counter()
    add(shuffled)
    build_time = time.perf_counter() - t0
    after_build = common.rss_mb()
    del shuffled
    print(f"[{config} seed {seed}] build {build_time:.1f}s", file=sys.stderr, flush=True)

    sweep = []
    for ef in EF_SEARCH:
        set_ef(ef)
        search(queries)  # warm-up
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            labels = search(queries)
            times.append(time.perf_counter() - t0)
        elapsed = min(times)
        found = np.where(labels >= 0, order[np.maximum(labels, 0)], -1)
        r1 = common.recall(base, queries, gt[:, :1], found[:, :1])
        r10 = common.recall(base, queries, gt, found)
        sweep.append({
            "ef_search": ef,
            "r_at_1": r1["recall_ids"],
            "r_at_1_dist": r1["recall"],
            "recall_at_10": r10["recall"],
            "ms_per_query": round(elapsed / len(queries) * 1e3, 5),
            "qps": round(len(queries) / elapsed, 1),
            "query_times_s": [round(t, 4) for t in times],
        })
        print(f"[{config} seed {seed}] ef {ef:4d} R@1 {r1['recall_ids']:.4f} "
              f"{elapsed / len(queries) * 1e3:.4f} ms/q", file=sys.stderr, flush=True)

    return {
        "insertion_order_seed": seed,
        "build_time_s": round(build_time, 2),
        "index_mb": round(after_build["rss_mb"] - before["rss_mb"], 1),
        "sweep": sweep,
    }


def summarize(builds: list[dict]) -> list[dict]:
    rows = []
    for i, ef in enumerate(EF_SEARCH):
        pts = [b["sweep"][i] for b in builds]
        rows.append({
            "ef_search": ef,
            "r_at_1_mean": round(float(np.mean([p["r_at_1"] for p in pts])), 4),
            "r_at_1_min": min(p["r_at_1"] for p in pts),
            "r_at_1_max": max(p["r_at_1"] for p in pts),
            "recall_at_10_mean": round(float(np.mean([p["recall_at_10"] for p in pts])), 4),
            "ms_per_query_median": round(float(np.median([p["ms_per_query"] for p in pts])), 5),
            "qps_median": round(float(np.median([p["qps"] for p in pts])), 1),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=0, help="base vectors (0 = all of SIFT1M)")
    ap.add_argument("--builds", type=int, default=3,
                    help="independent builds per library (insertion-order seeds 1..N)")
    ap.add_argument("--threads", type=int, default=20,
                    help="build and query threads (the wiki used 20)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed searches per efSearch; the fastest is kept")
    ap.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--out", help="output JSON (default bench/results/faiss_wiki_<n>.json)")
    args = ap.parse_args()
    if args.n == 0:
        args.n = common.SIFT.full_n

    base, queries = common.load(common.SIFT, args.n)
    common.ground_truth(base, queries)
    del base, queries

    import faiss
    import nmslib_rs

    ctx = mp.get_context("spawn")
    results = {}
    for config in args.configs:
        builds = []
        for seed in range(1, args.builds + 1):
            with ctx.Pool(1) as pool:
                builds.append(pool.apply(run_build, (config, args.n, seed, args.threads,
                                                     args.repeats)))
        results[config] = {"label": CONFIGS[config], "builds": builds,
                           "summary": summarize(builds)}

    result = {
        "benchmark": "FAISS wiki 'Indexing 1M vectors', HNSW Flat (benchs/bench_hnsw.py)",
        "dataset": common.SIFT.name,
        "n_base": args.n,
        "n_queries": 10_000,
        "k": K,
        "recall_metric": "R@1 by id, as bench_hnsw.py; also distance-based R@1 and recall@10",
        "params": {"M": M, "efConstruction": EF_CONSTRUCTION, "threads": args.threads,
                   "timing": f"one batch of all queries, fastest of {args.repeats}"},
        "versions": {"faiss": faiss.__version__, "faiss_compile": faiss.get_compile_options(),
                     "nmslib_rs": version("nmslib-rs"), "nmslib_rs_simd": nmslib_rs.simd_level()},
        "machine": common.machine_info(),
        "wiki_reference": {str(ef): {"ms_per_query": ms, "r_at_1": r} for ef, (ms, r) in WIKI.items()},
        "results": results,
    }
    out = args.out or common.RESULTS_DIR / f"faiss_wiki_{common.size_label(args.n)}.json"
    common.RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSIFT {common.size_label(args.n)}, M={M}, efConstruction={EF_CONSTRUCTION}, "
          f"{args.threads} threads, {args.builds} builds each (R@1 mean, median ms/query)")
    head = f"{'efSearch':>8} | {'wiki R@1':>8} {'wiki ms':>8}"
    for c in args.configs:
        head += f" | {c + ' R@1':>19} {'ms':>7}"
    print(head)
    for i, ef in enumerate(EF_SEARCH):
        line = f"{ef:>8} | {WIKI[ef][1]:>8.4f} {WIKI[ef][0]:>8.3f}"
        for c in args.configs:
            s = results[c]["summary"][i]
            line += f" | {s['r_at_1_mean']:>19.4f} {s['ms_per_query_median']:>7.4f}"
        print(line)
    for c in args.configs:
        bt = [b["build_time_s"] for b in results[c]["builds"]]
        mb = [b["index_mb"] for b in results[c]["builds"]]
        print(f"{c}: build {np.median(bt):.1f}s median, index {np.median(mb):.0f} MB")
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
