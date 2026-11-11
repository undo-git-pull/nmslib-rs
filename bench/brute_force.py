"""Exact search on SIFT or GloVe: nmslib_rs.BruteForceIndex against NMSLIB's
brute_force method and a float64 numpy scan.

Checks that both libraries return the exact neighbors (recall@10 against the
float64 ground truth) and times queries on one thread and on all cores.
Exact search is slow, so this uses a subset of the queries by default.

    python bench/brute_force.py --n 100000
    python bench/brute_force.py --dataset glove --n 100000 --queries 200
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

import common


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(common.DATASETS), default="sift")
    ap.add_argument("--n", type=int, default=100_000,
                    help="base vectors: a prefix of the dataset, or 0 for all of it")
    ap.add_argument("--queries", type=int, default=1000, help="number of queries to run")
    ap.add_argument("--skip-nmslib", action="store_true", help="do not run NMSLIB")
    args = ap.parse_args()
    ds = common.DATASETS[args.dataset]
    if args.n == 0:
        args.n = ds.full_n
    k = common.K

    base, queries = common.load(ds, args.n)
    queries = np.ascontiguousarray(queries[: args.queries])
    nq = len(queries)
    gt, numpy_s = timed(lambda: common.brute_force_knn(base, queries, k, ds.metric))

    import nmslib_rs

    rows = [("numpy float64, all cores", numpy_s, common.recall(base, queries, gt, gt, ds.metric))]

    index = nmslib_rs.BruteForceIndex(base.shape[1], ds.metric)
    index.add_items(base)
    for threads, label in ((1, "1 thread"), (0, f"{os.cpu_count()} threads")):
        (labels, _), secs = timed(lambda: index.knn_query(queries, k=k, num_threads=threads))
        rows.append((f"nmslib_rs, {label}", secs, common.recall(base, queries, gt, labels, ds.metric)))

    if not args.skip_nmslib:
        import nmslib

        ref = nmslib.init(method="brute_force", space=ds.nmslib_space)
        ref.addDataPointBatch(base)
        ref.createIndex()
        for threads, label in ((1, "1 thread"), (os.cpu_count(), f"{os.cpu_count()} threads")):
            res, secs = timed(lambda: ref.knnQueryBatch(queries, k=k, num_threads=threads))
            found = np.array([ids for ids, _ in res], dtype=np.int64)
            rows.append((f"NMSLIB brute_force, {label}", secs,
                         common.recall(base, queries, gt, found, ds.metric)))

    print(f"{ds.name}, n={args.n}, {nq} queries, k={k}, SIMD {nmslib_rs.simd_level()}")
    print(f"{'method':<34}{'QPS':>10}{'recall':>9}{'recall_ids':>12}")
    for name, secs, r in rows:
        print(f"{name:<34}{nq / secs:>10,.0f}{r['recall']:>9.4f}{r['recall_ids']:>12.4f}")


if __name__ == "__main__":
    main()
