"""Phase 6 gate checks for saved indexes on a full dataset.

Builds an index, saves it, loads it back memory-mapped and as a private copy,
and checks that both return exactly the labels and distances of the
in-memory index for every query at every efSearch. Reports file size, save
and load times, and query latency on a cold mapped index (file evicted from
the page cache with posix_fadvise, so no root is needed).

    python bench/check_saved.py --dataset sift
    python bench/check_saved.py --dataset glove

Keeps the saved file in bench/data for mmap_pss.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

import common
import nmslib_rs


def evict(path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=sorted(common.DATASETS), default="sift")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", help="JSON report (default results/saved_<dataset><n>.json)")
    args = ap.parse_args()
    ds = common.DATASETS[args.dataset]
    base, queries = common.load(ds, None)
    label = f"{args.dataset}_{common.size_label(len(base))}"
    path = common.DATA_DIR / f"nmslib_rs_{label}.nmslib_rs"

    index = nmslib_rs.Index(dim=base.shape[1], space=ds.metric, M=common.M,
                        ef_construction=common.EF_CONSTRUCTION, seed=1)
    t0 = time.perf_counter()
    index.add_items(base, num_threads=args.threads)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    index.save(path)
    save_s = time.perf_counter() - t0
    size_mb = path.stat().st_size / 2**20
    print(f"{label}: build {build_s:.1f}s, save {save_s:.2f}s, file {size_mb:.1f} MB", file=sys.stderr)

    loads = {}
    for mmap in (True, False):
        evict(path)
        t0 = time.perf_counter()
        loaded = nmslib_rs.Index.load(path, mmap=mmap)
        loads[mmap] = (loaded, time.perf_counter() - t0)

    # Cold mapped index: evict the file, load, and time the first queries.
    evict(path)
    cold = nmslib_rs.Index.load(path, mmap=True)
    cold.set_ef(80)
    t0 = time.perf_counter()
    cold.knn_query(queries[:1], k=10, num_threads=1)
    first_query_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    cold.knn_query(queries[1:1001], k=10, num_threads=1)
    next_1000_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    cold.knn_query(queries[1:1001], k=10, num_threads=1)
    warm_1000_ms = (time.perf_counter() - t0) * 1e3
    del cold

    rows = []
    for ef in common.EF_SEARCH:
        index.set_ef(ef)
        ref = index.knn_query(queries, k=common.K, num_threads=0)
        row = {"ef_search": ef}
        for mmap, (loaded, _) in loads.items():
            loaded.set_ef(ef)
            got = loaded.knn_query(queries, k=common.K, num_threads=0)
            same = bool((got[0] == ref[0]).all() and (got[1] == ref[1]).all())
            row["mmap_identical" if mmap else "copy_identical"] = same
        rows.append(row)
        print(f"  ef {ef:4d}: mmap identical {row['mmap_identical']}, copy identical "
              f"{row['copy_identical']}", file=sys.stderr)

    ok = all(r["mmap_identical"] and r["copy_identical"] for r in rows)
    report = {
        "dataset": ds.name,
        "n_base": len(base),
        "file": str(path.relative_to(common.BENCH_DIR.parent)),
        "file_mb": round(size_mb, 1),
        "build_s": round(build_s, 2),
        "build_threads": args.threads,
        "save_s": round(save_s, 2),
        "load_mmap_ms": round(loads[True][1] * 1e3, 2),
        "load_copy_s": round(loads[False][1], 2),
        "load_note": "file evicted from the page cache before each load",
        "cold_mmap_queries": {
            "ef_search": 80,
            "first_query_ms": round(first_query_ms, 2),
            "next_1000_queries_ms": round(next_1000_ms, 1),
            "same_1000_queries_warm_ms": round(warm_1000_ms, 1),
        },
        "identical_results": rows,
        "all_identical": ok,
        "machine": common.machine_info(),
    }
    out = args.out or common.RESULTS_DIR / f"saved_{label}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: report[k] for k in ("file_mb", "save_s", "load_mmap_ms", "load_copy_s",
                                             "cold_mmap_queries", "all_identical")}, indent=2))
    print(f"wrote {out}", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
