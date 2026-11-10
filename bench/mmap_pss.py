"""Phase 6 gate: memory of several processes sharing one memory-mapped index.

Starts N worker processes (spawn) in three groups and measures each worker's
RSS and PSS from /proc/<pid>/smaps_rollup while all workers of the group are
alive. PSS splits shared pages between the processes that map them, so the
group's summed PSS is its real footprint; RSS counts shared pages once per
process.

- baseline: imports nmslib_rs and NumPy and holds the queries, no index;
- mapped: Index.load(path, mmap=True), then every query at efSearch 640,
  which touches nearly every node of the index;
- copy: Index.load(path, mmap=False), the same queries (for contrast).

Gate: summed PSS of the mapped group <= 1.1 * (file size + summed PSS of the
baseline group).

    python bench/mmap_pss.py --index bench/data/nmslib_rs_sift_1M.nmslib_rs
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import common

MB = 1024.0


def mem_kb(path: str) -> dict:
    """Whole-process RSS/PSS, and RSS/PSS of the index file's mapping."""
    out = {}
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        key = line.split(":")[0]
        if key in ("Rss", "Pss"):
            out[key.lower()] = int(line.split()[1])
    out["map_rss"] = out["map_pss"] = 0
    inside = False
    for line in Path("/proc/self/smaps").read_text().splitlines():
        if line[0] in "0123456789abcdef" and "-" in line.split()[0]:
            inside = line.rstrip().endswith(path)
        elif inside and line.startswith(("Rss:", "Pss:")):
            out["map_" + line.split(":")[0].lower()] += int(line.split()[1])
    return out


def worker(mode: str, path: str, queries, done, measured, results) -> None:
    import nmslib_rs

    if mode != "baseline":
        index = nmslib_rs.Index.load(path, mmap=(mode == "mapped"))
        index.set_ef(640)
        index.knn_query(queries, k=10, num_threads=1)
    done.wait()
    results.put(mem_kb(path))
    measured.wait()


def run_group(mode: str, n: int, path: str, queries) -> list[dict]:
    ctx = mp.get_context("spawn")
    done, measured, results = ctx.Barrier(n), ctx.Barrier(n), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(mode, path, queries, done, measured, results))
             for _ in range(n)]
    for p in procs:
        p.start()
    rows = [results.get() for _ in procs]
    for p in procs:
        p.join()
        assert p.exitcode == 0, f"{mode} worker failed"
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", required=True, help="saved index file")
    ap.add_argument("--dataset", choices=sorted(common.DATASETS), default="sift")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", help="JSON report (default results/mmap_pss_<file stem>.json)")
    args = ap.parse_args()
    path = str(Path(args.index).resolve())
    file_mb = Path(path).stat().st_size / 2**20
    _, queries = common.load(common.DATASETS[args.dataset], 1)

    groups = {}
    for mode in ("baseline", "mapped", "copy"):
        rows = run_group(mode, args.procs, path, queries)
        groups[mode] = {
            "per_process_rss_mb": [round(r["rss"] / MB, 1) for r in rows],
            "per_process_pss_mb": [round(r["pss"] / MB, 1) for r in rows],
            "per_process_map_rss_mb": [round(r["map_rss"] / MB, 1) for r in rows],
            "per_process_map_pss_mb": [round(r["map_pss"] / MB, 1) for r in rows],
            "sum_rss_mb": round(sum(r["rss"] for r in rows) / MB, 1),
            "sum_pss_mb": round(sum(r["pss"] for r in rows) / MB, 1),
        }
        g = groups[mode]
        print(f"{mode:8} x{args.procs}: summed RSS {g['sum_rss_mb']:8.1f} MB, summed PSS "
              f"{g['sum_pss_mb']:8.1f} MB; index mapping per process RSS "
              f"{max(g['per_process_map_rss_mb']):6.1f} MB, PSS {max(g['per_process_map_pss_mb']):6.1f} MB",
              file=sys.stderr)

    limit = 1.1 * (file_mb + groups["baseline"]["sum_pss_mb"])
    ok = groups["mapped"]["sum_pss_mb"] <= limit
    report = {
        "index_file": path,
        "file_mb": round(file_mb, 1),
        "processes": args.procs,
        "queries_per_process": len(queries),
        "ef_search": 640,
        "groups": groups,
        "gate": {
            "rule": "sum PSS (mapped) <= 1.1 * (file size + sum PSS (baseline))",
            "limit_mb": round(limit, 1),
            "mapped_sum_pss_mb": groups["mapped"]["sum_pss_mb"],
            "pass": ok,
        },
        "machine": common.machine_info(),
    }
    out = args.out or common.RESULTS_DIR / f"mmap_pss_{Path(path).stem}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"file {file_mb:.1f} MB; gate limit {limit:.1f} MB; mapped sum PSS "
          f"{groups['mapped']['sum_pss_mb']:.1f} MB -> {'PASS' if ok else 'FAIL'}", file=sys.stderr)
    print(f"wrote {out}", file=sys.stderr)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
