"""Print NMSLIB and rhnsw recall@10 curves side by side, with QPS and parity.

    python bench/compare.py                 # 100k
    python bench/compare.py --size 1M

Exits non-zero if any efSearch fails the parity rule (PLAN.md phase 2).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import common


def mean_of(result: dict, key: str) -> dict[int, float]:
    return {ef: float(np.mean([b["sweep"][i][key] for b in result["builds"]]))
            for i, ef in enumerate(common.EF_SEARCH)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="100k", help="result label, e.g. 100k or 1M")
    ap.add_argument("--reference", help="NMSLIB JSON (default results/nmslib_<size>.json)")
    ap.add_argument("--candidate", help="rhnsw JSON (default results/rhnsw_<size>.json)")
    args = ap.parse_args()

    ref = json.loads(open(args.reference or common.RESULTS_DIR / f"nmslib_{args.size}.json").read())
    cand = json.loads(open(args.candidate or common.RESULTS_DIR / f"rhnsw_{args.size}.json").read())
    rows = common.parity(ref, cand["builds"])
    ref_qps, cand_qps = mean_of(ref, "qps"), mean_of(cand, "qps")

    def builds_line(name: str, r: dict) -> str:
        t = np.mean([b["build_time_s"] for b in r["builds"]])
        mem = np.mean([b["memory"]["index_mb"] for b in r["builds"]])
        peak = np.mean([b["memory"]["peak_rss_build_mb"] for b in r["builds"]])
        threads = r["params"].get("indexThreadQty", r["params"].get("build_threads"))
        return (f"{name:<7} {len(r['builds']):>2} builds  build {t:7.1f} s on {threads} thread(s)"
                f"  index memory {mem:7.1f} MB  peak RSS {peak:7.1f} MB")

    print(f"SIFT {args.size}, recall@10 (mean over builds), queries on 1 thread")
    print(builds_line("nmslib", ref))
    print(builds_line("rhnsw", cand))
    print()
    print(f"{'efSearch':>8} | {'nmslib':>8} {'rhnsw':>8} {'diff':>8} {'worst':>8} {'tol':>6} "
          f"{'parity':>6} | {'qps nmslib':>10} {'qps rhnsw':>10} {'ratio':>6}")
    print("-" * 97)
    for r in rows:
        ef = r["ef_search"]
        print(f"{ef:>8} | {r['reference_recall_mean']:8.5f} {r['recall_mean']:8.5f} "
              f"{r['diff_mean']:+8.5f} {r['worst_abs_diff']:8.5f} {r['tolerance']:6.3f} "
              f"{'pass' if r['pass'] else 'FAIL':>6} | {ref_qps[ef]:10.0f} {cand_qps[ef]:10.0f} "
              f"{cand_qps[ef] / ref_qps[ef]:5.2f}x")
    ok = all(r["pass"] for r in rows)
    print(f"\nparity: {'PASS' if ok else 'FAIL'} at {sum(r['pass'] for r in rows)}/{len(rows)} efSearch values")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
