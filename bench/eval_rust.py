"""Phase 3 evaluation: rhnsw (Rust, single-threaded build) vs the NMSLIB baseline.

Exports the SIFT prefix as raw f32, runs examples/eval_sift.rs for several
insertion orders, scores recall with common.recall (the same metric as the
baseline) and applies the parity rule from PLAN.md phase 2.

    python bench/eval_rust.py --n 100000 --builds 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import common

ROOT = common.BENCH_DIR.parent


def write_matrix(path: Path, a: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(np.array(a.shape, "<u4").tobytes())
        f.write(np.ascontiguousarray(a, "<f4").tobytes())


def label(n: int) -> str:
    return f"{n // 1000}k" if n < 1_000_000 else f"{n // 1_000_000}M"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--builds", type=int, default=3, help="insertion-order seeds 1..N, run in parallel")
    args = ap.parse_args()

    base, queries = common.load_sift(args.n)
    gt = common.ground_truth(base, queries)
    work = common.DATA_DIR / f"rust_eval_{label(args.n)}"
    work.mkdir(exist_ok=True)
    base_path, query_path = work / "base.f32", work / "queries.f32"
    write_matrix(base_path, base)
    write_matrix(query_path, queries)

    subprocess.run(["cargo", "build", "--release", "--example", "eval_sift"], cwd=ROOT, check=True)
    exe = ROOT / "target/release/examples/eval_sift"
    seeds = list(range(1, args.builds + 1))
    with ThreadPoolExecutor(len(seeds)) as pool:
        procs = list(pool.map(
            lambda s: subprocess.run([exe, base_path, query_path, work, str(s)], check=True), seeds))
    assert all(p.returncode == 0 for p in procs)

    builds = []
    for seed in seeds:
        summary = json.loads((work / f"summary_seed{seed}.json").read_text())
        for row in summary["sweep"]:
            raw = np.fromfile(work / f"ids_seed{seed}_ef{row['ef_search']}.i32", "<i4")
            row.update(common.recall(base, queries, gt, raw.reshape(len(queries), common.K).astype(np.int64)))
        builds.append(summary)

    baseline = json.loads((common.RESULTS_DIR / f"nmslib_{label(args.n)}.json").read_text())
    nmslib_band = {b["ef_search"]: b for b in baseline["noise_band"]}
    parity = []
    for i, ef in enumerate(common.EF_SEARCH):
        rs = [b["sweep"][i]["recall"] for b in builds]
        ref = nmslib_band[ef]
        mean = float(np.mean(rs))
        worst = max(abs(r - ref["recall_mean"]) for r in rs)
        parity.append({
            "ef_search": ef,
            "rhnsw_recall_mean": round(mean, 6),
            "rhnsw_recall_min": round(min(rs), 6),
            "rhnsw_recall_max": round(max(rs), 6),
            "nmslib_recall_mean": ref["recall_mean"],
            "diff_mean": round(mean - ref["recall_mean"], 6),
            "worst_abs_diff": round(worst, 6),
            "tolerance": ref["tolerance"],
            "pass": worst <= ref["tolerance"],
        })

    result = {
        "library": "rhnsw",
        "dataset": "sift-128-euclidean",
        "n_base": args.n,
        "n_queries": len(queries),
        "k": common.K,
        "recall_metric": baseline["recall_metric"],
        "query_threads": 1,
        "machine": common.machine_info(),
        "builds": builds,
        "parity_vs_nmslib": parity,
        "all_pass": all(p["pass"] for p in parity),
    }
    out = common.RESULTS_DIR / f"rhnsw_{label(args.n)}_phase3.json"
    out.write_text(json.dumps(result, indent=2))

    nm_qps = {r["ef_search"]: r["qps"] for r in baseline["builds"][0]["sweep"]}
    print(f"\nrhnsw {label(args.n)}: {len(builds)} builds, build time "
          f"{[b['build_time_s'] for b in builds]} s (1 thread)")
    print(f"{'ef':>4}  {'rhnsw':>8} {'nmslib':>8} {'diff':>8} {'worst':>8} {'tol':>6}  "
          f"{'pass':4}  {'qps rh':>8} {'qps nm':>8}")
    for p, i in zip(parity, range(len(parity))):
        print(f"{p['ef_search']:>4}  {p['rhnsw_recall_mean']:8.5f} {p['nmslib_recall_mean']:8.5f} "
              f"{p['diff_mean']:+8.5f} {p['worst_abs_diff']:8.5f} {p['tolerance']:6.3f}  "
              f"{'yes' if p['pass'] else 'NO':4}  {builds[0]['sweep'][i]['qps']:8.0f} "
              f"{nm_qps[p['ef_search']]:8.0f}")
    print(f"wrote {out}", file=sys.stderr)
    sys.exit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
