"""FAISS on the GPU against nmslib_rs on the CPU, on SIFT1M.

The faiss-gpu wheel has no GPU HNSW (and no CAGRA, which needs the conda
faiss-gpu-cuvs build), so the GPU side is GpuIndexFlatL2 (exact) and
GpuIndexIVFFlat with 4096 lists, the IVF setup from the FAISS wiki page
"Indexing 1M vectors". Scored like bench/faiss_wiki.py: R@1 by id, queries
from host memory, the 10k queries as one batch, fastest of several runs.

The CPU curves (FAISS HNSW and nmslib_rs, M=32, efConstruction=40, 20
threads) are read from faiss_wiki.py's results. A second sweep varies the
query batch size for GPU IVF and nmslib_rs at about the same R@1.

Needs a venv with faiss-gpu and nmslib_rs (faiss-gpu conflicts with faiss-cpu):

    .venv-faiss-gpu/bin/python bench/faiss_gpu.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version

import numpy as np

import common


K = 10
NLIST = 4096
NPROBE = [1, 4, 16, 32, 64, 128, 256, 512]
TRAIN_N = 200_000
BATCH_SIZES = [1, 10, 100, 1000, 10_000]
# For the batch sweep: about the same R@1 (~0.98) on both sides.
NMSLIB_RS_EF = 64
THREADS = 20
R1_TARGETS = [0.90, 0.95, 0.98, 0.99]


def timed(search, queries: np.ndarray, repeats: int) -> tuple[np.ndarray, float]:
    """Labels and the fastest of `repeats` searches, after one warm-up."""
    labels = search(queries)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        labels = search(queries)
        best = min(best, time.perf_counter() - t0)
    return labels, best


def point(base, queries, gt, labels, elapsed, **params) -> dict:
    r1 = common.recall(base, queries, gt[:, :1], labels[:, :1])
    r10 = common.recall(base, queries, gt, labels)
    return {**params, "r_at_1": r1["recall_ids"], "r_at_1_dist": r1["recall"],
            "recall_at_10": r10["recall"],
            "ms_per_query": round(elapsed / len(queries) * 1e3, 5),
            "qps": round(len(queries) / elapsed, 1)}


def batched(search, batch: int):
    def run(queries: np.ndarray) -> np.ndarray:
        return np.vstack([search(queries[s : s + batch]) for s in range(0, len(queries), batch)])
    return run


def fastest_at(points: list[dict], target: float) -> dict | None:
    ok = [p for p in points if p["r_at_1"] >= target]
    return min(ok, key=lambda p: p["ms_per_query"]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cpu-results", default=str(common.RESULTS_DIR / "faiss_wiki_1M.json"))
    ap.add_argument("--out", default=str(common.RESULTS_DIR / "faiss_gpu_1M.json"))
    args = ap.parse_args()

    import faiss
    import nmslib_rs

    base, queries = common.load(common.SIFT, None)
    gt = common.ground_truth(base, queries)
    dim = base.shape[1]
    res = faiss.StandardGpuResources()
    gpu_name = faiss.get_num_gpus() and __import__("subprocess").check_output(
        ["nvidia-smi", f"--id={args.device}", "--query-gpu=name", "--format=csv,noheader"],
        text=True).strip()

    # Exact search.
    flat_cfg = faiss.GpuIndexFlatConfig()
    flat_cfg.device = args.device
    flat = faiss.GpuIndexFlatL2(res, dim, flat_cfg)
    t0 = time.perf_counter()
    flat.add(base)
    flat_add = time.perf_counter() - t0
    labels, elapsed = timed(lambda q: flat.search(q, K)[1], queries, args.repeats)
    flat_point = point(base, queries, gt, labels, elapsed)
    print(f"[gpu flat] R@1 {flat_point['r_at_1']:.4f} {flat_point['ms_per_query']:.5f} ms/q",
          file=sys.stderr, flush=True)
    del flat

    # IVF4096,Flat.
    ivf_cfg = faiss.GpuIndexIVFFlatConfig()
    ivf_cfg.device = args.device
    ivf = faiss.GpuIndexIVFFlat(res, dim, NLIST, faiss.METRIC_L2, ivf_cfg)
    train = base[np.random.default_rng(0).choice(len(base), TRAIN_N, replace=False)]
    t0 = time.perf_counter()
    ivf.train(train)
    ivf_train = time.perf_counter() - t0
    t0 = time.perf_counter()
    ivf.add(base)
    ivf_add = time.perf_counter() - t0
    ivf_sweep = []
    for nprobe in NPROBE:
        ivf.nprobe = nprobe
        labels, elapsed = timed(lambda q: ivf.search(q, K)[1], queries, args.repeats)
        ivf_sweep.append(point(base, queries, gt, labels, elapsed, nprobe=nprobe))
        p = ivf_sweep[-1]
        print(f"[gpu ivf] nprobe {nprobe:4d} R@1 {p['r_at_1']:.4f} {p['ms_per_query']:.5f} ms/q",
              file=sys.stderr, flush=True)

    # Batch-size sweep: GPU IVF at the smallest nprobe reaching nmslib_rs's R@1 at
    # NMSLIB_RS_EF, and nmslib_rs (post=True) at that ef.
    hnsw = nmslib_rs.Index(dim=dim, space="l2", M=32, ef_construction=40, post=True)
    hnsw.add_items(base, num_threads=THREADS)
    hnsw.set_ef(NMSLIB_RS_EF)
    hnsw_search = lambda q: hnsw.knn_query(q, k=K, num_threads=THREADS)[0]
    hnsw_r1 = point(base, queries, gt, hnsw_search(queries), 1.0)["r_at_1"]
    match = next((p for p in ivf_sweep if p["r_at_1"] >= hnsw_r1), ivf_sweep[-1])
    ivf.nprobe = match["nprobe"]
    batch_sweep = []
    for batch in BATCH_SIZES:
        # Small batches are slow; time a subset so the sweep stays short.
        q = queries[: max(1000, min(len(queries), batch * 100))]
        g = gt[: len(q)]
        reps = 1 if batch == 1 else args.repeats
        for name, search in [("faiss_gpu_ivf", lambda x: ivf.search(x, K)[1]),
                             ("nmslib_rs_post", hnsw_search)]:
            labels, elapsed = timed(batched(search, batch), q, reps)
            batch_sweep.append(point(base, q, g, labels, elapsed, library=name, batch=batch,
                                     n_queries=len(q)))
            p = batch_sweep[-1]
            print(f"[batch {batch:5d}] {name:15s} R@1 {p['r_at_1']:.4f} {p['qps']:10.1f} qps",
                  file=sys.stderr, flush=True)

    cpu = json.loads(open(args.cpu_results).read())
    curves = {
        "faiss_gpu_ivf4096": ivf_sweep,
        **{f"{c}_cpu{THREADS}": [{"ef_search": s["ef_search"], "r_at_1": s["r_at_1_mean"],
                                  "recall_at_10": s["recall_at_10_mean"],
                                  "ms_per_query": s["ms_per_query_median"], "qps": s["qps_median"]}
                                 for s in cpu["results"][c]["summary"]]
           for c in cpu["results"]},
    }
    matched = [{"r_at_1_target": t,
                **{name: fastest_at(pts, t) for name, pts in curves.items()}} for t in R1_TARGETS]

    result = {
        "dataset": common.SIFT.name,
        "n_base": len(base),
        "n_queries": len(queries),
        "k": K,
        "recall_metric": "R@1 by id, as bench_hnsw.py; also distance-based R@1 and recall@10",
        "timing": f"queries and results in host memory, fastest of {args.repeats} after warm-up",
        "gpu": gpu_name,
        "versions": {"faiss": faiss.__version__, "nmslib_rs": version("nmslib-rs")},
        "machine": common.machine_info(),
        "gpu_flat": {"add_s": round(flat_add, 2), **flat_point},
        "gpu_ivf": {"nlist": NLIST, "train_n": TRAIN_N, "train_s": round(ivf_train, 2),
                    "add_s": round(ivf_add, 2), "sweep": ivf_sweep},
        "cpu_results_from": args.cpu_results,
        "matched_recall": matched,
        "batch_sweep": {"gpu_ivf_nprobe": match["nprobe"], "nmslib_rs_ef": NMSLIB_RS_EF,
                        "nmslib_rs_threads": THREADS, "points": batch_sweep},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nGPU: {gpu_name}. Flat (exact): R@1 {flat_point['r_at_1']:.4f}, "
          f"{flat_point['ms_per_query']:.5f} ms/q, {flat_point['qps']:,.0f} qps")
    print("\nFastest config reaching each R@1 (ms/query; batch of 10k queries)")
    names = list(curves)
    print(f"{'R@1 >=':>7} | " + " | ".join(f"{n:>26}" for n in names))
    for row in matched:
        cells = []
        for n in names:
            p = row[n]
            knob = "" if p is None else (f"np={p['nprobe']}" if "nprobe" in p else f"ef={p['ef_search']}")
            cells.append(f"{'-':>26}" if p is None else f"{p['ms_per_query']:.5f} ({knob}, {p['r_at_1']:.4f})".rjust(26))
        print(f"{row['r_at_1_target']:>7.2f} | " + " | ".join(cells))
    print(f"\nBatch sweep: GPU IVF nprobe={match['nprobe']}, nmslib_rs ef={NMSLIB_RS_EF} "
          f"on {THREADS} threads (qps)")
    for batch in BATCH_SIZES:
        pts = {p["library"]: p for p in batch_sweep if p["batch"] == batch}
        print(f"{batch:>6} | gpu ivf {pts['faiss_gpu_ivf']['qps']:>10,.0f} "
              f"(R@1 {pts['faiss_gpu_ivf']['r_at_1']:.3f}) | nmslib_rs "
              f"{pts['nmslib_rs_post']['qps']:>10,.0f} (R@1 {pts['nmslib_rs_post']['r_at_1']:.3f})")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
