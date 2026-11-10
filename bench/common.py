"""Dataset loading, ground truth and recall, shared by every benchmark script.

Recall is distance-based, as in ann-benchmarks: a returned neighbor counts
if its exact distance to the query is no larger than the true k-th nearest
distance. SIFT contains duplicate vectors, so id-based recall would penalize
returning a different copy of a tied vector. Id-based recall is reported too,
for reference.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"
SIFT_PATH = DATA_DIR / "sift-128-euclidean.hdf5"

K = 10
EF_SEARCH = [10, 20, 40, 80, 160, 320, 640]
M = 16
EF_CONSTRUCTION = 200
# Relative slack when comparing a returned distance to the k-th true distance,
# so float rounding on exact ties does not count as a miss.
TIE_RTOL = 1e-6


def load_sift(n: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Base vectors (first `n`, or all) and the 10k queries, as float32."""
    with h5py.File(SIFT_PATH, "r") as f:
        train = f["train"][:n] if n else f["train"][:]
        test = f["test"][:]
    return np.ascontiguousarray(train, np.float32), np.ascontiguousarray(test, np.float32)


def ground_truth(base: np.ndarray, queries: np.ndarray, k: int = K) -> np.ndarray:
    """Exact k-NN ids by brute force, in float64. Cached in bench/data."""
    cache = DATA_DIR / f"gt_sift_{len(base)}_k{k}.npy"
    if cache.exists():
        return np.load(cache)
    if len(base) == 1_000_000:
        with h5py.File(SIFT_PATH, "r") as f:
            gt = f["neighbors"][:, :k].astype(np.int64)
    else:
        gt = brute_force_knn(base, queries, k)
    np.save(cache, gt)
    return gt


def brute_force_knn(base: np.ndarray, queries: np.ndarray, k: int = K) -> np.ndarray:
    """Exact k-NN ids, nearest first, computed in float64."""
    b = base.astype(np.float64)
    b_norm = (b * b).sum(1)
    gt = np.empty((len(queries), k), np.int64)
    for s in range(0, len(queries), 500):
        q = queries[s : s + 500].astype(np.float64)
        d = b_norm[None, :] - 2.0 * q @ b.T
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.take_along_axis(d, part, 1).argsort(1)
        gt[s : s + 500] = np.take_along_axis(part, order, 1)
    return gt


def sq_dists(base: np.ndarray, queries: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Exact squared L2 distances in float64; -1 ids give +inf."""
    safe = np.where(ids < 0, 0, ids)
    diff = base[safe].astype(np.float64) - queries[:, None, :].astype(np.float64)
    d = (diff * diff).sum(-1)
    d[ids < 0] = np.inf
    return d


def recall(base: np.ndarray, queries: np.ndarray, gt: np.ndarray, found: np.ndarray) -> dict:
    """Distance-based and id-based recall@k of `found` (shape nq x k)."""
    k = gt.shape[1]
    kth = sq_dists(base, queries, gt).max(1)
    got = sq_dists(base, queries, found)
    hits = (got <= kth[:, None] * (1 + TIE_RTOL)).sum(1)
    id_hits = [len(set(f) & set(g)) for f, g in zip(found.tolist(), gt.tolist())]
    return {"recall": round(float(hits.mean() / k), 6),
            "recall_ids": round(float(np.mean(id_hits) / k), 6)}


def rss_mb() -> dict:
    """Current and peak resident set size of this process, in MB."""
    out = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            out[line.split(":")[0]] = int(line.split()[1]) / 1024
    return {"rss_mb": round(out["VmRSS"], 1), "peak_rss_mb": round(out["VmHWM"], 1)}


def reset_peak_rss() -> None:
    """Reset VmHWM to the current RSS (Linux >= 4.0)."""
    Path("/proc/self/clear_refs").write_text("5")


def machine_info() -> dict:
    cpu = next(
        (l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
         if l.startswith("model name")),
        platform.processor(),
    )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=BENCH_DIR, text=True
        ).strip()
    except Exception:
        commit = None
    return {
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": commit,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
