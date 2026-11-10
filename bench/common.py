"""Dataset loading, ground truth and recall, shared by every benchmark script.

Datasets: SIFT1M (L2) and GloVe-100 (angular, i.e. cosine distance), both
from ann-benchmarks. Recall is distance-based, as in ann-benchmarks: a
returned neighbor counts if its exact distance to the query is no larger than
the true k-th nearest distance. SIFT contains duplicate vectors, so id-based
recall would penalize returning a different copy of a tied vector. Id-based
recall is reported too, for reference.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
DATA_DIR = BENCH_DIR / "data"
RESULTS_DIR = BENCH_DIR / "results"


@dataclass(frozen=True)
class Dataset:
    name: str
    path: Path
    metric: str  # "l2" or "cosine"
    nmslib_space: str
    full_n: int
    # Prefix of result file names; empty for SIFT, which came first.
    label_prefix: str


DATASETS = {
    "sift": Dataset("sift-128-euclidean", DATA_DIR / "sift-128-euclidean.hdf5", "l2", "l2",
                    1_000_000, ""),
    "glove": Dataset("glove-100-angular", DATA_DIR / "glove-100-angular.hdf5", "cosine",
                     "cosinesimil", 1_183_514, "glove_"),
}
SIFT = DATASETS["sift"]

K = 10
EF_SEARCH = [10, 20, 40, 80, 160, 320, 640]
M = 16
EF_CONSTRUCTION = 200
# Relative slack when comparing a returned distance to the k-th true distance,
# so float rounding on exact ties does not count as a miss.
TIE_RTOL = 1e-6
# Minimum half-width of the parity band, in recall (0.5 percentage points).
MIN_TOLERANCE = 0.005


def load(ds: Dataset, n: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Base vectors (first `n`, or all) and the 10k queries, as float32."""
    with h5py.File(ds.path, "r") as f:
        train = f["train"][:n] if n else f["train"][:]
        test = f["test"][:]
    return np.ascontiguousarray(train, np.float32), np.ascontiguousarray(test, np.float32)


def load_sift(n: int | None) -> tuple[np.ndarray, np.ndarray]:
    return load(SIFT, n)


def ground_truth(base: np.ndarray, queries: np.ndarray, k: int = K, ds: Dataset = SIFT) -> np.ndarray:
    """Exact k-NN ids: the shipped neighbors for the full set, brute force in
    float64 for a prefix. Cached in bench/data."""
    stem = "sift" if ds is SIFT else ds.name
    cache = DATA_DIR / f"gt_{stem}_{len(base)}_k{k}.npy"
    if cache.exists():
        return np.load(cache)
    if len(base) == ds.full_n:
        with h5py.File(ds.path, "r") as f:
            gt = f["neighbors"][:, :k].astype(np.int64)
    else:
        gt = brute_force_knn(base, queries, k, ds.metric)
    np.save(cache, gt)
    return gt


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def brute_force_knn(base: np.ndarray, queries: np.ndarray, k: int = K,
                    metric: str = "l2") -> np.ndarray:
    """Exact k-NN ids, nearest first, computed in float64."""
    b = base.astype(np.float64)
    if metric == "cosine":
        b = _unit(b)
    b_norm = (b * b).sum(1)
    gt = np.empty((len(queries), k), np.int64)
    for s in range(0, len(queries), 500):
        q = queries[s : s + 500].astype(np.float64)
        if metric == "cosine":
            d = -(_unit(q) @ b.T)
        else:
            d = b_norm[None, :] - 2.0 * q @ b.T
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.take_along_axis(d, part, 1).argsort(1)
        gt[s : s + 500] = np.take_along_axis(part, order, 1)
    return gt


def dists(base: np.ndarray, queries: np.ndarray, ids: np.ndarray, metric: str = "l2") -> np.ndarray:
    """Exact distances in float64 (squared L2, or 1 - cosine); -1 ids give +inf."""
    safe = np.where(ids < 0, 0, ids)
    b = base[safe].astype(np.float64)
    q = queries[:, None, :].astype(np.float64)
    if metric == "cosine":
        d = 1.0 - (_unit(b) * _unit(q)).sum(-1)
    else:
        d = ((b - q) ** 2).sum(-1)
    d[ids < 0] = np.inf
    return d


def sq_dists(base: np.ndarray, queries: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return dists(base, queries, ids, "l2")


def recall(base: np.ndarray, queries: np.ndarray, gt: np.ndarray, found: np.ndarray,
           metric: str = "l2") -> dict:
    """Distance-based and id-based recall@k of `found` (shape nq x k)."""
    k = gt.shape[1]
    kth = dists(base, queries, gt, metric).max(1)
    got = dists(base, queries, found, metric)
    hits = (got <= kth[:, None] * (1 + TIE_RTOL)).sum(1)
    id_hits = [len(set(f) & set(g)) for f, g in zip(found.tolist(), gt.tolist())]
    return {"recall": round(float(hits.mean() / k), 6),
            "recall_ids": round(float(np.mean(id_hits) / k), 6)}


def size_label(n: int) -> str:
    if n < 1_000_000:
        return f"{n // 1000}k"
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}M"
    return f"{n / 1_000_000:.2f}M"


def result_path(library: str, ds: Dataset, n: int) -> Path:
    return RESULTS_DIR / f"{library}_{ds.label_prefix}{size_label(n)}.json"


def noise_band(builds: list[dict]) -> list[dict]:
    """Per-efSearch recall spread across builds, with the parity tolerance:
    the larger of the band's half-width around the mean and MIN_TOLERANCE."""
    band = []
    for i, ef in enumerate(EF_SEARCH):
        rs = [b["sweep"][i]["recall"] for b in builds]
        mean = float(np.mean(rs))
        band.append({
            "ef_search": ef,
            "recall_min": round(min(rs), 6),
            "recall_max": round(max(rs), 6),
            "recall_mean": round(mean, 6),
            "recall_std": round(float(np.std(rs, ddof=1)), 6) if len(rs) > 1 else 0.0,
            "spread": round(max(rs) - min(rs), 6),
            "tolerance": round(max(max(rs) - mean, mean - min(rs), MIN_TOLERANCE), 6),
        })
    return band


def parity(reference: dict, builds: list[dict]) -> list[dict]:
    """PLAN.md parity rule: every build's recall is within the reference
    (NMSLIB) tolerance of the reference mean, at each efSearch."""
    ref_band = {b["ef_search"]: b for b in reference["noise_band"]}
    rows = []
    for i, ef in enumerate(EF_SEARCH):
        rs = [b["sweep"][i]["recall"] for b in builds]
        ref = ref_band[ef]
        worst = max(abs(r - ref["recall_mean"]) for r in rs)
        rows.append({
            "ef_search": ef,
            "recall_mean": round(float(np.mean(rs)), 6),
            "reference_recall_mean": ref["recall_mean"],
            "diff_mean": round(float(np.mean(rs)) - ref["recall_mean"], 6),
            "worst_abs_diff": round(worst, 6),
            "tolerance": ref["tolerance"],
            "pass": bool(worst <= ref["tolerance"]),
        })
    return rows


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
