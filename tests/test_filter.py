"""Filtered search and graph diagnostics. Run with: pytest tests/"""

import numpy as np
import pytest

import nmslib_rs


def random(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


@pytest.fixture(scope="module")
def built():
    data = random(5000, 16)
    index = nmslib_rs.Index(dim=16, M=16, ef_construction=100, seed=0)
    index.add_items(data)
    exact = nmslib_rs.BruteForceIndex(dim=16)
    exact.add_items(data)
    return data, index, exact


def test_all_true_filter_matches_unfiltered(built):
    _, index, _ = built
    queries = random(200, 16, 1)
    mask = np.ones(len(index), dtype=bool)
    for ef in (10, 80):
        index.set_ef(ef)
        a = index.knn_query(queries, k=10)
        b = index.knn_query(queries, k=10, filter=mask)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_all_false_filter_returns_nothing(built):
    _, index, exact = built
    mask = np.zeros(len(index), dtype=bool)
    for ix in (index, exact):
        labels, dists = ix.knn_query(random(5, 16, 2), k=4, filter=mask)
        assert (labels == -1).all() and np.isinf(dists).all()


@pytest.mark.parametrize("fraction", [0.5, 0.1, 0.01])
def test_filtered_recall(built, fraction):
    _, index, exact = built
    mask = np.random.default_rng(3).random(len(index)) < fraction
    queries = random(200, 16, 4)
    index.set_ef(50)
    labels, _ = index.knn_query(queries, k=10, filter=mask)
    truth, _ = exact.knn_query(queries, k=10, filter=mask)
    assert mask[labels[labels >= 0]].all()
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, truth)])
    assert recall > 0.98


def test_brute_force_filter_is_exact(built):
    data, _, exact = built
    mask = np.random.default_rng(5).random(len(data)) < 0.2
    queries = random(20, 16, 6)
    labels, _ = exact.knn_query(queries, k=5, filter=mask)
    ids = np.flatnonzero(mask)
    d = ((queries[:, None, :] - data[ids][None, :, :]) ** 2).sum(-1)
    want = ids[np.argsort(d, axis=1, kind="stable")[:, :5]]
    assert (labels == want).mean() > 0.99


def test_max_candidates_limits_results(built):
    _, index, _ = built
    mask = np.random.default_rng(7).random(len(index)) < 0.01
    queries = random(50, 16, 8)
    index.set_ef(50)
    full, _ = index.knn_query(queries, k=10, filter=mask)
    capped, dists = index.knn_query(queries, k=10, filter=mask, max_candidates=5)
    assert (full >= 0).all()
    assert (capped == -1).any()
    assert np.isinf(dists[capped == -1]).all()
    assert mask[capped[capped >= 0]].all()


def test_tombstone_filter_hides_deleted_labels(built):
    data, index, _ = built
    deleted = np.arange(0, 100)
    alive = np.ones(len(index), dtype=bool)
    alive[deleted] = False
    labels, _ = index.knn_query(data[deleted], k=5, filter=alive)
    assert not np.isin(labels, deleted).any()


def test_mapped_and_in_memory_filtered_results_match(built, tmp_path):
    _, index, _ = built
    path = tmp_path / "f.nmslib_rs"
    index.save(path)
    mask = np.random.default_rng(9).random(len(index)) < 0.1
    queries = random(100, 16, 10)
    results = []
    for ix in (index, nmslib_rs.Index.load(path, mmap=True), nmslib_rs.Index.load(path, mmap=False)):
        ix.set_ef(40)
        results.append(ix.knn_query(queries, k=10, filter=mask, max_candidates=200))
    for labels, dists in results[1:]:
        assert (labels == results[0][0]).all() and (dists == results[0][1]).all()


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.ones(5000, np.int8), "dtype bool"),
        (np.ones(5000, np.float32), "dtype bool"),
        (np.ones(4999, bool), r"shape \(5000,\)"),
        (np.ones((5000, 1), bool), r"shape \(5000,\)"),
        (np.ones(10000, bool)[::2], "contiguous"),
        ([True] * 5000, "numpy.ndarray"),
    ],
)
def test_rejects_bad_filters(built, bad, match):
    _, index, exact = built
    q = random(1, 16)
    with pytest.raises(ValueError, match=match):
        index.knn_query(q, filter=bad)
    with pytest.raises(ValueError, match=match):
        exact.knn_query(q, filter=bad)


def test_rejects_negative_max_candidates(built):
    _, index, _ = built
    with pytest.raises(ValueError, match="max_candidates"):
        index.knn_query(random(1, 16), max_candidates=-1)


def test_graph_stats(built, tmp_path):
    _, index, _ = built
    stats = index.graph_stats()
    assert stats["nodes"] == stats["reached"] == 5000
    assert stats["unreachable"] == 0
    assert stats["one_way_links"] > 0
    assert stats["min_degree"] >= 1 and stats["mean_degree"] > stats["min_degree"]
    path = tmp_path / "g.nmslib_rs"
    index.save(path)
    for mmap in (True, False):
        assert nmslib_rs.Index.load(path, mmap=mmap).graph_stats() == stats


def test_self_recall(built):
    _, index, _ = built
    r = index.self_recall(ef=64)
    assert r["nodes"] == 5000
    assert r["top1"] >= 4990 and r["missing"] <= 5
    assert index.self_recall(ef=64, num_threads=1) == r
    with pytest.raises(ValueError):
        index.self_recall(ef=0)
    assert nmslib_rs.Index(dim=3).self_recall() == {"nodes": 0, "top1": 0, "missing": 0}
