"""Tests for nmslib_rs.BruteForceIndex. Run with: pytest tests/"""

import numpy as np
import pytest

import nmslib_rs


def random(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


def exact(data, queries, k, space):
    """Exact neighbors and distances in float64 with numpy."""
    d64, q64 = data.astype(np.float64), queries.astype(np.float64)
    if space == "cosine":
        d64 /= np.linalg.norm(d64, axis=1, keepdims=True)
        q64 /= np.linalg.norm(q64, axis=1, keepdims=True)
        dist = 1.0 - q64 @ d64.T
    else:
        dist = ((q64[:, None, :] - d64[None, :, :]) ** 2).sum(-1)
    ids = np.argsort(dist, axis=1, kind="stable")[:, :k]
    return ids, np.take_along_axis(dist, ids, 1)


@pytest.mark.parametrize("space", ["l2", "cosine"])
def test_matches_numpy(space):
    data, queries = random(3000, 24), random(40, 24, 1)
    index = nmslib_rs.BruteForceIndex(dim=24, space=space)
    index.add_items(data[:1000])
    index.add_items(data[1000:])
    assert len(index) == 3000 and index.dim == 24 and index.space == space
    labels, dists = index.knn_query(queries, k=10)
    assert labels.shape == (40, 10) and labels.dtype == np.int64
    assert dists.shape == (40, 10) and dists.dtype == np.float32
    want_ids, want_d = exact(data, queries, 10, space)
    np.testing.assert_allclose(dists, want_d, rtol=1e-4, atol=1e-5)
    # Float32 rounding can swap near ties; ids must match elsewhere.
    assert (labels == want_ids).mean() > 0.99


def test_thread_counts_agree():
    data, queries = random(2000, 16), random(100, 16, 2)
    index = nmslib_rs.BruteForceIndex(16)
    index.add_items(data)
    results = [index.knn_query(queries, k=5, num_threads=t) for t in (0, 1, 3)]
    for labels, dists in results[1:]:
        np.testing.assert_array_equal(labels, results[0][0])
        np.testing.assert_array_equal(dists, results[0][1])


def test_matches_index_with_large_ef():
    """Same kernels: HNSW with ef >= n returns the brute-force answer exactly."""
    data, queries = random(1500, 16), random(30, 16, 3)
    exact_index = nmslib_rs.BruteForceIndex(16, "l2")
    exact_index.add_items(data)
    hnsw = nmslib_rs.Index(16, "l2")
    hnsw.add_items(data)
    hnsw.set_ef(1500)
    for a, b in zip(hnsw.knn_query(queries, k=10), exact_index.knn_query(queries, k=10)):
        np.testing.assert_array_equal(a, b)


def test_fewer_items_than_k():
    index = nmslib_rs.BruteForceIndex(4)
    labels, dists = index.knn_query(random(2, 4), k=3)
    assert (labels == -1).all() and np.isinf(dists).all()
    index.add_items(random(2, 4))
    labels, dists = index.knn_query(random(1, 4, 1), k=3)
    assert sorted(labels[0, :2]) == [0, 1] and labels[0, 2] == -1
    assert np.isinf(dists[0, 2])


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        nmslib_rs.BruteForceIndex(0)
    with pytest.raises(ValueError):
        nmslib_rs.BruteForceIndex(4, space="ip")
    index = nmslib_rs.BruteForceIndex(4)
    with pytest.raises(ValueError):
        index.add_items(np.zeros((3, 5), np.float32))
    with pytest.raises(ValueError):
        index.add_items(np.zeros((3, 4), np.float64))
    with pytest.raises(ValueError):
        index.knn_query(np.zeros((1, 4), np.float32), k=0)


def test_repr():
    index = nmslib_rs.BruteForceIndex(8, "cosine")
    index.add_items(random(3, 8))
    assert repr(index) == 'nmslib_rs.BruteForceIndex(dim=8, space="cosine", len=3)'
