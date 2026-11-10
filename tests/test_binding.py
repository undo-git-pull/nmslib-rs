"""Tests for the rhnsw Python binding. Run with: pytest tests/"""

import threading
import time

import numpy as np
import pytest

import rhnsw


def random(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


def brute_force(data, queries, k):
    d = ((queries[:, None, :] - data[None, :, :]) ** 2).sum(-1)
    return np.argsort(d, axis=1)[:, :k]


@pytest.fixture(scope="module")
def built():
    data = random(5000, 16)
    index = rhnsw.Index(dim=16, space="l2", M=16, ef_construction=100, seed=0)
    index.add_items(data)
    return data, index


def test_api_shapes_and_types(built):
    data, index = built
    assert len(index) == 5000 and index.dim == 16 and index.space == "l2"
    index.set_ef(80)
    assert index.ef == 80
    labels, dists = index.knn_query(random(7, 16, 1), k=10)
    assert labels.shape == (7, 10) and labels.dtype == np.int64
    assert dists.shape == (7, 10) and dists.dtype == np.float32
    assert (np.diff(dists, axis=1) >= 0).all()


def test_recall_and_distances(built):
    data, index = built
    queries = random(100, 16, 2)
    index.set_ef(200)
    labels, dists = index.knn_query(queries, k=10)
    truth = brute_force(data, queries, 10)
    recall = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(labels, truth)])
    assert recall > 0.99
    # Distances are squared L2.
    exact = ((data[labels] - queries[:, None, :]) ** 2).sum(-1)
    np.testing.assert_allclose(dists, exact, rtol=1e-4)


def test_exact_match_is_first(built):
    data, index = built
    labels, dists = index.knn_query(data[:20], k=1)
    assert (labels[:, 0] == np.arange(20)).all()
    assert (dists[:, 0] == 0).all()


def test_labels_continue_across_add_calls():
    index = rhnsw.Index(dim=4, seed=1)
    a, b = random(50, 4, 3), random(50, 4, 4)
    index.add_items(a)
    index.add_items(b)
    assert len(index) == 100
    labels, _ = index.knn_query(b[:5], k=1)
    assert (labels[:, 0] == np.arange(50, 55)).all()


def test_fewer_items_than_k_pads():
    index = rhnsw.Index(dim=3)
    index.add_items(np.eye(3, dtype=np.float32))
    labels, dists = index.knn_query(np.zeros((1, 3), np.float32), k=5)
    assert sorted(labels[0, :3]) == [0, 1, 2]
    assert (labels[0, 3:] == -1).all() and np.isinf(dists[0, 3:]).all()


def test_empty_inputs():
    index = rhnsw.Index(dim=3)
    index.add_items(np.zeros((0, 3), np.float32))
    labels, dists = index.knn_query(np.zeros((2, 3), np.float32), k=2)
    assert (labels == -1).all()
    labels, _ = index.knn_query(np.zeros((0, 3), np.float32), k=2)
    assert labels.shape == (0, 2)


def test_cosine():
    index = rhnsw.Index(dim=2, space="cosine")
    index.add_items(np.array([[10, 0], [0, 3], [1, 1]], np.float32))
    labels, dists = index.knn_query(np.array([[0.5, 0]], np.float32), k=3)
    assert list(labels[0]) == [0, 2, 1]
    np.testing.assert_allclose(dists[0], [0, 1 - np.sqrt(0.5), 1], atol=1e-6)


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((4, 16), np.float64), "dtype float32"),
        (np.zeros((4, 8), np.float32), r"shape \(n, 16\)"),
        (np.zeros(16, np.float32), r"shape \(n, 16\)"),
        (np.zeros((4, 16, 1), np.float32), r"shape \(n, 16\)"),
        (np.zeros((16, 4), np.float32).T, "C-contiguous"),
        (np.zeros((4, 32), np.float32)[:, ::2], "C-contiguous"),
        (np.zeros((4, 16), ">f4"), "dtype float32"),
        ([[0.0] * 16], "numpy.ndarray"),
    ],
)
def test_rejects_bad_arrays(built, bad, match):
    _, index = built
    with pytest.raises(ValueError, match=match):
        index.add_items(bad)
    with pytest.raises(ValueError, match=match):
        index.knn_query(bad)


def test_rejects_bad_params(built):
    _, index = built
    for kwargs in [dict(dim=0), dict(dim=4, space="ip"), dict(dim=4, M=1), dict(dim=4, ef_construction=0)]:
        with pytest.raises(ValueError):
            rhnsw.Index(**kwargs)
    with pytest.raises(ValueError):
        index.set_ef(0)
    with pytest.raises(ValueError):
        index.knn_query(random(1, 16), k=0)


def test_releases_gil_during_build():
    """A Python thread keeps running while add_items works in Rust."""
    data = random(20000, 32, 5)
    index = rhnsw.Index(dim=32, ef_construction=200)
    ticks = 0
    done = threading.Event()

    def count():
        nonlocal ticks
        while not done.is_set():
            ticks += 1
            time.sleep(0.001)

    t = threading.Thread(target=count)
    t.start()
    start = time.perf_counter()
    index.add_items(data)
    elapsed = time.perf_counter() - start
    done.set()
    t.join()
    # With the GIL held the counter could not advance at all during the build.
    assert elapsed > 0.2
    assert ticks > 0.2 * elapsed / 0.001
