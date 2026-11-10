"""Save, load and memory-mapped sharing. Run with: pytest tests/"""

import multiprocessing as mp

import numpy as np
import pytest

import nmslib_rs


def random(n, dim, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim), dtype=np.float32)


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    data = random(4000, 16)
    index = nmslib_rs.Index(dim=16, space="cosine", M=16, ef_construction=100, seed=1)
    index.add_items(data)
    path = tmp_path_factory.mktemp("idx") / "cosine.nmslib_rs"
    index.save(path)
    return index, path, data


@pytest.mark.parametrize("mmap", [True, False])
def test_round_trip_is_exact(saved, mmap):
    index, path, _ = saved
    loaded = nmslib_rs.Index.load(path, mmap=mmap)
    assert loaded.read_only is mmap
    assert (len(loaded), loaded.dim, loaded.space) == (len(index), index.dim, index.space)
    queries = random(300, 16, 2)
    for ef in (10, 80):
        index.set_ef(ef)
        loaded.set_ef(ef)
        a, b = index.knn_query(queries, k=10), loaded.knn_query(queries, k=10)
        assert (a[0] == b[0]).all() and (a[1] == b[1]).all()


def test_load_accepts_str_paths(saved):
    _, path, _ = saved
    assert len(nmslib_rs.Index.load(str(path))) == 4000


def test_mapped_index_is_read_only(saved):
    _, path, data = saved
    mapped = nmslib_rs.Index.load(path)
    assert "read_only=True" in repr(mapped)
    with pytest.raises(RuntimeError, match="mmap=False"):
        mapped.add_items(data[:10])
    copy = nmslib_rs.Index.load(path, mmap=False)
    copy.add_items(data[:10])
    assert len(copy) == 4010


def test_bad_and_missing_files(tmp_path, saved):
    _, path, _ = saved
    bad = tmp_path / "bad.nmslib_rs"
    bad.write_bytes(b"not an index" * 100)
    with pytest.raises(ValueError, match="magic"):
        nmslib_rs.Index.load(bad)
    truncated = tmp_path / "truncated.nmslib_rs"
    truncated.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(ValueError):
        nmslib_rs.Index.load(truncated)
    with pytest.raises(FileNotFoundError):
        nmslib_rs.Index.load(tmp_path / "missing.nmslib_rs")


def test_save_replaces_a_mapped_file(tmp_path):
    path = tmp_path / "replace.nmslib_rs"
    first = nmslib_rs.Index(dim=8, seed=1)
    first.add_items(random(500, 8, 3))
    first.save(path)
    mapped = nmslib_rs.Index.load(path)
    queries = random(20, 8, 4)
    before = mapped.knn_query(queries, k=5)[0]
    second = nmslib_rs.Index(dim=8, seed=2)
    second.add_items(random(800, 8, 5))
    second.save(path)
    assert (mapped.knn_query(queries, k=5)[0] == before).all()
    assert len(nmslib_rs.Index.load(path)) == 800


def _query_mapped(args):
    path, queries = args
    index = nmslib_rs.Index.load(path, mmap=True)
    index.set_ef(50)
    return index.knn_query(queries, k=10, num_threads=1)[0]


def test_many_processes_share_one_file(saved):
    index, path, _ = saved
    queries = random(200, 16, 7)
    index.set_ef(50)
    expected = index.knn_query(queries, k=10, num_threads=1)[0]
    with mp.get_context("spawn").Pool(8) as pool:
        results = pool.map(_query_mapped, [(str(path), queries)] * 8)
    assert all((r == expected).all() for r in results)
