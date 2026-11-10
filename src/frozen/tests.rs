use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::hnsw::{Hnsw, Params};

fn temp_file(name: &str) -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!("rhnsw-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir.join(format!("{name}-{}.rhnsw", NEXT.fetch_add(1, Ordering::Relaxed)))
}

fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

/// M = 4 gives several upper levels at a few thousand nodes.
fn build(space: Space, post: bool) -> Hnsw {
    let params = Params { m: 4, ef_construction: 50, post, seed: 3 };
    let mut index = Hnsw::new(24, space, params);
    index.add(&random_data(3_000, 24, 11), 1);
    index
}

fn assert_same_results(a: &Hnsw, b: &Hnsw) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.dim(), b.dim());
    assert_eq!(a.space(), b.space());
    let (sa, sb) = (a.stats(), b.stats());
    assert_eq!(sa.nodes_per_level, sb.nodes_per_level);
    assert_eq!(sa.mean_degree, sb.mean_degree);
    let queries = random_data(200, a.dim(), 99);
    for ef in [10, 50, 200] {
        for q in queries.chunks_exact(a.dim()) {
            assert_eq!(a.knn_query(q, 10, ef), b.knn_query(q, 10, ef), "ef {ef}");
        }
    }
}

#[test]
fn round_trip_matches_in_memory_index() {
    for space in [Space::L2, Space::Cosine] {
        for post in [false, true] {
            let index = build(space, post);
            assert!(index.stats().nodes_per_level.len() >= 3, "want several upper levels");
            let path = temp_file("round-trip");
            index.save(&path).unwrap();
            for mmap in [true, false] {
                let loaded = Hnsw::load(&path, mmap).unwrap();
                assert_eq!(loaded.is_read_only(), mmap);
                assert_same_results(&index, &loaded);
            }
            fs::remove_file(&path).unwrap();
        }
    }
}

#[test]
fn resaving_a_mapped_index_gives_the_same_file() {
    let index = build(Space::L2, true);
    let (a, b) = (temp_file("first"), temp_file("second"));
    index.save(&a).unwrap();
    Hnsw::load(&a, true).unwrap().save(&b).unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
    // No temporary file is left behind.
    assert!(!temp_path(&b).exists());
}

#[test]
fn loaded_copy_can_grow_and_mapped_index_cannot() {
    // M = 16: with M = 4 a few added vectors are not found from the entry
    // point, loaded or not (8 of 500 in both cases when this was written).
    let mut index = Hnsw::new(24, Space::L2, Params { ef_construction: 50, seed: 3, ..Params::default() });
    index.add(&random_data(3_000, 24, 11), 1);
    let path = temp_file("grow");
    index.save(&path).unwrap();

    let mut copy = Hnsw::load(&path, false).unwrap();
    let extra = random_data(500, 24, 12);
    copy.add(&extra, 4);
    assert_eq!(copy.len(), 3_500);
    for (i, v) in extra.chunks_exact(24).enumerate() {
        assert_eq!(copy.knn_query(v, 1, 50)[0], (3_000 + i as u32, 0.0));
    }

    let mapped = Hnsw::load(&path, true).unwrap();
    assert!(mapped.is_read_only());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut mapped = mapped;
        mapped.add(&extra, 1);
    }));
    assert!(result.is_err());
}

#[test]
fn same_file_mapped_twice() {
    let index = build(Space::Cosine, false);
    let path = temp_file("twice");
    index.save(&path).unwrap();
    let (a, b) = (Hnsw::load(&path, true).unwrap(), Hnsw::load(&path, true).unwrap());
    assert_same_results(&a, &b);
}

#[test]
fn replacing_a_mapped_file_keeps_the_old_mapping_valid() {
    let path = temp_file("replace");
    let first = build(Space::L2, false);
    first.save(&path).unwrap();
    let mapped = Hnsw::load(&path, true).unwrap();

    let mut second = Hnsw::new(24, Space::L2, Params { m: 4, seed: 8, ..Params::default() });
    second.add(&random_data(1_000, 24, 13), 1);
    second.save(&path).unwrap();

    assert_same_results(&first, &mapped);
    assert_same_results(&second, &Hnsw::load(&path, true).unwrap());
}

#[test]
fn empty_index_round_trip() {
    let index = Hnsw::new(8, Space::L2, Params::default());
    let path = temp_file("empty");
    index.save(&path).unwrap();
    for mmap in [true, false] {
        let loaded = Hnsw::load(&path, mmap).unwrap();
        assert_eq!(loaded.len(), 0);
        assert!(loaded.knn_query(&[0.0; 8], 5, 10).is_empty());
    }
}

// --- malformed files ------------------------------------------------------

fn saved_bytes() -> (Hnsw, Vec<u8>) {
    let index = build(Space::L2, true);
    let path = temp_file("bytes");
    index.save(&path).unwrap();
    let bytes = fs::read(&path).unwrap();
    (index, bytes)
}

fn load_bytes(bytes: &[u8]) -> Result<Hnsw, LoadError> {
    let path = temp_file("malformed");
    fs::write(&path, bytes).unwrap();
    let result = Hnsw::load(&path, true);
    // Exercise the copying path on the same bytes too.
    let copied = Hnsw::load(&path, false);
    assert_eq!(result.is_ok(), copied.is_ok());
    result
}

fn expect_format_error(bytes: &[u8], contains: &str) {
    match load_bytes(bytes) {
        Err(LoadError::Format(msg)) => assert!(msg.contains(contains), "{msg:?} lacks {contains:?}"),
        Err(other) => panic!("expected a format error containing {contains:?}, got {other:?}"),
        Ok(_) => panic!("expected a format error containing {contains:?}, but it loaded"),
    }
}

fn put_u32(b: &mut [u8], at: usize, x: u32) {
    b[at..at + 4].copy_from_slice(&x.to_le_bytes());
}

fn put_u64(b: &mut [u8], at: usize, x: u64) {
    b[at..at + 8].copy_from_slice(&x.to_le_bytes());
}

/// Recomputes the header checksum after an edit, so the edit reaches the
/// field checks behind it.
fn reseal(b: &mut [u8]) {
    let sum = header_checksum(&b[..CHECKSUM_AT]);
    b[CHECKSUM_AT..HEADER_LEN].copy_from_slice(&sum.to_le_bytes());
}

#[test]
fn rejects_bad_headers() {
    let (_, good) = saved_bytes();
    let edit = |f: &dyn Fn(&mut Vec<u8>)| {
        let mut b = good.clone();
        f(&mut b);
        reseal(&mut b);
        b
    };
    expect_format_error(&edit(&|b| b[0] = b'X'), "magic");
    expect_format_error(&edit(&|b| put_u32(b, 8, VERSION + 1)), "version");
    // An edit that is not resealed fails the checksum.
    let mut b = good.clone();
    put_u32(&mut b, 16, 40);
    expect_format_error(&b, "checksum");
    expect_format_error(&edit(&|b| put_u32(b, 12, 7)), "flags");
    expect_format_error(&edit(&|b| put_u32(b, 16, 40)), "layout");
    expect_format_error(&edit(&|b| put_u32(b, 16, 0)), "dim");
    expect_format_error(&edit(&|b| put_u32(b, 20, 9)), "space");
    expect_format_error(&edit(&|b| put_u32(b, 24, 1)), "M");
    expect_format_error(&edit(&|b| put_u32(b, 36, 3_000)), "entry point");
    expect_format_error(&edit(&|b| put_u64(b, 40, 2_999)), "section");
    expect_format_error(&edit(&|b| put_u32(b, 48, 300)), "max level");
    expect_format_error(&edit(&|b| put_u64(b, 72, 257)), "levels section");
    expect_format_error(&edit(&|b| put_u64(b, 120, u64::MAX - 8)), "level0 section");
    expect_format_error(&edit(&|b| put_u64(b, 136, 12)), "header says");
}

#[test]
fn rejects_truncated_files() {
    let (_, good) = saved_bytes();
    let h = Header::decode(&good[..HEADER_LEN], good.len() as u64).unwrap();
    let mut cuts = vec![0, 1, 8, HEADER_LEN - 1, HEADER_LEN, good.len() - 1];
    for s in [h.levels, h.upper_offset, h.upper, h.level0] {
        cuts.extend([s.offset as usize, s.offset as usize + 1, (s.offset + s.len) as usize - 1]);
    }
    for cut in cuts {
        assert!(load_bytes(&good[..cut]).is_err(), "loaded a file cut at {cut}");
    }
}

#[test]
fn rejects_inconsistent_levels_and_offsets() {
    let (_, good) = saved_bytes();
    let h = Header::decode(&good[..HEADER_LEN], good.len() as u64).unwrap();
    let levels = h.levels.offset as usize;

    let mut b = good.clone();
    b[levels + 5] = (h.entry.max_level + 1) as u8;
    expect_format_error(&b, "above the max level");

    let mut b = good.clone();
    let id = (0..h.n).find(|&i| good[levels + i] > 0).unwrap();
    put_u64(&mut b, h.upper_offset.offset as usize + 8 * id, h.upper.len / 4);
    expect_format_error(&b, "upper-level lists");

    let mut b = good.clone();
    let entry = h.entry.node.unwrap() as usize;
    b[levels + entry] = 0;
    expect_format_error(&b, "entry point");
}

#[test]
fn corrupt_neighbor_lists_never_crash_queries() {
    let (_, good) = saved_bytes();
    let h = Header::decode(&good[..HEADER_LEN], good.len() as u64).unwrap();
    let mut rng = StdRng::seed_from_u64(5);
    let mut b = good.clone();
    // Random words over every list (counts and ids), in level 0 and above.
    for s in [h.upper, h.level0] {
        let words = s.len as usize / 4;
        for _ in 0..words / 3 {
            let at = s.offset as usize + 4 * rng.random_range(0..words);
            put_u32(&mut b, at, rng.random());
        }
    }
    let index = load_bytes(&b).expect("list contents are not validated at load");
    for q in random_data(200, 24, 6).chunks_exact(24) {
        let found = index.knn_query(q, 10, 50);
        assert!(found.iter().all(|&(id, _)| (id as usize) < h.n));
    }
    index.stats();
    // Saving a corrupted index writes only in-range ids.
    let path = temp_file("resave-corrupt");
    index.save(&path).unwrap();
    Hnsw::load(&path, true).unwrap();
}

#[test]
fn missing_file_is_an_io_error() {
    let path = temp_file("missing");
    assert!(matches!(Hnsw::load(&path, true), Err(LoadError::Io(_))));
}

