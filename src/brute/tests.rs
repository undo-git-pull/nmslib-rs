use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;
use crate::hnsw::{Hnsw, Params};
use crate::simd::{dot_scalar, l2_sq_scalar};

fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect()
}

/// Sorts every vector by scalar distance; the reference for the SIMD scan.
fn reference(data: &[f32], dim: usize, space: Space, q: &[f32], k: usize) -> Vec<u32> {
    let unit = |v: &[f32]| {
        let mut v = v.to_vec();
        normalize(&mut v);
        v
    };
    let q = match space {
        Space::L2 => q.to_vec(),
        Space::Cosine => unit(q),
    };
    let mut d: Vec<Cand> = data
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| Cand {
            dist: match space {
                Space::L2 => l2_sq_scalar(&q, v),
                Space::Cosine => 1.0 - dot_scalar(&q, &unit(v)),
            },
            id: i as u32,
        })
        .collect();
    d.sort_unstable();
    d.into_iter().take(k).map(|c| c.id).collect()
}

#[test]
fn matches_sorted_scan() {
    let dim = 37;
    for space in [Space::L2, Space::Cosine] {
        let data = random_data(2000, dim, 1);
        let queries = random_data(50, dim, 2);
        let mut index = BruteForce::new(dim, space);
        index.add(&data[..1000 * dim]);
        index.add(&data[1000 * dim..]);
        assert_eq!(index.len(), 2000);
        for q in queries.chunks_exact(dim) {
            let got: Vec<u32> = index.knn_query(q, 10).iter().map(|r| r.0).collect();
            // SIMD and scalar sums round differently; compare as sets of
            // the scalar top 10 allowing a swap at the boundary.
            let want = reference(&data, dim, space, q, 11);
            let missing = got.iter().filter(|id| !want.contains(id)).count();
            assert!(missing == 0, "{space:?}: got {got:?}, want {want:?}");
        }
    }
}

#[test]
fn results_are_sorted_with_ties_by_id() {
    // Four copies of the same point: equal distances come back in id order.
    let mut index = BruteForce::new(2, Space::L2);
    index.add(&[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]);
    let got = index.knn_query(&[1.0, 1.0], 4);
    assert_eq!(got, vec![(0, 0.0), (2, 0.0), (4, 0.0), (5, 0.0)]);
}

#[test]
fn small_and_empty_indexes() {
    let mut index = BruteForce::new(3, Space::L2);
    assert!(index.knn_query(&[0.0; 3], 5).is_empty());
    index.add(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]);
    let got = index.knn_query(&[1.5, 0.0, 0.0], 5);
    assert_eq!(got, vec![(1, 0.25), (0, 2.25)]);

    let mut labels = vec![0i64; 2 * 3];
    let mut dists = vec![0f32; 2 * 3];
    index.knn_query_batch(&[0.0; 6], 3, 1, &mut labels, &mut dists);
    assert_eq!(labels, vec![0, 1, -1, 0, 1, -1]);
    assert_eq!(dists[2], f32::INFINITY);
}

#[test]
fn batch_matches_single_queries_on_any_thread_count() {
    let dim = 16;
    let data = random_data(3000, dim, 3);
    let queries = random_data(64, dim, 4);
    let mut index = BruteForce::new(dim, Space::Cosine);
    index.add(&data);
    let k = 7;
    for threads in [0, 1, 3] {
        let mut labels = vec![0i64; 64 * k];
        let mut dists = vec![0f32; 64 * k];
        index.knn_query_batch(&queries, k, threads, &mut labels, &mut dists);
        for (i, q) in queries.chunks_exact(dim).enumerate() {
            let single = index.knn_query(q, k);
            let ids: Vec<i64> = single.iter().map(|r| r.0 as i64).collect();
            let ds: Vec<f32> = single.iter().map(|r| r.1).collect();
            assert_eq!(&labels[i * k..(i + 1) * k], &ids[..]);
            assert_eq!(&dists[i * k..(i + 1) * k], &ds[..]);
        }
    }
}

#[test]
fn hnsw_with_large_ef_finds_the_exact_answer() {
    // On a small set with ef far above n's diameter, HNSW should return what
    // brute force returns, with bit-identical distances (same kernels).
    let dim = 24;
    let data = random_data(1500, dim, 5);
    let queries = random_data(30, dim, 6);
    for space in [Space::L2, Space::Cosine] {
        let mut exact = BruteForce::new(dim, space);
        exact.add(&data);
        let mut hnsw = Hnsw::new(dim, space, Params::default());
        hnsw.add(&data, 1);
        for q in queries.chunks_exact(dim) {
            assert_eq!(hnsw.knn_query(q, 10, 1500), exact.knn_query(q, 10));
        }
    }
}
