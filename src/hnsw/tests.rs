use std::collections::{HashSet, VecDeque};

use fearless_simd::{Level, dispatch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;

fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn store_1d(points: &[f32]) -> Store {
    Store { dim: 1, space: Space::L2, data: points.to_vec() }
}

fn brute_force(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let mut d: Vec<Cand> = data
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| Cand { dist: crate::simd::l2_sq_scalar(q, v), id: i as u32 })
        .collect();
    d.sort_unstable();
    d.into_iter().take(k).map(|c| c.id).collect()
}

/// Points on a line, 0..5, linked as a path 0-1-2-3-4 at level 0.
fn path_graph() -> (Store, Graph) {
    let store = store_1d(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let mut g = Graph::new(2, 4);
    g.reserve_nodes(5);
    for id in 0..5 {
        g.init_node(id, 0);
    }
    for id in 0..4u32 {
        g.push_neighbor(id, 0, id + 1);
        g.push_neighbor(id + 1, 0, id);
    }
    g.entry = Some(0);
    (store, g)
}

#[test]
fn hand_built_graph_search() {
    let (store, g) = path_graph();
    let mut visited = VisitedSet::new(5);
    dispatch!(Level::new(), simd => {
        let q = [3.9f32];
        let ep = Cand { dist: store.dist(simd, &q, 0), id: 0 };
        // ef = 1 is a greedy walk along the path to the nearest point.
        let found = search_layer(simd, &store, &g, &mut visited, &q, ep, 1, 0);
        assert_eq!(found.iter().map(|c| c.id).collect::<Vec<_>>(), [4]);
        let found = search_layer(simd, &store, &g, &mut visited, &q, ep, 3, 0);
        assert_eq!(found.iter().map(|c| c.id).collect::<Vec<_>>(), [4, 3, 2]);
        assert!(found.windows(2).all(|w| w[0].dist <= w[1].dist));
        assert_eq!(greedy(simd, &store, &g, &q, ep, 0).id, 4);
    });
}

#[test]
fn heuristic_keeps_diverse_neighbors() {
    // Base at 0. Candidates at 1, 2, -1.5: 2 is closer to 1 than to the base,
    // so it is pruned; -1.5 lies on the other side and is kept.
    let store = store_1d(&[0.0, 1.0, 2.0, -1.5]);
    let cands = |ids: &[u32]| -> Vec<Cand> {
        let mut c: Vec<Cand> =
            ids.iter().map(|&id| Cand { dist: store.vector(id)[0].powi(2), id }).collect();
        c.sort_unstable();
        c
    };
    let mut out = Vec::new();
    dispatch!(Level::new(), simd => {
        select_neighbors(simd, &store, &cands(&[1, 2, 3]), 3, false, &mut out);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 3]);
        // Refill brings back the pruned candidate.
        select_neighbors(simd, &store, &cands(&[1, 2, 3]), 3, true, &mut out);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 3, 2]);
        // Fewer candidates than requested: returned unchanged, no pruning.
        select_neighbors(simd, &store, &cands(&[1, 2]), 3, false, &mut out);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 2]);
    });
}

#[test]
fn overflow_reprunes_without_refill() {
    // Node 0 at the origin already has 1 and 2 (cap 2 at level 0); adding 3,
    // which sits behind 1, prunes 3 and keeps the list at [1, 2].
    let store = store_1d(&[0.0, 1.0, -1.0, 1.5]);
    let mut g = Graph::new(1, 2);
    g.reserve_nodes(4);
    for id in 0..4 {
        g.init_node(id, 0);
    }
    g.push_neighbor(0, 0, 1);
    g.push_neighbor(0, 0, 2);
    dispatch!(Level::new(), simd => {
        add_link(simd, &store, &mut g, 0, 3, 0);
        let mut nbs = g.neighbors(0, 0).to_vec();
        nbs.sort();
        assert_eq!(nbs, [1, 2]);
        // Duplicates are refused.
        add_link(simd, &store, &mut g, 0, 1, 0);
        assert_eq!(g.neighbors(0, 0).len(), 2);
    });
}

fn check_invariants(index: &Hnsw) {
    let g = index.graph();
    let n = index.len() as u32;
    let entry = g.entry().expect("non-empty index has an entry point");
    assert_eq!(g.level(entry), g.max_level());
    for id in 0..n {
        assert!(g.level(id) <= g.max_level());
        for level in 0..=g.level(id) {
            let nbs = g.neighbors(id, level);
            assert!(nbs.len() <= g.cap(level), "node {id} level {level} degree {}", nbs.len());
            let unique: HashSet<_> = nbs.iter().collect();
            assert_eq!(unique.len(), nbs.len(), "duplicate edge at node {id} level {level}");
            for &nb in nbs {
                assert_ne!(nb, id, "self-loop at node {id} level {level}");
                assert!(nb < n);
                assert!(g.level(nb) >= level, "edge {id}->{nb} at level {level} above {nb}'s level");
            }
        }
    }
    // Every node is reachable from the entry point at level 0.
    let mut seen = vec![false; n as usize];
    let mut queue = VecDeque::from([entry]);
    seen[entry as usize] = true;
    while let Some(id) = queue.pop_front() {
        for &nb in g.neighbors(id, 0) {
            if !std::mem::replace(&mut seen[nb as usize], true) {
                queue.push_back(nb);
            }
        }
    }
    let unreached = seen.iter().filter(|s| !**s).count();
    assert_eq!(unreached, 0, "{unreached} nodes unreachable at level 0");
}

fn build(n: usize, dim: usize, post: bool) -> (Vec<f32>, Hnsw) {
    let data = random_data(n, dim, 5);
    let params = Params { m: 16, ef_construction: 100, post, seed: 9 };
    let mut index = Hnsw::new(dim, Space::L2, params);
    index.add(&data);
    (data, index)
}

#[test]
fn invariants_hold_after_10k_inserts() {
    let (_, index) = build(10_000, 16, false);
    check_invariants(&index);
    assert!(index.graph().max_level() >= 1);
}

#[test]
fn invariants_hold_after_10k_inserts_with_post() {
    let (_, index) = build(10_000, 16, true);
    check_invariants(&index);
}

#[test]
fn recall_at_10_above_0_99_on_10k() {
    let dim = 16;
    let (data, mut index) = build(10_000, dim, true);
    let queries = random_data(200, dim, 77);
    let mut hits = 0;
    for q in queries.chunks_exact(dim) {
        let truth: HashSet<u32> = brute_force(&data, dim, q, 10).into_iter().collect();
        let found = index.knn_query(q, 10, 200);
        assert_eq!(found.len(), 10);
        assert!(found.windows(2).all(|w| w[0].1 <= w[1].1));
        hits += found.iter().filter(|(id, _)| truth.contains(id)).count();
    }
    let recall = hits as f64 / (200.0 * 10.0);
    assert!(recall > 0.99, "recall@10 = {recall}");
}

#[test]
fn same_seed_same_graph() {
    let (_, a) = build(2_000, 8, true);
    let (_, b) = build(2_000, 8, true);
    let (ga, gb) = (a.graph(), b.graph());
    assert_eq!(ga.entry(), gb.entry());
    for id in 0..2_000 {
        assert_eq!(ga.level(id), gb.level(id));
        for level in 0..=ga.level(id) {
            assert_eq!(ga.neighbors(id, level), gb.neighbors(id, level));
        }
    }
}

#[test]
fn level_distribution_matches_ml() {
    // P(level >= 1) = exp(-1 / mL) = 1 / M.
    let (_, index) = build(20_000, 2, false);
    let g = index.graph();
    let upper = (0..20_000).filter(|&id| g.level(id) >= 1).count() as f64 / 20_000.0;
    assert!((upper - 1.0 / 16.0).abs() < 0.01, "fraction above level 0: {upper}");
}

#[test]
fn small_and_empty_indexes() {
    let mut index = Hnsw::new(3, Space::L2, Params::default());
    assert!(index.knn_query(&[0.0, 0.0, 0.0], 5, 10).is_empty());
    index.add(&[1.0, 0.0, 0.0]);
    assert_eq!(index.knn_query(&[0.0, 0.0, 0.0], 5, 10), [(0, 1.0)]);
    index.add(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.5]);
    let ids: Vec<u32> = index.knn_query(&[0.0, 0.0, 0.0], 5, 10).iter().map(|r| r.0).collect();
    assert_eq!(ids, [2, 0, 1]);
}

#[test]
fn cosine_ignores_magnitude() {
    let mut index = Hnsw::new(2, Space::Cosine, Params::default());
    index.add(&[10.0, 0.0, 0.0, 3.0, 1.0, 1.0]);
    let found = index.knn_query(&[0.5, 0.0], 3, 10);
    assert_eq!(found[0].0, 0);
    assert!(found[0].1.abs() < 1e-3);
    assert_eq!(found[1].0, 2);
    assert!((found[2].1 - 1.0).abs() < 1e-3, "orthogonal vector has distance 1");
}
