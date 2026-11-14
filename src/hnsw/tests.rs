use std::collections::{HashSet, VecDeque};

use fearless_simd::{Level, dispatch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use super::*;

fn random_data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n * dim)
        .map(|_| rng.random_range(-1.0f32..1.0))
        .collect()
}

fn store_1d(points: &[f32]) -> Store {
    Store {
        dim: 1,
        space: Space::L2,
        data: points.to_vec(),
    }
}

fn brute_force(data: &[f32], dim: usize, q: &[f32], k: usize) -> Vec<u32> {
    let mut d: Vec<Cand> = data
        .chunks_exact(dim)
        .enumerate()
        .map(|(i, v)| Cand {
            dist: crate::simd::l2_sq_scalar(q, v),
            id: i as u32,
        })
        .collect();
    d.sort_unstable();
    d.into_iter().take(k).map(|c| c.id).collect()
}

/// Points on a line, 0..5, linked as a path 0-1-2-3-4 at level 0.
fn path_graph() -> (Store, Graph) {
    let store = store_1d(&[0.0, 1.0, 2.0, 3.0, 4.0]);
    let mut g = Graph::new(2, 4);
    g.grow(&[0; 5]);
    for id in 0..4u32 {
        g.push_neighbor(id, 0, id + 1);
        g.push_neighbor(id + 1, 0, id);
    }
    g.set_entry(0, 0);
    (store, g)
}

#[test]
fn hand_built_graph_search() {
    let (store, g) = path_graph();
    let mut buf = SearchBuffers::default();
    buf.visited.ensure_capacity(5);
    let mut found = Vec::new();
    dispatch!(Level::new(), simd => {
        let q = [3.9f32];
        let ep = Cand { dist: store.dist(simd, &q, 0), id: 0 };
        // ef = 1 is a greedy walk along the path to the nearest point.
        let view = BuildView { store: &store, graph: &g };
        search_layer(simd, &view, &mut buf, &q, ep, 1, 0, &mut found);
        assert_eq!(found.iter().map(|c| c.id).collect::<Vec<_>>(), [4]);
        search_layer(simd, &view, &mut buf, &q, ep, 3, 0, &mut found);
        assert_eq!(found.iter().map(|c| c.id).collect::<Vec<_>>(), [4, 3, 2]);
        assert!(found.windows(2).all(|w| w[0].dist <= w[1].dist));
        assert_eq!(greedy(simd, &view, &q, ep, 0).id, 4);
    });
}

#[test]
fn heuristic_keeps_diverse_neighbors() {
    // Base at 0. Candidates at 1, 2, -1.5: 2 is closer to 1 than to the base,
    // so it is pruned; -1.5 lies on the other side and is kept.
    let store = store_1d(&[0.0, 1.0, 2.0, -1.5]);
    let cands = |ids: &[u32]| -> Vec<Cand> {
        let mut c: Vec<Cand> = ids
            .iter()
            .map(|&id| Cand {
                dist: store.vector(id)[0].powi(2),
                id,
            })
            .collect();
        c.sort_unstable();
        c
    };
    let (mut out, mut rej) = (Vec::new(), Vec::new());
    dispatch!(Level::new(), simd => {
        select_neighbors(simd, &store, &cands(&[1, 2, 3]), 3, false, &mut out, &mut rej);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 3]);
        // Refill brings back the pruned candidate.
        select_neighbors(simd, &store, &cands(&[1, 2, 3]), 3, true, &mut out, &mut rej);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 3, 2]);
        // Fewer candidates than requested: returned unchanged, no pruning.
        select_neighbors(simd, &store, &cands(&[1, 2]), 3, false, &mut out, &mut rej);
        assert_eq!(out.iter().map(|c| c.id).collect::<Vec<_>>(), [1, 2]);
    });
}

#[test]
fn overflow_reprunes_without_refill() {
    // Node 0 at the origin already has 1 and 2 (cap 2 at level 0); adding 3,
    // which sits behind 1, prunes 3 and keeps the list at [1, 2].
    let store = store_1d(&[0.0, 1.0, -1.0, 1.5]);
    let mut g = Graph::new(1, 2);
    g.grow(&[0; 4]);
    g.push_neighbor(0, 0, 1);
    g.push_neighbor(0, 0, 2);
    dispatch!(Level::new(), simd => {
        let mut buf = LinkBuffers::default();
        add_link(simd, &store, &g, 0, 3, 0, &mut buf);
        let mut nbs = g.neighbor_vec(0, 0);
        nbs.sort();
        assert_eq!(nbs, [1, 2]);
        // Duplicates are refused.
        add_link(simd, &store, &g, 0, 1, 0, &mut buf);
        assert_eq!(g.neighbor_vec(0, 0).len(), 2);
    });
}

fn check_invariants(index: &Hnsw) {
    check_invariants_allowing(index, 0.0);
}

/// Structural checks. `max_unreachable` is the fraction of nodes allowed to be
/// unreachable from the entry point at level 0. It is 0 for sequential builds
/// and post-merged builds. A parallel build without `post` can strand a few
/// nodes: nodes inserted at the same time do not see each other, and later
/// pruning can drop every edge into one of them (see diag_unreachable_by_threads).
fn check_invariants_allowing(index: &Hnsw, max_unreachable: f64) {
    let g = index.graph();
    let n = index.len() as u32;
    let entry = g.entry().expect("non-empty index has an entry point");
    assert_eq!(g.level(entry), g.max_level());
    for id in 0..n {
        assert!(g.level(id) <= g.max_level());
        for level in 0..=g.level(id) {
            let nbs = g.neighbor_vec(id, level);
            assert!(
                nbs.len() <= g.cap(level),
                "node {id} level {level} degree {}",
                nbs.len()
            );
            let unique: HashSet<_> = nbs.iter().collect();
            assert_eq!(
                unique.len(),
                nbs.len(),
                "duplicate edge at node {id} level {level}"
            );
            for &nb in &nbs {
                assert_ne!(nb, id, "self-loop at node {id} level {level}");
                assert!(nb < n);
                assert!(
                    g.level(nb) >= level,
                    "edge {id}->{nb} at level {level} above {nb}'s level"
                );
            }
        }
    }
    // Every node is reachable from the entry point at level 0.
    let unreached = index.graph_stats().unreachable;
    assert!(
        unreached as f64 <= max_unreachable * n as f64,
        "{unreached} of {n} nodes unreachable at level 0"
    );
}

fn build_threads(n: usize, dim: usize, post: bool, threads: usize) -> (Vec<f32>, Hnsw) {
    let data = random_data(n, dim, 5);
    let params = Params {
        m: 16,
        ef_construction: 100,
        post,
        seed: 9,
    };
    let mut index = Hnsw::new(dim, Space::L2, params);
    index.add(&data, threads);
    (data, index)
}

fn build(n: usize, dim: usize, post: bool) -> (Vec<f32>, Hnsw) {
    build_threads(n, dim, post, 1)
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
    let (data, index) = build(10_000, dim, true);
    assert_recall(&data, &index, dim);
}

fn assert_recall(data: &[f32], index: &Hnsw, dim: usize) {
    let queries = random_data(200, dim, 77);
    let mut hits = 0;
    for q in queries.chunks_exact(dim) {
        let truth: HashSet<u32> = brute_force(data, dim, q, 10).into_iter().collect();
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
            assert_eq!(ga.neighbor_vec(id, level), gb.neighbor_vec(id, level));
        }
    }
}

#[test]
fn level_distribution_matches_ml() {
    // P(level >= 1) = exp(-1 / mL) = 1 / M.
    let (_, index) = build(20_000, 2, false);
    let g = index.graph();
    let upper = (0..20_000).filter(|&id| g.level(id) >= 1).count() as f64 / 20_000.0;
    assert!(
        (upper - 1.0 / 16.0).abs() < 0.01,
        "fraction above level 0: {upper}"
    );
}

#[test]
fn small_and_empty_indexes() {
    let mut index = Hnsw::new(3, Space::L2, Params::default());
    assert!(index.knn_query(&[0.0, 0.0, 0.0], 5, 10).is_empty());
    index.add(&[1.0, 0.0, 0.0], 1);
    assert_eq!(index.knn_query(&[0.0, 0.0, 0.0], 5, 10), [(0, 1.0)]);
    index.add(&[0.0, 2.0, 0.0, 0.0, 0.0, 0.5], 4);
    let ids: Vec<u32> = index
        .knn_query(&[0.0, 0.0, 0.0], 5, 10)
        .iter()
        .map(|r| r.0)
        .collect();
    assert_eq!(ids, [2, 0, 1]);
}

#[test]
fn cosine_ignores_magnitude() {
    let mut index = Hnsw::new(2, Space::Cosine, Params::default());
    index.add(&[10.0, 0.0, 0.0, 3.0, 1.0, 1.0], 1);
    let found = index.knn_query(&[0.5, 0.0], 3, 10);
    assert_eq!(found[0].0, 0);
    assert!(found[0].1.abs() < 1e-3);
    assert_eq!(found[1].0, 2);
    assert!(
        (found[2].1 - 1.0).abs() < 1e-3,
        "orthogonal vector has distance 1"
    );
}

#[test]
fn parallel_build_invariants_and_recall() {
    for post in [false, true] {
        let (data, index) = build_threads(10_000, 16, post, 8);
        check_invariants_allowing(&index, if post { 0.0 } else { 0.01 });
        assert_recall(&data, &index, 16);
    }
}

#[test]
fn parallel_build_with_all_cores_and_incremental_add() {
    let dim = 16;
    let data = random_data(12_000, dim, 21);
    let mut index = Hnsw::new(
        dim,
        Space::L2,
        Params {
            ef_construction: 100,
            ..Params::default()
        },
    );
    index.add(&data[..6_000 * dim], 0);
    index.add(&data[6_000 * dim..], 0);
    assert_eq!(index.len(), 12_000);
    // The second add runs without post.
    check_invariants_allowing(&index, 0.01);
    assert_recall(&data, &index, dim);
}

#[test]
fn batch_query_matches_single_queries() {
    let dim = 16;
    let (_, index) = build_threads(5_000, dim, true, 8);
    let queries = random_data(300, dim, 31);
    let k = 10;
    for threads in [1, 4, 0] {
        let mut labels = vec![0i64; 300 * k];
        let mut dists = vec![0f32; 300 * k];
        index.knn_query_batch(&queries, k, 50, threads, &mut labels, &mut dists);
        for (i, q) in queries.chunks_exact(dim).enumerate() {
            let single = index.knn_query(q, k, 50);
            let ids: Vec<i64> = single.iter().map(|r| r.0 as i64).collect();
            assert_eq!(
                &labels[i * k..(i + 1) * k],
                &ids[..],
                "query {i}, threads {threads}"
            );
        }
    }
}

#[test]
fn batch_query_pads_missing_results() {
    let mut index = Hnsw::new(2, Space::L2, Params::default());
    index.add(&[0.0, 0.0, 1.0, 1.0], 1);
    let mut labels = vec![0i64; 2 * 3];
    let mut dists = vec![0f32; 2 * 3];
    index.knn_query_batch(&[0.0, 0.0, 1.0, 1.0], 3, 10, 2, &mut labels, &mut dists);
    assert_eq!(labels, [0, 1, -1, 1, 0, -1]);
    assert!(dists[2].is_infinite() && dists[5].is_infinite());
}

#[test]
fn concurrent_queries_from_many_threads() {
    let dim = 16;
    let (_, index) = build_threads(5_000, dim, false, 8);
    let queries = random_data(400, dim, 41);
    let expected: Vec<_> = queries
        .chunks_exact(dim)
        .map(|q| index.knn_query(q, 10, 40))
        .collect();
    std::thread::scope(|s| {
        for t in 0..8 {
            let (index, queries, expected) = (&index, &queries, &expected);
            s.spawn(move || {
                for (i, q) in queries.chunks_exact(dim).enumerate().skip(t).step_by(8) {
                    assert_eq!(index.knn_query(q, 10, 40), expected[i]);
                }
            });
        }
    });
}

/// Diagnostic, not a gate: unreachable level-0 nodes by thread count.
#[test]
#[ignore]
fn diag_unreachable_by_threads() {
    for threads in [1, 2, 8, 32] {
        for post in [false, true] {
            let (data, index) = build_threads(20_000, 16, post, threads);
            let g = index.graph();
            let n = index.len() as u32;
            let mut indeg = vec![0u32; n as usize];
            for id in 0..n {
                for nb in g.neighbors(id, 0) {
                    indeg[nb as usize] += 1;
                }
            }
            let mut seen = vec![false; n as usize];
            let entry = g.entry().unwrap();
            let mut queue = VecDeque::from([entry]);
            seen[entry as usize] = true;
            while let Some(id) = queue.pop_front() {
                for nb in g.neighbors(id, 0) {
                    if !std::mem::replace(&mut seen[nb as usize], true) {
                        queue.push_back(nb);
                    }
                }
            }
            let orphans: Vec<u32> = (0..n).filter(|&i| !seen[i as usize]).collect();
            let zero_in = orphans.iter().filter(|&&i| indeg[i as usize] == 0).count();
            let out_ok = orphans.iter().all(|&i| g.neighbors(i, 0).count() > 0);
            // Nearest other node for each orphan, and whether the orphan links to it.
            let mut near_links = 0;
            for &o in &orphans {
                let q = &data[o as usize * 16..(o as usize + 1) * 16];
                let nn = brute_force(&data, 16, q, 2)[1];
                if g.neighbors(nn, 0).any(|x| x == o) {
                    near_links += 1;
                }
            }
            println!(
                "threads {threads:2} post {post:5}: unreachable {:3}, of which zero in-degree {:3}, \
                 all have out-edges {out_ok}, nearest neighbor links back {near_links}",
                orphans.len(),
                zero_in
            );
        }
    }
}

#[test]
fn frozen_and_build_forms_give_identical_results() {
    for space in [Space::L2, Space::Cosine] {
        let mut index = Hnsw::new(
            16,
            space,
            Params {
                ef_construction: 100,
                ..Params::default()
            },
        );
        index.add(&random_data(5_000, 16, 51), 4);
        assert!(
            matches!(index.repr, Repr::Frozen(_)),
            "add leaves the index frozen"
        );
        let queries = random_data(200, 16, 52);
        let run = |ix: &Hnsw| -> Vec<_> {
            queries
                .chunks_exact(16)
                .map(|q| ix.knn_query(q, 10, 60))
                .collect()
        };
        let frozen = run(&index);
        index.thaw();
        assert!(matches!(index.repr, Repr::Build { .. }));
        assert_eq!(run(&index), frozen);
        index.freeze();
        assert_eq!(run(&index), frozen);
    }
}

#[test]
fn frozen_level0_starts_on_a_cache_line() {
    let mut index = Hnsw::new(24, Space::L2, Params::default());
    index.add(&random_data(1_000, 24, 53), 1);
    let Repr::Frozen(f) = &index.repr else {
        panic!("not frozen")
    };
    let v = f.view();
    assert_eq!(v.vector(0).as_ptr() as usize % 64, 0);
    assert_eq!(v.vector(999).as_ptr() as usize % 64, 0);
}

#[test]
fn graph_stats_finds_an_isolated_node() {
    // The path 0-1-2-3-4 plus node 5, which nothing links to.
    let store = store_1d(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut g = Graph::new(2, 4);
    g.grow(&[0; 6]);
    for id in 0..4u32 {
        g.push_neighbor(id, 0, id + 1);
        g.push_neighbor(id + 1, 0, id);
    }
    g.set_entry(0, 0);
    let h = health_of(&BuildView {
        store: &store,
        graph: &g,
    });
    assert_eq!(
        (
            h.nodes,
            h.reached,
            h.unreachable,
            h.one_way_links,
            h.min_degree
        ),
        (6, 5, 1, 0, 0)
    );
    assert!((h.mean_degree - 8.0 / 6.0).abs() < 1e-12);
    // A link out of 5 does not make it reachable, and has no reverse link.
    g.push_neighbor(5, 0, 4);
    let h = health_of(&BuildView {
        store: &store,
        graph: &g,
    });
    assert_eq!(
        (h.reached, h.unreachable, h.one_way_links, h.min_degree),
        (5, 1, 1, 1)
    );
}

#[test]
fn diagnostics_on_empty_index() {
    let index = Hnsw::new(4, Space::L2, Params::default());
    let h = index.graph_stats();
    assert_eq!(
        (h.nodes, h.reached, h.unreachable, h.min_degree),
        (0, 0, 0, 0)
    );
    assert_eq!(
        index.self_recall(10, 1),
        SelfRecall {
            nodes: 0,
            top1: 0,
            missing: 0
        }
    );
}

#[test]
fn every_node_reachable_after_several_adds() {
    // Sequential, so the result does not depend on thread timing; parallel
    // adds without post can strand a few nodes (diag_unreachable_by_threads).
    let dim = 16;
    let data = random_data(10_000, dim, 23);
    let mut index = Hnsw::new(
        dim,
        Space::L2,
        Params {
            ef_construction: 100,
            ..Params::default()
        },
    );
    for chunk in data.chunks(2_500 * dim) {
        index.add(chunk, 1);
    }
    let h = index.graph_stats();
    assert_eq!((h.nodes, h.reached, h.unreachable), (10_000, 10_000, 0));
}

#[test]
fn self_recall_on_random_data() {
    // Measured at ef 64: 10000 of 10000 first for the sequential post build,
    // 9998 (2 missing) for a parallel build without post.
    for (threads, post) in [(1, true), (8, false)] {
        let (_, index) = build_threads(10_000, 16, post, threads);
        let r = index.self_recall(64, 0);
        assert_eq!(r.nodes, 10_000);
        assert!(r.top1 >= 9_990 && r.missing <= 10, "{r:?}");
        assert_eq!(index.self_recall(64, 1), r);
    }
}

#[test]
fn self_recall_counts_a_duplicate_as_found() {
    // Every vector twice: a node often loses first place to its twin, which
    // has the same distance and a lower or higher id.
    let base = random_data(500, 8, 24);
    let data: Vec<f32> = base.iter().chain(&base).copied().collect();
    let mut index = Hnsw::new(8, Space::L2, Params::default());
    index.add(&data, 1);
    let r = index.self_recall(32, 1);
    assert_eq!((r.nodes, r.top1, r.missing), (1_000, 1_000, 0));
}

fn random_mask(n: usize, fraction: f64, seed: u64) -> Bitset {
    let mut rng = StdRng::seed_from_u64(seed);
    let bits: Vec<bool> = (0..n).map(|_| rng.random::<f64>() < fraction).collect();
    Bitset::from_bools(&bits)
}

#[test]
fn all_true_filter_matches_unfiltered_search() {
    let dim = 16;
    let (_, index) = build_threads(5_000, dim, true, 8);
    let all = Bitset::from_bools(&[true; 5_000]);
    let queries = random_data(200, dim, 62);
    for ef in [10, 50, 200] {
        for q in queries.chunks_exact(dim) {
            assert_eq!(
                index.knn_query_filtered(q, 10, ef, Some(&all), 0),
                index.knn_query(q, 10, ef),
                "ef {ef}"
            );
        }
    }
}

#[test]
fn all_false_filter_returns_nothing() {
    let dim = 16;
    let (_, index) = build_threads(2_000, dim, true, 4);
    let none = Bitset::new(2_000);
    let queries = random_data(20, dim, 63);
    let (mut labels, mut dists) = (vec![0i64; 20 * 5], vec![0f32; 20 * 5]);
    for cap in [0, 50] {
        index.knn_query_batch_filtered(
            &queries,
            5,
            20,
            Some(&none),
            cap,
            2,
            &mut labels,
            &mut dists,
        );
        assert!(labels.iter().all(|&l| l == -1));
        assert!(dists.iter().all(|d| d.is_infinite()));
    }
}

#[test]
fn filtered_recall_against_brute_force() {
    // Measured at ef 50 on this index, 500 queries: recall@10 1.0 at 50%,
    // 10% and 1% of labels passing (0.9556 / 0.9974 / 1.0 at ef 10).
    let dim = 16;
    let k = 10;
    let (data, index) = build_threads(10_000, dim, true, 8);
    let mut exact = crate::brute::BruteForce::new(dim, Space::L2);
    exact.add(&data);
    let queries = random_data(500, dim, 61);
    let nq = 500;
    for fraction in [0.5, 0.1, 0.01] {
        let filter = random_mask(10_000, fraction, 7);
        let (mut want, mut want_d) = (vec![0i64; nq * k], vec![0f32; nq * k]);
        exact.knn_query_batch_filtered(&queries, k, Some(&filter), 0, &mut want, &mut want_d);
        let (mut got, mut got_d) = (vec![0i64; nq * k], vec![0f32; nq * k]);
        index.knn_query_batch_filtered(&queries, k, 50, Some(&filter), 0, 0, &mut got, &mut got_d);
        assert!(got.iter().all(|&l| l >= 0 && filter.contains(l as u32)));
        let hits: usize = want
            .chunks(k)
            .zip(got.chunks(k))
            .map(|(w, g)| g.iter().filter(|x| w.contains(x)).count())
            .sum();
        let recall = hits as f64 / (nq * k) as f64;
        assert!(recall >= 0.99, "{fraction}: recall@10 = {recall}");
    }
}

#[test]
fn max_candidates_caps_expansions() {
    let dim = 16;
    let (_, index) = build_threads(10_000, dim, true, 8);
    let Repr::Frozen(f) = &index.repr else {
        panic!("not frozen")
    };
    let view = f.view();
    let filter = random_mask(10_000, 0.01, 8);
    let queries = random_data(50, dim, 64);
    let mut buf = SearchBuffers::default();
    buf.visited.ensure_capacity(10_000);
    let mut out = Vec::new();
    let ep_id = view.entry().node.unwrap();
    dispatch!(Level::new(), simd => {
        for q in queries.chunks_exact(dim) {
            let ep = Cand { dist: view.dist(simd, q, ep_id), id: ep_id };
            for filter in [Some(&filter), None] {
                let uncapped = search_layer_filtered(simd, &view, &mut buf, q, ep, 50, filter, 0, &mut out);
                for cap in [1, 10, 100, 1_000] {
                    let n = search_layer_filtered(simd, &view, &mut buf, q, ep, 50, filter, cap, &mut out);
                    assert_eq!(n, uncapped.min(cap));
                    assert!(out.iter().all(|c| filter.is_none_or(|f| f.contains(c.id))));
                }
            }
        }
    });
    // A tight cap can leave fewer than k results, all passing.
    let short = queries
        .chunks_exact(dim)
        .map(|q| index.knn_query_filtered(q, 10, 50, Some(&filter), 5))
        .inspect(|found| assert!(found.iter().all(|r| filter.contains(r.0))))
        .filter(|found| found.len() < 10)
        .count();
    assert!(short > 0);
}
