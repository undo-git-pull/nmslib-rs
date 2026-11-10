//! HNSW index, single-threaded, following NMSLIB's `hnsw` method.
//!
//! See NOTES.md for the NMSLIB behaviors matched here. In short: insert
//! selects `M` neighbors at every level with the pruning heuristic (no refill),
//! an overflowing list is re-pruned the same way, and the optional `post`
//! step (NMSLIB `post=2`) rebuilds the graph in reverse insertion order and
//! merges the two level-0 lists with the refilling heuristic.
//!
//! The SIMD level is chosen once per build or query, and every function that
//! computes distances runs inside it through `Simd::vectorize`.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use fearless_simd::{Level, Simd, dispatch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::simd::{dot_in, l2_sq_in};
use crate::visited::VisitedSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Squared Euclidean distance.
    L2,
    /// `1 - cos(a, b)`. Vectors are normalized on insert and on query.
    Cosine,
}

#[derive(Clone, Debug)]
pub struct Params {
    /// Neighbors selected per insert at every level. Upper-level lists are
    /// capped at `m`, level-0 lists at `2 * m`.
    pub m: usize,
    pub ef_construction: usize,
    /// NMSLIB's `post=2`: after the first build, build again in reverse
    /// insertion order and merge the two level-0 graphs. Roughly doubles build
    /// time. Applies to the first `add` on an empty index only.
    pub post: bool,
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, post: true, seed: 0 }
    }
}

/// A node id and its distance to some reference point. Ordered by distance,
/// then id, so heaps are deterministic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cand {
    pub dist: f32,
    pub id: u32,
}

impl Eq for Cand {}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then(self.id.cmp(&other.id))
    }
}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Vectors and the distance function.
struct Store {
    dim: usize,
    space: Space,
    data: Vec<f32>,
}

impl Store {
    fn vector(&self, id: u32) -> &[f32] {
        let start = id as usize * self.dim;
        &self.data[start..start + self.dim]
    }

    #[inline(always)]
    fn dist<S: Simd>(&self, simd: S, q: &[f32], id: u32) -> f32 {
        let v = self.vector(id);
        match self.space {
            Space::L2 => l2_sq_in(simd, q, v),
            Space::Cosine => 1.0 - dot_in(simd, q, v),
        }
    }
}

/// Neighbor lists. Level 0 is one flat array with `max_m0` slots per node and
/// a length array; upper levels are per node, `1 + max_m` slots per level
/// (a length followed by the ids).
#[derive(Clone)]
pub(crate) struct Graph {
    max_m: usize,
    max_m0: usize,
    levels: Vec<u8>,
    links0: Vec<u32>,
    len0: Vec<u32>,
    upper: Vec<Vec<u32>>,
    entry: Option<u32>,
    max_level: usize,
}

impl Graph {
    fn new(max_m: usize, max_m0: usize) -> Self {
        Self {
            max_m,
            max_m0,
            levels: Vec::new(),
            links0: Vec::new(),
            len0: Vec::new(),
            upper: Vec::new(),
            entry: None,
            max_level: 0,
        }
    }

    fn reserve_nodes(&mut self, n: usize) {
        self.levels.resize(n, 0);
        self.links0.resize(n * self.max_m0, 0);
        self.len0.resize(n, 0);
        self.upper.resize(n, Vec::new());
    }

    fn init_node(&mut self, id: u32, level: usize) {
        let i = id as usize;
        self.levels[i] = level as u8;
        self.len0[i] = 0;
        self.upper[i] = vec![0; level * (1 + self.max_m)];
    }

    pub(crate) fn cap(&self, level: usize) -> usize {
        if level == 0 { self.max_m0 } else { self.max_m }
    }

    pub(crate) fn level(&self, id: u32) -> usize {
        self.levels[id as usize] as usize
    }

    pub(crate) fn neighbors(&self, id: u32, level: usize) -> &[u32] {
        let i = id as usize;
        if level == 0 {
            let start = i * self.max_m0;
            &self.links0[start..start + self.len0[i] as usize]
        } else {
            let start = (level - 1) * (1 + self.max_m);
            let block = &self.upper[i][start..start + 1 + self.max_m];
            &block[1..1 + block[0] as usize]
        }
    }

    fn set_neighbors(&mut self, id: u32, level: usize, ids: &[u32]) {
        assert!(ids.len() <= self.cap(level));
        let i = id as usize;
        if level == 0 {
            let start = i * self.max_m0;
            self.links0[start..start + ids.len()].copy_from_slice(ids);
            self.len0[i] = ids.len() as u32;
        } else {
            let start = (level - 1) * (1 + self.max_m);
            let block = &mut self.upper[i][start..start + 1 + self.max_m];
            block[0] = ids.len() as u32;
            block[1..1 + ids.len()].copy_from_slice(ids);
        }
    }

    fn push_neighbor(&mut self, id: u32, level: usize, nb: u32) {
        let i = id as usize;
        if level == 0 {
            let len = self.len0[i] as usize;
            self.links0[i * self.max_m0 + len] = nb;
            self.len0[i] += 1;
        } else {
            let start = (level - 1) * (1 + self.max_m);
            let block = &mut self.upper[i][start..start + 1 + self.max_m];
            let len = block[0] as usize;
            block[1 + len] = nb;
            block[0] += 1;
        }
    }

    pub(crate) fn entry(&self) -> Option<u32> {
        self.entry
    }

    pub(crate) fn max_level(&self) -> usize {
        self.max_level
    }
}

pub struct Hnsw {
    store: Store,
    graph: Graph,
    params: Params,
    ml: f64,
    rng: StdRng,
    visited: VisitedSet,
    simd_level: Level,
}

impl Hnsw {
    pub fn new(dim: usize, space: Space, params: Params) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert!(params.m >= 2, "M must be at least 2");
        assert!(params.ef_construction >= 1, "ef_construction must be positive");
        Self {
            store: Store { dim, space, data: Vec::new() },
            graph: Graph::new(params.m, 2 * params.m),
            ml: 1.0 / (params.m as f64).ln(),
            rng: StdRng::seed_from_u64(params.seed),
            visited: VisitedSet::new(0),
            simd_level: Level::new(),
            params,
        }
    }

    pub fn dim(&self) -> usize {
        self.store.dim
    }

    pub fn len(&self) -> usize {
        self.store.data.len() / self.store.dim
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds `vectors` (row-major, `n * dim` floats). Ids continue from the
    /// current length. With `post`, the first add on an empty index also runs
    /// the reverse-order rebuild and level-0 merge.
    pub fn add(&mut self, vectors: &[f32]) {
        let dim = self.store.dim;
        assert!(vectors.len() % dim == 0, "vectors length is not a multiple of dim");
        let first_id = self.len();
        let n_new = vectors.len() / dim;
        let n = first_id + n_new;
        assert!(n <= u32::MAX as usize, "too many vectors");
        if n_new == 0 {
            return;
        }
        self.store.data.extend_from_slice(vectors);
        if self.store.space == Space::Cosine {
            for v in self.store.data[first_id * dim..].chunks_exact_mut(dim) {
                normalize(v);
            }
        }
        self.graph.reserve_nodes(n);
        self.visited.ensure_capacity(n);

        let post = self.params.post && first_id == 0;
        dispatch!(self.simd_level, simd => {
            for id in first_id..n {
                self.insert_node(simd, id as u32);
            }
            if post {
                self.post_merge(simd);
            }
        });
    }

    /// The `k` nearest neighbors of `query`, nearest first, as `(id, distance)`.
    /// L2 distances are squared. Searches level 0 with `max(ef_search, k)`.
    pub fn knn_query(&mut self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.store.dim, "query has the wrong dimension");
        let Some(entry) = self.graph.entry else { return Vec::new() };
        let mut q = query.to_vec();
        if self.store.space == Space::Cosine {
            normalize(&mut q);
        }
        let ef = ef_search.max(k).max(1);
        dispatch!(self.simd_level, simd => {
            let mut ep = Cand { dist: self.store.dist(simd, &q, entry), id: entry };
            for level in (1..=self.graph.max_level).rev() {
                ep = greedy(simd, &self.store, &self.graph, &q, ep, level);
            }
            let found = search_layer(simd, &self.store, &self.graph, &mut self.visited, &q, ep, ef, 0);
            found.into_iter().take(k).map(|c| (c.id, c.dist)).collect()
        })
    }

    fn random_level(&mut self) -> usize {
        // u in (0, 1], so ln(u) is finite.
        let u = 1.0 - self.rng.random::<f64>();
        ((-u.ln() * self.ml) as usize).min(u8::MAX as usize)
    }

    #[inline(always)]
    fn insert_node<S: Simd>(&mut self, simd: S, id: u32) {
        let level = self.random_level();
        self.graph.init_node(id, level);
        let Some(entry) = self.graph.entry else {
            self.graph.entry = Some(id);
            self.graph.max_level = level;
            return;
        };
        let store = &self.store;
        let graph = &mut self.graph;
        let q = store.vector(id);
        let max_level = graph.max_level;

        let mut ep = Cand { dist: store.dist(simd, q, entry), id: entry };
        for l in (level + 1..=max_level).rev() {
            ep = greedy(simd, store, graph, q, ep, l);
        }
        let mut selected = Vec::with_capacity(self.params.m);
        for l in (0..=level.min(max_level)).rev() {
            let found =
                search_layer(simd, store, graph, &mut self.visited, q, ep, self.params.ef_construction, l);
            select_neighbors(simd, store, &found, self.params.m, false, &mut selected);
            // NMSLIB links farthest first; the closest selected seeds the next level.
            for c in selected.iter().rev() {
                add_link(simd, store, graph, c.id, id, l);
                add_link(simd, store, graph, id, c.id, l);
            }
            ep = selected[0];
        }
        if level > max_level {
            graph.entry = Some(id);
            graph.max_level = level;
        }
    }

    /// NMSLIB `post=2`: rebuild in reverse order (id 0 first, then n-1 down to
    /// 1), then give every node except 0 the union of its two level-0 lists,
    /// pruned by the refilling heuristic to `max_m0`. Upper levels and the
    /// entry point come from the second build.
    #[inline(always)]
    fn post_merge<S: Simd>(&mut self, simd: S) {
        let n = self.len();
        let first = std::mem::replace(&mut self.graph, Graph::new(self.params.m, 2 * self.params.m));
        self.graph.reserve_nodes(n);
        self.insert_node(simd, 0);
        for id in (1..n as u32).rev() {
            self.insert_node(simd, id);
        }

        let store = &self.store;
        let second = &self.graph;
        let max_m0 = second.max_m0;
        let merged: Vec<Option<Vec<u32>>> = simd.vectorize(|| {
            let mut cands = Vec::new();
            let mut kept = Vec::with_capacity(max_m0);
            (0..n as u32)
                .map(|id| {
                    if id == 0 {
                        return None;
                    }
                    let q = store.vector(id);
                    cands.clear();
                    for &nb in first.neighbors(id, 0).iter().chain(second.neighbors(id, 0)) {
                        cands.push(Cand { dist: store.dist(simd, q, nb), id: nb });
                    }
                    cands.sort_unstable();
                    cands.dedup_by_key(|c| c.id);
                    select_neighbors(simd, store, &cands, max_m0, true, &mut kept);
                    Some(kept.iter().map(|c| c.id).collect())
                })
                .collect()
        });
        for (id, list) in merged.into_iter().enumerate() {
            if let Some(list) = list {
                self.graph.set_neighbors(id as u32, 0, &list);
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Level sizes and mean degrees, for comparing graph shape with NMSLIB.
    pub fn stats(&self) -> GraphStats {
        let g = &self.graph;
        let levels = if g.entry().is_some() { g.max_level() + 1 } else { 0 };
        let mut nodes_per_level = vec![0usize; levels];
        let mut edges_per_level = vec![0usize; levels];
        for id in 0..self.len() as u32 {
            for level in 0..=g.level(id) {
                nodes_per_level[level] += 1;
                edges_per_level[level] += g.neighbors(id, level).len();
            }
        }
        let mean_degree = nodes_per_level
            .iter()
            .zip(&edges_per_level)
            .map(|(&n, &e)| if n == 0 { 0.0 } else { e as f64 / n as f64 })
            .collect();
        GraphStats { nodes_per_level, mean_degree }
    }
}

#[derive(Clone, Debug)]
pub struct GraphStats {
    /// Nodes present at each level (a node at level `l` is in levels `0..=l`).
    pub nodes_per_level: Vec<usize>,
    pub mean_degree: Vec<f64>,
}

/// Greedy walk at one level: move to any strictly closer neighbor until none.
fn greedy<S: Simd>(simd: S, store: &Store, graph: &Graph, q: &[f32], ep: Cand, level: usize) -> Cand {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut cur = ep;
            loop {
                let mut changed = false;
                for &nb in graph.neighbors(cur.id, level) {
                    let d = store.dist(simd, q, nb);
                    if d < cur.dist {
                        cur = Cand { dist: d, id: nb };
                        changed = true;
                    }
                }
                if !changed {
                    return cur;
                }
            }
        },
    )
}

/// Beam search at one level from a single entry point. Returns up to `ef`
/// nodes, nearest first.
#[allow(clippy::too_many_arguments)]
fn search_layer<S: Simd>(
    simd: S,
    store: &Store,
    graph: &Graph,
    visited: &mut VisitedSet,
    q: &[f32],
    ep: Cand,
    ef: usize,
    level: usize,
) -> Vec<Cand> {
    simd.vectorize(
        #[inline(always)]
        || {
            visited.clear();
            visited.insert(ep.id);
            let mut candidates = BinaryHeap::from([Reverse(ep)]);
            let mut results = BinaryHeap::from([ep]);
            while let Some(Reverse(c)) = candidates.pop() {
                if c.dist > results.peek().unwrap().dist {
                    break;
                }
                for &nb in graph.neighbors(c.id, level) {
                    if !visited.insert(nb) {
                        continue;
                    }
                    let d = store.dist(simd, q, nb);
                    if results.len() < ef || d < results.peek().unwrap().dist {
                        let cand = Cand { dist: d, id: nb };
                        candidates.push(Reverse(cand));
                        results.push(cand);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
            results.into_sorted_vec()
        },
    )
}

/// The HNSW neighbor-selection heuristic (paper Algorithm 4, as in NMSLIB).
///
/// `cands` must be sorted nearest first by distance to the base point. A
/// candidate is kept unless it is strictly closer to an already kept one than
/// to the base. Stops at `nn` kept. With `refill`, rejected candidates fill
/// the remaining slots nearest first (NMSLIB heuristic 1); without it they
/// are dropped (heuristic 2). Fewer than `nn` candidates are kept unchanged.
fn select_neighbors<S: Simd>(
    simd: S,
    store: &Store,
    cands: &[Cand],
    nn: usize,
    refill: bool,
    out: &mut Vec<Cand>,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            out.clear();
            if cands.len() < nn {
                out.extend_from_slice(cands);
                return;
            }
            let mut rejected = Vec::new();
            for &c in cands {
                if out.len() >= nn {
                    break;
                }
                let v = store.vector(c.id);
                if out.iter().all(|k| store.dist(simd, v, k.id) >= c.dist) {
                    out.push(c);
                } else if refill {
                    rejected.push(c);
                }
            }
            for c in rejected {
                if out.len() >= nn {
                    break;
                }
                out.push(c);
            }
        },
    )
}

/// Adds `nb` to `id`'s list at `level`. On overflow, re-prunes the whole list
/// with the non-refilling heuristic, as NMSLIB's `addFriendlevel` does.
fn add_link<S: Simd>(simd: S, store: &Store, graph: &mut Graph, id: u32, nb: u32, level: usize) {
    simd.vectorize(
        #[inline(always)]
        || {
            let current = graph.neighbors(id, level);
            if current.contains(&nb) {
                return;
            }
            let cap = graph.cap(level);
            if current.len() < cap {
                graph.push_neighbor(id, level, nb);
                return;
            }
            let base = store.vector(id);
            let mut cands: Vec<Cand> = current
                .iter()
                .chain([&nb])
                .map(|&x| Cand { dist: store.dist(simd, base, x), id: x })
                .collect();
            cands.sort_unstable();
            let mut kept = Vec::with_capacity(cap);
            select_neighbors(simd, store, &cands, cap, false, &mut kept);
            let ids: Vec<u32> = kept.iter().map(|c| c.id).collect();
            graph.set_neighbors(id, level, &ids);
        },
    )
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests;
