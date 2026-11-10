//! HNSW index following NMSLIB's `hnsw` method, with parallel build and query.
//!
//! See NOTES.md for the NMSLIB behaviors matched here. In short: insert
//! selects `M` neighbors at every level with the pruning heuristic (no refill),
//! an overflowing list is re-pruned the same way, and the optional `post`
//! step (NMSLIB `post=2`) rebuilds the graph in reverse insertion order and
//! merges the two level-0 lists with the refilling heuristic.
//!
//! Concurrency, all in safe Rust: neighbor lists are `AtomicU32` slots with a
//! length slot. Writers change a node's list only under that node's mutex;
//! searches read without locking. A search racing with a writer can see an
//! old or partly rewritten list, but every value it reads is the id of an
//! inserted node, so the search stays valid. The entry point and max level sit
//! behind one mutex, held for a whole insert only by a node that raises the
//! max level (as in NMSLIB). Levels are drawn before inserting, so storage for
//! every node exists up front. With `threads = 1` the build is sequential and
//! deterministic for a given seed.
//!
//! The SIMD level is chosen once per insert or query, and every function that
//! computes distances runs inside it through `Simd::vectorize`.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Acquire, Ordering::Relaxed, Ordering::Release};
use std::sync::{Arc, Mutex};

use fearless_simd::{Level, Simd, dispatch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

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

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    node: Option<u32>,
    max_level: usize,
}

/// Neighbor lists. Each list is a block of `1 + cap` atomic slots: the length,
/// then the ids. Level 0 is one flat array with `1 + max_m0` slots per node;
/// upper levels share another flat array, `1 + max_m` slots per level, at
/// offsets fixed when the node's level is drawn.
pub(crate) struct Graph {
    max_m: usize,
    max_m0: usize,
    levels: Vec<u8>,
    links0: Vec<AtomicU32>,
    upper_start: Vec<usize>,
    upper: Vec<AtomicU32>,
    locks: Vec<Mutex<()>>,
    /// Entry point and max level packed as `max_level << 32 | node`, or
    /// `NO_ENTRY`. Inserts read it without locking; changing it requires
    /// `raise_lock`.
    entry: AtomicU64,
    raise_lock: Mutex<()>,
}

const NO_ENTRY: u64 = u64::MAX;

impl Graph {
    pub(crate) fn new(max_m: usize, max_m0: usize) -> Self {
        Self {
            max_m,
            max_m0,
            levels: Vec::new(),
            links0: Vec::new(),
            upper_start: vec![0],
            upper: Vec::new(),
            locks: Vec::new(),
            entry: AtomicU64::new(NO_ENTRY),
            raise_lock: Mutex::new(()),
        }
    }

    /// Adds nodes with the given levels and empty lists.
    pub(crate) fn grow(&mut self, levels: &[u8]) {
        for &level in levels {
            let start = *self.upper_start.last().unwrap();
            self.upper_start.push(start + level as usize * (1 + self.max_m));
            self.levels.push(level);
            self.locks.push(Mutex::new(()));
        }
        let n = self.levels.len();
        self.links0.resize_with(n * (1 + self.max_m0), || AtomicU32::new(0));
        self.upper.resize_with(*self.upper_start.last().unwrap(), || AtomicU32::new(0));
    }

    fn block(&self, id: u32, level: usize) -> &[AtomicU32] {
        let i = id as usize;
        if level == 0 {
            let w = 1 + self.max_m0;
            &self.links0[i * w..(i + 1) * w]
        } else {
            let w = 1 + self.max_m;
            let start = self.upper_start[i] + (level - 1) * w;
            &self.upper[start..start + w]
        }
    }

    pub(crate) fn cap(&self, level: usize) -> usize {
        if level == 0 { self.max_m0 } else { self.max_m }
    }

    pub(crate) fn level(&self, id: u32) -> usize {
        self.levels[id as usize] as usize
    }

    pub(crate) fn neighbors(&self, id: u32, level: usize) -> impl Iterator<Item = u32> + '_ {
        let block = self.block(id, level);
        let len = (block[0].load(Acquire) as usize).min(block.len() - 1);
        block[1..1 + len].iter().map(|slot| slot.load(Relaxed))
    }

    #[cfg(test)]
    pub(crate) fn neighbor_vec(&self, id: u32, level: usize) -> Vec<u32> {
        self.neighbors(id, level).collect()
    }

    /// Replaces a list. Callers hold the node's lock, or have exclusive access.
    pub(crate) fn set_neighbors(&self, id: u32, level: usize, ids: &[u32]) {
        let block = self.block(id, level);
        assert!(ids.len() < block.len(), "list over capacity");
        for (slot, &x) in block[1..].iter().zip(ids) {
            slot.store(x, Relaxed);
        }
        block[0].store(ids.len() as u32, Release);
    }

    /// Appends to a list with room left. Same locking rule as `set_neighbors`.
    pub(crate) fn push_neighbor(&self, id: u32, level: usize, nb: u32) {
        let block = self.block(id, level);
        let len = block[0].load(Relaxed) as usize;
        block[1 + len].store(nb, Relaxed);
        block[0].store(len as u32 + 1, Release);
    }

    fn entry_state(&self) -> Entry {
        match self.entry.load(Acquire) {
            NO_ENTRY => Entry::default(),
            packed => Entry { node: Some(packed as u32), max_level: (packed >> 32) as usize },
        }
    }

    /// Callers hold `raise_lock`, or have exclusive access.
    pub(crate) fn set_entry(&self, node: u32, max_level: usize) {
        self.entry.store((max_level as u64) << 32 | node as u64, Release);
    }

    pub(crate) fn entry(&self) -> Option<u32> {
        self.entry_state().node
    }

    pub(crate) fn max_level(&self) -> usize {
        self.entry_state().max_level
    }
}

/// Buffers for `search_layer`.
#[derive(Default)]
struct SearchBuffers {
    visited: VisitedSet,
    candidates: BinaryHeap<Reverse<Cand>>,
    results: BinaryHeap<Cand>,
}

/// Buffers for neighbor selection and linking.
#[derive(Default)]
struct LinkBuffers {
    ids: Vec<u32>,
    cands: Vec<Cand>,
    kept: Vec<Cand>,
    rejected: Vec<Cand>,
}

/// Per-thread working memory, reused across inserts and queries so the hot
/// paths do not allocate. With many threads, per-call allocation serializes
/// on the allocator's locks (it dominated parallel build time).
#[derive(Default)]
struct Scratch {
    search: SearchBuffers,
    found: Vec<Cand>,
    selected: Vec<Cand>,
    link: LinkBuffers,
    query: Vec<f32>,
}

/// Scratch buffers shared by builds and queries, checked out per thread.
#[derive(Default)]
struct ScratchPool {
    free: Mutex<Vec<Scratch>>,
}

struct PooledScratch<'a> {
    pool: &'a ScratchPool,
    scratch: Option<Scratch>,
}

impl ScratchPool {
    fn get(&self, n: usize) -> PooledScratch<'_> {
        let mut scratch = self.free.lock().unwrap().pop().unwrap_or_default();
        scratch.search.visited.ensure_capacity(n);
        PooledScratch { pool: self, scratch: Some(scratch) }
    }
}

impl Deref for PooledScratch<'_> {
    type Target = Scratch;
    fn deref(&self) -> &Scratch {
        self.scratch.as_ref().unwrap()
    }
}

impl DerefMut for PooledScratch<'_> {
    fn deref_mut(&mut self) -> &mut Scratch {
        self.scratch.as_mut().unwrap()
    }
}

impl Drop for PooledScratch<'_> {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            self.pool.free.lock().unwrap().push(scratch);
        }
    }
}

pub struct Hnsw {
    store: Store,
    graph: Graph,
    params: Params,
    ml: f64,
    rng: StdRng,
    scratch: ScratchPool,
    simd_level: Level,
    thread_pool: Mutex<Option<Arc<rayon::ThreadPool>>>,
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
            scratch: ScratchPool::default(),
            simd_level: Level::new(),
            thread_pool: Mutex::new(None),
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

    /// Adds `vectors` (row-major, `n * dim` floats). See [`Hnsw::add_vec`].
    pub fn add(&mut self, vectors: &[f32], threads: usize) {
        self.add_vec(vectors.to_vec(), threads);
    }

    /// Adds `vectors` (row-major, `n * dim` floats), taking ownership so the
    /// first batch is stored without another copy. Ids continue from the
    /// current length. `threads`: 1 is the sequential, deterministic build;
    /// 0 uses all cores; any other value uses that many threads. With `post`,
    /// the first add on an empty index also runs the reverse-order rebuild and
    /// level-0 merge.
    pub fn add_vec(&mut self, vectors: Vec<f32>, threads: usize) {
        let dim = self.store.dim;
        assert!(vectors.len() % dim == 0, "vectors length is not a multiple of dim");
        let first_id = self.len();
        let n = first_id + vectors.len() / dim;
        assert!(n <= u32::MAX as usize, "too many vectors");
        if n == first_id {
            return;
        }
        if self.store.data.is_empty() {
            self.store.data = vectors;
        } else {
            self.store.data.extend_from_slice(&vectors);
        }
        if self.store.space == Space::Cosine {
            for v in self.store.data[first_id * dim..].chunks_exact_mut(dim) {
                normalize(v);
            }
        }

        let levels: Vec<u8> = (first_id..n).map(|_| self.random_level()).collect();
        self.graph.grow(&levels);
        let order: Vec<u32> = (first_id as u32..n as u32).collect();
        self.insert_all(&order, threads);

        if self.params.post && first_id == 0 {
            let first = std::mem::replace(&mut self.graph, Graph::new(self.params.m, 2 * self.params.m));
            let levels: Vec<u8> = (0..n).map(|_| self.random_level()).collect();
            self.graph.grow(&levels);
            let reverse: Vec<u32> = std::iter::once(0).chain((1..n as u32).rev()).collect();
            self.insert_all(&reverse, threads);
            self.merge_level0(&first, threads);
        }
    }

    /// The `k` nearest neighbors of `query`, nearest first, as `(id, distance)`.
    /// L2 distances are squared. Searches level 0 with `max(ef_search, k)`.
    pub fn knn_query(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.store.dim, "query has the wrong dimension");
        let mut scratch = self.scratch.get(self.len());
        self.search(query, k, ef_search, &mut scratch).iter().map(|c| (c.id, c.dist)).collect()
    }

    /// Queries every row of `queries` and writes the results row by row into
    /// `labels` and `distances` (`k` per query, nearest first, -1 / inf where
    /// fewer than `k` are found). `threads` as in [`Hnsw::add_vec`].
    pub fn knn_query_batch(
        &self,
        queries: &[f32],
        k: usize,
        ef_search: usize,
        threads: usize,
        labels: &mut [i64],
        distances: &mut [f32],
    ) {
        let dim = self.store.dim;
        assert!(k > 0, "k must be positive");
        assert!(queries.len() % dim == 0, "queries length is not a multiple of dim");
        let nq = queries.len() / dim;
        assert!(labels.len() == nq * k && distances.len() == nq * k, "output buffers have the wrong size");
        let n = self.len();
        let one = |scratch: &mut Scratch, q: &[f32], labels: &mut [i64], dists: &mut [f32]| {
            labels.fill(-1);
            dists.fill(f32::INFINITY);
            for (j, c) in self.search(q, k, ef_search, scratch).iter().enumerate() {
                labels[j] = c.id as i64;
                dists[j] = c.dist;
            }
        };
        if threads == 1 {
            let mut scratch = self.scratch.get(n);
            for ((q, l), d) in queries.chunks_exact(dim).zip(labels.chunks_mut(k)).zip(distances.chunks_mut(k)) {
                one(&mut scratch, q, l, d);
            }
            return;
        }
        self.run_parallel(threads, || {
            queries
                .par_chunks_exact(dim)
                .zip(labels.par_chunks_mut(k))
                .zip(distances.par_chunks_mut(k))
                .for_each_init(|| self.scratch.get(n), |scratch, ((q, l), d)| one(scratch, q, l, d));
        });
    }

    /// The `k` nearest of `query`, nearest first, in `scratch.found`.
    fn search<'s>(&self, query: &[f32], k: usize, ef_search: usize, scratch: &'s mut Scratch) -> &'s [Cand] {
        let entry = self.graph.entry_state();
        scratch.found.clear();
        let Some(ep_id) = entry.node else { return &scratch.found };
        let q = if self.store.space == Space::Cosine {
            scratch.query.clear();
            scratch.query.extend_from_slice(query);
            normalize(&mut scratch.query);
            &scratch.query[..]
        } else {
            query
        };
        let ef = ef_search.max(k).max(1);
        let (store, graph) = (&self.store, &self.graph);
        dispatch!(self.simd_level, simd => {
            let mut ep = Cand { dist: store.dist(simd, q, ep_id), id: ep_id };
            for level in (1..=entry.max_level).rev() {
                ep = greedy(simd, store, graph, q, ep, level);
            }
            search_layer(simd, store, graph, &mut scratch.search, q, ep, ef, 0, &mut scratch.found);
        });
        scratch.found.truncate(k);
        &scratch.found
    }

    fn random_level(&mut self) -> u8 {
        // u in (0, 1], so ln(u) is finite.
        let u = 1.0 - self.rng.random::<f64>();
        ((-u.ln() * self.ml) as usize).min(u8::MAX as usize) as u8
    }

    /// Runs `f` on `threads` threads (0 = rayon's global pool, all cores).
    fn run_parallel<R: Send>(&self, threads: usize, f: impl FnOnce() -> R + Send) -> R {
        if threads == 0 {
            return f();
        }
        let pool = {
            let mut cached = self.thread_pool.lock().unwrap();
            match cached.as_ref() {
                Some(p) if p.current_num_threads() == threads => p.clone(),
                _ => {
                    let p = Arc::new(
                        rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool"),
                    );
                    *cached = Some(p.clone());
                    p
                }
            }
        };
        pool.install(f)
    }

    /// Inserts `order` (ids whose vectors and levels already exist). The first
    /// node of an empty graph goes in alone, as in NMSLIB, then the rest in
    /// parallel unless `threads == 1`.
    fn insert_all(&self, order: &[u32], threads: usize) {
        let n = self.len();
        let (store, graph, params, level) = (&self.store, &self.graph, &self.params, self.simd_level);
        let insert = |scratch: &mut Scratch, id: u32| {
            dispatch!(level, simd => insert_node(simd, store, graph, params, scratch, id));
        };
        let mut rest = order;
        if threads == 1 || graph.entry().is_none() {
            let mut scratch = self.scratch.get(n);
            let sequential = if threads == 1 { order.len() } else { 1.min(order.len()) };
            for &id in &order[..sequential] {
                insert(&mut scratch, id);
            }
            rest = &order[sequential..];
        }
        if !rest.is_empty() {
            // Workers take the next id from a shared counter, as NMSLIB's
            // ParallelFor does, so the graph grows in (nearly) insertion order.
            // Splitting `rest` into per-thread chunks instead grows many parts
            // of the order at once, which breaks the forward/reverse contrast
            // the post merge relies on and lowers recall.
            let next = AtomicUsize::new(0);
            self.run_parallel(threads, || {
                rayon::broadcast(|_| {
                    let mut scratch = self.scratch.get(n);
                    loop {
                        let i = next.fetch_add(1, Relaxed);
                        let Some(&id) = rest.get(i) else { break };
                        insert(&mut scratch, id);
                    }
                });
            });
        }
    }

    /// NMSLIB `post=2` merge: every node except 0 gets the union of its
    /// level-0 lists in `first` and in the current (reverse-order) graph,
    /// pruned by the refilling heuristic to `max_m0`. Each node reads and
    /// writes only its own lists, so nodes merge independently.
    fn merge_level0(&self, first: &Graph, threads: usize) {
        let (store, graph, level) = (&self.store, &self.graph, self.simd_level);
        let max_m0 = graph.max_m0;
        let merge = |buf: &mut LinkBuffers, id: u32| {
            dispatch!(level, simd => {
                let q = store.vector(id);
                buf.cands.clear();
                buf.cands.extend(
                    first
                        .neighbors(id, 0)
                        .chain(graph.neighbors(id, 0))
                        .map(|nb| Cand { dist: store.dist(simd, q, nb), id: nb }),
                );
                buf.cands.sort_unstable();
                buf.cands.dedup_by_key(|c| c.id);
                select_neighbors(simd, store, &buf.cands, max_m0, true, &mut buf.kept, &mut buf.rejected);
                buf.ids.clear();
                buf.ids.extend(buf.kept.iter().map(|c| c.id));
                graph.set_neighbors(id, 0, &buf.ids);
            });
        };
        let n = self.len() as u32;
        if threads == 1 {
            let mut buf = LinkBuffers::default();
            (1..n).for_each(|id| merge(&mut buf, id));
        } else {
            self.run_parallel(threads, || {
                (1..n).into_par_iter().for_each_init(LinkBuffers::default, |buf, id| merge(buf, id));
            });
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
                edges_per_level[level] += g.neighbors(id, level).count();
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

/// Inserts node `id`, whose vector and level already exist in the store and
/// graph. Safe to run concurrently for different ids.
#[inline(always)]
fn insert_node<S: Simd>(
    simd: S,
    store: &Store,
    graph: &Graph,
    params: &Params,
    scratch: &mut Scratch,
    id: u32,
) {
    let level = graph.level(id);
    let mut entry = graph.entry_state();
    // As in NMSLIB, only a node that may raise the max level takes the lock,
    // and keeps it until it is linked in, so raises happen one at a time.
    // Other inserts read the entry point without waiting.
    let held = if entry.node.is_none() || level > entry.max_level {
        let guard = graph.raise_lock.lock().unwrap();
        entry = graph.entry_state();
        (entry.node.is_none() || level > entry.max_level).then_some(guard)
    } else {
        None
    };
    let Some(entry_id) = entry.node else {
        graph.set_entry(id, level);
        return;
    };
    let max_level = entry.max_level;

    let q = store.vector(id);
    let mut ep = Cand { dist: store.dist(simd, q, entry_id), id: entry_id };
    for l in (level + 1..=max_level).rev() {
        ep = greedy(simd, store, graph, q, ep, l);
    }
    let Scratch { search, found, selected, link, .. } = scratch;
    for l in (0..=level.min(max_level)).rev() {
        search_layer(simd, store, graph, search, q, ep, params.ef_construction, l, found);
        select_neighbors(simd, store, found, params.m, false, selected, &mut link.rejected);
        // NMSLIB links farthest first; the closest selected seeds the next level.
        for c in selected.iter().rev() {
            add_link(simd, store, graph, c.id, id, l, link);
            add_link(simd, store, graph, id, c.id, l, link);
        }
        ep = selected[0];
    }
    if held.is_some() {
        graph.set_entry(id, level);
    }
}

/// Greedy walk at one level: move to any strictly closer neighbor until none.
fn greedy<S: Simd>(simd: S, store: &Store, graph: &Graph, q: &[f32], ep: Cand, level: usize) -> Cand {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut cur = ep;
            loop {
                let mut changed = false;
                for nb in graph.neighbors(cur.id, level) {
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

/// Beam search at one level from a single entry point. Writes up to `ef`
/// nodes, nearest first, to `out`.
#[allow(clippy::too_many_arguments)]
fn search_layer<S: Simd>(
    simd: S,
    store: &Store,
    graph: &Graph,
    buf: &mut SearchBuffers,
    q: &[f32],
    ep: Cand,
    ef: usize,
    level: usize,
    out: &mut Vec<Cand>,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let SearchBuffers { visited, candidates, results } = buf;
            visited.clear();
            visited.insert(ep.id);
            candidates.clear();
            results.clear();
            candidates.push(Reverse(ep));
            results.push(ep);
            while let Some(Reverse(c)) = candidates.pop() {
                if c.dist > results.peek().unwrap().dist {
                    break;
                }
                for nb in graph.neighbors(c.id, level) {
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
            out.clear();
            out.extend(results.drain());
            out.sort_unstable();
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
    rejected: &mut Vec<Cand>,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            out.clear();
            if cands.len() < nn {
                out.extend_from_slice(cands);
                return;
            }
            rejected.clear();
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
            for &c in rejected.iter() {
                if out.len() >= nn {
                    break;
                }
                out.push(c);
            }
        },
    )
}

/// Adds `nb` to `id`'s list at `level` under `id`'s lock. On overflow,
/// re-prunes the whole list with the non-refilling heuristic, as NMSLIB's
/// `addFriendlevel` does.
#[allow(clippy::too_many_arguments)]
fn add_link<S: Simd>(
    simd: S,
    store: &Store,
    graph: &Graph,
    id: u32,
    nb: u32,
    level: usize,
    buf: &mut LinkBuffers,
) {
    simd.vectorize(
        #[inline(always)]
        || {
            let LinkBuffers { ids, cands, kept, rejected } = buf;
            let _lock = graph.locks[id as usize].lock().unwrap();
            ids.clear();
            ids.extend(graph.neighbors(id, level));
            if ids.contains(&nb) {
                return;
            }
            let cap = graph.cap(level);
            if ids.len() < cap {
                graph.push_neighbor(id, level, nb);
                return;
            }
            let base = store.vector(id);
            cands.clear();
            cands.extend(ids.iter().chain([&nb]).map(|&x| Cand { dist: store.dist(simd, base, x), id: x }));
            cands.sort_unstable();
            select_neighbors(simd, store, cands, cap, false, kept, rejected);
            ids.clear();
            ids.extend(kept.iter().map(|c| c.id));
            graph.set_neighbors(id, level, ids);
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
