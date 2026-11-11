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
use std::sync::atomic::{
    AtomicU32, AtomicU64, AtomicUsize, Ordering::Acquire, Ordering::Relaxed, Ordering::Release,
};
use std::sync::{Arc, Mutex};

use fearless_simd::{Level, Simd, dispatch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

use std::path::Path;

use crate::frozen::{self, Frozen, LoadError, Meta};
use crate::simd::{dot_in, l2_sq_in, prefetch_in};
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
        Self {
            m: 16,
            ef_construction: 200,
            post: true,
            seed: 0,
        }
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
        self.dist
            .total_cmp(&other.dist)
            .then(self.id.cmp(&other.id))
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
pub(crate) struct Entry {
    pub(crate) node: Option<u32>,
    pub(crate) max_level: usize,
}

/// Read access to vectors and neighbor lists. Implemented by the build graph
/// and by a memory-mapped saved index (`frozen::FrozenView`), so both use one
/// search implementation.
pub(crate) trait View: Sync {
    fn space(&self) -> Space;
    fn len(&self) -> usize;
    fn entry(&self) -> Entry;
    fn level(&self, id: u32) -> usize;
    fn vector(&self, id: u32) -> &[f32];
    fn neighbors(&self, id: u32, level: usize) -> impl Iterator<Item = u32> + '_;
    /// Starts loading `id`'s list at `level` into cache.
    fn prefetch_list<S: Simd>(&self, simd: S, id: u32, level: usize);

    #[inline(always)]
    fn dist<S: Simd>(&self, simd: S, q: &[f32], id: u32) -> f32 {
        let v = self.vector(id);
        match self.space() {
            Space::L2 => l2_sq_in(simd, q, v),
            Space::Cosine => 1.0 - dot_in(simd, q, v),
        }
    }
}

/// The build-time index: vectors in `Store`, lists in the concurrent `Graph`.
struct BuildView<'a> {
    store: &'a Store,
    graph: &'a Graph,
}

impl View for BuildView<'_> {
    fn space(&self) -> Space {
        self.store.space
    }

    fn len(&self) -> usize {
        self.graph.levels.len()
    }

    fn entry(&self) -> Entry {
        self.graph.entry_state()
    }

    fn level(&self, id: u32) -> usize {
        self.graph.level(id)
    }

    #[inline(always)]
    fn vector(&self, id: u32) -> &[f32] {
        self.store.vector(id)
    }

    #[inline(always)]
    fn neighbors(&self, id: u32, level: usize) -> impl Iterator<Item = u32> + '_ {
        self.graph.neighbors(id, level)
    }

    #[inline(always)]
    fn prefetch_list<S: Simd>(&self, simd: S, id: u32, level: usize) {
        prefetch_in(simd, self.graph.list_memory(id, level));
    }
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
            self.upper_start
                .push(start + level as usize * (1 + self.max_m));
            self.levels.push(level);
            self.locks.push(Mutex::new(()));
        }
        let n = self.levels.len();
        self.links0
            .resize_with(n * (1 + self.max_m0), || AtomicU32::new(0));
        self.upper
            .resize_with(*self.upper_start.last().unwrap(), || AtomicU32::new(0));
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

    /// The memory holding a node's list, for prefetching.
    pub(crate) fn list_memory(&self, id: u32, level: usize) -> &[AtomicU32] {
        self.block(id, level)
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
            packed => Entry {
                node: Some(packed as u32),
                max_level: (packed >> 32) as usize,
            },
        }
    }

    /// Callers hold `raise_lock`, or have exclusive access.
    pub(crate) fn set_entry(&self, node: u32, max_level: usize) {
        self.entry
            .store((max_level as u64) << 32 | node as u64, Release);
    }

    pub(crate) fn entry(&self) -> Option<u32> {
        self.entry_state().node
    }

    #[cfg(test)]
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
    /// Unvisited neighbors of the node being expanded.
    fresh: Vec<u32>,
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

/// Work done by queries, with the `counters` feature.
#[cfg(feature = "counters")]
pub mod counters {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub static DISTANCES: AtomicU64 = AtomicU64::new(0);
    pub static EXPANSIONS: AtomicU64 = AtomicU64::new(0);
    pub fn take() -> (u64, u64) {
        (DISTANCES.swap(0, Relaxed), EXPANSIONS.swap(0, Relaxed))
    }
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
        PooledScratch {
            pool: self,
            scratch: Some(scratch),
        }
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

/// How an index holds its data. `Build` is the concurrent graph used while
/// adding; after each `add` the index is frozen into the interleaved,
/// cache-line-aligned file layout, in memory (see NOTES.md for the measured
/// effect). A loaded index is `Frozen` too, owned or memory-mapped
/// (read-only).
enum Repr {
    Build { store: Store, graph: Graph },
    Frozen(Frozen),
}

pub struct Hnsw {
    repr: Repr,
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
        assert!(
            params.ef_construction >= 1,
            "ef_construction must be positive"
        );
        Self {
            repr: Repr::Build {
                store: Store {
                    dim,
                    space,
                    data: Vec::new(),
                },
                graph: Graph::new(params.m, 2 * params.m),
            },
            ml: 1.0 / (params.m as f64).ln(),
            rng: StdRng::seed_from_u64(params.seed),
            scratch: ScratchPool::default(),
            simd_level: Level::new(),
            thread_pool: Mutex::new(None),
            params,
        }
    }

    pub fn dim(&self) -> usize {
        match &self.repr {
            Repr::Build { store, .. } => store.dim,
            Repr::Frozen(f) => f.header().dim,
        }
    }

    pub fn space(&self) -> Space {
        match &self.repr {
            Repr::Build { store, .. } => store.space,
            Repr::Frozen(f) => f.header().space,
        }
    }

    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Build { store, .. } => store.data.len() / store.dim,
            Repr::Frozen(f) => f.header().n,
        }
    }

    /// True for an index loaded with `mmap`; such an index cannot grow.
    pub fn is_read_only(&self) -> bool {
        matches!(&self.repr, Repr::Frozen(f) if f.is_mapped())
    }

    fn build_parts(&self) -> (&Store, &Graph) {
        match &self.repr {
            Repr::Build { store, graph } => (store, graph),
            Repr::Frozen(_) => unreachable!("building happens on a thawed index"),
        }
    }

    fn build_parts_mut(&mut self) -> (&mut Store, &mut Graph) {
        match &mut self.repr {
            Repr::Build { store, graph } => (store, graph),
            Repr::Frozen(_) => unreachable!("building happens on a thawed index"),
        }
    }

    fn meta(&self) -> Meta {
        Meta {
            dim: self.dim(),
            space: self.space(),
            m: self.params.m,
            ef_construction: self.params.ef_construction,
            post: self.params.post,
            seed: self.params.seed,
        }
    }

    /// Converts the build graph to the interleaved layout and drops it.
    /// Memory briefly holds both while the copy is made.
    fn freeze(&mut self) {
        if let Repr::Build { store, graph } = &self.repr {
            let frozen = Frozen::from_view(&BuildView { store, graph }, &self.meta());
            self.repr = Repr::Frozen(frozen);
        }
    }

    /// Converts a frozen in-memory index back to the build graph, to add to it.
    fn thaw(&mut self) {
        if let Repr::Frozen(f) = &self.repr {
            assert!(!f.is_mapped(), "index is memory-mapped and read-only");
            let v = f.view();
            let store = store_from_view(&v, self.dim());
            let graph = graph_from_view(&v, self.params.m);
            self.repr = Repr::Build { store, graph };
        }
    }

    /// Saves the index to `path` (see `frozen` for the format). Writes a
    /// temporary file in the same directory, then renames it over `path`.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let meta = self.meta();
        match &self.repr {
            Repr::Build { store, graph } => frozen::save(&BuildView { store, graph }, &meta, path),
            Repr::Frozen(f) => frozen::save(&f.view(), &meta, path),
        }
    }

    /// Loads a saved index. With `mmap`, the file is mapped read-only: pages
    /// are shared with every other process mapping it, nothing is read until
    /// queries touch it, and the index cannot grow. The file must not be
    /// modified while mapped. Without `mmap`, the file is read into memory
    /// and the index can grow. Adding to a loaded index draws levels from a
    /// fresh generator seeded with the saved seed.
    pub fn load(path: &Path, mmap: bool) -> Result<Hnsw, LoadError> {
        let frozen = if mmap {
            Frozen::map(path)?
        } else {
            Frozen::read(path)?
        };
        let h = frozen.header().clone();
        let params = Params {
            m: h.m,
            ef_construction: h.ef_construction,
            post: h.post,
            seed: h.seed,
        };
        let mut index = Hnsw::new(h.dim, h.space, params);
        index.repr = Repr::Frozen(frozen);
        Ok(index)
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
        assert!(!self.is_read_only(), "index is memory-mapped and read-only");
        let dim = self.dim();
        assert!(
            vectors.len() % dim == 0,
            "vectors length is not a multiple of dim"
        );
        let first_id = self.len();
        let n = first_id + vectors.len() / dim;
        assert!(n < u32::MAX as usize, "too many vectors");
        if n == first_id {
            return;
        }
        self.thaw();
        let levels: Vec<u8> = (first_id..n).map(|_| self.random_level()).collect();
        {
            let (store, graph) = self.build_parts_mut();
            if store.data.is_empty() {
                store.data = vectors;
            } else {
                store.data.extend_from_slice(&vectors);
            }
            if store.space == Space::Cosine {
                for v in store.data[first_id * dim..].chunks_exact_mut(dim) {
                    normalize(v);
                }
            }
            graph.grow(&levels);
        }
        let order: Vec<u32> = (first_id as u32..n as u32).collect();
        self.insert_all(&order, threads);

        if self.params.post && first_id == 0 {
            let levels: Vec<u8> = (0..n).map(|_| self.random_level()).collect();
            let (m, max_m0) = (self.params.m, 2 * self.params.m);
            let first = {
                let (_, graph) = self.build_parts_mut();
                let mut second = Graph::new(m, max_m0);
                second.grow(&levels);
                std::mem::replace(graph, second)
            };
            let reverse: Vec<u32> = std::iter::once(0).chain((1..n as u32).rev()).collect();
            self.insert_all(&reverse, threads);
            self.merge_level0(&first, threads);
        }
        self.freeze();
    }

    /// The `k` nearest neighbors of `query`, nearest first, as `(id, distance)`.
    /// L2 distances are squared. Searches level 0 with `max(ef_search, k)`.
    pub fn knn_query(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim(), "query has the wrong dimension");
        let mut scratch = self.scratch.get(self.len());
        self.search(query, k, ef_search, &mut scratch)
            .iter()
            .map(|c| (c.id, c.dist))
            .collect()
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
        let dim = self.dim();
        assert!(k > 0, "k must be positive");
        assert!(
            queries.len() % dim == 0,
            "queries length is not a multiple of dim"
        );
        let nq = queries.len() / dim;
        assert!(
            labels.len() == nq * k && distances.len() == nq * k,
            "output buffers have the wrong size"
        );
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
            for ((q, l), d) in queries
                .chunks_exact(dim)
                .zip(labels.chunks_mut(k))
                .zip(distances.chunks_mut(k))
            {
                one(&mut scratch, q, l, d);
            }
            return;
        }
        self.run_parallel(threads, || {
            queries
                .par_chunks_exact(dim)
                .zip(labels.par_chunks_mut(k))
                .zip(distances.par_chunks_mut(k))
                .for_each_init(
                    || self.scratch.get(n),
                    |scratch, ((q, l), d)| one(scratch, q, l, d),
                );
        });
    }

    /// The `k` nearest of `query`, nearest first, in `scratch.found`.
    fn search<'s>(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &'s mut Scratch,
    ) -> &'s [Cand] {
        match &self.repr {
            Repr::Build { store, graph } => {
                self.search_in(&BuildView { store, graph }, query, k, ef_search, scratch)
            }
            Repr::Frozen(f) => self.search_in(&f.view(), query, k, ef_search, scratch),
        }
    }

    fn search_in<'s, V: View>(
        &self,
        view: &V,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &'s mut Scratch,
    ) -> &'s [Cand] {
        let entry = view.entry();
        scratch.found.clear();
        let Some(ep_id) = entry.node else {
            return &scratch.found;
        };
        let q = if view.space() == Space::Cosine {
            scratch.query.clear();
            scratch.query.extend_from_slice(query);
            normalize(&mut scratch.query);
            &scratch.query[..]
        } else {
            query
        };
        let ef = ef_search.max(k).max(1);
        dispatch!(self.simd_level, simd => {
            let mut ep = Cand { dist: view.dist(simd, q, ep_id), id: ep_id };
            for level in (1..=entry.max_level).rev() {
                ep = greedy(simd, view, q, ep, level);
            }
            search_layer(simd, view, &mut scratch.search, q, ep, ef, 0, &mut scratch.found);
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
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(threads)
                            .build()
                            .expect("thread pool"),
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
        let (store, graph) = self.build_parts();
        let (params, level) = (&self.params, self.simd_level);
        let insert = |scratch: &mut Scratch, id: u32| {
            dispatch!(level, simd => insert_node(simd, store, graph, params, scratch, id));
        };
        let mut rest = order;
        if threads == 1 || graph.entry().is_none() {
            let mut scratch = self.scratch.get(n);
            let sequential = if threads == 1 {
                order.len()
            } else {
                1.min(order.len())
            };
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
        let (store, graph) = self.build_parts();
        let level = self.simd_level;
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
                (1..n)
                    .into_par_iter()
                    .for_each_init(LinkBuffers::default, |buf, id| merge(buf, id));
            });
        }
    }

    /// A copy of the graph in build form, for inspecting its structure.
    #[cfg(test)]
    pub(crate) fn graph(&self) -> Graph {
        match &self.repr {
            Repr::Build { store, graph } => {
                graph_from_view(&BuildView { store, graph }, self.params.m)
            }
            Repr::Frozen(f) => graph_from_view(&f.view(), self.params.m),
        }
    }

    /// Level sizes and mean degrees, for comparing graph shape with NMSLIB.
    pub fn stats(&self) -> GraphStats {
        match &self.repr {
            Repr::Build { store, graph } => stats_of(&BuildView { store, graph }),
            Repr::Frozen(f) => stats_of(&f.view()),
        }
    }
}

fn store_from_view<V: View>(v: &V, dim: usize) -> Store {
    let mut data = Vec::with_capacity(v.len() * dim);
    for id in 0..v.len() as u32 {
        data.extend_from_slice(v.vector(id));
    }
    Store {
        dim,
        space: v.space(),
        data,
    }
}

fn graph_from_view<V: View>(v: &V, m: usize) -> Graph {
    let levels: Vec<u8> = (0..v.len() as u32).map(|id| v.level(id) as u8).collect();
    let mut graph = Graph::new(m, 2 * m);
    graph.grow(&levels);
    let mut ids = Vec::new();
    for id in 0..v.len() as u32 {
        for level in 0..=v.level(id) {
            ids.clear();
            ids.extend(v.neighbors(id, level));
            graph.set_neighbors(id, level, &ids);
        }
    }
    let entry = v.entry();
    if let Some(e) = entry.node {
        graph.set_entry(e, entry.max_level);
    }
    graph
}

fn stats_of<V: View>(v: &V) -> GraphStats {
    {
        let entry = v.entry();
        let levels = if entry.node.is_some() {
            entry.max_level + 1
        } else {
            0
        };
        let mut nodes_per_level = vec![0usize; levels];
        let mut edges_per_level = vec![0usize; levels];
        for id in 0..v.len() as u32 {
            for level in 0..=v.level(id) {
                nodes_per_level[level] += 1;
                edges_per_level[level] += v.neighbors(id, level).count();
            }
        }
        let mean_degree = nodes_per_level
            .iter()
            .zip(&edges_per_level)
            .map(|(&n, &e)| if n == 0 { 0.0 } else { e as f64 / n as f64 })
            .collect();
        GraphStats {
            nodes_per_level,
            mean_degree,
        }
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

    let view = BuildView { store, graph };
    let q = store.vector(id);
    let mut ep = Cand {
        dist: store.dist(simd, q, entry_id),
        id: entry_id,
    };
    for l in (level + 1..=max_level).rev() {
        ep = greedy(simd, &view, q, ep, l);
    }
    let Scratch {
        search,
        found,
        selected,
        link,
        ..
    } = scratch;
    for l in (0..=level.min(max_level)).rev() {
        search_layer(simd, &view, search, q, ep, params.ef_construction, l, found);
        select_neighbors(
            simd,
            store,
            found,
            params.m,
            false,
            selected,
            &mut link.rejected,
        );
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
fn greedy<S: Simd, V: View>(simd: S, view: &V, q: &[f32], ep: Cand, level: usize) -> Cand {
    simd.vectorize(
        #[inline(always)]
        || {
            let mut cur = ep;
            loop {
                let mut changed = false;
                for nb in view.neighbors(cur.id, level) {
                    let d = view.dist(simd, q, nb);
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
fn search_layer<S: Simd, V: View>(
    simd: S,
    view: &V,
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
            let SearchBuffers {
                visited,
                candidates,
                results,
                fresh,
            } = buf;
            visited.clear();
            visited.insert(ep.id);
            candidates.clear();
            results.clear();
            candidates.push(Reverse(ep));
            results.push(ep);
            #[cfg(feature = "counters")]
            let (mut n_dist, mut n_exp) = (0u64, 0u64);
            while let Some(Reverse(c)) = candidates.pop() {
                if c.dist > results.peek().unwrap().dist {
                    break;
                }
                #[cfg(feature = "counters")]
                {
                    n_exp += 1;
                }
                // Mark and prefetch all unvisited neighbors first, then
                // compute distances, so their cache misses overlap.
                fresh.clear();
                fresh.extend(view.neighbors(c.id, level));
                for &nb in fresh.iter() {
                    prefetch_in(simd, visited.slot_memory(nb));
                }
                fresh.retain(|&nb| {
                    let new = visited.insert(nb);
                    if new {
                        prefetch_in(simd, view.vector(nb));
                    }
                    new
                });
                for &nb in fresh.iter() {
                    #[cfg(feature = "counters")]
                    {
                        n_dist += 1;
                    }
                    let d = view.dist(simd, q, nb);
                    if results.len() < ef || d < results.peek().unwrap().dist {
                        let cand = Cand { dist: d, id: nb };
                        candidates.push(Reverse(cand));
                        results.push(cand);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
                // Start loading the list of the node expanded next.
                if let Some(Reverse(next)) = candidates.peek() {
                    view.prefetch_list(simd, next.id, level);
                }
            }
            #[cfg(feature = "counters")]
            if level == 0 {
                counters::DISTANCES.fetch_add(n_dist, std::sync::atomic::Ordering::Relaxed);
                counters::EXPANSIONS.fetch_add(n_exp, std::sync::atomic::Ordering::Relaxed);
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
            let LinkBuffers {
                ids,
                cands,
                kept,
                rejected,
            } = buf;
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
            cands.extend(ids.iter().chain([&nb]).map(|&x| Cand {
                dist: store.dist(simd, base, x),
                id: x,
            }));
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
