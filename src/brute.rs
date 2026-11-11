//! Exact k-nearest-neighbor search by scanning every vector, the counterpart
//! of NMSLIB's `brute_force` method.
//!
//! It uses the same distance kernels, cosine normalization and tie order
//! (distance, then id) as [`Hnsw`](crate::hnsw::Hnsw), so its results are
//! the exact answer the graph search approximates. Use it to measure HNSW
//! recall, to debug a suspicious query, or as the baseline in benchmarks.
//! Each query costs `len * dim` multiply-adds.

use std::collections::BinaryHeap;

use fearless_simd::{Level, dispatch};
use rayon::prelude::*;

use crate::hnsw::{Cand, Space, normalize};
use crate::simd::{dot_in, l2_sq_in};

pub struct BruteForce {
    dim: usize,
    space: Space,
    /// Row-major vectors; normalized for `Space::Cosine`.
    data: Vec<f32>,
    simd_level: Level,
}

impl BruteForce {
    pub fn new(dim: usize, space: Space) -> Self {
        assert!(dim > 0, "dim must be positive");
        Self {
            dim,
            space,
            data: Vec::new(),
            simd_level: Level::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn space(&self) -> Space {
        self.space
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Appends `vectors` (row-major, `n * dim` floats). Ids continue from the
    /// current length.
    pub fn add(&mut self, vectors: &[f32]) {
        let dim = self.dim;
        assert!(
            vectors.len().is_multiple_of(dim),
            "vectors length is not a multiple of dim"
        );
        assert!(
            self.len() + vectors.len() / dim < u32::MAX as usize,
            "too many vectors"
        );
        let first = self.data.len();
        self.data.extend_from_slice(vectors);
        if self.space == Space::Cosine {
            for v in self.data[first..].chunks_exact_mut(dim) {
                normalize(v);
            }
        }
    }

    /// The `k` nearest neighbors of `query`, nearest first, as `(id, distance)`.
    /// L2 distances are squared.
    pub fn knn_query(&self, query: &[f32], k: usize) -> Vec<(u32, f32)> {
        assert_eq!(query.len(), self.dim, "query has the wrong dimension");
        let mut heap = BinaryHeap::with_capacity(k.min(self.len()) + 1);
        let mut q = Vec::new();
        self.scan(query, k, &mut q, &mut heap);
        heap.into_sorted_vec()
            .into_iter()
            .map(|c| (c.id, c.dist))
            .collect()
    }

    /// Queries every row of `queries` and writes the results row by row into
    /// `labels` and `distances` (`k` per query, nearest first, -1 / inf where
    /// the index holds fewer than `k` vectors). `threads`: 1 runs on the
    /// calling thread, 0 uses all cores, any other value that many threads.
    pub fn knn_query_batch(
        &self,
        queries: &[f32],
        k: usize,
        threads: usize,
        labels: &mut [i64],
        distances: &mut [f32],
    ) {
        let dim = self.dim;
        assert!(k > 0, "k must be positive");
        assert!(
            queries.len().is_multiple_of(dim),
            "queries length is not a multiple of dim"
        );
        let nq = queries.len() / dim;
        assert!(
            labels.len() == nq * k && distances.len() == nq * k,
            "output buffers have the wrong size"
        );
        let one = |scratch: &mut Scratch, q: &[f32], labels: &mut [i64], dists: &mut [f32]| {
            labels.fill(-1);
            dists.fill(f32::INFINITY);
            self.scan(q, k, &mut scratch.query, &mut scratch.heap);
            scratch.found.clear();
            scratch.found.extend(scratch.heap.drain());
            scratch.found.sort_unstable();
            for (j, c) in scratch.found.iter().enumerate() {
                labels[j] = c.id as i64;
                dists[j] = c.dist;
            }
        };
        let rows = queries
            .chunks_exact(dim)
            .zip(labels.chunks_mut(k))
            .zip(distances.chunks_mut(k));
        if threads == 1 {
            let mut scratch = Scratch::default();
            for ((q, l), d) in rows {
                one(&mut scratch, q, l, d);
            }
            return;
        }
        let mut run = || {
            queries
                .par_chunks_exact(dim)
                .zip(labels.par_chunks_mut(k))
                .zip(distances.par_chunks_mut(k))
                .for_each_init(Scratch::default, |scratch, ((q, l), d)| {
                    one(scratch, q, l, d)
                });
        };
        if threads == 0 {
            run();
        } else {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool")
                .install(run);
        }
    }

    /// Leaves the `k` nearest of `query` in `heap`, a max-heap on (distance,
    /// id). `q_buf` holds the normalized query for cosine.
    fn scan(&self, query: &[f32], k: usize, q_buf: &mut Vec<f32>, heap: &mut BinaryHeap<Cand>) {
        heap.clear();
        if k == 0 {
            return;
        }
        let q = if self.space == Space::Cosine {
            q_buf.clear();
            q_buf.extend_from_slice(query);
            normalize(q_buf);
            &q_buf[..]
        } else {
            query
        };
        let space = self.space;
        dispatch!(self.simd_level, simd => {
            for (id, v) in self.data.chunks_exact(self.dim).enumerate() {
                let dist = match space {
                    Space::L2 => l2_sq_in(simd, q, v),
                    Space::Cosine => 1.0 - dot_in(simd, q, v),
                };
                let c = Cand { dist, id: id as u32 };
                if heap.len() < k {
                    heap.push(c);
                } else if c < *heap.peek().expect("k > 0") {
                    *heap.peek_mut().expect("k > 0") = c;
                }
            }
        });
    }
}

/// Per-thread buffers for batch queries.
#[derive(Default)]
struct Scratch {
    query: Vec<f32>,
    heap: BinaryHeap<Cand>,
    found: Vec<Cand>,
}

#[cfg(test)]
mod tests;
