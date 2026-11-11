//! Query profiling on exported SIFT data: builds once (32 threads, post),
//! then times single-threaded queries per efSearch and, with the `counters`
//! feature, reports level-0 work per query.
//!
//!     cargo run --release --features counters --example query_profile -- <base.f32> <queries.f32> [reps]

use std::fs;
use std::path::Path;
use std::time::Instant;

use nmslib_rs::hnsw::{Hnsw, Params, Space};

fn read_matrix(path: &Path) -> (usize, usize, Vec<f32>) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let rows = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data = bytes[8..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    (rows, dim, data)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (_, dim, base) = read_matrix(Path::new(&args[1]));
    let (nq, _, queries) = read_matrix(Path::new(&args[2]));
    let reps: usize = args.get(3).map_or(3, |r| r.parse().unwrap());

    let mut index = Hnsw::new(
        dim,
        Space::L2,
        Params {
            seed: 1,
            ..Params::default()
        },
    );
    let t0 = Instant::now();
    index.add_vec(base, 32);
    eprintln!("build {:.1}s", t0.elapsed().as_secs_f64());
    #[cfg(feature = "counters")]
    nmslib_rs::hnsw::counters::take();

    // "hot": the first query repeated, so the graph region it touches stays
    // cached; the time per distance is then compute plus bookkeeping only.
    let queries = if args.get(4).map(String::as_str) == Some("hot") {
        queries[..dim].repeat(nq)
    } else {
        queries
    };
    let (k, mut labels, mut dists) = (10, vec![0i64; nq * 10], vec![0f32; nq * 10]);
    println!(
        "{:>4} {:>10} {:>9} {:>10} {:>9} {:>9}",
        "ef", "us/query", "qps", "dist/q", "exp/q", "ns/dist"
    );
    for ef in [10, 20, 40, 80, 160, 320, 640] {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t0 = Instant::now();
            index.knn_query_batch(&queries, k, ef, 1, &mut labels, &mut dists);
            best = best.min(t0.elapsed().as_secs_f64());
        }
        let us = best / nq as f64 * 1e6;
        #[cfg(feature = "counters")]
        let (d, e) = {
            let (d, e) = nmslib_rs::hnsw::counters::take();
            (d as f64 / (nq * reps) as f64, e as f64 / (nq * reps) as f64)
        };
        #[cfg(not(feature = "counters"))]
        let (d, e) = (f64::NAN, f64::NAN);
        println!(
            "{ef:>4} {us:>10.2} {:>9.0} {d:>10.1} {e:>9.1} {:>9.1}",
            1e6 / us,
            us * 1e3 / d
        );
    }
}
