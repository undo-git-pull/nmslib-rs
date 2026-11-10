//! Builds nmslib_rs on exported SIFT vectors and runs the efSearch sweep on one
//! thread. Writes the found ids per efSearch and a JSON summary; recall is
//! scored in Python with bench/common.py so it matches the NMSLIB baseline.
//!
//! Input files are written by bench/eval_rust.py: a little-endian header of
//! two u32 (rows, dim) followed by row-major f32.
//!
//!     cargo run --release --example eval_sift -- <base.f32> <queries.f32> <out_dir> <order_seed> [threads]
//!
//! `order_seed` 0 inserts in file order; otherwise the insertion order is a
//! seeded shuffle and ids are mapped back to file rows in the output.

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use nmslib_rs::hnsw::{Hnsw, Params, Space};

const K: usize = 10;
const EF_SEARCH: [usize; 7] = [10, 20, 40, 80, 160, 320, 640];

fn read_matrix(path: &Path) -> (usize, usize, Vec<f32>) {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let rows = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let data: Vec<f32> =
        bytes[8..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert_eq!(data.len(), rows * dim, "{}: size does not match header", path.display());
    (rows, dim, data)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() == 5 || args.len() == 6,
        "usage: eval_sift <base.f32> <queries.f32> <out_dir> <order_seed> [threads]"
    );
    let threads: usize = args.get(5).map_or(1, |t| t.parse().expect("threads must be an integer"));
    let (n, dim, base) = read_matrix(Path::new(&args[1]));
    let (nq, qdim, queries) = read_matrix(Path::new(&args[2]));
    assert_eq!(dim, qdim);
    let out_dir = Path::new(&args[3]);
    let seed: u64 = args[4].parse().expect("order_seed must be an integer");
    fs::create_dir_all(out_dir).unwrap();

    let mut order: Vec<usize> = (0..n).collect();
    if seed != 0 {
        order.shuffle(&mut StdRng::seed_from_u64(seed));
    }
    let mut shuffled = Vec::with_capacity(base.len());
    for &row in &order {
        shuffled.extend_from_slice(&base[row * dim..(row + 1) * dim]);
    }
    drop(base);

    let params = Params { seed, ..Params::default() };
    let mut index = Hnsw::new(dim, Space::L2, params.clone());
    let t0 = Instant::now();
    index.add_vec(shuffled, threads);
    let build_s = t0.elapsed().as_secs_f64();
    let stats = index.stats();
    eprintln!("[seed {seed}] build {build_s:.1}s, levels {:?}", stats.nodes_per_level);

    let mut sweep = Vec::new();
    for ef in EF_SEARCH {
        let mut ids = Vec::with_capacity(nq * K);
        let t0 = Instant::now();
        for q in queries.chunks_exact(dim) {
            let found = index.knn_query(q, K, ef);
            ids.extend(found.iter().map(|&(id, _)| order[id as usize] as i32));
            ids.extend(std::iter::repeat_n(-1, K - found.len()));
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        fs::write(out_dir.join(format!("ids_seed{seed}_ef{ef}.i32")), bytes).unwrap();
        eprintln!("[seed {seed}] ef {ef:4} qps {:9.1}", nq as f64 / secs);
        sweep.push(format!(
            "{{\"ef_search\": {ef}, \"qps\": {:.1}, \"query_time_s\": {secs:.3}}}",
            nq as f64 / secs
        ));
    }

    let json = format!(
        "{{\"insertion_order_seed\": {seed}, \"build_time_s\": {build_s:.2}, \
         \"build_threads\": {threads}, \"params\": {{\"M\": {}, \"efConstruction\": {}, \"post\": {}}}, \
         \"simd_level\": \"{}\", \"nodes_per_level\": {:?}, \"mean_degree\": [{}], \"sweep\": [{}]}}",
        params.m,
        params.ef_construction,
        params.post,
        nmslib_rs::simd::selected_level_name(),
        stats.nodes_per_level,
        stats.mean_degree.iter().map(|d| format!("{d:.3}")).collect::<Vec<_>>().join(", "),
        sweep.join(", "),
    );
    fs::File::create(out_dir.join(format!("summary_seed{seed}.json")))
        .and_then(|mut f| f.write_all(json.as_bytes()))
        .unwrap();
}
