use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use fearless_simd::{Level, dispatch};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use nmslib_rs::simd::{
    dot, dot_in, dot_scalar, l2_sq, l2_sq_in, l2_sq_scalar, selected_level_name,
};

fn random_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rng.random_range(-1.0f32..1.0)).collect()
}

fn kernels(c: &mut Criterion) {
    println!("selected SIMD level: {}", selected_level_name());
    let mut rng = StdRng::seed_from_u64(0);
    for dim in [128, 960] {
        let a = random_vec(&mut rng, dim);
        let b = random_vec(&mut rng, dim);
        let mut group = c.benchmark_group(format!("dim{dim}"));
        group.throughput(Throughput::Elements(dim as u64));
        group.bench_function("l2_sq_scalar", |bn| {
            bn.iter(|| l2_sq_scalar(black_box(&a), black_box(&b)))
        });
        group.bench_function("l2_sq_simd", |bn| bn.iter(|| l2_sq(black_box(&a), black_box(&b))));
        group.bench_function("dot_scalar", |bn| {
            bn.iter(|| dot_scalar(black_box(&a), black_box(&b)))
        });
        group.bench_function("dot_simd", |bn| bn.iter(|| dot(black_box(&a), black_box(&b))));
        group.finish();
    }
}

/// 1M l2_sq calls per iteration: dispatching on every call vs dispatching
/// once and running the whole loop inside the selected level.
fn dispatch_overhead(c: &mut Criterion) {
    const CALLS: usize = 1_000_000;
    let mut rng = StdRng::seed_from_u64(1);
    let mut group = c.benchmark_group("dispatch_1M_calls");
    group.sample_size(20);
    for dim in [4, 16, 128] {
        let a = random_vec(&mut rng, dim);
        let b = random_vec(&mut rng, dim);
        group.bench_with_input(BenchmarkId::new("per_call", dim), &dim, |bn, _| {
            bn.iter(|| {
                let mut s = 0.0f32;
                for _ in 0..CALLS {
                    s += l2_sq(black_box(&a), black_box(&b));
                }
                s
            })
        });
        group.bench_with_input(BenchmarkId::new("once", dim), &dim, |bn, _| {
            let level = Level::new();
            bn.iter(|| {
                dispatch!(level, simd => {
                    let mut s = 0.0f32;
                    for _ in 0..CALLS {
                        s += l2_sq_in(simd, black_box(&a), black_box(&b));
                    }
                    s
                })
            })
        });
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bn, _| {
            bn.iter(|| {
                let mut s = 0.0f32;
                for _ in 0..CALLS {
                    s += l2_sq_scalar(black_box(&a), black_box(&b));
                }
                s
            })
        });
    }
    group.finish();
    // Keep dot_in referenced so the once-dispatched API is exercised for both kernels.
    black_box(dispatch!(Level::new(), simd => dot_in(simd, &[1.0], &[1.0])));
}

criterion_group!(benches, kernels, dispatch_overhead);
criterion_main!(benches);
