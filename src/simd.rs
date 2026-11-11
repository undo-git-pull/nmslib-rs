//! Distance kernels on `&[f32]`, built on `fearless_simd`.
//!
//! The public `l2_sq` / `dot` functions dispatch to the best SIMD level on every
//! call. Hot loops that compute many distances should pick a level once with
//! [`Level::new`], enter it with `dispatch!`, and call the generic `*_in`
//! kernels from inside, which inline into the caller's target-feature context.

use fearless_simd::{Level, Simd, SimdBase, SimdFloat, dispatch};

/// Independent accumulators in the main loop, to hide FMA latency.
const UNROLL: usize = 4;

/// Squared Euclidean distance. Panics if the lengths differ.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    dispatch!(Level::new(), simd => l2_sq_in(simd, a, b))
}

/// Inner product. Panics if the lengths differ.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    dispatch!(Level::new(), simd => dot_in(simd, a, b))
}

/// `l2_sq` for a SIMD level the caller has already entered.
#[inline(always)]
pub fn l2_sq_in<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    reduce(
        simd,
        a,
        b,
        |x, y, acc| {
            let d = x - y;
            d.mul_add(d, acc)
        },
        |x, y| (x - y) * (x - y),
    )
}

/// `dot` for a SIMD level the caller has already entered.
#[inline(always)]
pub fn dot_in<S: Simd>(simd: S, a: &[f32], b: &[f32]) -> f32 {
    reduce(simd, a, b, |x, y, acc| x.mul_add(y, acc), |x, y| x * y)
}

/// Sums `step` over lanes of `a` and `b`: an unrolled loop over `UNROLL`
/// native vectors, then single native vectors, then a scalar tail.
#[inline(always)]
fn reduce<S: Simd>(
    simd: S,
    a: &[f32],
    b: &[f32],
    step: impl Fn(S::f32s, S::f32s, S::f32s) -> S::f32s,
    scalar: impl Fn(f32, f32) -> f32,
) -> f32 {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    let lanes = S::f32s::LEN;
    let block = lanes * UNROLL;

    let mut acc = [S::f32s::splat(simd, 0.0); UNROLL];
    let mut a_blocks = a.chunks_exact(block);
    let mut b_blocks = b.chunks_exact(block);
    for (ab, bb) in (&mut a_blocks).zip(&mut b_blocks) {
        for (j, acc) in acc.iter_mut().enumerate() {
            let r = j * lanes..(j + 1) * lanes;
            let x = S::f32s::from_slice(simd, &ab[r.clone()]);
            let y = S::f32s::from_slice(simd, &bb[r]);
            *acc = step(x, y, *acc);
        }
    }

    let mut a_vecs = a_blocks.remainder().chunks_exact(lanes);
    let mut b_vecs = b_blocks.remainder().chunks_exact(lanes);
    for (av, bv) in (&mut a_vecs).zip(&mut b_vecs) {
        let x = S::f32s::from_slice(simd, av);
        let y = S::f32s::from_slice(simd, bv);
        acc[0] = step(x, y, acc[0]);
    }

    let total = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    let tail: f32 = a_vecs
        .remainder()
        .iter()
        .zip(b_vecs.remainder())
        .map(|(&x, &y)| scalar(x, y))
        .sum();
    total.reduce_sum() + tail
}

/// Hints the CPU to start loading the first cache line of `data`. Graph
/// search issues this for every neighbor before computing any distance, so
/// the memory misses overlap instead of happening one by one; one line per
/// vector measured best (the hardware fetches the following lines). Uses
/// `prefetcht0` on x86 and `prfm pldl1keep` on aarch64; a no-op elsewhere.
#[inline(always)]
pub fn prefetch_in<S: Simd, T>(simd: S, data: &[T]) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(sse2) = simd.level().as_sse2() {
        prefetch_sse2(sse2, data.as_ptr().cast());
    }
    #[cfg(target_arch = "aarch64")]
    {
        let _ = simd;
        prefetch_aarch64(data.as_ptr().cast());
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = (simd, data);
}

/// `prfm pldl1keep`: the aarch64 prefetch-to-L1 hint. Rust's prefetch
/// intrinsic for ARM is unstable, so this uses inline assembly. On Apple M1
/// at SIFT 1M, queries were memory-bound and level with NMSLIB (which
/// prefetches) until this was added; see NOTES.md.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn prefetch_aarch64(ptr: *const u8) {
    // SAFETY: `prfm` is a hint. It reads no data into registers, writes
    // nothing, and never faults, even for an invalid address (here the
    // pointer comes from a live slice anyway). It touches no stack or flags.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, readonly, preserves_flags),
        );
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fearless_simd::kernel!(
    #[inline(always)]
    fn prefetch_sse2(sse2: Sse2, ptr: *const i8) {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::{_MM_HINT_T0, _mm_prefetch};
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let _ = sse2;
        _mm_prefetch::<_MM_HINT_T0>(ptr);
    }
);

/// Name of the SIMD level `Level::new()` selects on this machine.
pub fn selected_level_name() -> &'static str {
    let level = Level::new();
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if level.as_avx512().is_some() {
            return "AVX-512";
        }
        if level.as_avx2().is_some() {
            return "AVX2+FMA";
        }
        if level.as_sse4_2().is_some() {
            return "SSE4.2";
        }
        if level.as_sse2().is_some() {
            return "SSE2";
        }
    }
    #[cfg(target_arch = "aarch64")]
    if level.as_neon().is_some() {
        return "NEON";
    }
    if level.is_fallback() {
        "scalar fallback"
    } else {
        "unknown"
    }
}

/// Scalar reference for `l2_sq`, used in tests and benchmarks.
pub fn l2_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// Scalar reference for `dot`, used in tests and benchmarks.
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vector lengths differ");
    a.iter().zip(b).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const DIMS: [usize; 11] = [1, 3, 7, 15, 16, 17, 100, 128, 129, 768, 960];
    const SCALES: [f32; 4] = [1.0, 1e-3, 1e3, 1e-15];

    /// Scalar references computed in f64, so the comparison measures the
    /// kernel's error rather than the f32 reference's.
    fn l2_sq_f64(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
            .sum()
    }

    fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum()
    }

    fn abs_f64(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 * y as f64).abs())
            .sum()
    }

    fn random_vec(rng: &mut StdRng, dim: usize, scale: f32) -> Vec<f32> {
        (0..dim)
            .map(|_| rng.random_range(-1.0f32..1.0) * scale)
            .collect()
    }

    /// Relative tolerance of 1e-4. For dot products, cancellation can make
    /// the result tiny, so the bound is relative to `sum |a_i * b_i|`.
    fn assert_close(got: f32, want: f64, magnitude: f64, what: &str) {
        let err = (got as f64 - want).abs();
        assert!(
            err <= 1e-4 * magnitude.max(f64::MIN_POSITIVE),
            "{what}: got {got}, want {want}, rel err {}",
            err / magnitude
        );
    }

    fn check(a: &[f32], b: &[f32]) {
        let ctx = format!("dim {}", a.len());
        let want = l2_sq_f64(a, b);
        assert_close(l2_sq(a, b), want, want, &format!("l2_sq {ctx}"));
        assert_close(
            l2_sq_scalar(a, b),
            want,
            want,
            &format!("l2_sq_scalar {ctx}"),
        );
        let mag = abs_f64(a, b);
        assert_close(dot(a, b), dot_f64(a, b), mag, &format!("dot {ctx}"));
        assert_close(
            dot_scalar(a, b),
            dot_f64(a, b),
            mag,
            &format!("dot_scalar {ctx}"),
        );
    }

    #[test]
    fn matches_scalar_on_listed_dims() {
        let mut rng = StdRng::seed_from_u64(42);
        for dim in DIMS {
            for scale in SCALES {
                for _ in 0..20 {
                    let a = random_vec(&mut rng, dim, scale);
                    let b = random_vec(&mut rng, dim, scale);
                    check(&a, &b);
                    let simd = l2_sq(&a, &b) as f64;
                    let scalar = l2_sq_scalar(&a, &b) as f64;
                    assert!(
                        (simd - scalar).abs() <= 1e-4 * scalar,
                        "dim {dim}: {simd} vs {scalar}"
                    );
                }
            }
        }
    }

    #[test]
    fn mixed_magnitudes() {
        let mut rng = StdRng::seed_from_u64(7);
        for dim in DIMS {
            let a: Vec<f32> = (0..dim)
                .map(|i| rng.random_range(-1.0f32..1.0) * SCALES[i % SCALES.len()])
                .collect();
            let b = random_vec(&mut rng, dim, 1.0);
            check(&a, &b);
        }
    }

    #[test]
    fn exact_on_small_integers() {
        for dim in DIMS {
            let a: Vec<f32> = (0..dim).map(|i| (i % 5) as f32).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i % 3) as f32).collect();
            assert_eq!(l2_sq(&a, &b), l2_sq_scalar(&a, &b), "dim {dim}");
            assert_eq!(dot(&a, &b), dot_scalar(&a, &b), "dim {dim}");
        }
    }

    #[test]
    fn identical_vectors_have_zero_distance() {
        let mut rng = StdRng::seed_from_u64(3);
        for dim in DIMS {
            let a = random_vec(&mut rng, dim, 10.0);
            assert_eq!(l2_sq(&a, &a), 0.0);
        }
    }

    #[test]
    fn zero_length_returns_zero() {
        assert_eq!(l2_sq(&[], &[]), 0.0);
        assert_eq!(dot(&[], &[]), 0.0);
        assert_eq!(l2_sq_scalar(&[], &[]), 0.0);
        assert_eq!(dot_scalar(&[], &[]), 0.0);
    }

    #[test]
    #[should_panic(expected = "vector lengths differ")]
    fn mismatched_lengths_panic() {
        l2_sq(&[1.0, 2.0], &[1.0]);
    }

    /// `Level::new()` only exercises the best level, so run each lower x86
    /// level this CPU supports as well.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn every_available_x86_level_matches() {
        fn run<S: Simd>(simd: S, rng: &mut StdRng) {
            for dim in DIMS {
                let a = random_vec(rng, dim, 1.0);
                let b = random_vec(rng, dim, 1.0);
                let l2 = simd.vectorize(|| l2_sq_in(simd, &a, &b));
                let d = simd.vectorize(|| dot_in(simd, &a, &b));
                let want = l2_sq_f64(&a, &b);
                assert_close(l2, want, want, &format!("{simd:?} l2_sq dim {dim}"));
                assert_close(
                    d,
                    dot_f64(&a, &b),
                    abs_f64(&a, &b),
                    &format!("{simd:?} dot dim {dim}"),
                );
            }
        }
        let level = Level::new();
        let mut rng = StdRng::seed_from_u64(11);
        let mut ran = 0;
        if let Some(s) = level.as_sse2() {
            run(s, &mut rng);
            ran += 1;
        }
        if let Some(s) = level.as_sse4_2() {
            run(s, &mut rng);
            ran += 1;
        }
        if let Some(s) = level.as_avx2() {
            run(s, &mut rng);
            ran += 1;
        }
        if let Some(s) = level.as_avx512() {
            run(s, &mut rng);
            ran += 1;
        }
        assert!(ran > 0);
        println!("checked {ran} x86 SIMD levels");
    }

    #[test]
    fn reports_a_level() {
        let name = selected_level_name();
        assert_ne!(name, "unknown");
        println!("selected SIMD level: {name}");
    }

    fn vec_pair() -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
        (
            0usize..=2048,
            prop_oneof![Just(1.0f32), Just(1e-3), Just(1e3)],
        )
            .prop_flat_map(|(len, scale)| {
                let elem = (-1.0f32..1.0).prop_map(move |x| x * scale);
                (
                    proptest::collection::vec(elem.clone(), len),
                    proptest::collection::vec(elem, len),
                )
            })
    }

    proptest! {
        #[test]
        fn prop_matches_scalar((a, b) in vec_pair()) {
            check(&a, &b);
        }
    }
}
