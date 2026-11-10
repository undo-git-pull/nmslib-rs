pub mod simd;

use pyo3::prelude::*;

#[pymodule]
mod rhnsw {
    use pyo3::prelude::*;

    /// SIMD instruction set the distance kernels use on this machine.
    #[pyfunction]
    fn simd_level() -> &'static str {
        crate::simd::selected_level_name()
    }
}
