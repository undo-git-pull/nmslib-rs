pub mod frozen;
pub mod hnsw;
pub mod simd;
pub mod visited;

use pyo3::prelude::*;

#[pymodule]
mod nmslib_rs {
    use numpy::ndarray::Array2;
    use numpy::{
        IntoPyArray, PyArray2, PyArrayDescrMethods, PyArrayMethods, PyUntypedArray,
        PyUntypedArrayMethods, dtype,
    };
    use std::path::PathBuf;

    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    use crate::frozen::LoadError;
    use crate::hnsw::{Hnsw, Params, Space};

    /// SIMD instruction set the distance kernels use on this machine.
    #[pyfunction]
    fn simd_level() -> &'static str {
        crate::simd::selected_level_name()
    }

    /// HNSW index over float32 vectors.
    ///
    /// Distances returned by `knn_query` are squared L2 for `space="l2"` and
    /// `1 - cosine similarity` for `space="cosine"`. Labels are the insertion
    /// positions of the vectors, starting at 0 and continuing across calls to
    /// `add_items`.
    #[pyclass(module = "nmslib_rs")]
    struct Index {
        inner: Hnsw,
        ef: usize,
    }

    #[pymethods]
    impl Index {
        /// `post=True` matches NMSLIB's `post=2` (a second, reverse-order build
        /// merged into level 0). It applies to the first `add_items` call only.
        #[new]
        #[pyo3(signature = (dim, space = "l2", M = 16, ef_construction = 200, seed = 0, post = true))]
        #[allow(non_snake_case)]
        fn new(
            dim: usize,
            space: &str,
            M: usize,
            ef_construction: usize,
            seed: u64,
            post: bool,
        ) -> PyResult<Self> {
            let space = match space {
                "l2" => Space::L2,
                "cosine" => Space::Cosine,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "space must be \"l2\" or \"cosine\", got {other:?}"
                    )));
                }
            };
            if dim == 0 {
                return Err(PyValueError::new_err("dim must be positive"));
            }
            if M < 2 {
                return Err(PyValueError::new_err(format!("M must be at least 2, got {M}")));
            }
            if ef_construction == 0 {
                return Err(PyValueError::new_err("ef_construction must be positive"));
            }
            let params = Params { m: M, ef_construction, post, seed };
            Ok(Self { inner: Hnsw::new(dim, space, params), ef: 10 })
        }

        /// Saves the index to `path`. Writes a temporary file in the same
        /// directory and renames it over `path`, so processes that have the
        /// old file memory-mapped keep a valid mapping.
        fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
            let inner = &self.inner;
            py.detach(|| inner.save(&path))?;
            Ok(())
        }

        /// Loads an index saved with `save`.
        ///
        /// With `mmap=True` the file is mapped read-only: every process that
        /// maps the same file shares one copy in the OS page cache, loading
        /// reads almost nothing until queries touch the data, and `add_items`
        /// raises. The file must not be modified or truncated while it is
        /// mapped (replacing it with `save` is fine). With `mmap=False` the
        /// index is copied into memory and can grow.
        ///
        /// Raises `ValueError` if the file is not a valid index this version
        /// can read, and `OSError` if it cannot be read.
        #[staticmethod]
        #[pyo3(signature = (path, mmap = true))]
        fn load(py: Python<'_>, path: PathBuf, mmap: bool) -> PyResult<Self> {
            let inner = py.detach(|| Hnsw::load(&path, mmap)).map_err(|e| match e {
                LoadError::Io(e) => PyErr::from(e),
                LoadError::Format(msg) => PyValueError::new_err(format!("{}: {msg}", path.display())),
            })?;
            Ok(Self { inner, ef: 10 })
        }

        /// Adds vectors, a C-contiguous float32 array of shape (n, dim).
        /// `num_threads`: 0 uses all cores; 1 is the sequential build, which
        /// gives the same graph for the same seed.
        ///
        /// When the build finishes, the index is rearranged into the same
        /// interleaved, cache-line-aligned layout as a saved file. On large
        /// indexes that makes queries faster (about 25% on SIFT1M, 5% on
        /// GloVe-100). That step copies the index once (memory briefly holds
        /// both), and a later add_items converts it back first.
        #[pyo3(signature = (vectors, num_threads = 0))]
        fn add_items(
            &mut self,
            py: Python<'_>,
            vectors: &Bound<'_, PyAny>,
            num_threads: usize,
        ) -> PyResult<()> {
            if self.inner.is_read_only() {
                return Err(PyRuntimeError::new_err(
                    "index is memory-mapped and read-only; load it with mmap=False to add items",
                ));
            }
            let (_, data) = copy_matrix(vectors, self.inner.dim(), "vectors")?;
            let inner = &mut self.inner;
            py.detach(|| inner.add_vec(data, num_threads));
            Ok(())
        }

        /// Sets the query beam width. The search uses max(ef, k).
        fn set_ef(&mut self, ef: usize) -> PyResult<()> {
            if ef == 0 {
                return Err(PyValueError::new_err("ef must be positive"));
            }
            self.ef = ef;
            Ok(())
        }

        /// Returns (labels, distances), each of shape (n_queries, k), nearest
        /// first. Labels are int64; when the index holds fewer than k vectors
        /// the missing entries are -1 with distance inf. `num_threads`: 0 uses
        /// all cores. Several Python threads may query the same index at once.
        #[pyo3(signature = (queries, k = 10, num_threads = 0))]
        fn knn_query<'py>(
            &self,
            py: Python<'py>,
            queries: &Bound<'py, PyAny>,
            k: usize,
            num_threads: usize,
        ) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<f32>>)> {
            if k == 0 {
                return Err(PyValueError::new_err("k must be positive"));
            }
            let (nq, data) = copy_matrix(queries, self.inner.dim(), "queries")?;
            let (ef, inner) = (self.ef, &self.inner);
            let (labels, dists) = py.detach(|| {
                let mut labels = vec![-1i64; nq * k];
                let mut dists = vec![f32::INFINITY; nq * k];
                inner.knn_query_batch(&data, k, ef, num_threads, &mut labels, &mut dists);
                (labels, dists)
            });
            let labels = Array2::from_shape_vec((nq, k), labels).expect("shape matches");
            let dists = Array2::from_shape_vec((nq, k), dists).expect("shape matches");
            Ok((labels.into_pyarray(py), dists.into_pyarray(py)))
        }

        #[getter]
        fn dim(&self) -> usize {
            self.inner.dim()
        }

        #[getter]
        fn space(&self) -> &'static str {
            match self.inner.space() {
                Space::L2 => "l2",
                Space::Cosine => "cosine",
            }
        }

        /// True for an index loaded with `mmap=True`.
        #[getter]
        fn read_only(&self) -> bool {
            self.inner.is_read_only()
        }

        #[getter]
        fn ef(&self) -> usize {
            self.ef
        }

        fn __len__(&self) -> usize {
            self.inner.len()
        }

        fn __repr__(&self) -> String {
            format!(
                "nmslib_rs.Index(dim={}, space={:?}, len={}, ef={}{})",
                self.inner.dim(),
                self.space(),
                self.inner.len(),
                self.ef,
                if self.inner.is_read_only() { ", read_only=True" } else { "" }
            )
        }
    }

    /// Validates a (n, dim) C-contiguous float32 array and copies it out while
    /// the GIL is held, so no Python thread can modify it during a GIL-free
    /// build or search. Returns the row count and the data.
    fn copy_matrix(obj: &Bound<'_, PyAny>, dim: usize, what: &str) -> PyResult<(usize, Vec<f32>)> {
        let fix = "convert with np.ascontiguousarray(x, dtype=np.float32)";
        let arr = obj.cast::<PyUntypedArray>().map_err(|_| {
            let ty = obj.get_type().name().map(|n| n.to_string()).unwrap_or_default();
            PyValueError::new_err(format!("{what} must be a numpy.ndarray, got {ty}"))
        })?;
        if arr.ndim() != 2 || arr.shape()[1] != dim {
            return Err(PyValueError::new_err(format!(
                "{what} must have shape (n, {dim}), got {:?}",
                arr.shape()
            )));
        }
        if !arr.dtype().is_equiv_to(&dtype::<f32>(obj.py())) {
            return Err(PyValueError::new_err(format!(
                "{what} must have dtype float32, got {}; {fix}",
                arr.dtype()
            )));
        }
        if !arr.is_c_contiguous() {
            return Err(PyValueError::new_err(format!("{what} must be C-contiguous; {fix}")));
        }
        let typed = arr.cast::<PyArray2<f32>>()?;
        let view = typed.try_readonly().map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
        let slice = view.as_slice().map_err(|e| PyValueError::new_err(format!("{what}: {e}")))?;
        Ok((arr.shape()[0], slice.to_vec()))
    }
}
