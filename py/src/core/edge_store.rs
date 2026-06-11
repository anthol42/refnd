use pyo3::prelude::*;
use pyo3::exceptions::{PyIndexError, PyIOError};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use refnd_core::core::EdgeStore as CoreEdgeStore;
use super::leiden::CsrGraph;

/// A compact, flat list of weighted directed edges between integer node IDs.
///
/// ``EdgeStore`` is the central data carrier in refnd: it is produced by
/// ``exact_edges`` and ``HNSWState.edges``, consumed by ``CsrGraph``, and can be
/// persisted to disk for later reuse.
///
/// Each edge is a triple ``(src, dst, weight)`` where ``src`` and ``dst`` are
/// zero-based node indices in ``[0, node_count)`` and ``weight`` is a ``float32``
/// similarity score (higher = more similar, unless the graph is built with
/// ``is_weight_distance=True``, in such case it's a real distance ``[0, ∞)`` that will be converted
/// to a similarity score).
///
/// Example::
///
///     from refnd.core import EdgeStore
///
///     store = EdgeStore(node_count=3, edges=[(0, 1, 0.9), (1, 2, 0.7)])
///     print(len(store))     # 2
///     print(store[0])       # (0, 1, 0.9)
///     for src, dst, w in store:
///         print(src, dst, w)
#[gen_stub_pyclass]
#[pyclass(module = "refnd.core", from_py_object)]
#[derive(Clone)]
pub struct EdgeStore {
    pub inner: CoreEdgeStore,
}

#[gen_stub_pymethods]
#[pymethods]
impl EdgeStore {
    /// Create a new EdgeStore.
    ///
    /// Args:
    ///     node_count: Total number of nodes in the graph (must be ≥ the largest node ID + 1).
    ///     edges: List of ``(src, dst, weight)`` triples. Weights are ``float32`` similarity scores.
    #[new]
    pub fn new(node_count: usize, edges: Vec<(usize, usize, f32)>) -> Self {
        Self { inner: CoreEdgeStore::new(node_count, edges) }
    }

    /// Return all edges as a list of ``(src, dst, weight)`` triples.
    #[pyo3(signature = ())]
    pub fn edges(&self) -> Vec<(usize, usize, f32)> {
        self.inner.edges()
    }

    /// Return the total number of nodes this store was created with.
    #[pyo3(signature = ())]
    pub fn node_count(&self) -> usize {self.inner.node_count}

    /// Build a ``CsrGraph`` from this edge store.
    ///
    /// Args:
    ///     use_weight: If ``True``, edge weights are used for graph operations (e.g. strength).
    ///                 If ``False``, all edges are treated as unweighted (weight = 1.0).
    ///     is_weight_distance: If ``True``, edges weights are normalized to have a maximal bound of
    ///                         1 using this formula: ``1.0 / (1.0 + w)`` since the Leiden algorithm works on
    ///                         similarity graphs.
    ///
    /// Returns:
    ///     A ``CsrGraph`` backed by this edge list.
    #[pyo3(signature = (weighted = true, is_weight_distance=true))]
    fn graph(&self, weighted: bool, is_weight_distance: bool) -> CsrGraph {
        CsrGraph { inner: self.inner.graph(weighted, is_weight_distance) }
    }

    /// Serialize this EdgeStore to disk. It supports two file formats: ``text`` with
    /// ``.edgelist`` extension or ``binary`` with ``.edgestr`` extension. Binary is usually 2x
    /// more space efficient at the cost of not being human-readable.
    ///
    /// Args:
    ///     path: Destination file path.
    ///
    /// Raises:
    ///     IOError: On any I/O failure.
    /// Example::
    ///
    ///     from refnd.core import EdgeStore
    ///
    ///     store = EdgeStore(node_count=3, edges=[(0, 1, 0.9), (1, 2, 0.7)])
    ///     store.save("my/path/myedges.edgelist") # Text format
    ///     store.save("my/path/myedges.edgestr")  # Bin format
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load an EdgeStore that was previously saved with ``EdgeStore.save``. It infers the format
    /// from the extension of the path, ``.edgelist`` or ``.edgestr``.
    ///
    /// Args:
    ///     path: Path to the file produced by ``save``.
    ///
    /// Returns:
    ///     The deserialized ``EdgeStore``.
    ///
    /// Raises:
    ///     IOError: If the file cannot be read or the format is invalid.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        CoreEdgeStore::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __getitem__(&self, idx: isize) -> PyResult<(usize, usize, f32)> {
        let n = self.inner.len();
        let i = if idx < 0 { n as isize + idx } else { idx } as usize;
        if i >= n {
            return Err(PyIndexError::new_err(format!("index {idx} out of range for EdgeStore of length {n}")));
        }
        Ok(self.inner.get(i))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> EdgeStoreIter {
        EdgeStoreIter { edges: slf.inner.edges(), pos: 0 }
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

#[gen_stub_pyclass]
#[pyclass(module = "refnd.core")]
pub struct EdgeStoreIter {
    edges: Vec<(usize, usize, f32)>,
    pos: usize,
}

#[pymethods]
impl EdgeStoreIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self) -> Option<(usize, usize, f32)> {
        if self.pos < self.edges.len() {
            let e = self.edges[self.pos];
            self.pos += 1;
            Some(e)
        } else {
            None
        }
    }
}
