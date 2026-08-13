use pyo3::prelude::*;
use pyo3::exceptions::{PyIndexError, PyIOError, PyValueError};
use pyo3::Borrowed;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use pyo3_stub_gen::impl_stub_type;
use refnd_core::core::EdgeStore as CoreEdgeStore;
use super::leiden::{CsrGraph, INWeightType};

/// A compact, flat list of weighted directed edges between integer node IDs.
///
/// ``EdgeStore`` is the central data carrier in refnd: it is produced by
/// ``exact_edges`` and ``HNSWState.edges``, consumed by ``CsrGraph``, and can be
/// persisted to disk for later reuse.
///
/// Each edge is a triple ``(src, dst, weight)`` where ``src`` and ``dst`` are
/// zero-based node indices in ``[0, node_count)`` and ``weight`` is a ``float32``
/// value.
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
///
///     # numpy-style boolean masking: keep only the edges where mask[i] is True
///     store[[True, False]]   # EdgeStore with only (0, 1, 0.9)
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
    pub fn new(node_count: usize, edges: Vec<(u32, u32, f32)>) -> Self {
        Self { inner: CoreEdgeStore::new(node_count, edges) }
    }

    /// Return all edges as a list of ``(src, dst, weight)`` triples.
    #[pyo3(signature = ())]
    pub fn edges(&self) -> Vec<(u32, u32, f32)> {
        self.inner.edges().to_vec()
    }

    /// Return the total number of nodes this store was created with.
    #[pyo3(signature = ())]
    pub fn node_count(&self) -> usize { self.inner.node_count }

    /// Build a ``CsrGraph`` from this edge store.
    ///
    /// Args:
    ///     inweight_type: How to convert the raw edge weights to similarity-like weights.
    ///                    See ``INWeightType``. Pass ``INWeightType.Unweighted`` to ignore
    ///                    weights entirely (every edge gets weight ``1.0``). Defaults to
    ///                    ``INWeightType.Distance``, meaning weights are assumed to be a distance
    ///                    measurement, and are converted to similarities.
    ///
    /// Returns:
    ///     A ``CsrGraph`` backed by this edge list.
    #[pyo3(signature = (inweight_type = INWeightType::Distance))]
    fn graph(&self, inweight_type: INWeightType) -> CsrGraph {
        CsrGraph { inner: self.inner.graph(inweight_type.into()) }
    }

    /// Serialize this EdgeStore to disk. It supports two file formats: ``text`` with
    /// ``.edgelist`` extension or ``binary`` with ``.edgestr`` extension. Binary is usually 2x
    /// more space efficient at the cost of not being human-readable.
    ///
    /// **Version compatibility (binary format only):** The binary ``.edgestr`` format is versioned
    /// to the exact package version and is not forward or backward compatible. A file saved with
    /// a different package version will fail to load with a version mismatch error. The text
    /// ``.edgelist`` format has no version stamp and may be portable across versions.
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
    /// **Version compatibility (binary format only):** The binary ``.edgestr`` format is versioned
    /// to the exact package version. A file saved with a different package version will fail to
    /// load with a version mismatch error. There is no forward or backward compatibility. The text
    /// ``.edgelist`` format is not versioned and may be portable across versions.
    ///
    /// Args:
    ///     path: Path to the file produced by ``save``.
    ///
    /// Returns:
    ///     The deserialized ``EdgeStore``.
    ///
    /// Raises:
    ///     IOError: If the file cannot be read, the format is invalid, or (for ``.edgestr`` files)
    ///              the saved version does not match the running package version.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        CoreEdgeStore::load(path)
            .map(|inner| Self { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Index by position (``store[3]``, supports negative indices) or by a boolean
    /// mask (``store[mask]``, one bool per edge), analogous to numpy's ``arr[mask]``.
    /// A mask keeps only the edges where ``mask[i]`` is ``True``; ``node_count`` is
    /// left unchanged.
    fn __getitem__(&self, idx: EdgeIndex) -> PyResult<EdgeOrStore> {
        match idx {
            EdgeIndex::Position(idx) => {
                let n = self.inner.len();
                let i = if idx < 0 { n as isize + idx } else { idx };
                if i < 0 || i as usize >= n {
                    return Err(PyIndexError::new_err(format!("index {idx} out of range for EdgeStore of length {n}")));
                }
                Ok(EdgeOrStore::Edge(self.inner.get(i as usize)))
            }
            EdgeIndex::Mask(mask) => {
                if mask.len() != self.inner.len() {
                    return Err(PyValueError::new_err(format!(
                        "boolean mask length {} does not match EdgeStore length {}", mask.len(), self.inner.len()
                    )));
                }
                Ok(EdgeOrStore::Store(Self { inner: self.inner.mask(&mask) }))
            }
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> EdgeStoreIter {
        EdgeStoreIter { edges: slf.inner.edges().to_vec(), pos: 0 }
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.inner)
    }
}

// ── `__getitem__` index / return types ──────────────────────────────────────
//
// Mirrors numpy's `arr[idx]` overload: an int returns a single element, a
// boolean mask returns a filtered copy of the array.

/// Either a position (`store[3]`) or a boolean mask (`store[[True, False]]`).
enum EdgeIndex {
    Position(isize),
    Mask(Vec<bool>),
}

impl_stub_type!(EdgeIndex = isize | Vec<bool>);

impl<'a, 'py> FromPyObject<'a, 'py> for EdgeIndex {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(i) = ob.extract::<isize>() {
            return Ok(EdgeIndex::Position(i));
        }
        // numpy bool array — extract via its buffer rather than as a Vec<bool> directly
        if ob.hasattr("dtype")? {
            let arr: numpy::PyReadonlyArray1<bool> = ob.extract()?;
            return Ok(EdgeIndex::Mask(arr.as_slice()?.to_vec()));
        }
        let mask: Vec<bool> = ob.extract().map_err(|_| {
            PyValueError::new_err(
                "expected an int or a boolean mask (list, numpy bool array) for EdgeStore indexing",
            )
        })?;
        Ok(EdgeIndex::Mask(mask))
    }
}

enum EdgeOrStore {
    Edge((u32, u32, f32)),
    Store(EdgeStore),
}

impl_stub_type!(EdgeOrStore = (u32, u32, f32) | EdgeStore);

impl<'py> IntoPyObject<'py> for EdgeOrStore {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            EdgeOrStore::Edge(e) => Ok(e.into_pyobject(py).unwrap().into_any()),
            EdgeOrStore::Store(s) => Ok(Py::new(py, s)?.into_bound(py).into_any()),
        }
    }
}

#[gen_stub_pyclass]
#[pyclass(module = "refnd.core")]
pub struct EdgeStoreIter {
    edges: Vec<(u32, u32, f32)>,
    pos: usize,
}

#[pymethods]
impl EdgeStoreIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> { slf }

    fn __next__(&mut self) -> Option<(u32, u32, f32)> {
        if self.pos < self.edges.len() {
            let e = self.edges[self.pos];
            self.pos += 1;
            Some(e)
        } else {
            None
        }
    }
}
