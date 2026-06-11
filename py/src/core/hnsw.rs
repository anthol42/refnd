use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use refnd_core::core::hnsw::{HNSWState as HNSWStateCore, HNSWIndex as HNSWIndexCore, HNSWConfig as HNSWConfigCore};
use refnd_core::kernels::proteins::parasail::{GlobalAligner, LocalAligner};
use refnd_core::kernels::molecules::tanimoto::Tanimoto;
use super::edge_store::EdgeStore;
use super::_utils::{logfacto_progress_bar, linear_progress_bar};
use super::super::utils::{BitFingerprint, RealFingerprint};
use super::super::kernels::{
    KernelVariant,
    protein::sequence::{
        GlobalAligner as _GlobalAligner,
        LocalAligner as _LocalAligner,
    },
    molecules::{TanimotoReal as _TanimotoReal, TanimotoBit as _TanimotoBit},
};

/// Configuration for the HNSW approximate nearest-neighbour index.
///
/// All parameters have sensible defaults; in most cases you only need to ``ef_construction``, and
/// ``proximity_threshold``.
///
/// **Parameters:**
///
/// - ``proximity_threshold`` *(float)* — Distance threshold under which a connection is established in the proximity graph.
/// - ``ef_construction`` *(int)* — Candidate list size during build. Default ``64``.
/// - ``m`` *(int)* — Bidirectional links per node at layers > 0.
///   Higher = better recall, more memory, slower build. Typical: 8–64.
/// - ``m_max`` *(int)* — Maximum connections per node at layers > 0. Usually equal to ``m``.
/// - ``m_max0`` *(int)* — Maximum connections at layer 0. Usually ``2 * m``.
/// - ``m_l`` *(float)* — Level generation factor. Usually ``0.36 ≈ 1 / ln(m)``.
/// - ``ef_init`` *(int)* — Candidate list size during initial-layer insertion. Almost always 1 for
///   a greedy search.
/// - ``extend_candidates`` *(bool)* — Use neighbors of neighbors as candidates, making search more exhaustive.
/// - ``keep_pruned_connections`` *(bool)* — Retain discarded candidates to fill up to ``m`` connections when not enough connections are found.
/// - ``cache_capacity`` *(int)* — Maximum cached kernel scores. Increasing it increase the memory
///   footprint, but also cache hits, which can improve runtime performances for computationally expensive kernels.
/// - ``cache_shards`` *(int)* — Number of cache shards (reduces lock contention).
/// - ``n_threads`` *(int)* — Threads used during build. ``0`` = all available cores.
/// - ``shuffle`` *(bool)* — Shuffle insertion order before building. Can create a less biased
///   graph, but reduces cache hits, which increases runtime.
/// - ``use_heuristic`` *(bool)* — Use the heuristic neighbour-selection from the paper (recommended).
/// - ``strict_ef`` *(bool)* — If ``True``, enforces the result set size to exactly ``ef`` during search. Empirically, setting this to ``False`` can improve runtime performance, as it allows halving ``ef_construction`` without sacrificing accuracy.
/// - ``threshold_based_neighbourhood`` *(bool)* — Select a minimum of ``m`` neighbors like the classic algorithm, but doesn't bound the neighbourhood size as all candidates that are closer than the threshold are kept.
#[gen_stub_pyclass]
#[pyclass(module = "refnd.core", from_py_object)]
#[derive(Clone)]
pub struct HNSWConfig {
    inner: HNSWConfigCore,
}

#[gen_stub_pymethods]
#[pymethods]
impl HNSWConfig {
    /// Create an HNSWConfig. See class docstring for parameter descriptions.
    #[new]
    #[pyo3(signature = (
        proximity_threshold = 0.5,
        ef_construction = 64,
        m = 16,
        m_max = 16,
        m_max0 = 32,
        m_l = 0.36,
        ef_init = 1,
        extend_candidates = false,
        keep_pruned_connections = true,
        cache_capacity = 2_000_000,
        cache_shards = 64,
        n_threads = 0,
        shuffle = false,
        use_heuristic = true,
        strict_ef = false,
        threshold_based_neighbourhood = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        proximity_threshold: f32, ef_construction: usize,
        m: usize, m_max: usize, m_max0: usize, m_l: f64,
        ef_init: usize,
        extend_candidates: bool, keep_pruned_connections: bool,
        cache_capacity: usize, cache_shards: usize, n_threads: usize,
        shuffle: bool, use_heuristic: bool,
        strict_ef: bool, threshold_based_neighbourhood: bool,
    ) -> Self {
        HNSWConfig {
            inner: HNSWConfigCore {
                m, m_max, m_max0, m_l, ef_init, ef_construction,
                extend_candidates, keep_pruned_connections,
                cache_capacity, cache_shards, proximity_threshold,
                n_threads, shuffle, use_heuristic,
                strict_ef, threshold_based_neighbourhood,
            },
        }
    }

    #[getter] fn m(&self) -> usize { self.inner.m }
    #[getter] fn m_max(&self) -> usize { self.inner.m_max }
    #[getter] fn m_max0(&self) -> usize { self.inner.m_max0 }
    #[getter] fn m_l(&self) -> f64 { self.inner.m_l }
    #[getter] fn ef_init(&self) -> usize { self.inner.ef_init }
    #[getter] fn ef_construction(&self) -> usize { self.inner.ef_construction }
    #[getter] fn extend_candidates(&self) -> bool { self.inner.extend_candidates }
    #[getter] fn keep_pruned_connections(&self) -> bool { self.inner.keep_pruned_connections }
    #[getter] fn cache_capacity(&self) -> usize { self.inner.cache_capacity }
    #[getter] fn cache_shards(&self) -> usize { self.inner.cache_shards }
    #[getter] fn proximity_threshold(&self) -> f32 { self.inner.proximity_threshold }
    #[getter] fn n_threads(&self) -> usize { self.inner.n_threads }
    #[getter] fn shuffle(&self) -> bool { self.inner.shuffle }
    #[getter] fn use_heuristic(&self) -> bool { self.inner.use_heuristic }
    #[getter] fn strict_ef(&self) -> bool { self.inner.strict_ef }
    #[getter] fn threshold_based_neighbourhood(&self) -> bool { self.inner.threshold_based_neighbourhood }

    /// Return the configuration as a plain Python dict.
    fn dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("m", self.inner.m)?;
        d.set_item("m_max", self.inner.m_max)?;
        d.set_item("m_max0", self.inner.m_max0)?;
        d.set_item("m_l", self.inner.m_l)?;
        d.set_item("ef_init", self.inner.ef_init)?;
        d.set_item("ef_construction", self.inner.ef_construction)?;
        d.set_item("extend_candidates", self.inner.extend_candidates)?;
        d.set_item("keep_pruned_connections", self.inner.keep_pruned_connections)?;
        d.set_item("cache_capacity", self.inner.cache_capacity)?;
        d.set_item("cache_shards", self.inner.cache_shards)?;
        d.set_item("proximity_threshold", self.inner.proximity_threshold)?;
        d.set_item("n_threads", self.inner.n_threads)?;
        d.set_item("shuffle", self.inner.shuffle)?;
        d.set_item("use_heuristic", self.inner.use_heuristic)?;
        d.set_item("strict_ef", self.inner.strict_ef)?;
        d.set_item("threshold_based_neighbourhood", self.inner.threshold_based_neighbourhood)?;
        Ok(d.into())
    }

    fn __str__(&self) -> String { format!("{:?}", self.inner) }
    fn __repr__(&self) -> String { format!("{}", self.inner) }
}



/// Read-only snapshot of the HNSW graph structure after a build.
///
/// Obtained via ``HNSWState.index``. Primarily useful for inspection,
/// serialisation, and debugging; normal users interact with ``HNSWState``
/// instead.
///
/// Properties:
///     dataset_size (int): Number of items indexed.
///     layers (list): Nested multi-layer adjacency list ``layers[layer][node] = [neighbor_ids]``.
///     entry_point (tuple | None): ``(layer, node_id)`` of the global entry point, or ``None`` if empty.
///     max_layers (int): Number of layers in the hierarchy.
///     proximity_edges (list): List of ``((src, dst), score)`` for proximity-threshold edges.
///     config (HNSWConfig): The config used to build this index.
#[gen_stub_pyclass]
#[pyclass(module = "refnd.core")]
pub struct HNSWIndex {
    inner: HNSWIndexCore,
}

#[gen_stub_pymethods]
#[pymethods]
impl HNSWIndex {
    #[getter] pub fn dataset_size(&self) -> usize { self.inner.dataset_size }
    #[getter] pub fn layers(&self) -> Vec<Vec<Vec<usize>>> { self.inner.layers.clone() }
    #[getter] pub fn entry_point(&self) -> Option<(usize, usize)> { self.inner.entry_point }
    #[getter] pub fn max_layers(&self) -> usize { self.inner.max_layers }
    #[getter] pub fn proximity_edges(&self) -> Vec<((usize, usize), f32)> { self.inner.proximity_edges.clone() }
    #[getter] pub fn config(&self) -> HNSWConfig { HNSWConfig { inner: self.inner.config.clone() } }

    /// Save the object to a binary representation (e.g. *.hnsw* file).
    ///
    /// Args:
    ///     path: Destination file path (recommended with a .hnsw extension)
    ///
    /// Raises:
    ///     RuntimeError: On serialisation or I/O failure.
    pub fn save(&self, path: String) -> PyResult<()> {
        self.inner.save(&path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Load an index previously saved with ``HNSWIndex.save``.
    ///
    /// Args:
    ///     path: Path to the saved file.
    ///
    /// Returns:
    ///     The deserialised ``HNSWIndex``.
    ///
    /// Raises:
    ///     RuntimeError: If the file cannot be read or the format is invalid.
    #[staticmethod]
    pub fn load(path: String) -> PyResult<Self> {
        HNSWIndexCore::load(&path)
            .map(|inner| HNSWIndex { inner })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    pub fn __str__(&self) -> String { format!("{:?}", self.inner) }
    pub fn __repr__(&self) -> String { format!("{}", self.inner) }
}

enum HNSWType {
    ProteinGlobal(HNSWStateCore<String, GlobalAligner>),
    ProteinLocal(HNSWStateCore<String, LocalAligner>),
    TanimotoBit(HNSWStateCore<BitFingerprint, Tanimoto>),
    TanimotoReal(HNSWStateCore<RealFingerprint, Tanimoto>),
}

/// Expands a `HNSWState::new(data, kernel, config)` constructor for each KernelVariant.
/// The kernel is instantiated from Python with forwarded *args/**kwargs.
/// `data.extract(py)?` is inlined per-arm so the type is inferred from the constructor signature.
macro_rules! hnsw_new {
    ($ctor:path; $py:expr, $which:expr, $data:expr, $config:expr, $args:expr, $kwargs:expr;
     $($variant:ident : $kernel:ty),+ $(,)?) => {
        match $which {
            $(
                KernelVariant::$variant => {
                    let _obj = $py.get_type::<$kernel>().call($args, $kwargs)?;
                    let _aligner: ::pyo3::PyRef<$kernel> = _obj.extract()?;
                    HNSWType::$variant($ctor($data.extract($py)?, _aligner.inner.clone(), $config))
                }
            )+
        }
    };
}

/// Expands a `HNSWState::load(path, data, config, kernel)` call for each KernelVariant.
macro_rules! hnsw_load {
    ($py:expr, $which:expr, $path:expr, $data:expr, $config:expr, $args:expr, $kwargs:expr;
     $($variant:ident : $kernel:ty),+ $(,)?) => {
        match $which {
            $(
                KernelVariant::$variant => {
                    let _obj = $py.get_type::<$kernel>().call($args, $kwargs)?;
                    let _aligner: ::pyo3::PyRef<$kernel> = _obj.extract()?;
                    HNSWType::$variant(
                        HNSWStateCore::load($path, $data.extract($py)?, $config, _aligner.inner.clone())
                            .map_err(|e| ::pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    )
                }
            )+
        }
    };
}

/// Dispatches a method call to the inner HNSWState for each HNSWType variant.
/// `$args` is a parenthesized token tree, e.g. `(k, ef)` or `()`.
/// This avoids mixing two independent repetition depths in one `$()+` block.
macro_rules! hnsw_dispatch {
    ($inner:expr, $method:ident $args:tt;
     $($variant:ident : $kernel:ty),+ $(,)?) => {
        match &$inner {
            $(HNSWType::$variant(inner) => inner.$method $args,)+
        }
    };
}

macro_rules! hnsw_dispatch_mut {
    ($inner:expr, $method:ident $args:tt;
     $($variant:ident : $kernel:ty),+ $(,)?) => {
        match &mut $inner {
            $(HNSWType::$variant(inner) => inner.$method $args,)+
        }
    };
}

/// Stateful Hierarchical Navigable Small World (HNSW) approximate nearest-neighbour index. This
/// class is used to build and search in the index, whereas ``HNSWIndex`` is a read-only *internal*
/// representation of the index used for saving / loading / structure exploration.
///
/// ``HNSWState`` wraps the dataset and the graph index together. The typical
/// workflow is:
///
/// 1. Construct with the data and a ``KernelVariant``.
/// 2. Call ``build()`` to insert all items and form the graph.
/// 3. Call ``search()`` to query, or ``edges()`` to extract the proximity graph
///    for downstream clustering / splitting.
///
/// HNSW parameters (``proximity_threshold``, ``ef_construction``, …) are the same as ``HNSWConfig``
/// and can be passed directly to the constructor as keyword arguments.
///
/// Args:
///     variant: Kernel to use (e.g.  ``KernelVariant.ProteinGlobal``, ``KernelVariant.ProteinLocal``, ``KernelVariant.TanimotoBit``, *etc*).
///     data: The dataset — a list of items matching the kernel type (e.g. ``list[str]`` or ``list[np.ndarray]``).
///     proximity_threshold, ef_construction, m, m_max, m_max0, m_l, ef_init, extend_candidates,
///         keep_pruned_connections, cache_capacity, cache_shards,
///         n_threads, shuffle, use_heuristic, strict_ef,
///         threshold_based_neighbourhood: See ``HNSWConfig`` for descriptions.
///
/// Properties:
///     config (HNSWConfig): The config in use.
///     index (HNSWIndex): A snapshot of the current graph structure.
///
/// Example::
///
///     from refnd import HNSWState, KernelVariant
///
///     seqs = ["MKTAYIAK", "MKTAYIAKQR", "ACDEFGHIKLM", "MKTAYIAKQRQIS"]
///     state = HNSWState(KernelVariant.ProteinGlobal, seqs, proximity_threshold=0.3, ef_construction=64)
///     state.build()
///     results = state.search(["MKTAYIAK"], k=2)
///     # results[0] -> [(0, 1.0), (1, 0.88)]
///
///     store = state.edges()        # EdgeStore for graph-based splitting
///     state.save("index.hnsw")
///     state2 = HNSWState.load(KernelVariant.ProteinGlobal, "index.hnsw", seqs)
#[gen_stub_pyclass]
#[pyclass(module = "refnd.core")]
pub struct HNSWState {
    inner: HNSWType,
    n: usize,
    config: HNSWConfig,
}

#[gen_stub_pymethods]
#[pymethods]
impl HNSWState {
    #[new]
    #[pyo3(signature = (
        variant, data,
        *args,
        proximity_threshold = 0.5,
        ef_construction = 64,
        m = 16,
        m_max = 16,
        m_max0 = 32,
        m_l = 0.36,
        ef_init = 1,
        extend_candidates = false,
        keep_pruned_connections = true,
        cache_capacity = 2_000_000,
        cache_shards = 64,
        n_threads = 0,
        shuffle = false,
        use_heuristic = true,
        strict_ef = false,
        threshold_based_neighbourhood = false,
        **kwargs
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        variant: KernelVariant,
        data: Py<PyAny>,
        py: Python,
        args: &Bound<'_, PyTuple>,
        proximity_threshold: f32,
        ef_construction: usize,
        m: usize,
        m_max: usize,
        m_max0: usize,
        m_l: f64,
        ef_init: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
        cache_capacity: usize,
        cache_shards: usize,
        n_threads: usize,
        shuffle: bool,
        use_heuristic: bool,
        strict_ef: bool,
        threshold_based_neighbourhood: bool,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let n = data.bind(py).len()?;
        let config = HNSWConfigCore {
            m, m_max, m_max0, m_l, ef_init, ef_construction,
            extend_candidates, keep_pruned_connections,
            cache_capacity, cache_shards, proximity_threshold,
            n_threads, shuffle, use_heuristic,
            strict_ef, threshold_based_neighbourhood,
        };
        let config_py = HNSWConfig { inner: config.clone() };
        let inner = hnsw_new!(
            HNSWStateCore::new; py, variant, data, config, args, kwargs;
            ProteinGlobal:_GlobalAligner,
            ProteinLocal:_LocalAligner,
            TanimotoBit:_TanimotoBit,
            TanimotoReal:_TanimotoReal
        );
        Ok(HNSWState { inner, n, config: config_py })
    }

    /// Build the HNSW index by inserting all data items.
    ///
    /// Must be called before ``search`` or ``edges``. Calling ``build`` a second
    /// time raises a ``RuntimeError``; construct a new ``HNSWState`` instead.
    ///
    /// Args:
    ///     progress: Display a progress bar. Defaults to ``True``.
    ///
    /// Raises:
    ///     RuntimeError: If the index has already been built.
    #[pyo3(signature = (progress = true))]
    pub fn build(&mut self, progress: bool) -> PyResult<()> {
        let pb = if progress { Some(logfacto_progress_bar(self.n, "Building index")) } else { None };
        hnsw_dispatch_mut!(
            self.inner, build(pb.as_ref());
            ProteinGlobal:_GlobalAligner,
            ProteinLocal:_LocalAligner,
            TanimotoBit:_TanimotoBit,
            TanimotoReal:_TanimotoReal
        ).map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        if let Some(pb) = pb { pb.finish() };
        Ok(())
    }

    /// Search the index for approximate nearest neighbours.
    ///
    /// For each query item, returns the ``k`` most similar items in the dataset,
    /// sorted by descending similarity. The quality of the approximation is
    /// controlled by ``ef``: larger values explore more candidates and improve
    /// recall at the cost of speed.
    ///
    /// Args:
    ///     queries: List of query items (same type as the indexed data).
    ///     k: Number of nearest neighbours to return per query. Defaults to ``1``.
    ///     ef: Size of the dynamic candidate list during search. Must be ≥ ``k``.
    ///         Defaults to ``64``.
    ///     threads: Number of parallel threads. ``0`` = all available cores.
    ///     progress: Display a progress bar. Defaults to ``True``.
    ///
    /// Returns:
    ///     A list of length ``len(queries)``. Each element is a sorted list of
    ///     up to ``k`` tuples ``(dataset_index, similarity_score)``.
    ///
    /// Raises:
    ///     RuntimeError: If ``build`` has not been called yet.
    #[pyo3(signature = (queries, k = 1, ef = 64, threads = 0, progress = true))]
    pub fn search(
        &self,
        py: Python,
        queries: Py<PyAny>,
        k: usize,
        ef: usize,
        threads: usize,
        progress: bool,
    ) -> PyResult<Vec<Vec<(usize, f32)>>> {
        let n = queries.bind(py).len()?;
        let pb = if progress { Some(linear_progress_bar(n, "Searching")) } else { None };
        // Direct match: the concrete inner type per arm lets the compiler infer
        // the element type of `queries.extract(py)?` without a hardcoded annotation.
        // hnsw_dispatch! can't be used here because `tt`-captured args are opaque
        // to rustc's type inference when `extract`'s output type must flow from
        // the arm's `inner: &HNSWState<T, _>`.
        let res = match &self.inner {
            HNSWType::ProteinGlobal(inner) => inner.parallel_search(queries.extract::<Vec<_>>(py)?.as_slice(), k, ef, threads, pb.as_ref()),
            HNSWType::ProteinLocal(inner)  => inner.parallel_search(queries.extract::<Vec<_>>(py)?.as_slice(), k, ef, threads, pb.as_ref()),
            HNSWType::TanimotoBit(inner)   => inner.parallel_search(queries.extract::<Vec<_>>(py)?.as_slice(), k, ef, threads, pb.as_ref()),
            HNSWType::TanimotoReal(inner)  => inner.parallel_search(queries.extract::<Vec<_>>(py)?.as_slice(), k, ef, threads, pb.as_ref()),
        }.map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        if let Some(pb) = pb { pb.finish() };
        Ok(res)
    }

    /// Extract the proximity graph as an ``EdgeStore``.
    ///
    /// Returns all edges found during ``build``. All distance measurements below the
    /// ``proximity_threshold``.
    ///
    /// Returns:
    ///     An ``EdgeStore`` with ``node_count = dataset_size``.
    pub fn edges(&self) -> EdgeStore {
        let edges = hnsw_dispatch!(
            self.inner, edges();
            ProteinGlobal:_GlobalAligner,
            ProteinLocal:_LocalAligner,
            TanimotoBit:_TanimotoBit,
            TanimotoReal:_TanimotoReal
        );
        EdgeStore::new(self.n, edges)
    }

    /// Return the adjacency lists for a specific HNSW layer.
    ///
    /// Args:
    ///     layer_idx: Zero-based layer index (0 = base layer with most nodes).
    ///
    /// Returns:
    ///     A list of length ``dataset_size`` where element ``i`` is the list of
    ///     neighbour IDs of node ``i`` at this layer. Node IDs are their index in the original dataset.
    ///
    /// Raises:
    ///     IndexError: If ``layer_idx`` is out of range.
    pub fn get_layer(&self, layer_idx: usize) -> PyResult<Vec<Vec<usize>>> {
        hnsw_dispatch!(
            self.inner, get_layer(layer_idx);
            ProteinGlobal:_GlobalAligner,
            ProteinLocal:_LocalAligner,
            TanimotoBit:_TanimotoBit,
            TanimotoReal:_TanimotoReal
        ).map_err(pyo3::exceptions::PyIndexError::new_err)
    }

    /// Serialize the full state (index + config) to a binary file.
    ///
    /// The saved file can be loaded back with ``HNSWState.load``. The original
    /// data must be provided again at load time (it is not embedded in the file).
    ///
    /// Args:
    ///     path: Destination file path.
    ///
    /// Raises:
    ///     RuntimeError: On serialisation or I/O failure.
    pub fn save(&self, path: String) -> PyResult<()> {
        hnsw_dispatch!(
            self.inner, save(&path);
            ProteinGlobal:_GlobalAligner,
            ProteinLocal:_LocalAligner,
            TanimotoBit:_TanimotoBit,
            TanimotoReal:_TanimotoReal
        )
        .map_err(|e: Box<dyn std::error::Error>| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Load an HNSWState form a binary file saved with ``HNSWState.save``.
    ///
    /// Args:
    ///     variant: Must match the kernel used during the original build.
    ///     path: Path to the saved file.
    ///     data: The original dataset (required to re-attach the kernel).
    ///
    /// Returns:
    ///     The restored ``HNSWState``, ready to call ``search`` or ``edges`` if the index was built.
    ///
    /// Raises:
    ///     RuntimeError: If the file cannot be read or the format is invalid.
    #[staticmethod]
    #[pyo3(signature = (variant, path, data, *args, **kwargs))]
    pub fn load(
        variant: KernelVariant,
        path: String,
        data: Py<PyAny>,
        py: Python,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let n = data.bind(py).len()?;
        let inner = hnsw_load!(
            py, variant, path, data, None::<HNSWConfigCore>, args, kwargs;
            ProteinGlobal:_GlobalAligner,
            ProteinLocal:_LocalAligner,
            TanimotoBit:_TanimotoBit,
            TanimotoReal:_TanimotoReal
        );
        let config = HNSWConfig {
            inner: hnsw_dispatch!(
                inner, config();
                ProteinGlobal:_GlobalAligner,
                ProteinLocal:_LocalAligner,
                TanimotoBit:_TanimotoBit,
                TanimotoReal:_TanimotoReal
            ).clone(),
        };
        Ok(HNSWState { inner, n, config })
    }

    /// Whether ``build`` has been called on this index.
    #[getter]
    pub fn is_built(&self) -> bool {
        match &self.inner {
            HNSWType::ProteinGlobal(inner) => inner.has_been_built,
            HNSWType::ProteinLocal(inner)  => inner.has_been_built,
            HNSWType::TanimotoBit(inner)   => inner.has_been_built,
            HNSWType::TanimotoReal(inner)  => inner.has_been_built,
        }
    }

    /// Config used to initiate this object.
    #[getter]
    pub fn config(&self) -> HNSWConfig {
        self.config.clone()
    }

    /// HNSWIndex snapshot.
    #[getter]
    pub fn index(&self) -> HNSWIndex {
        HNSWIndex {
            inner: hnsw_dispatch!(
                self.inner, index();
                ProteinGlobal:_GlobalAligner,
                ProteinLocal:_LocalAligner,
                TanimotoBit:_TanimotoBit,
                TanimotoReal:_TanimotoReal
            ),
        }
    }
}

