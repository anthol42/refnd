use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use refnd_core::core::exact::{
    exact_edges as exact_edges_core, exact_edges_total,
    exact_nearest_neighbors as exact_nearest_neighbors_core, exact_nearest_neighbors_total,
};
use crate::kernels::{
    KernelVariant,
    protein::sequence::{GlobalAligner, LocalAligner},
    molecules::{TanimotoBit, TanimotoReal}
};
use crate::utils::{BitFingerprint, RealFingerprint};
use super::_utils::linear_progress_bar;
use super::edge_store::EdgeStore;

/// Construct a kernel from Python *args/**kwargs and call `$func` with it injected
/// between the `[$($pre),*]` and `[$($post),*]` argument groups.
/// `$pre` / `$post` token trees expand inside each arm so type inference flows from
/// the concrete kernel (e.g. `data.extract(py)?` resolves to the right `Vec<T>`).
macro_rules! call_generic {
    ($func:path;
     $py:expr, $T:ty, $pykernel:ty, $args:expr, $kwargs:expr;
     $($data:expr),*;
     $($post:expr),*) => {{

        let _obj = $py.get_type::<$pykernel>().call($args, $kwargs)?;
        let _ref: ::pyo3::PyRef<$pykernel> = _obj.extract()?;
        let _kernel = _ref.inner.clone();

        $func($(&$data.extract::<Vec<$T>>($py)?),*, &_kernel, $($post),*)
    }};
}


/// Compute all pairs of data points whose similarity exceeds a threshold (exact, brute-force).
///
/// Evaluates every unordered pair ``(i, j)`` with ``i < j`` and records an edge
/// when ``kernel(data[i], data[j]) <= proximity_threshold``. This is O(n²) in the number
/// of data points; prefer the approximate``HNSWState`` for large datasets.
///
/// Extra positional and keyword arguments are forwarded to the kernel constructor.
///
/// Args:
///     variant: Which kernel to use (``KernelVariant.ProteinGlobal`` or
///              ``KernelVariant.ProteinLocal``).
///     data: Sequence of data items (e.g. ``list[str]`` for protein sequences).
///     proximity_threshold: Maximum distance for an edge to be recorded.
///     threads: Number of parallel threads. ``0`` uses all available cores.
///     progress: Show a progress bar. Defaults to ``True``.
///
/// Returns:
///     An ``EdgeStore`` containing all edges whose distance ≤ ``proximity_threshold``.
///
/// Example::
///
///     from refnd import KernelVariant, exact_edges
///
///     seqs = ["MKTAYIAK", "MKTAYIAKQR", "ACDEFGHIKLM"]
///     store = exact_edges(KernelVariant.ProteinGlobal, seqs, proximity_threshold=0.5)
///     print(len(store))   # number of similar pairs
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
#[pyo3(signature = (variant, data, proximity_threshold = 0.5, threads = 0, progress = true, *args, **kwargs))]
pub fn exact_edges(
    py: Python,
    variant: KernelVariant,
    data: Py<PyAny>,
    proximity_threshold: f32,
    threads: usize,
    progress: bool,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<EdgeStore> {
    let n = data.bind(py).len()?;
    let pb = if progress {
        let pb = linear_progress_bar(exact_edges_total(n) as usize, "Computing edges");
        pb.set_length(exact_edges_total(n));
        Some(pb)
    } else {
        None
    };
    let edges = match variant {
        KernelVariant::ProteinGlobal => {call_generic!(exact_edges_core;
            py, String, GlobalAligner, args, kwargs; data; proximity_threshold, threads, pb.as_ref())}
        KernelVariant::ProteinLocal => {call_generic!(exact_edges_core;
            py, String, LocalAligner, args, kwargs; data; proximity_threshold, threads, pb.as_ref())}
        KernelVariant::TanimotoBit => {call_generic!(exact_edges_core;
            py, BitFingerprint, TanimotoBit, args, kwargs; data; proximity_threshold, threads, pb.as_ref())}
        KernelVariant::TanimotoReal => {call_generic!(exact_edges_core;
            py, RealFingerprint, TanimotoReal, args, kwargs; data; proximity_threshold, threads, pb.as_ref())}
    };
    if let Some(pb) = pb { pb.finish(); }
    Ok(EdgeStore::new(n, edges))
}

/// Find the k nearest neighbors of each query in a reference set (exact, brute-force).
///
/// For every query item, scores it against every reference item using the
/// chosen kernel and returns the top-k references sorted by ascending
/// distance. Complexity is O(n_queries × n_references); prefer
/// ``HNSWState.search`` for approximate nearest neighbors with large datasets.
///
/// Extra positional and keyword arguments are forwarded to the kernel constructor.
///
/// Args:
///     variant: Which kernel to use (e.g. ``KernelVariant.ProteinGlobal`` or
///              ``KernelVariant.TanimotoBit``).
///     queries: Sequence of query items.
///     references: Sequence of reference items to search over. List of items of the same type of the selected kernel variant.
///     k: Number of nearest neighbors to return per query. List of items of the same type of the selected kernel variant.
///     threads: Number of parallel threads. ``0`` uses all available cores.
///     progress: Show a progress bar. Defaults to ``True``.
///
/// Returns:
///     A list of length ``len(queries)``. Each element is a list of up to ``k``
///     tuples ``(reference_index, similarity_score)`` sorted by ascending distance.
///
/// Example::
///
///     from refnd import KernelVariant, exact_nearest_neighbors
///
///     queries = ["MKTAYIAK"]
///     refs    = ["MKTAYIAKQR", "ACDEFGHIKLM", "MKTAYIAKQRQ"]
///     results = exact_nearest_neighbors(
///         KernelVariant.ProteinGlobal, queries, refs, k=2
///     )
///     # results[0] -> [(0, 0.20), (2, 0.27)]
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
#[pyo3(signature = (variant, queries, references, k, threads = 0, progress = true, *args, **kwargs))]
pub fn exact_nearest_neighbors(
    py: Python,
    variant: KernelVariant,
    queries: Py<PyAny>,
    references: Py<PyAny>,
    k: usize,
    threads: usize,
    progress: bool,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<Vec<(usize, f32)>>> {
    let nq = queries.bind(py).len()?;
    let nr = references.bind(py).len()?;
    let pb = if progress {
        let total = exact_nearest_neighbors_total(nq, nr);
        let pb = linear_progress_bar(total as usize, "Computing kNN");
        pb.set_length(total);
        Some(pb)
    } else {
        None
    };
    let result = match variant {
        KernelVariant::ProteinGlobal => {call_generic!(exact_nearest_neighbors_core;
            py, String, GlobalAligner, args, kwargs; queries, references; k, threads, pb.as_ref())}
        KernelVariant::ProteinLocal => {call_generic!(exact_nearest_neighbors_core;
            py, String, LocalAligner, args, kwargs; queries, references; k, threads, pb.as_ref())}
        KernelVariant::TanimotoBit => {call_generic!(exact_nearest_neighbors_core;
            py, BitFingerprint, TanimotoBit, args, kwargs; queries, references; k, threads, pb.as_ref())}
        KernelVariant::TanimotoReal => {call_generic!(exact_nearest_neighbors_core;
            py, RealFingerprint, TanimotoReal, args, kwargs; queries, references; k, threads, pb.as_ref())}
    };
    if let Some(pb) = pb { pb.finish(); }
    Ok(result)
}
