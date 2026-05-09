use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use refnd_core::core::{
    partition_dataset,
    find_connected_components
};
use super::leiden::CsrGraph;

/// Split a clustered dataset into train and test index sets along clusters
///
/// Clusters are kept intact — no cluster is split across train and test.
/// The function greedily and randomly assigns whole clusters to the test set until the test size
/// is larger than ``test_ratio``, then assigns the rest to train.
///
/// This is the key function for leakage-free dataset splitting: run
/// ``find_communities``, ``connected_components``, or your own clustering, then call ``partition``
/// to obtain indices you can use to slice your data arrays.
///
/// Args:
///     clusters: A list of cluster IDs, one per data point (as returned by
///               ``find_communities`` or ``connected_components``). Length must
///               equal the number of nodes in ``graph``.
///     graph: The graph used to derive community structure. Its node count
///            must match ``len(clusters)``.
///     test_ratio: Target fraction of data points to place in the test set.
///                 The actual fraction may differ slightly because clusters
///                 are indivisible. Defaults to ``0.2``.
///     seed: Optional integer seed for reproducible cluster shuffling.
///           Pass ``None`` for a random assignment.
///     post_filtering: If ``True``, remove train nodes that share an edge with
///                     any test node (stricter leakage prevention at the
///                     cost of a smaller train set). Use only when clusters are found from
///                     communities (``find_communities``). It won't have any effect when used with
///                     ``connected_components`` as this prevents inter-cluster connections.
///
/// Returns:
///     A tuple ``(train_indices, test_indices)`` of zero-based data-point indices.
///
/// Raises:
///     ValueError: If ``clusters`` length does not match ``graph.n``, or if
///                 ``test_ratio`` is not in ``(0, 1)``.
///
/// Example::
///
///     from refnd.core import (
///         EdgeStore, CsrGraph, find_communities, partition
///     )
///
///     store = EdgeStore(6, [(0,1,0.9),(1,2,0.8),(3,4,0.7),(4,5,0.6)])
///     g = CsrGraph(store, use_weight=True, is_weight_distance=False)
///     clusters = find_communities(g) # or connected_components(g)
///     train_idx, test_idx = partition(clusters, g, test_ratio=0.3, seed=42)
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
#[pyo3(signature = (clusters, graph, test_ratio = 0.2, seed = None, post_filtering = false))]
pub fn partition(
    clusters: Vec<usize>,
    graph: &CsrGraph,
    test_ratio: f64,
    seed: Option<usize>,
    post_filtering: bool,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    partition_dataset(clusters, &graph.inner, test_ratio as f32, seed, post_filtering)
        .map_err(|e| PyValueError::new_err(e))
}

/// Finds the connected components of a graph.
///
/// Runs a standard BFS/union-find over the graph and assigns each node to
/// its component.
///
/// Args:
///     graph: The graph to find components.
///
/// Returns:
///     A list of length ``graph.n`` where element ``i`` is the component ID of
///     node ``i``. IDs are arbitrary non-negative integers.
///
/// Example::
///
///     from refnd.core import EdgeStore, connected_components
///
///     store = EdgeStore(6, [(0,1,0.9),(1,2,0.8),(3,4,0.7),(4,5,0.6)])
///     g = store.graph()
///     clusters = connected_components(g) # [0, 0, 0, 1, 1, 1]
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
pub fn connected_components(graph: &CsrGraph) -> Vec<usize> {
    find_connected_components(&graph.inner)
}
