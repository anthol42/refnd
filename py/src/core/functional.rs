use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use refnd_core::core::{
    partition_dataset,
    find_connected_components,
    largest_cluster,
};
use super::leiden::CsrGraph;

/// Split a clustered dataset into train and test index sets.
///
/// Clusters are kept intact — no cluster is split across train and test.
/// The function greedily assigns whole clusters to the test set until
/// ``test_ratio`` is reached, then assigns the rest to train.
///
/// This is the key function for leakage-free dataset splitting: run
/// ``find_communities`` (or your own clustering), then call ``partition`` to
/// obtain indices you can use to slice your data arrays.
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
///     post_filtering: If ``True``, remove test nodes that share an edge with
///                     any train node (stricter leakage prevention at the
///                     cost of a smaller test set).
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
///     labels = find_communities(g)
///     train_idx, test_idx = partition(labels, g, test_ratio=0.3, seed=42)
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
#[pyo3(signature = (clusters, graph, test_ratio = 0.2, seed = None, post_filtering = false))]
pub fn partition(
    clusters: Vec<usize>,
    graph: &CsrGraph,
    test_ratio: f32,
    seed: Option<usize>,
    post_filtering: bool,
) -> PyResult<(Vec<usize>, Vec<usize>)> {
    partition_dataset(clusters, &graph.inner, test_ratio, seed, post_filtering)
        .map_err(|e| PyValueError::new_err(e))
}

/// Compute the connected components of an unweighted graph.
///
/// Runs a standard BFS/union-find over the graph and assigns each node to
/// its component. Use the result with ``largest_component`` or ``partition``.
///
/// Args:
///     graph: The graph to analyse.
///
/// Returns:
///     A list of length ``graph.n`` where element ``i`` is the component ID of
///     node ``i``. IDs are arbitrary non-negative integers.
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
pub fn connected_components(graph: &CsrGraph) -> Vec<usize> {
    find_connected_components(&graph.inner)
}

/// Return the ID and size of the largest cluster.
///
/// Convenience helper — iterates over a cluster-label vector and finds the
/// most populous label.
///
/// Args:
///     clusters: A list of cluster IDs (e.g. from ``connected_components`` or
///               ``find_communities``).
///
/// Returns:
///     A tuple ``(cluster_id, size)`` for the largest cluster.
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
pub fn largest_component(clusters: Vec<usize>) -> (usize, usize) {
    largest_cluster(&clusters)
}
