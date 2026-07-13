use std::collections::BTreeMap;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods, gen_stub_pyfunction};
use super::edge_store::EdgeStore;
use refnd_core::core::leiden::{CsrGraph as CoreCsrGraph, LeidenObjective as CoreLeidenObjective, INWeightType as CoreINWeightType, find_communities as CoreFindCommunities};

/// Objective function used by the Leiden community-detection algorithm.
///
/// - ``Modularity`` — Maximise Newman-Girvan modularity. Good default for most graphs.
/// - ``CPM`` — Constant Potts Model. Finds communities of a fixed internal density.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.core")]
#[derive(Clone, Copy, PartialEq)]
pub enum LeidenObjective {
    Modularity,
    CPM,
}

impl From<LeidenObjective> for CoreLeidenObjective {
    fn from(o: LeidenObjective) -> Self {
        match o {
            LeidenObjective::Modularity => CoreLeidenObjective::Modularity,
            LeidenObjective::CPM => CoreLeidenObjective::CPM,
        }
    }
}

/// How to convert a raw ``EdgeStore`` weight into the similarity-like edge weight
/// used internally by ``CsrGraph`` and the Leiden algorithm.
///
/// - ``Similarity`` — the raw weight is already a similarity; used as-is.
/// - ``Distance`` — the raw weight is a distance in ``[0, ∞)``; mapped to
///   ``1 / (1 + w)`` so closer nodes get a higher weight.
/// - ``SimilarityComplement`` — the raw weight is ``1 - similarity``; mapped
///   back to a similarity via ``1 - w``.
/// - ``Unweighted`` — the raw weight is ignored; every edge weight is set to ``1.0``.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.core")]
#[derive(Clone, Copy, PartialEq)]
pub enum INWeightType {
    Similarity,
    Distance,
    SimilarityComplement,
    Unweighted,
}

impl From<INWeightType> for CoreINWeightType {
    fn from(t: INWeightType) -> Self {
        match t {
            INWeightType::Similarity => CoreINWeightType::Similarity,
            INWeightType::Distance => CoreINWeightType::Distance,
            INWeightType::SimilarityComplement => CoreINWeightType::SimilarityComplement,
            INWeightType::Unweighted => CoreINWeightType::Unweighted,
        }
    }
}

/// Compressed Sparse Row graph built from an ``EdgeStore``. A CSR graph is compact, and highly
/// efficient for traversal, exploiting cache structure and maximizing cache hits. However, it is
/// immutable, so adding a new node or edge require building the full graph again.
///
/// ``CsrGraph`` is an immutable adjacency structure. Because of its property of being very
/// efficient for traversal, it is used by graph algorithms such as ``partition`` and
/// ``connected_components``.
///
/// Properties:
///     - ``n``: Number of nodes.
///     - ``m``: Sum of all edge weights (or total edge count when ``inweight_type=INWeightType.Unweighted``).
///
/// Example::
///
///     from refnd.core import EdgeStore, CsrGraph
///
///     store = EdgeStore(4, [(0,1,0.9),(1,2,0.8),(2,3,0.6)])
///     g = CsrGraph(store)
///     print(g.n)               # 4
///     print(g.neighbors(1))    # [(0, 0.9), (2, 0.8)]
///     print(g.strength(1))     # 1.7
#[gen_stub_pyclass]
#[pyclass(module = "refnd.core")]
pub struct CsrGraph {
    pub inner: CoreCsrGraph,
}

#[gen_stub_pymethods]
#[pymethods]
impl CsrGraph {
    /// Build a CsrGraph from an EdgeStore.
    ///
    /// Args:
    ///     edges: The edge store to build the graph from.
    ///     inweight_type: How to convert the raw ``EdgeStore`` weights to
    ///                    similarity-like weights. See ``INWeightType``. Pass
    ///                    ``INWeightType.Unweighted`` to ignore weights entirely
    ///                    (every edge gets weight ``1.0``). Defaults to
    ///                    ``INWeightType.Distance``.
    #[new]
    #[pyo3(signature = (edges, inweight_type = INWeightType::Distance))]
    fn new(edges: EdgeStore, inweight_type: INWeightType) -> Self {
        Self { inner: CoreCsrGraph::new(edges.node_count(), &edges.edges(), inweight_type.into()) }
    }

    /// Return the neighbors of node ``v`` as a list of ``(node_id, weight)`` pairs.
    ///
    /// Args:
    ///     v: Zero-based node index.
    fn neighbors(&self, v: usize) -> Vec<(u32, f32)> {
        self.inner.neighbors(v).to_vec()
    }

    /// Return the strength (weighted degree) of node ``v``.
    ///
    /// When the graph is unweighted this equals the degree (number of neighbors).
    ///
    /// Args:
    ///     v: Zero-based node index.
    fn strength(&self, v: usize) -> f32 {
        self.inner.strength(v)
    }

    /// Extract a subgraph composed of a subset of nodes.
    ///
    /// New node ids are assigned contiguously in the order ``nodes`` is given.
    ///
    /// All edges pointing outside the subgraph are ignores.
    ///
    /// Args:
    ///     nodes: Node ids to keep, in the desired new order.
    ///
    /// Returns:
    ///     A tuple ``(subgraph, mapping)`` where ``mapping`` is a dict from old node id
    ///     to new node id.
    ///
    /// Example::
    ///
    ///     sub, mapping = g.subgraph([2, 0, 3])
    ///     print(sub.n)      # 3
    ///     print(mapping)    # {0: 1, 2: 0, 3: 2}
    fn subgraph(&self, nodes: Vec<usize>) -> (CsrGraph, BTreeMap<usize, usize>) {
        let (inner, mapping) = self.inner.subgraph(&nodes);
        (CsrGraph { inner }, mapping)
    }

    /// Number of nodes
    #[getter]
    fn n(&self) -> usize { self.inner.n }

    /// Total weight (each edge counted once)
    #[getter]
    fn m(&self) -> f32 { self.inner.m }
}

/// Detect communities in a graph with the Leiden algorithm.
///
/// The Leiden algorithm is an improvement over Louvain that guarantees
/// well-connected communities. Use the returned cluster labels with
/// ``partition`` to produce train/test splits that consider community boundaries.
///
/// Args:
///     graph: The graph to partition into communities.
///     gamma: Resolution parameter. Higher values produce more, smaller communities.
///            For ``Modularity``, typical range is 0.5–2.0; default ``1.0``.
///            For ``CPM``, it represents the minimum internal edge density.
///     beta: Randomness parameter controlling the refinement phase.
///           Smaller values yield more deterministic results. Default ``0.01``.
///     n_iterations: Number of optimisation passes. More iterations improve
///                   quality at the cost of runtime.
///     objective: ``LeidenObjective.Modularity`` (default) or ``LeidenObjective.CPM``.
///
/// Returns:
///     A list of length ``graph.n`` where element ``i`` is the cluster ID of node ``i``.
///     Cluster IDs are arbitrary non-negative integers.
///
/// Example::
///
///     from refnd.core import CsrGraph, EdgeStore, find_communities, LeidenObjective, INWeightType
///
///     store = EdgeStore(4, [(0,1,0.9),(1,2,0.8),(2,3,0.6)])
///     g = CsrGraph(store, inweight_type=INWeightType.Similarity)
///     clusters = find_communities(g, gamma=1.0, n_iterations=20) # e.g. [0, 1, 2, 3]
#[gen_stub_pyfunction(module = "refnd.core")]
#[pyfunction]
#[pyo3(signature = (graph, gamma = 1.0, beta = 0.01, n_iterations = 2, objective = LeidenObjective::Modularity))]
pub fn find_communities(
    graph: &CsrGraph,
    gamma: f32,
    beta: f64,
    n_iterations: usize,
    objective: LeidenObjective,
) -> Vec<usize> {
    CoreFindCommunities(graph.inner.clone(), gamma, beta, n_iterations, objective.into())
}
