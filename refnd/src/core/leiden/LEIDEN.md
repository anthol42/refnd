## Key design decisions (Leiden)

**Two-phase iterative refinement.**
Each iteration has two phases:
- **`fastmove_nodes`**: Deterministic greedy local optimization. Nodes are shuffled and each node is moved to the neighbouring cluster that maximizes gain (or a new empty cluster). Neighbouring clusters are only those adjacent to the node in the graph, reducing the search space.
- **`merge_nodes`**: Probabilistic refinement within each cluster. Singleton clusters are dissolved, and nodes are probabilistically reassigned based on exponential likelihood weighting (scaled by the `beta` temperature parameter). This prevents premature convergence and finds better local optima.

**Hierarchical multi-level aggregation.**
After each iteration, clusters are aggregated into a coarser graph (super-nodes represent clusters). The algorithm repeats on the coarse graph. This multi-scale approach reduces computational cost and improves solution quality. The process continues until the partition stabilizes (no changes) or the number of clusters equals the number of nodes.

**Scratch buffer allocation for hot loops.**
Performance-critical loops (`fastmove_nodes`, `merge_nodes`, `aggregate`) pre-allocate all scratch buffers:
- `weight_to_cluster`: cumulative edge weights from a node to each neighbouring cluster.
- `is_neighbor_cluster`: `FixedBitSet` for O(1) membership checks (avoid repeated vector searches).
- `neighbor_clusters`: reusable vector of candidate cluster indices.
- Buffers are cleared and reused each iteration, avoiding allocation churn.

**Compact CSR graph representation.**
Graphs are stored in Compressed Sparse Row (CSR) format: `offsets` array (row pointers) + `adj` array (neighbor, weight pairs). This is memory-efficient for sparse graphs and enables cache-friendly sequential access. Self-loops are handled specially: counted once (not twice) when building adjacency lists.

**Two objective functions with different semantics.**
- **Modularity** (`LeidenObjective::Modularity`): Weighted by node strengths (sum of incident edge weights). Resolution parameter is normalized by total graph strength. Suitable for detecting communities in networks with heterogeneous node degrees.
- **CPM** (`LeidenObjective::CPM`, Constant Potts Model): All nodes have unit weight. Resolution is absolute. Simpler semantics, useful for uniform community detection.

**Probabilistic refinement via exponential likelihood weighting.**
In `merge_nodes`, cluster choices are weighted by `exp(diff / beta)`, where `diff` is the modularity (or CPM) gain and `beta` is a temperature parameter. High `beta` makes the choice more uniform; low `beta` favors the best cluster. This stochasticity helps escape local optima.

**Cluster ID reindexing and compaction.**
After `fastmove_nodes` and `merge_nodes`, cluster IDs are reindexed to [0..k) to ensure dense numbering (no gaps). This simplifies downstream aggregation and memory use.

## Structure
```
src/
  core/
    leiden/
      mod.rs
      csr_graph.rs        # CsrGraph: CSR sparse graph representation
      leiden.rs           # find_communities(): Leiden community detection
      utils.rs            # reindex_membership() helper
```
