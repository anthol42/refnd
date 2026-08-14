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

**Deterministic subgraph extraction.**
`CsrGraph::subgraph(nodes: &[usize])` builds the induced subgraph on `nodes`, reindexing to contiguous ids in the order of the provided `nodes`. Returns the subgraph plus a `BTreeMap<usize, u32>` from old id to new id; `BTreeMap` keeps the mapping's iteration order reproducible.

**Two objective functions with different semantics.**
- **Modularity** (`LeidenObjective::Modularity`): Weighted by node strengths (sum of incident edge weights). Resolution parameter is normalized by total graph strength. Suitable for detecting communities in networks with heterogeneous node degrees.
- **CPM** (`LeidenObjective::CPM`, Constant Potts Model): All nodes have unit weight. Resolution is absolute. Simpler semantics, useful for uniform community detection.

**Probabilistic refinement via exponential likelihood weighting.**
In `merge_nodes`, cluster choices are weighted by `exp(diff / beta)`, where `diff` is the modularity (or CPM) gain and `beta` is a temperature parameter. High `beta` makes the choice more uniform; low `beta` favors the best cluster. This stochasticity helps escape local optima.

**Cluster ID reindexing and compaction.**
After `fastmove_nodes` and `merge_nodes`, cluster IDs are reindexed to [0..k) to ensure dense numbering (no gaps). This simplifies downstream aggregation and memory use.

**`self.membership` is only ever written by the flatten step in `find_partition`.**
It must be written at *every* level where the loop continues (`continue_clustering`), including level 0. Reference implementations (e.g. igraph's C `community_leiden`) alias their output `membership` pointer directly onto the level-0 working buffer, so `fastmove_vertices`'s in-place writes are automatically visible with no copy needed -- that's why they can special-case level 0 as a no-op. This port instead clones `self.membership` into `aggregated_membership` up front, so level 0's result has no other path back to `self.membership`; skipping that write (as a naive line-for-line port of the C guard would) silently discards the whole clustering whenever the hierarchy happens to converge in exactly two levels. Level 0's copy is a plain `copy_from_slice` rather than the general indexed gather used for level > 0, since `super_node_map` is still the identity mapping at that point in the loop -- if that ordering ever changes, this shortcut needs revisiting.

**`fastmove_nodes`'s stay-in-place baseline must use the same penalized diff formula as every candidate cluster.**
The gain formula is `diff = E(v, C) - resolution * node_weight(v) * cluster_weight(C)`. The initial `max_diff` (representing "leave `v` in `current_cluster`") has to be computed with that same formula applied to `current_cluster`, exactly like every other candidate in the `neighbor_clusters` loop -- matching igraph's `leiden_fastmove_vertices`, which computes its pre-loop baseline the identical way (`leiden.c`, "Calculate maximum diff"). Using the raw, un-penalized `weight_to_cluster[current_cluster]` as the baseline (as an earlier version of this code did) gives staying put a discount no other candidate gets, biasing the local search toward staying in worse clusters. The effect isn't visible on trivial/well-separated graphs (any real improvement clears the inflated bar anyway) but shows up as a measurable, consistent quality gap on anything with real optimization difficulty -- caught by comparing achieved objective values (not partition identity) against igraph over repeated runs, see `pytests/test_leiden_accuracy.py`.

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
