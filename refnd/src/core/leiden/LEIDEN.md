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

**`leidenp.rs` is a parallel fork of `leiden.rs`, kept algorithmically identical except where noted below.**
`leiden.rs` is the sequential reference implementation and is never modified as part of the parallelization work; `leidenp.rs` (`fast_find_communities`) is being incrementally rewritten phase-by-phase to use `rayon`, capped at 8 threads via a dedicated `rayon::ThreadPool` built once in `fast_find_communities` and stored on `LeidenState`. Progress so far:

- **`merge_nodes` -- parallelized.** Clusters partition the node set, so each cluster's local refinement is independent and runs concurrently via `cluster_scratch.par_iter_mut()`. Two things needed redesigning versus the sequential version: (1) the shared `refined_membership` scratch buffer (indexed by global node id) is now a per-thread `thread_local!` (`MERGE_SCRATCH`), since a single shared buffer would alias across threads -- reused across all calls on that worker thread, resized (never shrunk) lazily to the current level's node count; (2) each cluster now returns its own locally-compacted refined ids ([0, k_i)) instead of writing directly into a globally-offset shared counter, since threading one running offset through concurrent calls isn't possible -- a cheap sequential prefix-sum over the per-cluster counts afterward reconciles these into one globally-contiguous id range. This is the only serial part of the phase.
- **`aggregate` -- parallelized.** Each super-node `c`'s work (scan its members' edges, sum weight to neighbouring super-nodes, pick a representative) is independent across `c` and runs concurrently via `refined_clusters.par_iter()`. Same shared-scratch problem as `merge_nodes`: `weight_to_cluster`/`is_neighbor_cluster` (sized to the *new*, coarser graph's node count) become a per-thread `thread_local!` (`AGGREGATE_SCRATCH`), reused across calls and lazily grown (never shrunk). Each cluster returns its own local edge list instead of appending to one shared `aggregated_edges`; concatenated by a cheap sequential pass afterward (order-preserving, so the result matches the sequential version's edge order exactly, not just its edge set).
- **`fastmove_nodes` -- parallelized, with a different design than the other two phases.** `merge_nodes`/`aggregate` stay data-parallel over already-disjoint work with a small serial reconciliation pass at the end; that shape doesn't work here because the natural per-node reconciliation is O(n), not O(clusters) -- an earlier attempt at a frozen-snapshot-decide-then-serial-apply design (mirroring the other two phases) hit real, measured problems: batched decisions caused genuine oscillation (neighbours swapping into each other's clusters based on stale data), and the serial reconciliation pass's O(n) cost meant it was *slower* than the sequential version at production scale (confirmed via `top -H`: only 1 of 8 threads ever busy). What's implemented instead follows two published, benchmarked reference implementations (NetworKit's PLM; the GVE-Leiden paper/implementation, arxiv.org/abs/2312.13936, github.com/puzzlef/leiden-communities-openmp): a **fused** parallel decide+apply with no separate phases at all. Every node is swept in parallel each round; each thread scans its own node's neighbours, decides the best move, and commits it immediately via atomics (`membership`/`cluster_weights` become `Vec<AtomicU32>` for the call's duration, `Ordering::Relaxed` throughout). An `affected: Vec<AtomicBool>` flag per node (cleared when processed, set on a mover's neighbours) replaces the explicit active-list both the abandoned design and the references' predecessors use. `MAX_FASTMOVE_ROUNDS` still caps the loop -- both references cap their own move phase the same way, confirming this is standard practice for the technique, not a hack. One detail neither reference needed: "become a new singleton cluster" is handled by reserving `graph.n + v` as node `v`'s own private, collision-free cluster id, avoiding the need for a shared/serial id allocator -- cluster ids in this function range over `[0, 2*graph.n)`, not `[0, graph.n)`.

All three phases are now parallelized. See `LEIDEN_IMPROVEMENTS.md` (repo root) for the full history and benchmark results.

## Structure
```
src/
  core/
    leiden/
      mod.rs
      csr_graph.rs        # CsrGraph: CSR sparse graph representation
      leiden.rs           # find_communities(): Leiden community detection (sequential reference)
      leidenp.rs           # fast_find_communities(): parallel fork of leiden.rs, phase-by-phase
      utils.rs            # reindex_membership() helper
```
