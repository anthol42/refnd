# refnd.core

## Graph construction

### *class* refnd.core.EdgeStore(node_count: int, edges: Sequence[tuple[int, int, float]])

Bases: `object`

A compact, flat list of weighted directed edges between integer node IDs.

`EdgeStore` is the central data carrier in refnd: it is produced by
`exact_edges` and `HNSWState.edges`, consumed by `CsrGraph`, and can be
persisted to disk for later reuse.

Each edge is a triple `(src, dst, weight)` where `src` and `dst` are
zero-based node indices in `[0, node_count)` and `weight` is a `float32`
value.

Example:

```default
from refnd.core import EdgeStore

store = EdgeStore(node_count=3, edges=[(0, 1, 0.9), (1, 2, 0.7)])
print(len(store))     # 2
print(store[0])       # (0, 1, 0.9)
for src, dst, w in store:
    print(src, dst, w)

# numpy-style boolean masking: keep only the edges where mask[i] is True
store[[True, False]]   # EdgeStore with only (0, 1, 0.9)
```

#### edges() → list[tuple[int, int, float]]

Return all edges as a list of `(src, dst, weight)` triples.

#### graph(inweight_type: [INWeightType](#refnd.core.INWeightType) = INWeightType.Distance) → [CsrGraph](#refnd.core.CsrGraph)

Build a `CsrGraph` from this edge store.

* **Parameters:**
  **inweight_type** – How to convert the raw edge weights to similarity-like weights.
  See `INWeightType`. Pass `INWeightType.Unweighted` to ignore
  weights entirely (every edge gets weight `1.0`). Defaults to
  `INWeightType.Distance`, meaning weights are assumed to be a distance
  measurement, and are converted to similarities.
* **Returns:**
  A `CsrGraph` backed by this edge list.

#### *static* load(path: str) → [EdgeStore](#refnd.core.EdgeStore)

Load an EdgeStore that was previously saved with `EdgeStore.save`. It infers the format
from the extension of the path, `.edgelist` or `.edgestr`.

**Version compatibility (binary format only):** The binary `.edgestr` format is versioned
to the exact package version. A file saved with a different package version will fail to
load with a version mismatch error. There is no forward or backward compatibility. The text
`.edgelist` format is not versioned and may be portable across versions.

* **Parameters:**
  **path** – Path to the file produced by `save`.
* **Returns:**
  The deserialized `EdgeStore`.
* **Raises:**
  **IOError** – If the file cannot be read, the format is invalid, or (for `.edgestr` files)
      the saved version does not match the running package version.

#### node_count() → int

Return the total number of nodes this store was created with.

#### save(path: str) → None

Serialize this EdgeStore to disk. It supports two file formats: `text` with
`.edgelist` extension or `binary` with `.edgestr` extension. Binary is usually 2x
more space efficient at the cost of not being human-readable.

**Version compatibility (binary format only):** The binary `.edgestr` format is versioned
to the exact package version and is not forward or backward compatible. A file saved with
a different package version will fail to load with a version mismatch error. The text
`.edgelist` format has no version stamp and may be portable across versions.

* **Parameters:**
  **path** – Destination file path.
* **Raises:**
  **IOError** – On any I/O failure.

Example:

```default
from refnd.core import EdgeStore

store = EdgeStore(node_count=3, edges=[(0, 1, 0.9), (1, 2, 0.7)])
store.save("my/path/myedges.edgelist") # Text format
store.save("my/path/myedges.edgestr")  # Bin format
```

#### return *: [EdgeStore](#refnd.core.EdgeStore)*

### *class* refnd.core.CsrGraph(edges: [EdgeStore](#refnd.core.EdgeStore), inweight_type: [INWeightType](#refnd.core.INWeightType) = INWeightType.Distance)

Bases: `object`

Compressed Sparse Row graph built from an `EdgeStore`. A CSR graph is compact, and highly
efficient for traversal, exploiting cache structure and maximizing cache hits. However, it is
immutable, so adding a new node or edge require building the full graph again.

`CsrGraph` is an immutable adjacency structure. Because of its property of being very
efficient for traversal, it is used by graph algorithms such as `partition` and
`connected_components`.

Properties:
: - `n`: Number of nodes.
  - `m`: Sum of all edge weights (or total edge count when `inweight_type=INWeightType.Unweighted`).

Example:

```default
from refnd.core import EdgeStore, CsrGraph

store = EdgeStore(4, [(0,1,0.9),(1,2,0.8),(2,3,0.6)])
g = CsrGraph(store)
print(g.n)               # 4
print(g.neighbors(1))    # [(0, 0.9), (2, 0.8)]
print(g.strength(1))     # 1.7
```

#### m

Total weight (each edge counted once)

#### n

Number of nodes

#### neighbors(v: int) → list[tuple[int, float]]

Return the neighbors of node `v` as a list of `(node_id, weight)` pairs.

* **Parameters:**
  **v** – Zero-based node index.

#### strength(v: int) → float

Return the strength (weighted degree) of node `v`.

When the graph is unweighted this equals the degree (number of neighbors).

* **Parameters:**
  **v** – Zero-based node index.

#### subgraph(nodes: Sequence[int]) → tuple[[CsrGraph](#refnd.core.CsrGraph), dict[int, int]]

Extract a subgraph composed of a subset of nodes.

New node ids are assigned contiguously in the order `nodes` is given.

All edges pointing outside the subgraph are ignores.

* **Parameters:**
  **nodes** – Node ids to keep, in the desired new order.
* **Returns:**
  A tuple `(subgraph, mapping)` where `mapping` is a dict from old node id
  to new node id.

Example:

```default
sub, mapping = g.subgraph([2, 0, 3])
print(sub.n)      # 3
print(mapping)    # {0: 1, 2: 0, 3: 2}
```

#### edges *: [EdgeStore](#refnd.core.EdgeStore)*

#### inweight_type *: [INWeightType](#refnd.core.INWeightType)*

#### return *: [CsrGraph](#refnd.core.CsrGraph)*

## HNSW approximate nearest neighbours

### *class* refnd.core.HNSWConfig(proximity_threshold: float = 0.0, ef_construction: int = 64, m: int = 16, m_max: int = 16, m_max0: int = 32, m_l: float = 0.36, ef_init: int = 1, extend_candidates: bool = False, keep_pruned_connections: bool = True, keep_all_edges: bool = True, cache_capacity: int = 2000000, cache_shards: int = 64, n_threads: int = 0, shuffle: bool = False, use_heuristic: bool = True, strict_ef: bool = False, threshold_based_neighbourhood: bool = False)

Bases: `object`

Configuration for the HNSW approximate nearest-neighbour index.

All parameters have sensible defaults; in most cases you only need to `ef_construction`, and
`proximity_threshold`.

**Parameters:**

- `proximity_threshold`  *(float)* — Distance threshold under which a connection is established in the proximity graph.
- `ef_construction`  *(int)* — Candidate list size during build. Default `64`.
- `m`  *(int)* — Bidirectional links per node at layers > 0.
  Higher = better recall, more memory, slower build. Typical: 8–64.
- `m_max`  *(int)* — Maximum connections per node at layers > 0. Usually equal to `m`.
- `m_max0`  *(int)* — Maximum connections at layer 0. Usually `2 * m`.
- `m_l`  *(float)* — Level generation factor. Usually `0.36 ≈ 1 / ln(m)`.
- `ef_init`  *(int)* — Candidate list size during initial-layer insertion. Almost always 1 for
  a greedy search.
- `extend_candidates`  *(bool)* — Use neighbors of neighbors as candidates, making search more exhaustive.
- `keep_pruned_connections`  *(bool)* — Retain discarded candidates to fill up to `m` connections when not enough connections are found.
- `keep_all_edges`  *(bool)* — Record every below-threshold distance computed during build into
  the proximity graph returned by `edges()`. Default `True`. Setting it to `False` makes
  `edges()` return `None`, but removes a per-hit locked hashmap insert from the hot path.
  Worth doing when the proximity graph is not required.
- `cache_capacity`  *(int)* — Maximum cached kernel scores. Increasing it increase the memory
  footprint, but also cache hits, which can improve runtime performances for computationally expensive kernels.
  Set to `0` to disable the cache entirely — every distance is recomputed directly, which is
  faster for cheap kernels (e.g. `TanimotoBit`/`TanimotoReal`) where cache overhead exceeds the
  cost of the kernel itself.
- `cache_shards`  *(int)* — Number of cache shards (reduces lock contention).
- `n_threads`  *(int)* — Threads used during build. `0` = all available cores.
- `shuffle`  *(bool)* — Shuffle insertion order before building. Can create a less biased
  graph, but reduces cache hits, which increases runtime.
- `use_heuristic`  *(bool)* — Use the heuristic neighbour-selection from the paper (recommended).
- `strict_ef`  *(bool)* — If `True`, enforces the result set size to exactly `ef` during search. Empirically, setting this to `False` can improve runtime performance, as it allows halving `ef_construction` without sacrificing accuracy.
- `threshold_based_neighbourhood`  *(bool)* — Select a minimum of `m` neighbors like the classic algorithm, but doesn’t bound the neighbourhood size as all candidates that are closer than the threshold are kept.

#### cache_capacity *: int*

#### cache_shards *: int*

#### dict() → dict

Return the configuration as a plain Python dict.

#### ef_construction *: int*

#### ef_init *: int*

#### extend_candidates *: bool*

#### keep_all_edges *: bool*

#### keep_pruned_connections *: bool*

#### m *: int*

#### m_l *: float*

#### m_max *: int*

#### m_max0 *: int*

#### n_threads *: int*

#### proximity_threshold *: float*

#### shuffle *: bool*

#### strict_ef *: bool*

#### threshold_based_neighbourhood *: bool*

#### use_heuristic *: bool*

#### return *: [HNSWConfig](#refnd.core.HNSWConfig)*

### *class* refnd.core.HNSWState(variant: [kernels.KernelVariant](kernels.md#refnd.kernels.KernelVariant), data: Any, \*args: Any, proximity_threshold: float = 0.0, ef_construction: int = 64, m: int = 16, m_max: int = 16, m_max0: int = 32, m_l: float = 0.36, ef_init: int = 1, extend_candidates: bool = False, keep_pruned_connections: bool = True, keep_all_edges: bool = True, cache_capacity: int = 2000000, cache_shards: int = 64, n_threads: int = 0, shuffle: bool = False, use_heuristic: bool = True, strict_ef: bool = False, threshold_based_neighbourhood: bool = False, \*\*kwargs: Any)

Bases: `object`

Stateful Hierarchical Navigable Small World (HNSW) approximate nearest-neighbour index. This
class is used to build and search in the index, whereas `HNSWIndex` is a read-only *internal*
representation of the index used for saving / loading / structure exploration.

`HNSWState` wraps the dataset and the graph index together. The typical
workflow is:

1. Construct with the data and a `KernelVariant`.
2. Call `build()` to insert all items and form the graph.
3. Call `search()` to query, or `edges()` to extract the proximity graph
   for downstream clustering / splitting.

HNSW parameters (`proximity_threshold`, `ef_construction`, …) are the same as `HNSWConfig`
and can be passed directly to the constructor as keyword arguments.

`data` may be any Python iterable — a `list`, a `tuple`, a generator, anything with
`__iter__`. It’s drained one item at a time into the constructor’s own Rust-owned copy,
so a generator never needs to be fully materialized into a Python-side list first — both
copies never need to coexist at once. That matters when a single item’s Python
representation is large enough that two full-dataset copies wouldn’t fit in memory
together. When `data` supports `len()` (a `list`/`tuple`), that’s used to
pre-size the Rust-owned copy exactly; it doesn’t change how `data` is walked.

* **Parameters:**
  * **variant** – Kernel to use (e.g.  `KernelVariant.AlignmentGlobal`, `KernelVariant.AlignmentLocal`, `KernelVariant.TanimotoBit`, *etc*).
  * **data** – The dataset — a list of items matching the kernel type (e.g. `list[str]` or
    `list[np.ndarray]`), or any other Python iterable of such items (e.g. a generator).
  * **proximity_threshold, ef_construction, m, m_max, m_max0, m_l, ef_init, extend_candidates,** – keep_pruned_connections, keep_all_edges, cache_capacity, cache_shards,
    n_threads, shuffle, use_heuristic, strict_ef,
    threshold_based_neighbourhood: See `HNSWConfig` for descriptions.

Properties:
: config (HNSWConfig): The config in use.
  index (HNSWIndex): A snapshot of the current graph structure.

Example:

```default
from refnd import HNSWState, KernelVariant

seqs = ["MKTAYIAK", "MKTAYIAKQR", "ACDEFGHIKLM", "MKTAYIAKQRQIS"]
state = HNSWState(KernelVariant.AlignmentGlobal, seqs, proximity_threshold=0.3, ef_construction=64)
state.build()
results = state.search(["MKTAYIAK"], k=2)
# results[0] -> [(0, 1.0), (1, 0.88)]

store = state.edges()        # EdgeStore for graph-based splitting
state.save("index.hnsw")
state2 = HNSWState.load(KernelVariant.AlignmentGlobal, "index.hnsw", seqs)
```

#### build(progress: bool = True) → None

Build the HNSW index by inserting all data items.

Must be called before `search` or `edges`. Calling `build` a second
time raises a `RuntimeError`; construct a new `HNSWState` instead.

* **Parameters:**
  **progress** – Display a progress bar. Defaults to `True`.
* **Raises:**
  **RuntimeError** – If the index has already been built.

#### config

Config used to initiate this object.

#### edges() → Optional[[EdgeStore](#refnd.core.EdgeStore)]

Extract the proximity graph as an `EdgeStore`.

Returns all edges found during `build`. All distance measurements below the
`proximity_threshold`.

* **Returns:**
  An `EdgeStore` with `node_count = dataset_size`, or `None` if the index was built
  with `keep_all_edges=False`.

#### extend_build(data: Any, progress: bool = True) → None

Extend an already-built index with more data: appended to the dataset, then
inserted into the graph. Existing nodes and edges are untouched.

`data` accepts any Python iterable, same as the constructor (see `HNSWState`’s
class docstring) – drained one item at a time so a generator never needs to be
fully materialized into a Python-side list first.

* **Parameters:**
  * **data** – The new items to add (same type as the original dataset).
  * **progress** – Display a progress bar. Defaults to `True`.
* **Raises:**
  **RuntimeError** – If `build` has not been called yet.

Example:

```default
from refnd import HNSWState, KernelVariant

seqs = ["MKTAYIAK", "MKTAYIAKQR", "ACDEFGHIKLM"]
state = HNSWState(KernelVariant.AlignmentGlobal, seqs, proximity_threshold=0.3)
state.build()

more_seqs = ["MKTAYIAKQRQIS", "ACDEFGHIKLMNP"]
state.extend_build(more_seqs)
state.index.dataset_size  # 5

results = state.search(["MKTAYIAKQRQIS"], k=2)
# results[0] -> [(3, 0.0), (1, 0.23)] -- found itself among the new items
```

#### get_layer(layer_idx: int = 0, directed: bool = False, weights: bool = True, progress: bool = True) → [EdgeStore](#refnd.core.EdgeStore)

Return the adjacency lists for a specific HNSW layer.

* **Parameters:**
  **layer_idx** – Zero-based layer index (0 = base layer with most nodes).
* **Returns:**
  A list of length `dataset_size` where element `i` is the list of
  neighbour IDs of node `i` at this layer. Node IDs are their index in the original dataset.
* **Raises:**
  **IndexError** – If `layer_idx` is out of range.

Edges of one HNSW layer, as an `EdgeStore`.

* **Parameters:**
  * **layer_idx** – Zero-based layer index (0 = base layer with most nodes).
  * **directed** – If `True`, every edge is returned exactly as internally
    recorded – an undirected connection contributes one entry per
    endpoint, i.e. `(x, y)` is distinct from `(y, x)`. If
    `False` (default), edges are canonicalized and deduplicated,
    so each undirected pair appears exactly once.
  * **weights** – If `True` (default), each edge’s weight is its real
    distance, computed on the fly since not stored in the hierarchical graph.
    If `False`, every edge gets weight
    `1.0` – much cheaper when the real distance isn’t needed.
  * **progress** – Display a progress bar while distances are computed.
    Only meaningful when `weights=True`. Defaults to `False`.
* **Returns:**
  An `EdgeStore` with `node_count = dataset_size`.
* **Raises:**
  **IndexError** – If `layer_idx` is out of range.

#### index

HNSWIndex snapshot.

#### is_built

Whether `build` has been called on this index.

#### *static* load(variant: [kernels.KernelVariant](kernels.md#refnd.kernels.KernelVariant), path: str, data: Any, \*args: Any, \*\*kwargs: Any) → [HNSWState](#refnd.core.HNSWState)

Load an HNSWState form a binary file saved with `HNSWState.save`.

**Version compatibility:** The binary format is versioned to the exact package version.
A file saved with a different package version (older or newer) will fail to load with
a version mismatch error. There is no forward or backward compatibility guarantee during
the unstable pre-0.1.0 phase.

`data` accepts any Python iterable, same as the constructor (see `HNSWState`’s
class docstring) — including a generator re-reading a cache file.

* **Parameters:**
  * **variant** – Must match the kernel used during the original build.
  * **path** – Path to the saved file.
  * **data** – The original dataset (required to re-attach the kernel).
* **Returns:**
  The restored `HNSWState`, ready to call `search` or `edges` if the index was built.
* **Raises:**
  **RuntimeError** – If the file cannot be read, the format is invalid, or the saved
      version does not match the running package version.

#### save(path: str) → None

Serialize the full state (index + config) to a binary file.

The saved file can be loaded back with `HNSWState.load`. The original
data must be provided again at load time (it is not embedded in the file).

**Version compatibility:** The binary format is versioned to the exact package version.
A file saved with a different package version (older or newer) will fail to load with
a version mismatch error. There is no forward or backward compatibility guarantee during
the unstable pre-0.1.0 phase.

* **Parameters:**
  **path** – Destination file path.
* **Raises:**
  **RuntimeError** – On serialisation or I/O failure.

#### search(queries: Any, k: int = 1, ef: int = 64, threads: int = 0, progress: bool = True) → list[list[tuple[int, float]]]

Search the index for approximate nearest neighbours.

For each query item, returns the `k` most similar items in the dataset,
sorted by descending similarity. The quality of the approximation is
controlled by `ef`: larger values explore more candidates and improve
recall at the cost of speed.

* **Parameters:**
  * **queries** – List of query items (same type as the indexed data).
  * **k** – Number of nearest neighbours to return per query. Defaults to `1`.
  * **ef** – Size of the dynamic candidate list during search. Must be ≥ `k`.
    Defaults to `64`.
  * **threads** – Number of parallel threads. `0` = all available cores.
  * **progress** – Display a progress bar. Defaults to `True`.
* **Returns:**
  A list of length `len(queries)`. Each element is a sorted list of
  up to `k` tuples `(dataset_index, similarity_score)`.
* **Raises:**
  **RuntimeError** – If `build` has not been called yet.

#### variant *: [KernelVariant](kernels.md#refnd.kernels.KernelVariant)*

#### data *: Any*

#### args *: Any*

#### proximity_threshold *: float*

#### ef_construction *: int*

#### m *: int*

#### m_max *: int*

#### m_max0 *: int*

#### m_l *: float*

#### ef_init *: int*

#### extend_candidates *: bool*

#### keep_pruned_connections *: bool*

#### keep_all_edges *: bool*

#### cache_capacity *: int*

#### cache_shards *: int*

#### n_threads *: int*

#### shuffle *: bool*

#### use_heuristic *: bool*

#### strict_ef *: bool*

#### threshold_based_neighbourhood *: bool*

#### kwargs *: Any*

#### return *: [HNSWState](#refnd.core.HNSWState)*

### *class* refnd.core.HNSWIndex

Bases: `object`

Read-only snapshot of the HNSW graph structure after a build.

Obtained via `HNSWState.index`. Primarily useful for inspection,
serialisation, and debugging; normal users interact with `HNSWState`
instead.

Properties:
: dataset_size (int): Number of items indexed.
  layers (list): Nested multi-layer adjacency list `layers[layer][node] = [neighbor_ids]`.
  entry_point (tuple | None): `(layer, node_id)` of the global entry point, or `None` if empty.
  max_layers (int): Number of layers in the hierarchy.
  proximity_edges (list): List of `((src, dst), score)` for proximity-threshold edges.
  config (HNSWConfig): The config used to build this index.

#### config

#### dataset_size

#### entry_point

#### layers

Nested multi-layer adjacency list, layers[layer][node] = [neighbor_ids]. Layers
above 0 may be stored sparsely internally (most nodes aren’t present at those
layers); this densifies them into the full node-id-indexed shape on access.

#### *static* load(path: str) → [HNSWIndex](#refnd.core.HNSWIndex)

Load an index previously saved with `HNSWIndex.save`.

**Version compatibility:** The binary format is versioned to the exact package version.
A file saved with a different package version (older or newer) will fail to load with
a version mismatch error. There is no forward or backward compatibility guarantee during
the unstable pre-0.1.0 phase.

* **Parameters:**
  **path** – Path to the saved file.
* **Returns:**
  The deserialised `HNSWIndex`.
* **Raises:**
  **RuntimeError** – If the file cannot be read, the format is invalid, or the saved
      version does not match the running package version.

#### max_layers

#### proximity_edges

#### save(path: str) → None

Save the object to a binary representation (e.g.  *.hnsw* file).

The binary format is versioned to the exact package version and is not
forward or backward compatible. A file saved with version X can only be
loaded by the exact same version X. Files saved with a different version
will fail to load with a version mismatch error.

* **Parameters:**
  **path** – Destination file path (recommended with a .hnsw extension)
* **Raises:**
  **RuntimeError** – On serialisation or I/O failure.

## Graph algorithms

### *class* refnd.core.INWeightType

Bases: `object`

How to convert a raw `EdgeStore` weight into the similarity-like edge weight
used internally by `CsrGraph` and the Leiden algorithm.

- `Similarity` — the raw weight is already a similarity; used as-is.
- `Distance` — the raw weight is a distance in `[0, ∞)`; mapped to
  `1 / (1 + w)` so closer nodes get a higher weight.
- `SimilarityComplement` — the raw weight is `1 - similarity`; mapped
  back to a similarity via `1 - w`.
- `Unweighted` — the raw weight is ignored; every edge weight is set to `1.0`.

#### Distance *= INWeightType.Distance*

#### Similarity *= INWeightType.Similarity*

#### SimilarityComplement *= INWeightType.SimilarityComplement*

#### Unweighted *= INWeightType.Unweighted*

### *class* refnd.core.LeidenObjective

Bases: `object`

Objective function used by the Leiden community-detection algorithm.

- `Modularity` — Maximise Newman-Girvan modularity. Good default for most graphs.
- `CPM` — Constant Potts Model. Finds communities of a fixed internal density.

#### CPM *= LeidenObjective.CPM*

#### Modularity *= LeidenObjective.Modularity*

### refnd.core.find_communities(graph: [CsrGraph](#refnd.core.CsrGraph), gamma: float = 1.0, beta: float = 0.01, n_iterations: int = 2, objective: [LeidenObjective](#refnd.core.LeidenObjective) = LeidenObjective.Modularity) → list[int]

Detect communities in a graph with the Leiden algorithm.

The Leiden algorithm is an improvement over Louvain that guarantees
well-connected communities. Use the returned cluster labels with
`partition` to produce train/test splits that consider community boundaries.

* **Parameters:**
  * **graph** – The graph to partition into communities.
  * **gamma** – Resolution parameter. Higher values produce more, smaller communities.
    For `Modularity`, typical range is 0.5–2.0; default `1.0`.
    For `CPM`, it represents the minimum internal edge density.
  * **beta** – Randomness parameter controlling the refinement phase.
    Smaller values yield more deterministic results. Default `0.01`.
  * **n_iterations** – Number of optimisation passes. More iterations improve
    quality at the cost of runtime.
  * **objective** – `LeidenObjective.Modularity` (default) or `LeidenObjective.CPM`.
* **Returns:**
  A list of length `graph.n` where element `i` is the cluster ID of node `i`.
  Cluster IDs are arbitrary non-negative integers.

Example:

```default
from refnd.core import CsrGraph, EdgeStore, find_communities, LeidenObjective, INWeightType

store = EdgeStore(4, [(0,1,0.9),(1,2,0.8),(2,3,0.6)])
g = CsrGraph(store, inweight_type=INWeightType.Similarity)
clusters = find_communities(g, gamma=1.0, n_iterations=20) # e.g. [0, 1, 2, 3]
```

### refnd.core.find_components(graph: [CsrGraph](#refnd.core.CsrGraph)) → list[int]

Finds the connected components of a graph.

Runs a standard BFS/union-find over the graph and assigns each node to
its component.

* **Parameters:**
  **graph** – The graph to find components.
* **Returns:**
  A list of length `graph.n` where element `i` is the component ID of
  node `i`. IDs are arbitrary non-negative integers.

Example:

```default
from refnd.core import EdgeStore, find_components

store = EdgeStore(6, [(0,1,0.9),(1,2,0.8),(3,4,0.7),(4,5,0.6)])
g = store.graph()
clusters = find_components(g) # [0, 0, 0, 1, 1, 1]
```

### refnd.core.partition(clusters: Sequence[int], graph: [CsrGraph](#refnd.core.CsrGraph), test_ratio: float = 0.2, seed: Optional[int] = None, post_filtering: bool = False) → tuple[list[int], list[int]]

Split a clustered dataset into train and test index sets along clusters

Clusters are kept intact — no cluster is split across train and test.
The function greedily and randomly assigns whole clusters to the test set until the test size
is larger than `test_ratio`, then assigns the rest to train.

This is the key function for leakage-free dataset splitting: run
`find_communities`, `find_components`, or your own clustering, then call `partition`
to obtain indices you can use to slice your data arrays.

* **Parameters:**
  * **clusters** – A list of cluster IDs, one per data point (as returned by
    `find_communities` or `connected_components`). Length must
    equal the number of nodes in `graph`.
  * **graph** – The graph used to derive community structure. Its node count
    must match `len(clusters)`.
  * **test_ratio** – Target fraction of data points to place in the test set.
    The actual fraction may differ slightly because clusters
    are indivisible. Defaults to `0.2`.
  * **seed** – Optional integer seed for reproducible cluster shuffling.
    Pass `None` for a random assignment.
  * **post_filtering** – If `True`, remove train nodes that share an edge with
    any test node (stricter leakage prevention at the
    cost of a smaller train set). Use only when clusters are found from
    communities (`find_communities`). It won’t have any effect when used with
    `connected_components` as this prevents inter-cluster connections.
* **Returns:**
  A tuple `(train_indices, test_indices)` of zero-based data-point indices.
* **Raises:**
  **ValueError** – If `clusters` length does not match `graph.n`, or if
      `test_ratio` is not in `(0, 1)`.

Example:

```default
from refnd.core import (
    EdgeStore, CsrGraph, find_communities, partition, INWeightType
)

store = EdgeStore(6, [(0,1,0.9),(1,2,0.8),(3,4,0.7),(4,5,0.6)])
g = CsrGraph(store, inweight_type=INWeightType.Similarity)
clusters = find_communities(g) # or connected_components(g)
train_idx, test_idx = partition(clusters, g, test_ratio=0.3, seed=42)
```

## Exact search

### refnd.core.exact_edges(variant: [kernels.KernelVariant](kernels.md#refnd.kernels.KernelVariant), data: Any, proximity_threshold: float = 0.5, n_threads: int = 0, progress: bool = True, \*args: Any, \*\*kwargs: Any) → [EdgeStore](#refnd.core.EdgeStore)

Compute all pairs of data points whose similarity exceeds a threshold (exact, brute-force).

Evaluates every unordered pair `(i, j)` with `i < j` and records an edge
when `kernel(data[i], data[j]) <= proximity_threshold`. This is O(n²) in the number
of data points; prefer the approximate\`\`HNSWState\`\` for large datasets.

Extra positional and keyword arguments are forwarded to the kernel constructor.

* **Parameters:**
  * **variant** – Which kernel to use (`KernelVariant.AlignmentGlobal` or
    `KernelVariant.AlignmentLocal`).
  * **data** – Sequence of data items (e.g. `list[str]` for protein sequences).
  * **proximity_threshold** – Maximum distance for an edge to be recorded.
  * **n_threads** – Number of parallel threads. `0` uses all available cores.
  * **progress** – Show a progress bar. Defaults to `True`.
* **Returns:**
  An `EdgeStore` containing all edges whose distance ≤ `proximity_threshold`.

Example:

```default
from refnd import KernelVariant, exact_edges

seqs = ["MKTAYIAK", "MKTAYIAKQR", "ACDEFGHIKLM"]
store = exact_edges(KernelVariant.AlignmentGlobal, seqs, proximity_threshold=0.5)
print(len(store))   # number of similar pairs
```

### refnd.core.exact_nearest_neighbors(variant: [kernels.KernelVariant](kernels.md#refnd.kernels.KernelVariant), queries: Any, references: Any, k: int, threads: int = 0, progress: bool = True, \*args: Any, \*\*kwargs: Any) → list[list[tuple[int, float]]]

Find the k nearest neighbors of each query in a reference set (exact, brute-force).

For every query item, scores it against every reference item using the
chosen kernel and returns the top-k references sorted by ascending
distance. Complexity is O(n_queries × n_references); prefer
`HNSWState.search` for approximate nearest neighbors with large datasets.

Extra positional and keyword arguments are forwarded to the kernel constructor.

* **Parameters:**
  * **variant** – Which kernel to use (`KernelVariant.AlignmentGlobal` or
    `KernelVariant.AlignmentsLocal`).
  * **queries** – Sequence of query items.
  * **references** – Sequence of reference items to search over. List of items of the same type of the selected kernel variant.
  * **k** – Number of nearest neighbors to return per query. List of items of the same type of the selected kernel variant.
  * **threads** – Number of parallel threads. `0` uses all available cores.
  * **progress** – Show a progress bar. Defaults to `True`.
* **Returns:**
  A list of length `len(queries)`. Each element is a list of up to `k`
  tuples `(reference_index, similarity_score)` sorted by ascending distance.

Example:

```default
from refnd import KernelVariant, exact_nearest_neighbors

queries = ["MKTAYIAK"]
refs    = ["MKTAYIAKQR", "ACDEFGHIKLM", "MKTAYIAKQRQ"]
results = exact_nearest_neighbors(
    KernelVariant.AlignmentGlobal, queries, refs, k=2
)
# results[0] -> [(0, 0.20), (2, 0.27)]
```
