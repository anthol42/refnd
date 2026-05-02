## Key design decisions

**Parallelism without a global lock.**
`HGraph` uses one `parking_lot::RwLock` per node. `add_edge` always locks `min(u,v)` before `max(u,v)` — the only deadlock rule. The entry point is a `Mutex<Option<Loc>>` (updated O(log N) times total, contention is negligible).

**Thread-local hot path.**
Each rayon worker owns its `ScratchBuffers` (heaps, bitset, snapshot vecs) and `StdRng`. Nothing shared in the inner loop except the graph locks, the sharded distance cache, and the `DashMap` for proximity edges.

**Sharded distance cache.**
Distances are cached in a shared `ShardedCache` — `N` independent `quick_cache::sync::Cache` shards. Pair `(i, j)` (with `i ≤ j`) always routes to shard `i & (N-1)` (fast power-of-two modulo). With 64 shards and 8 threads, the probability of two threads hitting the same shard simultaneously is ~12%, vs 100% for a single shared cache.

**Snapshot before FFI.**
`neighbors_snapshot()` clones a node's neighbour list under a brief read lock, then releases before calling into the distance function. Holding a lock across FFI would serialize all threads.

**Log-factorial ETA.**
Insert `k` costs O(log k), so total work ∝ log(N!). The progress bar computes `fraction_done = log(i!) / log(N!)` via Stirling's approximation and derives ETA from that — accurate from the first few seconds (5s).

**Neighbour selection strategies.**
Three strategies are available, selected per-insertion via config flags:
- `use_heuristic = true` (default) — `select_neighbors`: Algorithm 4 from the HNSW paper. Candidates are accepted only if they are closer to the query than to any already-selected neighbour, promoting diversity.
- `use_heuristic = false` — `select_neighbors_simple`: Algorithm 3, plain top-`m` by distance. Faster build, lower graph quality.
- `threshold_based_neighbourhood = true` (layer 0 only) — `select_threshold_neighbors`: same heuristic as Algorithm 4 but without a hard upper bound on neighbourhood size. Every candidate whose distance to the query is below `threshold` is kept, with a guaranteed minimum of `m` neighbours. Produces a denser layer 0, which improves proximity-graph recall at the cost of higher memory and build time.

**Query-time search API.**
`HNSWState::search(&T, &mut ScratchBuffers, k, ef_search)` performs nearest-neighbor lookup for an external query (not inserted in the graph). It does a greedy descent on upper layers, then a best-first search on layer 0, and returns `(index, distance)` sorted by ascending distance.

## Structure
```
src/
  core/
    hnsw/
      mod.rs                     HNSWState, HGraph, EntryPoint, ScratchBuffers
      config.rs                  HNSWConfig (builder pattern)
      build.rs                   Parallel build loop + progress bar
      insert.rs                  insert_parallel: layer assignment, search, edge wiring
      search_layer.rs            Greedy k-NN search within one layer
      select_neighbors.rs        Neighbour selection 
```
