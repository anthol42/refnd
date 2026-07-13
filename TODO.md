# TODO

- [X] **`CsrGraph::subgraph(nodes: list[int]) -> (CsrGraph, dict[int, int])`** — Add a native subgraph method to `CsrGraph` that accepts a list of node IDs, filters edges to those within the set, remaps IDs to a contiguous `[0, len(nodes))` range, and returns the new graph alongside the old→new node ID mapping. This avoids callers having to reach back into the `EdgeStore` and redo the remapping manually.
- [X] Add a way to build a CSRGraph by using the complement of the distance for similarities used as distance: `w = 1 - d`
- [X] **Memory: `CsrGraph::adj` — use `(u32, f32)` instead of `(usize, f32)`** — `(usize, f32)` is 16 bytes due to alignment padding; `(u32, f32)` is 8 bytes. For 1B undirected edges (stored twice), this halves `adj` from 32 GB → 16 GB. Requires updating `neighbors()` return type and all call sites to cast `u as usize` where used as an index (leiden.rs, connected_components.rs, partition.rs).
- [ ] ~**Memory: `CsrGraph::new` — accept an iterator / raw `&[(u32, u32, f32)]` instead of `&[(usize, usize, f32)]`** — `EdgeStore::graph()` currently calls `self.edges()` which allocates a full `Vec<(usize, usize, f32)>` just to widen u32→usize before passing to `CsrGraph::new`. For 1B edges this wastes 20 GB. Fix by adding a `CsrGraph::from_u32_edges(n, edges: &[(u32, u32, f32)], ...)` constructor and dispatching to it from `EdgeStore::graph()` when using U32 storage. The internal leiden aggregation (`CsrGraph::new` with small community-count graphs) can keep the existing `usize` path.~
- [X] Optimize HNSW with molecules, should be faster than naive
- [X] Optimize Leiden
- [X] Refactor HNSW: Change usize to u32, and if the speedup is meaningful, merge it.
- [X] Move structure US score outside of alignment in core 
- [ ] Use `sphinx-llm` to build LLM-friendly docs
- [ ] Add to ablation: Test if scaling ef_construction with nlog(n) is enough to maintain recall
