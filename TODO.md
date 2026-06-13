# TODO

- [ ] **`CsrGraph::subgraph(nodes: list[int]) -> (CsrGraph, dict[int, int])`** — Add a native subgraph method to `CsrGraph` that accepts a list of node IDs, filters edges to those within the set, remaps IDs to a contiguous `[0, len(nodes))` range, and returns the new graph alongside the old→new node ID mapping. This avoids callers having to reach back into the `EdgeStore` and redo the remapping manually.
- [ ] Add a way to build a CSRGraph by using the complement of the distance for similarities used as distance: `w = 1 - d`

