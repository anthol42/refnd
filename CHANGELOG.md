# Changelog

## 0.0.3

- Building an HNSW index is significantly faster, especially on large datasets — roughly
  2x faster at 8 million items in testing for fingerprints graphs, with the improvement growing larger still at
  bigger scales.
- Building, saving, and loading an HNSW index all use substantially less memory, especially
  on large datasets. Internal graph storage no longer reserves space for connections that
  were never made, cutting that overhead by roughly 8x at large scale; saving and loading
  now stream to and from disk instead of holding multiple full copies of the index in
  memory at once. Together, these make it possible to build and save indexes that
  previously ran out of memory.
- `HNSWState(...)` now accepts any Python iterable for `data`, not just a sized sequence
  (checked at runtime via `len()`). A non-sized iterable (e.g. a generator) is drained one
  item at a time instead of requiring the caller to first materialize a full list, so a
  large dataset never needs a fully-materialized Python-side copy to coexist with the
  Rust-owned copy the constructor builds either way — halving peak memory for datasets
  where a single item's Python representation is large.
- `HNSWState.load(...)` now accepts any Python iterable for `data` too (previously it
  required a sized sequence like a list) — a generator can be used to re-read cached data
  without first materializing it as a full list.
- Constructing a `BitFingerprint` from a numpy array (`BitFingerprint.from_np(...)`) is up
  to 15x faster.
