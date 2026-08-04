# Changelog

## 0.0.3

- `HNSWState(...)` now accepts any Python iterable for `data`, not just a sized sequence
  (checked at runtime via `len()`). A non-sized iterable (e.g. a generator) is drained one
  item at a time instead of requiring the caller to first materialize a full list, so a
  large dataset never needs a fully-materialized Python-side copy to coexist with the
  Rust-owned copy the constructor builds either way — halving peak memory for datasets
  where a single item's Python representation is large.
