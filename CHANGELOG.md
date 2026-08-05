# Changelog

## 0.0.3

- `HNSWState(...)` and `HNSWState.load(...)` now accept any Python iterable for `data` —
  list, tuple, generator, anything with `__iter__` — and always drain it one item at a time
  into their own Rust-owned copy, rather than checking `len()` to pick between a sized
  bulk-extract path and an unsized streaming fallback. The two paths did the same per-item
  extraction work either way (there's no bulk/vectorized extraction for a sequence of
  wrapped objects like `BitFingerprint`), so the "fast path" bought nothing; dropping it
  means list, tuple, and generator inputs are all handled identically, and a generator never
  needs to be fully materialized into a Python-side list first — the win that matters when a
  single item's Python representation is large enough that two full-dataset copies wouldn't
  fit in memory together. `size_hint()` still gets an exact capacity hint for a plain list or
  tuple either way, since CPython's own sequence iterators implement `__length_hint__`.
- Fixed an O(n²) blowup in `HNSWState.build()`: the per-thread `visited` set used during
  graph search was sized to the *total declared dataset size* and cleared (an O(capacity)
  sweep) multiple times per insertion, regardless of how many nodes the graph actually
  contained yet — making per-insertion cost scale with total N instead of current graph
  size. Replaced with a generation-counter-based `VisitedSet` (amortized O(1) `clear()`).
  Measured on a 49.2M-item dataset with heavy near-duplicate clustering: `build()` dropped
  from a 97+ hour (never completed) trajectory to 977.7s — roughly 359x.
- `BitFingerprint`'s internal bitset is now backed by `SmallVec<[u64; 32]>` instead of
  `fixedbitset::FixedBitSet` (itself `Vec`-backed). Fingerprints at or under 2048 bits
  (i.e. virtually all real Morgan/RDKit fingerprints) are stored inline with no separate
  heap allocation per fingerprint; larger ones transparently spill to the heap, so no
  bit-length flexibility is lost. Measured: ~603 bytes/fingerprint -> ~338 bytes/fingerprint
  for a 2048-bit fingerprint, cutting the Rust-owned copy's memory roughly in half at
  million-plus-fingerprint scale.
- `HNSWState`'s per-layer graph storage is now sparse above layer 0: layers only ever hold
  an exponentially shrinking fraction of nodes (`~exp(-layer / m_l)`), but were previously
  allocated as `Vec<RwLock<Vec<u32>>>` sized to the *full* dataset regardless — at
  `max_layers=9` and `n=98.4M`, that's `9 * 98.4M * 32 bytes` (`size_of::<RwLock<Vec<u32>>>()`)
  ≈ 26.4GB of empty slots. Layers above 0 are now backed by `DashMap<u32, RwLock<Vec<u32>>>`,
  storing only nodes that actually have neighbors at that layer. The on-disk format
  (`LayerData::Dense`/`Sparse`) preserves the same distinction instead of densifying sparse
  layers back out at save time.
- `HNSWState::save()` no longer builds a full in-memory snapshot (`HNSWIndex`) and a
  separate full in-memory serialized byte buffer before writing to disk. Both were
  redundant with the live in-memory graph, so all three were resident simultaneously at
  the point of highest memory pressure (end of `build()`) — the actual cause of two
  consecutive full-scale (98.4M-item) OOM failures during `save()`, even after `build()`
  itself completed cleanly. `save()` now streams each field, and each graph layer
  individually, straight to a `BufWriter<File>` via `bincode::encode_into_std_write`. The
  on-disk bytes are unchanged from the previous format.
- `HNSWState.load()` no longer OOMs at BELKA scale either, on the mirror-image problem to
  `save()`'s: it read the whole file into one `Vec<u8>` and `bincode::decode_from_slice`'d
  the *entire* graph into one `HNSWIndex` snapshot before converting any of it into the live
  graph, and separately required `data` to already be a sized Python sequence (`len()`),
  forcing the caller to fully materialize the dataset as a Python-side list first. Both are
  fixed the same way as `save()`: `load()` now streams the file field-by-field and
  layer-by-layer via `bincode::decode_from_std_read`, converting each layer straight into its
  live `LayerStorage` before decoding the next, and — see above — always drains `data`
  through the iterator protocol so a streaming source never needs a fully materialized
  Python-side copy either. Verified at full BELKA scale (98.4M items, ~52GB live graph):
  completes in 219s with no OOM.
