mod build;
mod config;
mod hnsw_index;
mod insert;
mod search_layer;
mod select_neighbors;
mod search;

pub use hnsw_index::HNSWIndex;
pub(crate) use hnsw_index::current_crate_version;

// ── Lock contention monitoring ────────────────────────────────────────────────

pub struct LockStat {
    pub wait_ns: std::sync::atomic::AtomicU64,
    pub calls: std::sync::atomic::AtomicU64,
}

impl LockStat {
    pub const fn new() -> Self {
        Self {
            wait_ns: std::sync::atomic::AtomicU64::new(0),
            calls: std::sync::atomic::AtomicU64::new(0),
        }
    }

    #[cfg(feature = "monitor")]
    pub fn record(&self, ns: u64) {
        self.wait_ns.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
        self.calls.fetch_add(1,  std::sync::atomic::Ordering::Relaxed);
    }

    pub fn report(&self, name: &str) {
        let c = self.calls.load(std::sync::atomic::Ordering::Relaxed);
        if c == 0 { return; }
        eprintln!("{name}: {c} calls, avg {:.0}ns/call, total {:.1}ms",
            self.wait_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / c as f64,
            self.wait_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6);
    }
}

/// Time `$expr`, record the wait into `$stat`, and return the expression's value.
/// Compiled away entirely when the `monitor` feature is off.
#[cfg(feature = "monitor")]
macro_rules! measure {
    ($expr:expr, $stat:expr) => {{
        let __t = std::time::Instant::now();
        let __v = $expr;
        $stat.record(__t.elapsed().as_nanos() as u64);
        __v
    }};
}

#[cfg(not(feature = "monitor"))]
macro_rules! measure {
    ($expr:expr, $stat:expr) => { $expr };
}

#[allow(unused_imports)]
pub(crate) use measure;

pub static STAT_ADD_EDGE_LO:          LockStat = LockStat::new();
pub static STAT_ADD_EDGE_HI:          LockStat = LockStat::new();
pub static STAT_SET_NEIGHBOURHOOD:    LockStat = LockStat::new();
pub static STAT_SNAPSHOT:             LockStat = LockStat::new();
pub static STAT_DASHMAP:              LockStat = LockStat::new();
pub static STAT_CACHE_HIT:            LockStat = LockStat::new();
pub static STAT_CACHE_MISS:           LockStat = LockStat::new();
pub static STAT_ALIGNMENT:            LockStat = LockStat::new();
pub static STAT_CACHE_GET:            LockStat = LockStat::new();
pub static STAT_CACHE_INSERT:         LockStat = LockStat::new();

use fixedbitset::FixedBitSet;
use std::collections::BinaryHeap;
use std::cmp::{Reverse, Ordering};
use std::cell::RefCell;
use std::hash::{BuildHasher, Hasher};
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::core::Distance;
use quick_cache::sync::Cache;
pub use config::HNSWConfig;

/// Hasher for `(u32, u32)` node-pair keys, used for the internal bucket
/// placement of the per-shard hashmaps in [`ShardedCache`] and
/// [`ShardedEdgeSet`] (shard *selection* itself is a plain `key.0 & mask` in
/// both, not this hasher — see their docs).
///
/// `Hash` for a tuple feeds each field through `write_u32` in order, so
/// `write_u32` packs them into a single u64 (`first << 32 | second`) with
/// just a shift and an OR. `finish()` then runs one multiply + xor-shift
/// (Fibonacci hashing) over the packed value. This is *not* optional: node
/// ids are far smaller than 2^32, so the packed value's real entropy sits in
/// the low bits of each half — hash table implementations generally rely on
/// the high bits too (e.g. for SwissTable-style probing), which would
/// otherwise be constant zero for any dataset under ~2M nodes. One multiply +
/// xor-shift is enough to spread that entropy across all 64 bits and is still
/// a couple orders of magnitude cheaper than a general-purpose hasher.
#[derive(Default, Clone, Copy)]
pub(crate) struct PairHasher(u64);

impl Hasher for PairHasher {
    fn finish(&self) -> u64 {
        let mut h = self.0.wrapping_mul(0x9E3779B97F4A7C15);
        h ^= h >> 32;
        h
    }
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("PairHasher only supports write_u32, fed by hashing a (u32, u32) key");
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0 << 32) | i as u64;
    }
}

#[derive(Default, Clone, Copy)]
pub(crate) struct PairBuildHasher;

impl BuildHasher for PairBuildHasher {
    type Hasher = PairHasher;
    fn build_hasher(&self) -> PairHasher { PairHasher::default() }
}

/// A sharded distance cache: N independent `Cache` instances, each with its own
/// internal locks and LRU state. Pair `(i, j)` (with i ≤ j) always routes to
/// shard `i & mask`, which is equivalent to `i % n_shards` but faster since
/// n_shards is a power of two and the AND replaces a division.
///
/// With 64 shards and 8 threads, the probability that two threads hit the same
/// shard simultaneously is ~12%, vs 100% for a single shared cache.
struct ShardedCache {
    /// Empty when `cache_capacity == 0` — caching is disabled, every lookup is a no-op.
    shards: Vec<Cache<(u32, u32), f32, quick_cache::UnitWeighter, PairBuildHasher>>,
    /// n_shards - 1: used for the fast-modulo AND
    mask: u32,
}

impl ShardedCache {
    fn new(total_capacity: usize, n_shards: usize) -> Self {
        if total_capacity == 0 {
            return Self { shards: Vec::new(), mask: 0 };
        }
        // Round up to the nearest power of two so `key.0 & mask` is valid
        let n_shards = n_shards.next_power_of_two();
        let per_shard = (total_capacity / n_shards).max(1);
        Self {
            shards: (0..n_shards)
                .map(|_| Cache::with(per_shard, per_shard as u64, Default::default(), PairBuildHasher, Default::default()))
                .collect(),
            mask: (n_shards - 1) as u32,
        }
    }

    #[inline]
    fn get(&self, key: &(u32, u32)) -> Option<f32> {
        if self.shards.is_empty() {
            return None;
        }
        // key.0 & mask is equivalent to key.0 % n_shards, but faster (single AND vs division)
        self.shards[(key.0 & self.mask) as usize].get(key)
    }

    #[inline]
    fn insert(&self, key: (u32, u32), val: f32) {
        if self.shards.is_empty() {
            return;
        }
        // key.0 & mask is equivalent to key.0 % n_shards, but faster (single AND vs division)
        self.shards[(key.0 & self.mask) as usize].insert(key, val);
    }
}

/// Concurrent store for `proximity_edges`: routes to a shard the same cheap way as
/// [`ShardedCache`] (`key.0 & mask`, no hashing needed to pick the shard), but each
/// shard is a plain `HashMap` behind a `Mutex` rather than a `DashMap`.
///
/// This is write-mostly, dump-once-at-the-end usage — build() never looks a key
/// up, only inserts and (at the very end) iterates everything. A `HashMap` per
/// shard still dedups redundant re-inserts of the same pair (recomputed by two
/// different node insertions) so memory doesn't grow with duplicate writes, but
/// skips `DashMap`'s own internal shard-selection hashing entirely — we already
/// know which shard a key belongs to from `key.0` alone.
struct ShardedEdgeSet {
    shards: Vec<Mutex<std::collections::HashMap<(u32, u32), f32, PairBuildHasher>>>,
    mask: u32,
}

impl ShardedEdgeSet {
    fn new(n_shards: usize) -> Self {
        let n_shards = n_shards.next_power_of_two();
        Self {
            shards: (0..n_shards)
                .map(|_| Mutex::new(std::collections::HashMap::with_hasher(PairBuildHasher)))
                .collect(),
            mask: (n_shards - 1) as u32,
        }
    }

    #[inline]
    fn insert(&self, key: (u32, u32), val: f32) {
        self.shards[(key.0 & self.mask) as usize].lock().insert(key, val);
    }

    fn iter_all(&self) -> impl Iterator<Item = ((u32, u32), f32)> + '_ {
        self.shards.iter().flat_map(|s| {
            s.lock().iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>()
        })
    }
}

/// Max-heap: largest element at the top (BinaryHeap default)
pub type MaxHeap<T> = BinaryHeap<T>;

/// Min-heap: smallest element at the top (zero-cost via Reverse)
pub type MinHeap<T> = BinaryHeap<Reverse<T>>;

#[derive(Copy, Clone)]
struct Candidate {
    pub idx: u32,
    /// Distance between node idx and query
    pub distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

#[derive(Copy, Clone)]
struct Loc {
    pub(super) layer: usize,
    pub(super) node: u32,
}

/// Hierarchical Graph with per-node RwLock for concurrent access
struct HGraph {
    /// Shape(N_layers, N_nodes): each node's neighbor list is individually locked
    layers: Vec<Vec<RwLock<Vec<u32>>>>,
}

impl HGraph {
    pub fn with_capacity(n_layers: usize, n_nodes: usize) -> HGraph {
        HGraph {
            layers: (0..n_layers)
                .map(|_| (0..n_nodes).map(|_| RwLock::new(Vec::new())).collect())
                .collect(),
        }
    }

    /// Clone the neighbor list under a brief read lock into `buffer`, then release.
    pub fn neighbors_snapshot(&self, layer: usize, node: u32, buffer: &mut Vec<u32>) {
        buffer.clone_from(&*measure!(self.layers[layer][node as usize].read(), STAT_SNAPSHOT));
    }

    pub fn neighbors_len(&self, layer: usize, node: u32) -> usize {
        self.layers[layer][node as usize].read().len()
    }

    /// Add a bidirectional edge. Always locks min(from, to) first to prevent deadlocks.
    pub fn add_edge(&self, layer: usize, from: u32, to: u32) {
        if from == to {
            return;
        }
        let (lo, hi) = if from < to { (from, to) } else { (to, from) };
        let mut lo_guard = measure!(self.layers[layer][lo as usize].write(), STAT_ADD_EDGE_LO);
        let mut hi_guard = measure!(self.layers[layer][hi as usize].write(), STAT_ADD_EDGE_HI);
        lo_guard.push(hi);
        hi_guard.push(lo);
    }

    pub fn set_neighbourhood(&self, layer: usize, node: u32, neighbourhood: &[u32]) {
        let mut guard = measure!(self.layers[layer][node as usize].write(), STAT_SET_NEIGHBOURHOOD);
        guard.clear();
        guard.extend_from_slice(neighbourhood);
    }
}

/// Entry point protected by a mutex — updates are O(log N) total, contention is negligible.
struct EntryPoint {
    inner: Mutex<Option<Loc>>,
}

impl EntryPoint {
    fn new() -> Self {
        Self { inner: Mutex::new(None) }
    }

    fn get(&self) -> Option<(u32, usize)> {
        self.inner.lock().map(|loc| (loc.node, loc.layer))
    }

    /// Update to (new_node, new_layer) only if new_layer exceeds the current layer.
    fn try_update(&self, new_layer: usize, new_node: u32) {
        let mut guard = self.inner.lock();
        let should_update = match *guard {
            None => true,
            Some(loc) => new_layer > loc.layer,
        };
        if should_update {
            *guard = Some(Loc { layer: new_layer, node: new_node });
        }
    }
}

pub struct ScratchBuffers {
    visited: FixedBitSet,
    candidates: MinHeap<Candidate>,
    discarded_candidates: MinHeap<Candidate>,
    nearest_neighbors: MaxHeap<Candidate>,
    selected_neighbors: Vec<u32>,
    neighbors: Vec<u32>,
    /// Snapshot buffer: holds a node's neighbor list cloned under a read lock
    snapshot: Vec<u32>,
    /// Inner snapshot buffer used inside select_neighbors for the extend_candidates path
    inner_snapshot: Vec<u32>,
}

impl ScratchBuffers {
    pub fn with_capacity(n_nodes: usize, ef: usize, m_max: usize) -> Self {
        ScratchBuffers {
            visited: FixedBitSet::with_capacity(n_nodes),
            candidates: MinHeap::with_capacity(ef),
            discarded_candidates: MinHeap::with_capacity(ef),
            nearest_neighbors: MaxHeap::with_capacity(ef),
            selected_neighbors: Vec::with_capacity(m_max),
            neighbors: Vec::with_capacity(m_max),
            snapshot: Vec::new(),
            inner_snapshot: Vec::new(),
        }
    }

    /// Grow the visited bitset if it is smaller than `n_nodes`.
    fn ensure_capacity(&mut self, n_nodes: usize) {
        if self.visited.len() < n_nodes {
            self.visited.grow(n_nodes);
        }
    }

    fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.discarded_candidates.clear();
        self.nearest_neighbors.clear();
        self.selected_neighbors.clear();
        self.neighbors.clear();
        self.snapshot.clear();
        self.inner_snapshot.clear();
    }

    fn is_clear(&self) -> bool {
        self.visited.is_clear()
            && self.candidates.is_empty()
            && self.discarded_candidates.is_empty()
            && self.nearest_neighbors.is_empty()
            && self.selected_neighbors.is_empty()
            && self.neighbors.is_empty()
    }
}

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(rand::random::<u64>()));
    static SCRATCH: RefCell<Option<ScratchBuffers>> = const { RefCell::new(None) };
}

pub struct HNSWState<T: Sync, D: Distance<T>> {
    data: Vec<T>,
    hgraph: HGraph,
    entry_point: EntryPoint,
    config: HNSWConfig,
    /// Pre-allocated maximum number of layers — eliminates dynamic resize during build
    max_layers: usize,
    distance: D,
    /// Sharded distance cache: reduces contention vs a single shared cache.
    dist_cache: ShardedCache,
    /// All pairs whose distance is below config.proximity_threshold
    proximity_edges: ShardedEdgeSet,
    pub has_been_built: bool,
}

impl<T: Sync, D: Distance<T>> HNSWState<T, D> {
    pub fn new(data: Vec<T>, distance: D, config: HNSWConfig) -> Self {
        let len = data.len();
        let max_layers = ((len as f64).ln() * config.m_l).ceil() as usize + 2;
        let max_layers = max_layers.max(1);
        Self {
            hgraph: HGraph::with_capacity(max_layers, len),
            entry_point: EntryPoint::new(),
            dist_cache: ShardedCache::new(config.cache_capacity, config.cache_shards),
            proximity_edges: ShardedEdgeSet::new(config.cache_shards),
            max_layers,
            data,
            config,
            distance,
            has_been_built: false,
        }
    }

    pub fn query_distance(&self, query: &T, y: u32) -> f32 {
        self.distance.call(query, &self.data[y as usize])
    }
    pub fn distance(&self, x: u32, y: u32) -> f32 {
        let key = if x <= y { (x, y) } else { (y, x) };

        // Fast path: shared cache hit (no FFI call, no allocation)
        if let Some(d) = measure!(self.dist_cache.get(&key), STAT_CACHE_GET) {
            measure!((), STAT_CACHE_HIT);
            return d;
        }
        measure!((), STAT_CACHE_MISS);

        // Slow path: compute via FFI, then cache for all threads
        let d = measure!(self.distance.call(&self.data[key.0 as usize], &self.data[key.1 as usize]), STAT_ALIGNMENT);
        measure!(self.dist_cache.insert(key, d), STAT_CACHE_INSERT);
        if self.config.keep_all_edges && d < self.config.proximity_threshold {
            measure!(self.proximity_edges.insert(key, d), STAT_DASHMAP);
        }
        d
    }

    fn min_distance_with_many(&self, x: u32, ys: &[u32]) -> f32 {
        let mut min_distance = f32::MAX;
        for &y in ys {
            let dist = self.distance(x, y);
            min_distance = min_distance.min(dist);
        }
        min_distance
    }

    /// Returns all edges with distance below `config.proximity_threshold`, but only if
    /// `keep_all_edges` is True, otherwise it returns None.
    pub fn edges(&self) -> Option<Vec<(u32, u32, f32)>> {
        if self.config.keep_all_edges {
            Some(
                self.proximity_edges
                    .iter_all()
                    .map(|((u, v), w)| (u, v, w))
                    .collect()
            )
        }else { None }
    }

    pub fn get_layer(&self, layer_idx: usize) -> Result<Vec<Vec<u32>>, String> {
        if layer_idx >= self.hgraph.layers.len() {
            return Err(format!(
                "layer index {} out of range: index has {} layers (0..{})",
                layer_idx, self.hgraph.layers.len(), self.hgraph.layers.len().saturating_sub(1)
            ));
        }
        Ok(self.hgraph.layers[layer_idx]
            .iter()
            .map(|node| node.read().clone())
            .collect())
    }

    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }

    pub fn index(&self) -> HNSWIndex {
        HNSWIndex {
            crate_version: hnsw_index::current_crate_version(),
            dataset_size: self.data.len(),
            layers: self.hgraph.layers.iter()
                .map(|layer| layer.iter().map(|node| node.read().clone()).collect())
                .collect(),
            entry_point: self.entry_point.get(),
            config: self.config.clone(),
            max_layers: self.max_layers,
            proximity_edges: self.proximity_edges.iter_all().collect(),
            has_been_built: self.has_been_built,
        }
    }

    /// Serialize the index to `path` using bincode.
    ///
    /// The data and distance function are not stored — pass them back to
    /// [`HNSWState::load`]. The distance cache is discarded; it repopulates
    /// on demand.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        self.index().save(path)
    }

    /// Deserialize an index written by [`HNSWState::save`] and reconstruct
    /// the full state.
    ///
    /// `data` must be the same dataset used during the original build.
    /// `distance` must be the same kernel. A size mismatch between the
    /// saved index and `data` is returned as an error.
    pub fn load(
        path:     impl AsRef<std::path::Path>,
        data:     Vec<T>,
        config: Option<HNSWConfig>,
        distance: D,
    ) -> Result<Self, Box<dyn std::error::Error>>
    {
        let index = HNSWIndex::load(path)?;

        // Sanity checks
        // The on-disk format is stable from v0.1.0 onward; reject if either the file
        // or the running crate predates that (pre the u32 node-id refactor).
        let running_version = hnsw_index::current_crate_version();
        if index.crate_version != running_version {
            return Err(format!(
                "index format mismatch: saved with refnd v{}.{}.{}, running v{}.{}.{} — \
                 one of them predates the stable index format. Rebuild the index.",
                index.crate_version.0, index.crate_version.1, index.crate_version.2,
                running_version.0, running_version.1, running_version.2,
            ).into());
        }
        if index.dataset_size != data.len() {
            return Err(format!(
                "dataset size mismatch: index was built on {} points, got {}. Consider\
                 deleting the current index to refresh it, or changing the index filepath.",
                index.dataset_size,
                data.len()
            ).into());
        }
        if let Some(cfg) = config && cfg != index.config{
            return Err(format!(
                "Config mismatch: The current config and index config are not the same. Consider \
                 deleting the current index to refresh it, or changing the index filepath.",
            ).into());
        }

        let hgraph = HGraph {
            layers: index.layers.into_iter()
                .map(|layer| layer.into_iter().map(|nbrs| RwLock::new(nbrs)).collect())
                .collect(),
        };

        let entry_point = EntryPoint::new();
        if let Some((node, layer)) = index.entry_point {
            entry_point.try_update(layer, node);
        }

        let proximity_edges = ShardedEdgeSet::new(index.config.cache_shards);
        for (key, val) in index.proximity_edges {
            proximity_edges.insert(key, val);
        }

        let dist_cache = ShardedCache::new(index.config.cache_capacity, index.config.cache_shards);

        Ok(Self {
            data,
            hgraph,
            entry_point,
            max_layers: index.max_layers,
            distance,
            dist_cache,
            proximity_edges,
            config: index.config,
            has_been_built: index.has_been_built,
        })
    }
}
