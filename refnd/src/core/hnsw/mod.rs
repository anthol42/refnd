mod build;
mod config;
mod hnsw_index;
mod insert;
mod search_layer;
mod select_neighbors;
mod search;

pub use hnsw_index::HNSWIndex;

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
use parking_lot::{Mutex, RwLock};
use rand::rngs::StdRng;
use rand::SeedableRng;
use crate::core::Distance;
use rustc_hash::FxBuildHasher;
use dashmap::DashMap;
use quick_cache::sync::Cache;
pub use config::HNSWConfig;

/// A sharded distance cache: N independent `Cache` instances, each with its own
/// internal locks and LRU state. Pair `(i, j)` (with i ≤ j) always routes to
/// shard `i & mask`, which is equivalent to `i % n_shards` but faster since
/// n_shards is a power of two and the AND replaces a division.
///
/// With 64 shards and 8 threads, the probability that two threads hit the same
/// shard simultaneously is ~12%, vs 100% for a single shared cache.
struct ShardedCache {
    shards: Vec<Cache<(usize, usize), f32>>,
    /// n_shards - 1: used for the fast-modulo AND
    mask: usize,
}

impl ShardedCache {
    fn new(total_capacity: usize, n_shards: usize) -> Self {
        // Round up to the nearest power of two so `key.0 & mask` is valid
        let n_shards = n_shards.next_power_of_two();
        let per_shard = (total_capacity / n_shards).max(1);
        Self {
            shards: (0..n_shards).map(|_| Cache::new(per_shard)).collect(),
            mask: n_shards - 1,
        }
    }

    #[inline]
    fn get(&self, key: &(usize, usize)) -> Option<f32> {
        // key.0 & mask is equivalent to key.0 % n_shards, but faster (single AND vs division)
        self.shards[key.0 & self.mask].get(key)
    }

    #[inline]
    fn insert(&self, key: (usize, usize), val: f32) {
        // key.0 & mask is equivalent to key.0 % n_shards, but faster (single AND vs division)
        self.shards[key.0 & self.mask].insert(key, val);
    }
}

/// Max-heap: largest element at the top (BinaryHeap default)
pub type MaxHeap<T> = BinaryHeap<T>;

/// Min-heap: smallest element at the top (zero-cost via Reverse)
pub type MinHeap<T> = BinaryHeap<Reverse<T>>;

#[derive(Copy, Clone)]
struct Candidate {
    pub idx: usize,
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
    pub(super) node: usize,
}

/// Hierarchical Graph with per-node RwLock for concurrent access
struct HGraph {
    /// Shape(N_layers, N_nodes): each node's neighbor list is individually locked
    layers: Vec<Vec<RwLock<Vec<usize>>>>,
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
    pub fn neighbors_snapshot(&self, layer: usize, node: usize, buffer: &mut Vec<usize>) {
        buffer.clone_from(&*measure!(self.layers[layer][node].read(), STAT_SNAPSHOT));
    }

    pub fn neighbors_len(&self, layer: usize, node: usize) -> usize {
        self.layers[layer][node].read().len()
    }

    /// Add a bidirectional edge. Always locks min(from, to) first to prevent deadlocks.
    pub fn add_edge(&self, layer: usize, from: usize, to: usize) {
        if from == to {
            return;
        }
        let (lo, hi) = if from < to { (from, to) } else { (to, from) };
        let mut lo_guard = measure!(self.layers[layer][lo].write(), STAT_ADD_EDGE_LO);
        let mut hi_guard = measure!(self.layers[layer][hi].write(), STAT_ADD_EDGE_HI);
        lo_guard.push(hi);
        hi_guard.push(lo);
    }

    pub fn set_neighbourhood(&self, layer: usize, node: usize, neighbourhood: &[usize]) {
        let mut guard = measure!(self.layers[layer][node].write(), STAT_SET_NEIGHBOURHOOD);
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

    fn get(&self) -> Option<(usize, usize)> {
        self.inner.lock().map(|loc| (loc.node, loc.layer))
    }

    /// Update to (new_node, new_layer) only if new_layer exceeds the current layer.
    fn try_update(&self, new_layer: usize, new_node: usize) {
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
    selected_neighbors: Vec<usize>,
    neighbors: Vec<usize>,
    /// Snapshot buffer: holds a node's neighbor list cloned under a read lock
    snapshot: Vec<usize>,
    /// Inner snapshot buffer used inside select_neighbors for the extend_candidates path
    inner_snapshot: Vec<usize>,
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
    proximity_edges: DashMap<(usize, usize), f32, FxBuildHasher>,
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
            proximity_edges: DashMap::with_hasher(FxBuildHasher),
            max_layers,
            data,
            config,
            distance,
        }
    }

    pub fn query_distance(&self, query: &T, y: usize) -> f32 {
        self.distance.call(query, &self.data[y])
    }
    pub fn distance(&self, x: usize, y: usize) -> f32 {
        let key = if x <= y { (x, y) } else { (y, x) };

        // Fast path: shared cache hit (no FFI call, no allocation)
        if let Some(d) = measure!(self.dist_cache.get(&key), STAT_CACHE_GET) {
            measure!((), STAT_CACHE_HIT);
            return d;
        }
        measure!((), STAT_CACHE_MISS);

        // Slow path: compute via FFI, then cache for all threads
        let d = measure!(self.distance.call(&self.data[key.0], &self.data[key.1]), STAT_ALIGNMENT);
        measure!(self.dist_cache.insert(key, d), STAT_CACHE_INSERT);
        if d < self.config.proximity_threshold {
            measure!(self.proximity_edges.insert(key, d), STAT_DASHMAP);
        }
        d
    }

    fn min_distance_with_many(&self, x: usize, ys: &[usize]) -> f32 {
        let mut min_distance = f32::MAX;
        for &y in ys {
            let dist = self.distance(x, y);
            min_distance = min_distance.min(dist);
        }
        min_distance
    }

    /// Returns all edges with distance below `config.proximity_threshold`
    pub fn edges(&self) -> Vec<(usize, usize, f32)> {
        self.proximity_edges
            .iter()
            .map(|entry| {
                let &(u, v) = entry.key();
                let &w = entry.value();
                (u, v, w)
            })
            .collect()
    }

    pub fn get_layer(&self, layer_idx: usize) -> Vec<Vec<usize>> {
        self.hgraph.layers[layer_idx]
            .iter()
            .map(|node| node.read().clone())
            .collect()
    }

    pub fn config(&self) -> &HNSWConfig {
        &self.config
    }

    pub fn index(&self) -> HNSWIndex {
        HNSWIndex {
            dataset_size: self.data.len(),
            layers: self.hgraph.layers.iter()
                .map(|layer| layer.iter().map(|node| node.read().clone()).collect())
                .collect(),
            entry_point: self.entry_point.get(),
            config: self.config.clone(),
            max_layers: self.max_layers,
            proximity_edges: self.proximity_edges.iter()
                .map(|e| (*e.key(), *e.value()))
                .collect(),
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

        let proximity_edges: DashMap<(usize, usize), f32, FxBuildHasher> =
            DashMap::with_hasher(FxBuildHasher);
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
        })
    }
}
