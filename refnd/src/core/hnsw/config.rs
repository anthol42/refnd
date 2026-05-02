use std::fmt;

#[derive(Clone, bincode::Encode, bincode::Decode, PartialEq)]
pub struct HNSWConfig{
    pub m: usize,
    pub m_max: usize,
    pub m_max0: usize,
    pub m_l: f64,
    pub ef_init: usize,
    pub ef_construction: usize,
    pub extend_candidates: bool, // Extends candidates with their neighbors when finding neighbors
    pub keep_pruned_connections: bool,
    /// Total number of entries across all cache shards
    pub cache_capacity: usize,
    /// Number of independent cache shards (rounded up to next power of two).
    /// More shards = less contention, but more memory overhead. Default: 64.
    pub cache_shards: usize,
    /// Edges with distance below this threshold are stored in the proximity edge set
    pub proximity_threshold: f32,
    /// Number of threads for parallel build. 0 = use all available cores (Rayon default).
    pub n_threads: usize,
    /// Shuffle insertion order before building. Improves graph quality at the cost of cache locality.
    pub shuffle: bool,
    /// Use the heuristic neighbor selection (Algorithm 4). When false, uses the simple
    /// nearest-neighbor selection (Algorithm 3) — faster but lower graph quality.
    pub use_heuristic: bool,
    /// If true, the ef maximum list size is enforced, even if the candidate to drop 
    /// has a distance to the query lower than the threshold
    pub strict_ef: bool,
    /// If true, the neighbours of a node are selected if their distance is under the threshold.
    /// Only valid for layer 0.
    pub threshold_based_neighbourhood: bool,
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 16,
            m_max0: 32,
            m_l: 0.36, // 1.0 / ln(m)
            ef_init: 1,
            ef_construction: 128,
            extend_candidates: false,
            keep_pruned_connections: true,
            cache_capacity: 2_000_000,
            cache_shards: 64,
            proximity_threshold: 0.5,
            n_threads: 0,
            shuffle: false,
            use_heuristic: true,
            strict_ef: false,
            threshold_based_neighbourhood: false,
        }
    }
}

impl HNSWConfig {
    pub fn set_m(&mut self, m: usize) -> &mut Self {
        self.m = m;
        self
    }
    pub fn set_m_max(&mut self, m: usize) -> &mut Self {
        self.m_max = m;
        self
    }
    pub fn set_m_max0(&mut self, m: usize) -> &mut Self {
        self.m_max0 = m;
        self
    }
    pub fn set_m_l(&mut self, ml: f64) -> &mut Self {
        self.m_l = ml;
        self
    }
    pub fn set_ef_init(&mut self, ef: usize) -> &mut Self {
        self.ef_init = ef;
        self
    }
    pub fn set_ef_construction(&mut self, ef: usize) -> &mut Self {
        self.ef_construction = ef;
        self
    }
    pub fn set_extend_candidates(&mut self, extend: bool) -> &mut Self {
        self.extend_candidates = extend;
        self
    }
    pub fn set_keep_pruned_connections(&mut self, keep: bool) -> &mut Self {
        self.keep_pruned_connections = keep;
        self
    }
    pub fn set_cache_capacity(&mut self, capacity: usize) -> &mut Self {
        self.cache_capacity = capacity;
        self
    }
    pub fn set_cache_shards(&mut self, shards: usize) -> &mut Self {
        self.cache_shards = shards;
        self
    }
    pub fn set_proximity_threshold(&mut self, threshold: f32) -> &mut Self {
        self.proximity_threshold = threshold;
        self
    }
    pub fn set_n_threads(&mut self, n: usize) -> &mut Self {
        self.n_threads = n;
        self
    }
    pub fn set_shuffle(&mut self, shuffle: bool) -> &mut Self {
        self.shuffle = shuffle;
        self
    }
    pub fn set_use_heuristic(&mut self, use_heuristic: bool) -> &mut Self {
        self.use_heuristic = use_heuristic;
        self
    }
}

impl fmt::Display for HNSWConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HNSWConfig(m={}, m_max={}, m_max0={}, ef_c={}, threshold={}, kpc={})",
            self.m, self.m_max, self.m_max0,
            self.ef_construction, self.proximity_threshold, self.keep_pruned_connections,
        )
    }
}

impl fmt::Debug for HNSWConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HNSWConfig(\n\
            \x20 m={}, m_max={}, m_max0={}, m_l={},\n\
            \x20 ef_init={}, ef_construction={},\n\
            \x20 extend_candidates={}, keep_pruned_connections={},\n\
            \x20 cache_capacity={}, cache_shards={},\n\
            \x20 proximity_threshold={}, n_threads={},\n\
            \x20 shuffle={}, use_heuristic={},\n\
            \x20 strict_ef={}, threshold_based_neighbourhood={}\n\
            )",
            self.m, self.m_max, self.m_max0, self.m_l,
            self.ef_init, self.ef_construction,
            self.extend_candidates, self.keep_pruned_connections,
            self.cache_capacity, self.cache_shards,
            self.proximity_threshold, self.n_threads,
            self.shuffle, self.use_heuristic,
            self.strict_ef, self.threshold_based_neighbourhood,
        )
    }
}