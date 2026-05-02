use std::collections::BTreeMap;
use clap::Args;
use refnd::core::leiden::LeidenObjective;
use refnd::core::hnsw::HNSWConfig;
use crate::fields;

#[derive(Args)]
#[command(next_help_heading = "HNSW Options")]
pub struct HnswArgs{
    /// Max bidirectional links per node per layer
    #[arg(long, default_value_t = 16, value_name = "INT")]
    pub m: usize,

    /// Max links per node on layers above 0 (usually m)
    #[arg(long, default_value_t = 16, value_name = "INT")]
    pub m_max: usize,

    /// Max links per node on layer 0 (usually 2×m)
    #[arg(long, default_value_t = 32, value_name = "INT")]
    pub m_max0: usize,

    /// Level-generation multiplier (1/ln(m) ≈ 0.36 for m=16)
    #[arg(long, default_value_t = 0.36, value_name = "FLOAT")]
    pub m_l: f64,

    /// ef value used during search at non-construction layers
    #[arg(long, default_value_t = 1, value_name = "INT")]
    pub ef_init: usize,

    /// Dynamic candidate list size during index construction
    #[arg(long, default_value_t = 128, value_name = "INT")]
    pub ef_construction: usize,

    /// Extend neighbour candidates with their neighbours during selection
    #[arg(long)]
    pub extend_candidates: bool,

    /// Prevents keeping pruned connections when the candidate list is exhausted
    #[arg(long)]
    pub no_keep_pruned_connections: bool,

    /// Total distance-cache entries across all shards
    #[arg(long, default_value_t = 2_000_000, value_name = "INT")]
    pub cache_capacity: usize,

    /// Number of cache shards (rounded to next power-of-two)
    #[arg(long, default_value_t = 64, value_name = "INT")]
    pub cache_shards: usize,

    /// Shuffle insertion order before building (better quality, ~25 % slower)
    #[arg(long)]
    pub shuffle: bool,

    /// Use simple nearest-neighbour selection (faster build, lower quality)
    #[arg(long)]
    pub simple_neighbors: bool,

    /// If enabled, ef-construction is strictly enforced, meaning that even a candidate is under 
    /// the threshold, it can be dropped if the list is larger than ef-construction.
    #[arg(long)]
    pub strict_ef: bool,

    /// Select layer-0 neighbours by distance threshold instead of capping at m_max0
    #[arg(long)]
    pub threshold_based_neighbourhood: bool,
}

pub fn build_hnsw_config(hnsw: &HnswArgs, proximity_threshold: f32, threads: usize) -> HNSWConfig {
    HNSWConfig {
        m:                             hnsw.m,
        m_max:                         hnsw.m_max,
        m_max0:                        hnsw.m_max0,
        m_l:                           hnsw.m_l,
        ef_init:                       hnsw.ef_init,
        ef_construction:               hnsw.ef_construction,
        extend_candidates:             hnsw.extend_candidates,
        keep_pruned_connections:       !hnsw.no_keep_pruned_connections,
        cache_capacity:                hnsw.cache_capacity,
        cache_shards:                  hnsw.cache_shards,
        proximity_threshold,
        n_threads:                     threads,
        shuffle:                       hnsw.shuffle,
        use_heuristic:                 !hnsw.simple_neighbors,
        strict_ef:                     hnsw.strict_ef,
        threshold_based_neighbourhood: hnsw.threshold_based_neighbourhood,
    }
}

pub fn hnsw_params(hnsw: &HnswArgs) -> BTreeMap<String, String> {
    fields!(hnsw;
            m, m_max, m_max0, m_l, ef_init, ef_construction,
            extend_candidates, no_keep_pruned_connections as "keep_pruned_connections" => !,
            cache_capacity, cache_shards, shuffle, simple_neighbors,
            strict_ef, threshold_based_neighbourhood,
        )
}

#[derive(Args)]
#[command(next_help_heading = "Leiden Options")]
pub struct LeidenArgs{
    /// Use Leiden community detection instead of connected components (default: connected components)
    #[arg(long)]
    pub leiden: bool,

    /// Leiden resolution, the larger it is, the larger the communities will be.
    #[arg(long, default_value_t = 1., value_name = "FLOAT")]
    pub resolution: f32,

    /// Leiden beta parameter, the larger it is, the more explorative the algorithm will be (slower).
    #[arg(long, default_value_t = 0.01, value_name = "FLOAT")]
    pub beta: f64,

    /// Community partition objective: modularity or cpm (Constant Potts Model)
    #[arg(long, default_value = "modularity", value_name = "MODE")]
    pub leiden_objective: LeidenObjective,

    /// Number of iterations
    #[arg(long, default_value_t = 2, value_name = "INT")]
    pub leiden_iteration: usize,

    /// Make the graph weighted before detecting communities
    #[arg(long, value_name = "FLAG")]
    pub weighted: bool
}

pub fn leiden_params(leiden: &LeidenArgs) -> BTreeMap<String, String> {
    fields!(leiden;
            leiden,
            resolution, beta,
            leiden_objective as "objective" => "{:?}",
            leiden_iteration as "iterations",
        )
}
