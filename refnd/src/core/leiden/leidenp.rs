use super::{CsrGraph, INWeightType, reindex_membership};
use crate::core::hnsw::measure;
#[cfg(feature = "monitor")]
use crate::core::hnsw::LockStat;
use fixedbitset::FixedBitSet;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;
use super::leiden::LeidenObjective;

/// Parallel sections never spawn more than this many worker threads.
const MAX_THREADS: usize = 8;

/// Round cap for `fastmove_nodes`'s parallel local-moving loop. Even with
/// near-real-time atomic visibility between threads (see `fastmove_nodes`),
/// synchronous parallel local-moving can still settle into a small persistent
/// back-and-forth that never reaches a literal fixed point (confirmed
/// empirically on a toy graph: raising this cap 20x didn't shrink the
/// residual further, so it's a genuine steady-state cycle, not slow
/// convergence). This bounds the wasted work on that residual rather than
/// looping forever. Not a workaround specific to this port: both NetworKit's
/// PLM and the published GVE-Leiden design (github.com/puzzlef/leiden-
/// communities-openmp, arxiv.org/abs/2312.13936) cap their own move phase the
/// same way, for the same reason -- losing the sequential algorithm's
/// monotonic-improvement guarantee under concurrent updates is a known,
/// accepted property of the technique, not a bug to eliminate outright.
const MAX_FASTMOVE_ROUNDS: usize = 100;

#[cfg(feature = "monitor")]
pub static STAT_FASTMOVE:   LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_MERGE:      LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_AGGREGATE:  LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_REINDEX:    LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_FLATTEN:    LockStat = LockStat::new();

thread_local! {
    /// Per-thread scratch reused across all `merge_nodes` calls handled by this
    /// worker: `local[v]` is `v`'s local refined-cluster label. Indexed by global
    /// node id (like the original single shared buffer), but kept one-per-thread
    /// so concurrent clusters never alias the same slots -- resized (never
    /// shrunk) lazily to the current level's node count on first use.
    static MERGE_SCRATCH: RefCell<Vec<u32>> = RefCell::new(Vec::new());

    /// Per-thread scratch reused across all `aggregate` per-cluster calls handled
    /// by this worker: `weight_to_cluster[c2]` accumulates edge weight to
    /// candidate neighbour super-node `c2`, `is_neighbor_cluster` dedupes it.
    /// Indexed by refined-cluster id (the *new*, coarser graph's node count),
    /// resized (never shrunk) lazily on first use -- same reasoning as
    /// `MERGE_SCRATCH`.
    static AGGREGATE_SCRATCH: RefCell<(Vec<f32>, FixedBitSet)> = RefCell::new((Vec::new(), FixedBitSet::new()));

    /// Per-thread scratch reused across all `decide_move` calls handled by this
    /// worker, indexed by cluster id. Sized to `2*graph.n` (not `graph.n`) --
    /// see `fastmove_nodes` for why the id space is doubled.
    static FASTMOVE_SCRATCH: RefCell<(Vec<f32>, FixedBitSet)> = RefCell::new((Vec::new(), FixedBitSet::new()));
}

/// Atomically adds `val` to the f32 stored (as bits) in `cell` -- std has no
/// `AtomicF32`. Used for `fastmove_nodes`'s cluster-weight bookkeeping, which
/// multiple threads update concurrently.
fn atomic_f32_add(cell: &AtomicU32, val: f32) {
    cell.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
        Some((f32::from_bits(bits) + val).to_bits())
    }).unwrap();
}

struct LeidenConfig {
    pub resolution: f32,
    pub beta: f64
}
struct LeidenState {
    graph: CsrGraph,
    node_weight: Vec<f32>,
    membership: Vec<u32>,
    pool: rayon::ThreadPool,
}

impl LeidenState {
    pub fn find_partition(&mut self, config: &LeidenConfig) -> bool{
        // Initialize temporary buffers
        let mut refined_membership: Vec<u32> = vec![0; self.graph.n];
        let mut cluster_scratch: Vec<Vec<u32>> = vec![vec![]; self.graph.n]; // clusters
        let mut super_node_map: Vec<u32> = (0..self.graph.n as u32).collect(); // aggregate_vertex
        let mut aggregated_node_weights: Vec<f32> = self.node_weight.clone(); // i_vertex_out_weight
        let mut aggregated_membership: Vec<u32> = self.membership.clone(); // i_membership
        let mut aggregated_graph = self.graph.clone(); // i_graph and i_edge_weights

        // Ensure the cluster_ids are from [0 to k)
        #[allow(unused_assignments)]
        let mut nb_clusters = reindex_membership(&mut aggregated_membership, aggregated_graph.n);

        let mut changed = false;
        let mut did_changed;
        let mut continue_clustering;
        let mut level = 0;
        loop {
            // Move nodes in order to increase the quality
            (did_changed, nb_clusters) = measure!(self.fastmove_nodes(
                &aggregated_graph,
                &aggregated_node_weights,
                &config,
                &mut aggregated_membership,
            ), STAT_FASTMOVE);
            changed = changed || did_changed;

            continue_clustering = nb_clusters < aggregated_graph.n;
            if continue_clustering {
                // Flatten membership. At level 0, `super_node_map` is still the identity
                // mapping ((0..n).collect(), set before this loop and only ever mutated
                // further down in this same iteration), so `aggregated_membership[super_node_map[v]]`
                // degenerates to `aggregated_membership[v]` for every node -- a straight copy.
                // Doing it as a copy instead of the general indexed gather lets the compiler
                // emit a single memcpy instead of a per-node indirect load.
                measure!({
                    if level > 0 {
                        for node_id in 0..self.graph.n {
                            let super_node_id = super_node_map[node_id] as usize;
                            self.membership[node_id] = aggregated_membership[super_node_id];
                        }
                    } else {
                        self.membership.copy_from_slice(&aggregated_membership);
                    }
                    self.retrieve_clusters(&mut cluster_scratch, &aggregated_membership);
                }, STAT_FLATTEN);
                // ensure refined membership is correct size
                refined_membership.truncate(aggregated_graph.n);
                // Refine each cluster. Clusters are disjoint node sets, so each one's
                // local refinement (`merge_nodes`) is independent -- run them in
                // parallel, each returning its own locally-compacted refined ids
                // ([0, k_i)) instead of writing straight into a shared/globally-offset
                // buffer. A cheap sequential pass then turns those per-cluster id
                // spaces into one globally-contiguous range (`nb_refined_clusters`),
                // which is the only genuinely serial part of this phase.
                let state = &*self;
                let cluster_results: Vec<(Vec<u32>, usize)> = state.pool.install(|| {
                    cluster_scratch[..nb_clusters]
                        .par_iter_mut()
                        .enumerate()
                        .map(|(cluster_idx, members)| measure!(
                            state.merge_nodes(
                                &aggregated_graph,
                                &aggregated_node_weights,
                                members,
                                &aggregated_membership,
                                cluster_idx as u32,
                                &config,
                            ),
                            STAT_MERGE
                        ))
                        .collect()
                });
                let mut nb_refined_clusters = 0usize;
                for (cluster_idx, (local_ids, k_i)) in cluster_results.into_iter().enumerate() {
                    for (&v, local_id) in cluster_scratch[cluster_idx].iter().zip(local_ids) {
                        refined_membership[v as usize] = nb_refined_clusters as u32 + local_id;
                    }
                    nb_refined_clusters += k_i;
                    cluster_scratch[cluster_idx].clear();
                }

                // If the refinement didn't aggregate any cluster, we aggregate based on the
                // original clustering obtained by fastmove_nodes
                if nb_refined_clusters >= aggregated_graph.n {
                    refined_membership = aggregated_membership.clone();
                    nb_refined_clusters = nb_clusters;
                }

                // Compute super node mapping
                measure!({
                    for node_id in 0..self.graph.n {
                        let super_node_id = super_node_map[node_id] as usize;
                        super_node_map[node_id] = refined_membership[super_node_id];
                    }
                }, STAT_REINDEX);
                (aggregated_graph, aggregated_membership, aggregated_node_weights) = measure!(self.aggregate(
                    &aggregated_graph,
                    &aggregated_node_weights,
                    &aggregated_membership,
                    &refined_membership,
                    nb_refined_clusters
                ), STAT_AGGREGATE);

                level += 1
            }
            // Optimization ended for this pass, we quit the loop
            if !continue_clustering { break; }
        }

        changed
    }

    fn retrieve_clusters(&self, cluster_scratch: &mut Vec<Vec<u32>>, aggregated_membership: &Vec<u32>) {
        for (node_id, &membership) in aggregated_membership.iter().enumerate() {
            cluster_scratch[membership as usize].push(node_id as u32);
        }
    }

    /// Local-moving phase, redesigned around a fused parallel decide+apply loop
    /// instead of the sequential version's async FIFO queue (every move
    /// immediately visible to the next) -- following the design used by real
    /// parallel Louvain/Leiden implementations (NetworKit's PLM; the published
    /// GVE-Leiden, github.com/puzzlef/leiden-communities-openmp): every node is
    /// swept in parallel each round; each thread scans its own node's
    /// neighbours, decides the best move, and commits it immediately via
    /// atomics, all within the same pass -- no separate frozen-snapshot decide
    /// phase followed by a serial apply/reconciliation phase. This matters
    /// operationally, not just architecturally: an earlier version of this
    /// function *did* freeze a snapshot and reconcile in a serial pass, and its
    /// serial apply cost scaled with the size of the active set each round --
    /// which is the *entire graph* on round 1. At production scale (50M
    /// nodes) that serial pass alone dominated the whole call, leaving 7 of 8
    /// worker threads idle and making the "parallel" version slower than the
    /// sequential one. The fused design keeps every round's work -- including
    /// the neighbour scan needed to find requeue candidates -- inside the
    /// parallel sweep.
    ///
    /// Instead of an explicit shrinking active-node list, an `affected` flag
    /// (one per node, cleared when processed, set on a mover's neighbours)
    /// tracks who needs reconsidering -- same technique both reference
    /// implementations use. `membership`/`cluster_weights` become atomic for
    /// the duration of this call so concurrent threads can read/update them
    /// safely; updates are `Relaxed` since nothing here needs stronger
    /// ordering than "eventually visible to other threads" -- there's no
    /// synchronization-dependent invariant beyond the atomics' own
    /// read-modify-write correctness. (Unlike the sequential version, there's
    /// no `cluster_degree`/empty-cluster-recycling bookkeeping to maintain
    /// here -- see the singleton-id note below for why.)
    ///
    /// "Become a new singleton" needs a cluster id nobody else could possibly
    /// claim at the same time, without a shared/serial allocator -- solved by
    /// reserving `graph.n + v` as node `v`'s own private singleton id: unique
    /// by construction, so claiming it needs no synchronization at all. Hence
    /// cluster ids in this function range over `[0, 2*graph.n)`, not
    /// `[0, graph.n)`.
    ///
    /// Trade-off: two nodes can still race to move into the same cluster (or
    /// swap into each other's) based on a momentarily-stale read, since
    /// there's no per-decision re-validation here -- accepted rather than
    /// guarded against, same as both reference implementations, because with
    /// atomics the staleness window is tiny (a handful of concurrent
    /// instructions) rather than a full round, so it self-corrects almost
    /// immediately via the next round's `affected` wake-ups instead of
    /// settling into a sustained oscillation. `MAX_FASTMOVE_ROUNDS` bounds the
    /// residual either way. `pytests/test_leiden_accuracy.py` is there to
    /// catch it if any of this ever drifts quality beyond normal stochastic
    /// noise.
    fn fastmove_nodes(&self, graph: &CsrGraph,
                      node_weights: &Vec<f32>,
                      config: &LeidenConfig,
                      membership: &mut Vec<u32>) -> (bool, usize){
        // Cluster ids [0, graph.n) are real clusters; [graph.n, 2*graph.n) are
        // each node's reserved private singleton id (node v -> graph.n + v).
        let id_space = 2 * graph.n;

        let cluster_weights: Vec<AtomicU32> = (0..id_space).map(|_| AtomicU32::new(0.0f32.to_bits())).collect();
        for v in 0..graph.n {
            let c = membership[v] as usize;
            atomic_f32_add(&cluster_weights[c], node_weights[v]);
        }
        let membership_atomic: Vec<AtomicU32> = membership.iter().map(|&m| AtomicU32::new(m)).collect();
        let affected: Vec<AtomicBool> = (0..graph.n).map(|_| AtomicBool::new(true)).collect();

        let changed = AtomicBool::new(false);
        let mut round = 0;
        loop {
            round += 1;
            let any_moved = AtomicBool::new(false);
            self.pool.install(|| {
                (0..graph.n).into_par_iter().for_each(|v| {
                    if !affected[v].swap(false, Ordering::Relaxed) { return; }
                    let Some(target) = self.decide_move(graph, node_weights, config, id_space, &membership_atomic, &cluster_weights, v) else { return; };

                    // decide_move only ever returns a target different from v's
                    // current cluster, and v is owned by exactly one task per
                    // round, so membership_atomic[v] can't have changed since.
                    let current = membership_atomic[v].load(Ordering::Relaxed) as usize;
                    atomic_f32_add(&cluster_weights[current], -node_weights[v]);
                    atomic_f32_add(&cluster_weights[target], node_weights[v]);
                    membership_atomic[v].store(target as u32, Ordering::Relaxed);
                    changed.store(true, Ordering::Relaxed);
                    any_moved.store(true, Ordering::Relaxed);

                    for &(u, _) in graph.neighbors(v) {
                        affected[u as usize].store(true, Ordering::Relaxed);
                    }
                });
            });
            if !any_moved.load(Ordering::Relaxed) || round >= MAX_FASTMOVE_ROUNDS { break; }
        }

        for (m, a) in membership.iter_mut().zip(&membership_atomic) {
            *m = a.load(Ordering::Relaxed);
        }
        let nb_clusters = reindex_membership(membership, id_space);
        (changed.load(Ordering::Relaxed), nb_clusters)
    }

    /// Scores node `v`'s candidate moves against the current (concurrently
    /// updated, not frozen) cluster state, returning the best target cluster
    /// id, or `None` to stay. Safe to call concurrently for different `v`:
    /// every shared read is atomic, and the only mutable state is the
    /// per-thread `FASTMOVE_SCRATCH` buffer, which different nodes on the same
    /// thread reuse sequentially and different threads never share.
    fn decide_move(&self, graph: &CsrGraph,
                   node_weights: &Vec<f32>,
                   config: &LeidenConfig,
                   id_space: usize,
                   membership: &Vec<AtomicU32>,
                   cluster_weights: &Vec<AtomicU32>,
                   v: usize) -> Option<usize> {
        let current_cluster = membership[v].load(Ordering::Relaxed) as usize;
        let singleton_cluster = graph.n + v;
        let weight = |c: usize| f32::from_bits(cluster_weights[c].load(Ordering::Relaxed));

        FASTMOVE_SCRATCH.with(|scratch| {
            let (weight_to_cluster, is_neighbor_cluster) = &mut *scratch.borrow_mut();
            if weight_to_cluster.len() < id_space {
                weight_to_cluster.resize(id_space, 0.0);
                is_neighbor_cluster.grow(id_space);
            }
            let mut neighbor_clusters: Vec<u32> = Vec::new();

            for &(u, w) in graph.neighbors(v) {
                let u = u as usize;
                if u != v {
                    let c = membership[u].load(Ordering::Relaxed) as usize;
                    if !is_neighbor_cluster.put(c) {
                        neighbor_clusters.push(c as u32);
                    }
                    weight_to_cluster[c] += w;
                }
            }

            // Calculate the score for each cluster to find the best one. Every
            // candidate is scored against a cluster weight that EXCLUDES v's
            // own contribution -- v isn't a member of any other candidate
            // cluster, so `current_cluster` must be treated the same way for a
            // fair comparison. "Become a new singleton" is evaluated first
            // (matching the sequential version's ordering, for the same
            // tie-breaking behaviour) -- its diff is always exactly 0, since an
            // empty cluster has no weight and no edges point to it yet.
            let current_cluster_weight = weight(current_cluster) - node_weights[v];
            let mut best: Option<usize> = None;
            let mut max_diff = weight_to_cluster[current_cluster] -
                config.resolution * (node_weights[v] * current_cluster_weight);
            if 0.0 > max_diff && current_cluster != singleton_cluster {
                best = Some(singleton_cluster);
                max_diff = 0.0;
            }
            for &c in &neighbor_clusters {
                let c = c as usize;
                let cw = if c == current_cluster { current_cluster_weight } else { weight(c) };
                let diff = weight_to_cluster[c] -
                    config.resolution * (node_weights[v] * cw);
                // Only consider positive improvements
                if diff > max_diff {
                    best = if c == current_cluster { None } else { Some(c) };
                    max_diff = diff;
                }
                weight_to_cluster[c] = 0.0;
                is_neighbor_cluster.set(c, false);
            }

            best
        })
    }

    /// Refines one (disjoint) cluster's members in isolation and returns the
    /// resulting locally-compacted refined-cluster id for each member (aligned
    /// 1:1 with `cluster_members`'s order on return, i.e. post-shuffle) plus
    /// the number of distinct ids used ([0, count)). Safe to call concurrently
    /// for different clusters: all scratch is either function-local (sized to
    /// this cluster only, same as the original) or the per-thread
    /// `MERGE_SCRATCH` buffer, which different clusters on the same thread
    /// reuse sequentially and different threads never share.
    fn merge_nodes(&self, graph: &CsrGraph,
                      node_weights: &Vec<f32>,
                      cluster_members: &mut Vec<u32>,
                      membership: &Vec<u32>,
                      cluster_idx: u32,
                      config: &LeidenConfig) -> (Vec<u32>, usize) {
        let n = cluster_members.len();
        // Weight of cluster. Sum of weights of all nodes
        let mut cluster_weights = vec![0.0f32; n]; // cluster_out_weights
        let mut cluster_degree = vec![0u32; n]; // nb_vertices_per_cluster
        // Sum of weight of all edges from a cluster going to another cluster
        let mut cluster_out_weight = vec![0.0f32; n]; // external_edge_weight_per_cluster_in_subset

        MERGE_SCRATCH.with(|scratch| {
            let mut refined_membership = scratch.borrow_mut();
            if refined_membership.len() < graph.n {
                refined_membership.resize(graph.n, 0);
            }

            let mut total_node_weight: f32 = 0.0;
            for (c, &v) in cluster_members.iter().enumerate() {
                let v = v as usize;
                refined_membership[v] = c as u32;
                cluster_weights[c] += node_weights[v];
                total_node_weight += node_weights[v];
                cluster_degree[c] += 1;

                // Find neighbours clusters
                for &(u, w) in graph.neighbors(v) {
                    let u = u as usize;
                    if u != v && membership[u] == cluster_idx {
                        cluster_out_weight[c] += w;
                    }
                }
            }

            let mut non_singleton_cluster = FixedBitSet::with_capacity(n);

            // Preallocate scratch buffers for the hot main loop
            // Contains the total weight of nodes going to cluster at index c
            let mut weight_to_cluster = vec![0.0f32; n]; // edge_weights_per_cluster or E(v, C)
            let mut is_neighbor_cluster = FixedBitSet::with_capacity(n); // neighbor_cluster_added
            let mut neighbor_clusters: Vec<u32> = Vec::with_capacity(n);

            // Cumulative likelihood
            let mut cum_likelihood: Vec<f64> = Vec::with_capacity(n); // cum_trans_diff

            // Main loop in a random order
            cluster_members.shuffle(&mut rng());

            for &v in cluster_members.iter() {
                let v = v as usize;
                let current_cluster = refined_membership[v] as usize;
                let node_weight_prod = cluster_weights[current_cluster] * (total_node_weight - cluster_weights[current_cluster]);

                if !non_singleton_cluster.contains(current_cluster) &&
                    cluster_out_weight[current_cluster] >= node_weight_prod * config.resolution {
                    // Remove the node from the cluster.
                    // Since cluster is a singleton, the cluster weight and degree becomes 0
                    cluster_weights[current_cluster] = 0.0;
                    cluster_degree[current_cluster] = 0;

                    // Find neighbouring clusters, and also add the current cluster to ensure the node
                    // can stay in its current cluster
                    neighbor_clusters.push(current_cluster as u32);
                    is_neighbor_cluster.set(current_cluster, true);
                    for &(u, w) in graph.neighbors(v) {
                        let u = u as usize;
                        if u != v && membership[u] == cluster_idx {
                            let c = refined_membership[u] as usize;
                            if !is_neighbor_cluster.put(c) {
                                neighbor_clusters.push(c as u32);
                            }
                            weight_to_cluster[c] += w;
                        }
                    }

                    // Calculate diffs and likelihoods
                    let mut best_cluster = current_cluster;
                    let mut max_diff = 0.0f32;
                    let mut total_cum_likelihood = 0.0f64;
                    for &c in &neighbor_clusters {
                        let c = c as usize;
                        let node_weight_prod = cluster_weights[c] * (total_node_weight - cluster_weights[c]);

                        if cluster_out_weight[c] >= config.resolution * node_weight_prod {
                            let diff = weight_to_cluster[c] - config.resolution * (node_weights[v] * cluster_weights[c]);
                            if diff > max_diff {
                                best_cluster = c;
                                max_diff = diff;
                            }
                            if diff >= 0.0 {
                                total_cum_likelihood += ((diff as f64) / config.beta).exp();
                            }
                        }
                        cum_likelihood.push(total_cum_likelihood);
                        // Reset scratch buffers
                        weight_to_cluster[c] = 0.0;
                        is_neighbor_cluster.set(c, false);
                    }
                    let chosen_cluster = if total_cum_likelihood.is_finite() {
                        let r = rng().random_range(0.0..total_cum_likelihood);
                        let chosen_idx = cum_likelihood.partition_point(|&x| x < r);
                        neighbor_clusters[chosen_idx] as usize
                    } else {
                        best_cluster
                    };
                    // Reset the remaining scratch buffers
                    neighbor_clusters.clear();
                    cum_likelihood.clear();

                    // Move node to randomly chosen cluster
                    cluster_weights[chosen_cluster] += node_weights[v];
                    cluster_degree[chosen_cluster] += 1;
                    // Update the cluster_out_weight state as the sum of edge weight going out of
                    // clusters may have changed
                    if chosen_cluster != current_cluster {
                        for &(u, w) in graph.neighbors(v) {
                            let u = u as usize;
                            if membership[u] == cluster_idx {
                                if refined_membership[u] as usize == chosen_cluster {
                                    cluster_out_weight[chosen_cluster] -= w;
                                }else {
                                    cluster_out_weight[chosen_cluster] += w;
                                }
                            }
                        }
                        refined_membership[v] = chosen_cluster as u32;
                        non_singleton_cluster.set(chosen_cluster, true);
                    }
                }
            }

            self.clean_refined_membership(cluster_members, &refined_membership)
        })
    }

    /// Compacts `cluster_members`' refined-cluster labels (read from
    /// `refined_membership`, indexed by global node id) into a dense local
    /// range [0, count). Returns the compacted id for each member, in
    /// `cluster_members`'s current order, plus the distinct-id count.
    fn clean_refined_membership(&self, cluster_members: &Vec<u32>,
                                refined_membership: &Vec<u32>) -> (Vec<u32>, usize) {
        let mut new_cluster = vec![0u32; cluster_members.len()];
        let mut nb_local_clusters = 0u32;
        for &v in cluster_members {
            let c = refined_membership[v as usize] as usize;
            if new_cluster[c] == 0 {
                nb_local_clusters += 1;
                new_cluster[c] = nb_local_clusters;
            }
        }
        let compacted = cluster_members.iter()
            .map(|&v| new_cluster[refined_membership[v as usize] as usize] - 1)
            .collect();
        (compacted, nb_local_clusters as usize)
    }

    fn aggregate(&self,
                 graph: &CsrGraph,
                 node_weights: &Vec<f32>,
                 membership: &Vec<u32>,
                 refined_membership: &Vec<u32>,
                 nb_refined_clusters: usize) -> (CsrGraph, Vec<u32>, Vec<f32>) {
        let mut refined_clusters: Vec<Vec<u32>> = vec![Vec::new(); nb_refined_clusters];
        self.retrieve_clusters(&mut refined_clusters, refined_membership);

        // Each super-node `c`'s work (scan its members' edges, sum their weight,
        // pick a representative) only reads shared state and writes to its own
        // `c`-indexed output slot -- independent across clusters, so runs in
        // parallel. Each task returns its own local edge list instead of pushing
        // into one shared `aggregated_edges`; concatenated below.
        let results: Vec<(Vec<(u32, u32, f32)>, f32, u32)> = self.pool.install(|| {
            refined_clusters.par_iter().enumerate().map(|(c, refined_cluster)| {
                AGGREGATE_SCRATCH.with(|scratch| {
                    let (weight_to_cluster, is_neighbor_cluster) = &mut *scratch.borrow_mut();
                    if weight_to_cluster.len() < nb_refined_clusters {
                        weight_to_cluster.resize(nb_refined_clusters, 0.0);
                        is_neighbor_cluster.grow(nb_refined_clusters);
                    }
                    let mut neighbor_clusters: Vec<u32> = Vec::new();

                    let mut local_edges = Vec::new();
                    let mut node_weight_sum = 0.0f32;
                    // Iterate on all nodes in refined cluster to get neighbour cluster and weights
                    for &v in refined_cluster {
                        let v = v as usize;
                        // Then iterate on edges to find neighbour clusters
                        for &(u, w) in graph.neighbors(v) {
                            let c2 = refined_membership[u as usize] as usize;
                            // To consider each edge once
                            if c2 > c {
                                if !is_neighbor_cluster.put(c2) {
                                    neighbor_clusters.push(c2 as u32);
                                }
                                weight_to_cluster[c2] += w;
                            }
                        }
                        node_weight_sum += node_weights[v];
                    }

                    // Actually add edges
                    for &c2 in &neighbor_clusters {
                        let c2 = c2 as usize;
                        local_edges.push((c as u32, c2 as u32, weight_to_cluster[c2]));

                        // Reset scratch buffer
                        weight_to_cluster[c2] = 0.0;
                        is_neighbor_cluster.set(c2, false);
                    }

                    // Representative membership of super node
                    (local_edges, node_weight_sum, membership[refined_cluster[0] as usize])
                })
            }).collect()
        });

        let mut aggregated_edges: Vec<(u32, u32, f32)> = Vec::new();
        let mut aggregated_node_weights: Vec<f32> = vec![0.0; nb_refined_clusters];
        let mut aggregated_membership: Vec<u32> = vec![0; nb_refined_clusters];
        for (c, (local_edges, node_weight_sum, representative)) in results.into_iter().enumerate() {
            aggregated_edges.extend(local_edges);
            aggregated_node_weights[c] = node_weight_sum;
            aggregated_membership[c] = representative;
        }

        (CsrGraph::new(nb_refined_clusters, &aggregated_edges, INWeightType::Similarity),
        aggregated_membership,
        aggregated_node_weights)
    }
}

pub fn fast_find_communities(graph: CsrGraph, gamma: f32, beta: f64, n_iterations: usize,
                        objective: LeidenObjective) -> Vec<usize> {
    let (resolution, node_weights) = match objective {
        LeidenObjective::Modularity => {
            let node_strengths: Vec<_> = (0..graph.n).into_iter().map(|v| graph.strength(v)).collect();
            (gamma / node_strengths.iter().sum::<f32>(), node_strengths)
        }
        LeidenObjective::CPM => {
            (gamma, vec![1.0f32; graph.n])
        }
    };
    let membership: Vec<u32> = (0..graph.n as u32).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(MAX_THREADS)
        .build()
        .expect("failed to build leidenp thread pool");
    let mut leiden_state = LeidenState{graph, node_weight: node_weights, membership, pool};
    let config = LeidenConfig{resolution, beta: beta};

    for _ in 0..(if n_iterations > 0 {n_iterations} else {usize::MAX}) {
        let changed = leiden_state.find_partition(&config);
        if !changed { break; }
    }

    #[cfg(feature = "monitor")]
    {
        eprintln!("── Leiden timings ──────────────────────────────");
        STAT_FASTMOVE.report("  fastmove_nodes");
        STAT_MERGE.report("  merge_nodes    ");
        STAT_AGGREGATE.report("  aggregate      ");
        STAT_FLATTEN.report("  flatten/retrieve");
        STAT_REINDEX.report("  supernode_remap");
        eprintln!("────────────────────────────────────────────────");
    }

    // Convert u32 membership back to usize for the public API
    leiden_state.membership.into_iter().map(|x| x as usize).collect()
}
