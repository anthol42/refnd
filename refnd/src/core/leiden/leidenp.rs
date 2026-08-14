use super::{CsrGraph, INWeightType, reindex_membership};
use crate::core::hnsw::measure;
#[cfg(feature = "monitor")]
use crate::core::hnsw::LockStat;
use fixedbitset::FixedBitSet;
use std::cell::RefCell;
use std::collections::VecDeque;
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;
use super::leiden::LeidenObjective;

/// Parallel sections never spawn more than this many worker threads.
const MAX_THREADS: usize = 8;

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

    fn fastmove_nodes(&self, graph: &CsrGraph,
                      node_weights: &Vec<f32>,
                      config: &LeidenConfig,
                      membership: &mut Vec<u32>) -> (bool, usize){
        let mut changed = false;
        // 1 if node is NOT in the queue. 0 otherwise. All initialized to 0 as they are all in the queue
        let mut is_node_stable = FixedBitSet::with_capacity(graph.n);

        // Shuffle nodes, then add to the queue
        let mut nodes: Vec<u32> = (0..graph.n as u32).collect();
        nodes.shuffle(&mut rng());
        let mut unstable_nodes = VecDeque::from_iter(nodes.into_iter());

        // This contains the weight of the cluster, the sum of weights of each node
        let mut cluster_weights = vec![0.0f32; graph.n]; // cluster_out_weights
        let mut cluster_degree = vec![0u32; graph.n]; // nb_vertices_per_cluster
        for v in 0..graph.n {
            let c = membership[v] as usize;
            cluster_weights[c] += node_weights[v];
            cluster_degree[c] += 1;
        }

        // This vector is used as a stack (FILO). It contains the idx of empty clusters for id recycling
        let mut empty_clusters: Vec<u32> = Vec::with_capacity(graph.n);
        for c in 0..graph.n {
            if cluster_degree[c] == 0 {
                empty_clusters.push(c as u32);
            }
        }
        // Preallocate scratch buffers for the hot main loop
        // Contains the total weight of nodes going to cluster at index c
        let mut weight_to_cluster = vec![0.0f32; graph.n]; // edge_weights_per_cluster or E(v, C)
        let mut is_neighbor_cluster = FixedBitSet::with_capacity(graph.n); // neighbor_cluster_added
        let mut neighbor_clusters: Vec<u32> = Vec::with_capacity(graph.n);

        while let Some(v) = unstable_nodes.pop_front() {
            let v = v as usize;
            let current_cluster = membership[v] as usize;
            // Remove node from current cluster
            cluster_weights[current_cluster] -= node_weights[v];
            cluster_degree[current_cluster] -= 1;
            if cluster_degree[current_cluster] == 0 {
                empty_clusters.push(current_cluster as u32);
            }

            // Find neighboring clusters, and weights to them from current node v
            // We also need to consider the case to moving the node v to a new empty cluster, so
            // let's do that first
            let empty_cluster = empty_clusters.pop().unwrap() as usize;
            neighbor_clusters.push(empty_cluster as u32);
            is_neighbor_cluster.set(empty_cluster, true);

            for &(u, w) in graph.neighbors(v) {
                let u = u as usize;
                if u != v {
                    let c = membership[u] as usize;
                    if !is_neighbor_cluster.put(c) {
                        neighbor_clusters.push(c as u32);
                    }
                    weight_to_cluster[c] += w;
                }
            }

            // Calculate the score for each cluster to find the best one
            let mut best_cluster = current_cluster;
            let mut max_diff = weight_to_cluster[current_cluster] -
                config.resolution * (node_weights[v] * cluster_weights[current_cluster]);
            for &c in &neighbor_clusters {
                let c = c as usize;
                let diff = weight_to_cluster[c] -
                    config.resolution * (node_weights[v] * cluster_weights[c]);
                // Only consider positive improvements
                if diff > max_diff {
                    best_cluster = c;
                    max_diff = diff;
                }
                weight_to_cluster[c] = 0.0;
                is_neighbor_cluster.set(c, false);
            }
            neighbor_clusters.clear();

            // Move node to best cluster
            cluster_weights[best_cluster] += node_weights[v];
            cluster_degree[best_cluster] += 1;

            // If we did not use the empty cluster, put it back on the stack for a later reuse
            if best_cluster != empty_cluster {
                empty_clusters.push(empty_cluster as u32);
            }

            // Mark node as stable as it is not in the queue anymore
            is_node_stable.set(v, true);

            // Add stable neighbors (not in queue) that are not part of the new cluster to the queue to check them again
            if best_cluster != current_cluster {
                changed = true;
                membership[v] = best_cluster as u32;

                for &(u, _) in graph.neighbors(v) {
                    let u = u as usize;
                    if is_node_stable.contains(u) && membership[u] as usize != best_cluster {
                        unstable_nodes.push_back(u as u32);
                        is_node_stable.set(u, false);
                    }
                }
            }

        }

        let nb_clusters = reindex_membership(membership, graph.n);
        (changed, nb_clusters)
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

        let mut aggregated_edges: Vec<(u32, u32, f32)> = Vec::new();
        let mut aggregated_node_weights: Vec<f32> = vec![0.0; nb_refined_clusters];
        let mut aggregated_membership: Vec<u32> = vec![0; nb_refined_clusters];

        // Preallocate scratch buffers
        // Contains the total weight of nodes going to cluster at index c
        let mut weight_to_cluster: Vec<f32> = vec![0.0; nb_refined_clusters];
        let mut is_neighbor_cluster = FixedBitSet::with_capacity(nb_refined_clusters); // neighbor_cluster_added
        let mut neighbor_clusters: Vec<u32> = Vec::with_capacity(nb_refined_clusters);

        for (c, refined_cluster) in refined_clusters.iter().enumerate() {
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

                aggregated_node_weights[c] += node_weights[v];
            }

            // Actually add edges
            for &c2 in &neighbor_clusters {
                let c2 = c2 as usize;
                aggregated_edges.push((c as u32, c2 as u32, weight_to_cluster[c2]));

                // Reset scratch buffer
                weight_to_cluster[c2] = 0.0;
                is_neighbor_cluster.set(c2, false);
            }
            neighbor_clusters.clear();

            // Set membership of super node
            aggregated_membership[c] = membership[refined_cluster[0] as usize];
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
