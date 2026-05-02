use super::{CsrGraph, reindex_membership};
use fixedbitset::FixedBitSet;
use std::collections::VecDeque;
use rand::prelude::*;
use rand::rng;
use clap::ValueEnum;

struct LeidenConfig {
    pub resolution: f32,
    pub beta: f64
}
struct LeidenState {
    graph: CsrGraph,
    node_weight: Vec<f32>,
    membership: Vec<usize>,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum LeidenObjective{
    Modularity,
    CPM
}

impl LeidenState {
    pub fn find_partition(&mut self, config: &LeidenConfig) -> bool{
        // Initialize temporary buffers
        let mut refined_membership: Vec<usize> = vec![0; self.graph.n];
        let mut cluster_scratch: Vec<Vec<usize>> = vec![vec![]; self.graph.n]; // clusters
        let mut super_node_map: Vec<usize> = (0..self.graph.n).collect(); // aggregate_vertex
        let mut aggregated_node_weights: Vec<f32> = self.node_weight.clone(); // i_vertex_out_weight
        let mut aggregated_membership: Vec<usize> = self.membership.clone(); // i_membership
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
            (did_changed, nb_clusters) = self.fastmove_nodes(
                &aggregated_graph,
                &aggregated_node_weights,
                &config,
                &mut aggregated_membership,
            );
            changed = changed || did_changed;

            continue_clustering = nb_clusters < aggregated_graph.n;
            if continue_clustering {
                // Flatten membership
                if level > 0 {
                    for node_id in 0..self.graph.n {
                        let super_node_id = super_node_map[node_id];
                        self.membership[node_id] = aggregated_membership[super_node_id];
                    }
                }
                self.retrieve_clusters(&mut cluster_scratch, &aggregated_membership);
                // ensure refined membership is correct size
                refined_membership.truncate(aggregated_graph.n);
                // Refine each cluster
                let mut nb_refined_clusters = 0;
                for cluster_idx in 0..nb_clusters {
                    nb_refined_clusters = self.merge_nodes(
                        &aggregated_graph,
                        &aggregated_node_weights,
                        &mut cluster_scratch[cluster_idx],
                        &aggregated_membership,
                        cluster_idx,
                        &config,
                        nb_refined_clusters,
                        &mut refined_membership
                    );
                    cluster_scratch[cluster_idx].clear()
                }

                // If the refinement didn't aggregate any cluster, we aggregate based on the
                // original clustering obtained by fastmove_nodes
                if nb_refined_clusters >= aggregated_graph.n {
                    refined_membership = aggregated_membership.clone();
                    nb_refined_clusters = nb_clusters;
                }

                // Compute super node mapping
                for node_id in 0..self.graph.n {
                    let super_node_id = super_node_map[node_id];
                    super_node_map[node_id] = refined_membership[super_node_id];
                }
                (aggregated_graph, aggregated_membership, aggregated_node_weights) = self.aggregate(
                    &aggregated_graph,
                    &aggregated_node_weights,
                    &aggregated_membership,
                    &refined_membership,
                    nb_refined_clusters
                );

                level += 1
            }
            // Optimization ended for this pass, we quit the loop
            if !continue_clustering { break; }
        }

        changed
    }
    fn retrieve_clusters(&self, cluster_scratch: &mut Vec<Vec<usize>>, aggregated_membership: &Vec<usize>) {
        for (node_id, membership) in aggregated_membership.iter().enumerate() {
            cluster_scratch[*membership].push(node_id);
        }
    }

    fn fastmove_nodes(&self, graph: &CsrGraph,
                      node_weights: &Vec<f32>,
                      config: &LeidenConfig,
                      membership: &mut Vec<usize>) -> (bool, usize){
        let mut changed = false;
        // 1 if node is NOT in the queue. 0 otherwise. All initialized to 0 as they are all in the queue
        let mut is_node_stable = FixedBitSet::with_capacity(graph.n);

        // Shuffle nodes, then add to the queue
        let mut nodes: Vec<usize> = (0..graph.n).collect();
        nodes.shuffle(&mut rng());
        let mut unstable_nodes = VecDeque::from_iter(nodes.into_iter());

        // This contains the weight of the cluster, the sum of weights of each node
        let mut cluster_weights = vec![0.0f32; graph.n]; // cluster_out_weights
        let mut cluster_degree = vec![0usize; graph.n]; // nb_vertices_per_cluster
        for v in 0..graph.n {
            let c = membership[v];
            cluster_weights[c] += node_weights[v];
            cluster_degree[c] += 1;
        }

        // This vector is used as a stack (FILO). It contains the idx of empty clusters for id recycling
        let mut empty_clusters: Vec<usize> = Vec::with_capacity(graph.n);
        for c in 0..graph.n {
            if cluster_degree[c] == 0 {
                empty_clusters.push(c);
            }
        }
        // Preallocate scratch buffers for the hot main loop
        // Contains the total weight of nodes going to cluster at index c
        let mut weight_to_cluster = vec![0.0f32; graph.n]; // edge_weights_per_cluster or E(v, C)
        let mut is_neighbor_cluster = FixedBitSet::with_capacity(graph.n); // neighbor_cluster_added
        let mut neighbor_clusters: Vec<usize> = Vec::with_capacity(graph.n);

        while let Some(v) = unstable_nodes.pop_front() {
            let current_cluster = membership[v];
            // Remove node from current cluster
            cluster_weights[current_cluster] -= node_weights[v];
            cluster_degree[current_cluster] -= 1;
            if cluster_degree[current_cluster] == 0 {
                empty_clusters.push(current_cluster);
            }

            // Find neighboring clusters, and weights to them from current node v
            // We also need to consider the case to moving the node v to a new empty cluster, so
            // let's do that first
            let empty_cluster = empty_clusters.pop().unwrap();
            neighbor_clusters.push(empty_cluster);
            is_neighbor_cluster.set(empty_cluster, true);

            for (u, w) in graph.neighbors(v) {
                if *u != v {
                    let c = membership[*u];
                    if !is_neighbor_cluster.put(c) {
                        neighbor_clusters.push(c);
                    }
                    weight_to_cluster[c] += w;
                }
            }

            // Calculate the score for each cluster to find the best one
            let mut best_cluster = current_cluster;
            // ΔH = E(v, C) - γ(k_v * k_C)
            let mut max_diff = weight_to_cluster[current_cluster];
            for c in &neighbor_clusters {
                let diff = weight_to_cluster[*c] -
                    config.resolution * (node_weights[v] * cluster_weights[*c]);
                // Only consider positive improvements
                if diff > max_diff {
                    best_cluster = *c;
                    max_diff = diff;
                }
                weight_to_cluster[*c] = 0.0;
                is_neighbor_cluster.set(*c, false);
            }
            neighbor_clusters.clear();

            // Move node to best cluster
            cluster_weights[best_cluster] += node_weights[v];
            cluster_degree[best_cluster] += 1;

            // If we did not use the empty cluster, put it back on the stack for a later reuse
            if best_cluster != empty_cluster {
                empty_clusters.push(empty_cluster);
            }

            // Mark node as stable as it is not in the queue anymore
            is_node_stable.set(v, true);

            // Add stable neighbors (not in queue) that are not part of the new cluster to the queue to check them again
            if best_cluster != current_cluster {
                changed = true;
                membership[v] = best_cluster;

                for (u, _) in graph.neighbors(v) {
                    if is_node_stable.contains(*u) && membership[*u] != best_cluster {
                        unstable_nodes.push_back(*u);
                        is_node_stable.set(*u, false);
                    }
                }
            }

        }

        let nb_clusters = reindex_membership(membership, graph.n);
        (changed, nb_clusters)
    }

    fn merge_nodes(&self, graph: &CsrGraph,
                      node_weights: &Vec<f32>,
                      cluster_members: &mut Vec<usize>,
                      membership: &Vec<usize>,
                      cluster_idx: usize,
                      config: &LeidenConfig,
                      nb_refined_clusters: usize,
                      refined_membership: &mut Vec<usize>) -> usize {
        let n = cluster_members.len();
        // Weight of cluster. Sum of weights of all nodes
        let mut cluster_weights = vec![0.0f32; n]; // cluster_out_weights
        let mut cluster_degree = vec![0usize; n]; // nb_vertices_per_cluster
        // Sum of weight of all edges from a cluster going to another cluster
        let mut cluster_out_weight = vec![0.0f32; n]; // external_edge_weight_per_cluster_in_subset

        let mut total_node_weight: f32 = 0.0;
        for (c, v) in cluster_members.iter().enumerate() {
            refined_membership[*v] = c;
            cluster_weights[c] += node_weights[*v];
            total_node_weight += node_weights[*v];
            cluster_degree[c] += 1;

            // Find neighbours clusters
            for (u, w) in graph.neighbors(*v) {
                if u != v && membership[*u] == cluster_idx {
                    cluster_out_weight[c] += w;
                }
            }
        }

        let mut non_singleton_cluster = FixedBitSet::with_capacity(n);

        // Preallocate scratch buffers for the hot main loop
        // Contains the total weight of nodes going to cluster at index c
        let mut weight_to_cluster = vec![0.0f32; n]; // edge_weights_per_cluster or E(v, C)
        let mut is_neighbor_cluster = FixedBitSet::with_capacity(n); // neighbor_cluster_added
        let mut neighbor_clusters: Vec<usize> = Vec::with_capacity(n);

        // Cumulative likelihood
        let mut cum_likelihood: Vec<f64> = Vec::with_capacity(n); // cum_trans_diff

        // Main loop in a random order
        cluster_members.shuffle(&mut rng());

        for v in cluster_members.iter() {
            let current_cluster = refined_membership[*v];
            let node_weight_prod = cluster_weights[current_cluster] * (total_node_weight - cluster_weights[current_cluster]);

            if !non_singleton_cluster.contains(current_cluster) &&
                cluster_out_weight[current_cluster] >= node_weight_prod * config.resolution {
                // Remove the node from the cluster.
                // Since cluster is a singleton, the cluster weight and degree becomes 0
                cluster_weights[current_cluster] = 0.0;
                cluster_degree[current_cluster] = 0;

                // Find neighbouring clusters, and also add the current cluster to ensure the node
                // can stay in its current cluster
                neighbor_clusters.push(current_cluster);
                is_neighbor_cluster.set(current_cluster, true);
                for (u, w) in graph.neighbors(*v) {
                    if *u != *v && membership[*u] == cluster_idx {
                        let c = refined_membership[*u];
                        if !is_neighbor_cluster.put(c) {
                            neighbor_clusters.push(c);
                        }
                        weight_to_cluster[c] += w;
                    }
                }

                // Calculate diffs and likelihoods
                let mut best_cluster = current_cluster;
                let mut max_diff = 0.0f32;
                let mut total_cum_likelihood = 0.0f64;
                for c in &neighbor_clusters {
                    let node_weight_prod = cluster_weights[*c] * (total_node_weight - cluster_weights[*c]);

                    if cluster_out_weight[*c] >= config.resolution * node_weight_prod {
                        let diff = weight_to_cluster[*c] - config.resolution * (node_weights[*v] * cluster_weights[*c]);
                        if diff > max_diff {
                            best_cluster = *c;
                            max_diff = diff;
                        }
                        if diff >= 0.0 {
                            total_cum_likelihood += ((diff as f64) / config.beta).exp();
                        }
                    }
                    cum_likelihood.push(total_cum_likelihood);
                    // Reset scratch buffers
                    weight_to_cluster[*c] = 0.0;
                    is_neighbor_cluster.set(*c, false);
                }
                let chosen_cluster = if total_cum_likelihood.is_finite() {
                    let r = rng().random_range(0.0..total_cum_likelihood);
                    let chosen_idx = cum_likelihood.partition_point(|&x| x < r);
                    neighbor_clusters[chosen_idx]
                } else {
                    best_cluster
                };
                // Reset the remaining scratch buffers
                neighbor_clusters.clear();
                cum_likelihood.clear();

                // Move node to randomly chosen cluster
                cluster_weights[chosen_cluster] += node_weights[*v];
                cluster_degree[chosen_cluster] += 1;
                // Update the cluster_out_weight state as the sum of edge weight going out of
                // clusters may have changed
                if chosen_cluster != current_cluster {
                    for (u, w) in graph.neighbors(*v) {
                        if membership[*u] == cluster_idx {
                            if refined_membership[*u] == chosen_cluster {
                                cluster_out_weight[chosen_cluster] -= w;
                            }else {
                                cluster_out_weight[chosen_cluster] += w;
                            }
                        }
                    }
                    refined_membership[*v] = chosen_cluster;
                    non_singleton_cluster.set(chosen_cluster, true);
                }
            }
        }

        self.clean_refined_membership(&cluster_members, refined_membership, nb_refined_clusters)
    }

    fn clean_refined_membership(&self, cluster_members: &Vec<usize>,
                                refined_membership: &mut Vec<usize>,
                                mut nb_refined_clusters: usize) -> usize {
        let mut new_cluster = vec![0usize; refined_membership.len()];
        nb_refined_clusters += 1;
        // Fill new_cluster / cluster mapping
        for v in cluster_members {
            let c = refined_membership[*v];
            if new_cluster[c] == 0 {
                new_cluster[c] = nb_refined_clusters;
                nb_refined_clusters += 1;
            }
        }
        // Assign new clusters
        for v in cluster_members {
            let c = refined_membership[*v];
            refined_membership[*v] = new_cluster[c] - 1;
        }
        nb_refined_clusters -= 1;

        nb_refined_clusters
    }
    fn aggregate(&self,
                 graph: &CsrGraph,
                 node_weights: &Vec<f32>,
                 membership: &Vec<usize>,
                 refined_membership: &Vec<usize>,
                 nb_refined_clusters: usize) -> (CsrGraph, Vec<usize>, Vec<f32>) {
        let mut refined_clusters: Vec<Vec<usize>> = vec![Vec::new(); nb_refined_clusters];
        self.retrieve_clusters(&mut refined_clusters, refined_membership);

        let mut aggregated_edges: Vec<(usize, usize, f32)> = Vec::new();
        let mut aggregated_node_weights: Vec<f32> = vec![0.0; nb_refined_clusters];
        let mut aggregated_membership: Vec<usize> = vec![0; nb_refined_clusters];

        // Preallocate scratch buffers
        // Contains the total weight of nodes going to cluster at index c
        let mut weight_to_cluster: Vec<f32> = vec![0.0; nb_refined_clusters];
        let mut is_neighbor_cluster = FixedBitSet::with_capacity(nb_refined_clusters); // neighbor_cluster_added
        let mut neighbor_clusters: Vec<usize> = Vec::with_capacity(nb_refined_clusters);

        for (c, refined_cluster) in refined_clusters.iter().enumerate() {
            // Iterate on all nodes in refined cluster to get neighbour cluster and weights
            for v in refined_cluster {
                // Then iterate on edges to find neighbour clusters
                for (u, w) in graph.neighbors(*v) {
                    let c2 = refined_membership[*u];
                    // To consider each edge once
                    if c2 > c {
                        if !is_neighbor_cluster.put(c2) {
                            neighbor_clusters.push(c2);
                        }
                        weight_to_cluster[c2] += w;
                    }
                }

                aggregated_node_weights[c] += node_weights[*v];
            }

            // Actually add edges
            for c2 in &neighbor_clusters {
                aggregated_edges.push((c, *c2, weight_to_cluster[*c2]));

                // Reset scratch buffer
                weight_to_cluster[*c2] = 0.0;
                is_neighbor_cluster.set(*c2, false);
            }
            neighbor_clusters.clear();

            // Set membership of super node
            aggregated_membership[c] = membership[refined_cluster[0]];
        }

        (CsrGraph::new(nb_refined_clusters, &aggregated_edges, false, false),
        aggregated_membership,
        aggregated_node_weights)
    }
}

pub fn find_communities(graph: CsrGraph, gamma: f32, beta: f64, n_iterations: usize,
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
    let membership: Vec<usize> = (0..graph.n).collect();
    let mut leiden_state = LeidenState{graph, node_weight: node_weights, membership};
    let config = LeidenConfig{resolution, beta: beta};

    for _ in 0..(if n_iterations > 0 {n_iterations} else {usize::MAX}) {
        let changed = leiden_state.find_partition(&config);
        if !changed { break; }
    }

    leiden_state.membership
}