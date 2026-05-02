use super::{HNSWState, ScratchBuffers, Candidate};
use rand::prelude::*;
use crate::core::Distance;

impl<T: Sync, D: Distance<T>> HNSWState<T, D> {
    pub fn insert_parallel(&self, query_idx: usize, scratch_buffers: &mut ScratchBuffers, rng: &mut impl Rng) {
        debug_assert!(scratch_buffers.is_clear(), "scratch_buffers are expected to be empty on entry");
        scratch_buffers.ensure_capacity(self.data.len());

        let layer = if let Some((ep_node, ep_layer)) = self.entry_point.get() {
            let layer = self.sample_layer_with(rng);

            scratch_buffers.nearest_neighbors.push(Candidate {
                idx: ep_node,
                distance: self.distance(query_idx, ep_node),
            });
            // Coarse search: descend from entry point layer down to the new node's layer + 1
            if ep_layer > layer {
                for l in (layer + 1..=ep_layer).rev() {
                    self.search_layer(
                        query_idx, l, self.config.ef_init, self.config.proximity_threshold, self.config.strict_ef,
                        &mut scratch_buffers.visited,
                        &mut scratch_buffers.candidates,
                        &mut scratch_buffers.nearest_neighbors,
                        &mut scratch_buffers.snapshot,
                    );
                }
            }
            // Fine search: descend from min(layer, ep_layer) to 0, connecting edges
            for l in (0..=layer.min(ep_layer)).rev() {
                self.search_layer(
                    query_idx, l, self.config.ef_construction, self.config.proximity_threshold, self.config.strict_ef,
                    &mut scratch_buffers.visited,
                    &mut scratch_buffers.candidates,
                    &mut scratch_buffers.nearest_neighbors,
                    &mut scratch_buffers.snapshot,
                );
                scratch_buffers.neighbors.extend(
                    scratch_buffers.nearest_neighbors.iter().map(|c| c.idx)
                );
                scratch_buffers.selected_neighbors.clear();
                if l == 0 && self.config.threshold_based_neighbourhood {
                    self.select_threshold_neighbors(
                        query_idx, l, self.config.m, self.config.proximity_threshold,
                        &scratch_buffers.neighbors,
                        &mut scratch_buffers.candidates,
                        &mut scratch_buffers.discarded_candidates,
                        &mut scratch_buffers.visited,
                        &mut scratch_buffers.selected_neighbors,
                        &mut scratch_buffers.inner_snapshot,
                    );
                } else if self.config.use_heuristic {
                    self.select_neighbors(
                        query_idx, l, self.config.m,
                        &scratch_buffers.neighbors,
                        &mut scratch_buffers.candidates,
                        &mut scratch_buffers.discarded_candidates,
                        &mut scratch_buffers.visited,
                        &mut scratch_buffers.selected_neighbors,
                        &mut scratch_buffers.inner_snapshot,
                    );
                } else {
                    self.select_neighbors_simple(
                        query_idx, self.config.m,
                        &scratch_buffers.neighbors,
                        &mut scratch_buffers.candidates,
                        &mut scratch_buffers.selected_neighbors,
                    );
                }
                // Add bidirectional edges between the selected neighbors and the query
                for &e in scratch_buffers.selected_neighbors.iter() {
                    self.hgraph.add_edge(l, e, query_idx);
                }
                // Shrink connections for neighbors that exceeded M_max
                scratch_buffers.neighbors.clone_from(&scratch_buffers.selected_neighbors);
                for &e in scratch_buffers.neighbors.iter() {
                    let m_max = if l == 0 { self.config.m_max0 } else { self.config.m_max };
                    if self.hgraph.neighbors_len(l, e) > m_max {
                        // Snapshot e's neighbors, then select the best m_max of them
                        self.hgraph.neighbors_snapshot(l, e, &mut scratch_buffers.snapshot);
                        scratch_buffers.selected_neighbors.clear();
                        if l == 0 && self.config.threshold_based_neighbourhood {
                            self.select_threshold_neighbors(
                                e, l, m_max, self.config.proximity_threshold,
                                &scratch_buffers.snapshot,
                                &mut scratch_buffers.candidates,
                                &mut scratch_buffers.discarded_candidates,
                                &mut scratch_buffers.visited,
                                &mut scratch_buffers.selected_neighbors,
                                &mut scratch_buffers.inner_snapshot,
                            );
                        } else if self.config.use_heuristic {
                            self.select_neighbors(
                                e, l, m_max,
                                &scratch_buffers.snapshot,
                                &mut scratch_buffers.candidates,
                                &mut scratch_buffers.discarded_candidates,
                                &mut scratch_buffers.visited,
                                &mut scratch_buffers.selected_neighbors,
                                &mut scratch_buffers.inner_snapshot,
                            );
                        } else {
                            self.select_neighbors_simple(
                                e, m_max,
                                &scratch_buffers.snapshot,
                                &mut scratch_buffers.candidates,
                                &mut scratch_buffers.selected_neighbors,
                            );
                        }
                        self.hgraph.set_neighbourhood(l, e, &scratch_buffers.selected_neighbors);
                    }
                }
                scratch_buffers.neighbors.clear();
            }
            layer
        } else {
            // First node: no connections possible, just establish the entry point
            self.sample_layer_with(rng)
        };

        // Update the entry point if this node was inserted at a higher layer
        self.entry_point.try_update(layer, query_idx);

        // Housekeeping: reset all scratch buffers
        scratch_buffers.clear();
    }

    fn sample_layer_with(&self, rng: &mut impl Rng) -> usize {
        // Guard against sampling 0, which would return usize::MAX
        let r = rng.random::<f64>().max(f64::MIN_POSITIVE);
        let layer = (-r.ln() * self.config.m_l).floor() as usize;
        // Hard cap: probability of hitting this is negligible but it prevents index OOB
        layer.min(self.max_layers - 1)
    }
}
