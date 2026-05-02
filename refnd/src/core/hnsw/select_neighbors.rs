use std::cmp::Reverse;
use fixedbitset::FixedBitSet;
use crate::core::Distance;
use super::{HNSWState, Candidate, MinHeap};

impl<T: Sync, D: Distance<T>> HNSWState<T, D> {
    /// All scratch buffers and `selected_neighbors` must be clear on entry.
    /// Clears `candidates`, `discarded_candidates`, and `visited` before returning.
    /// `selected_neighbors` is the output — modified in place.
    pub(super) fn select_neighbors(
        &self,
        query: usize,
        layer: usize,
        m: usize,
        nearest_neighbors: &[usize],
        // Scratch buffers
        candidates: &mut MinHeap<Candidate>,
        discarded_candidates: &mut MinHeap<Candidate>,
        visited: &mut FixedBitSet,
        // Output
        selected_neighbors: &mut Vec<usize>,
        // Snapshot buffer for the extend_candidates path
        inner_snapshot: &mut Vec<usize>,
    ) {
        debug_assert!(selected_neighbors.is_empty(), "Expect selected_neighbors to be empty on entry");
        debug_assert!(candidates.is_empty(), "Expect candidates to be empty on entry");
        debug_assert!(discarded_candidates.is_empty(), "Expect discarded_candidates to be empty on entry");
        debug_assert!(visited.is_clear(), "Expect visited bitset to be empty on entry");

        for &neighbor in nearest_neighbors {
            let dist_neighbor2query = self.distance(query, neighbor);
            candidates.push(Reverse(Candidate { idx: neighbor, distance: dist_neighbor2query }));
            visited.set(neighbor, true);
        }

        if self.config.extend_candidates {
            for &neighbor in nearest_neighbors {
                // Snapshot under read lock, release before calling distance()
                self.hgraph.neighbors_snapshot(layer, neighbor, inner_snapshot);
                for &neighbor_of_neighbor in inner_snapshot.iter() {
                    if !visited.put(neighbor_of_neighbor) {
                        let dist = self.distance(query, neighbor_of_neighbor);
                        candidates.push(Reverse(Candidate { idx: neighbor_of_neighbor, distance: dist }));
                    }
                }
                inner_snapshot.clear();
            }
        }

        while let Some(Reverse(candidate)) = candidates.pop() {
            if selected_neighbors.len() >= m {
                break;
            }
            if selected_neighbors.is_empty()
                || candidate.distance < self.min_distance_with_many(candidate.idx, selected_neighbors)
            {
                selected_neighbors.push(candidate.idx);
            } else {
                discarded_candidates.push(Reverse(candidate));
            }
        }

        if self.config.keep_pruned_connections {
            while let Some(Reverse(candidate)) = discarded_candidates.pop() {
                if selected_neighbors.len() >= m {
                    break;
                }
                selected_neighbors.push(candidate.idx);
            }
        }

        // Housekeeping: discarded, candidates, and visited are not output
        candidates.clear();
        discarded_candidates.clear();
        visited.clear();
    }

    /// Algorithm 3 from the HNSW paper: return the M nearest elements from `nearest_neighbors`.
    /// No heuristic pruning — just sort by distance and take the top-m.
    /// Clears `candidates` before returning.
    pub(super) fn select_neighbors_simple(
        &self,
        query: usize,
        m: usize,
        nearest_neighbors: &[usize],
        candidates: &mut MinHeap<Candidate>,
        selected_neighbors: &mut Vec<usize>,
    ) {
        debug_assert!(selected_neighbors.is_empty());
        debug_assert!(candidates.is_empty());

        for &neighbor in nearest_neighbors {
            let dist = self.distance(query, neighbor);
            candidates.push(Reverse(Candidate { idx: neighbor, distance: dist }));
        }
        while let Some(Reverse(c)) = candidates.pop() {
            if selected_neighbors.len() >= m {
                break;
            }
            selected_neighbors.push(c.idx);
        }
        candidates.clear();
    }

    /// Select a minimum of m neighbors like select_neighbors_simple, but doesn't bound the
    /// neighbourhood size as all candidates that are closer than the threshold are kept.
    /// This function should only be called on layer0, as it will make a denser layer.
    /// Heuristic neighbour selection (analogous to `select_neighbors` / Algorithm 4) but without
    /// a hard upper bound on neighbourhood size: every candidate whose distance to `query` is
    /// below `threshold` is kept, with a guaranteed minimum of `m` neighbours.
    /// Should only be called on layer 0 (produces a denser neighbourhood than upper layers need).
    pub(super) fn select_threshold_neighbors(
        &self,
        query: usize,
        layer: usize,
        m: usize,
        threshold: f32,
        nearest_neighbors: &[usize],
        // Scratch buffers
        candidates: &mut MinHeap<Candidate>,
        discarded_candidates: &mut MinHeap<Candidate>,
        visited: &mut FixedBitSet,
        // Output
        selected_neighbors: &mut Vec<usize>,
        // Snapshot buffer for the extend_candidates path
        inner_snapshot: &mut Vec<usize>,
    ) {
        debug_assert!(selected_neighbors.is_empty(), "Expect selected_neighbors to be empty on entry");
        debug_assert!(candidates.is_empty(), "Expect candidates to be empty on entry");
        debug_assert!(discarded_candidates.is_empty(), "Expect discarded_candidates to be empty on entry");
        debug_assert!(visited.is_clear(), "Expect visited bitset to be empty on entry");

        for &neighbor in nearest_neighbors {
            let dist = self.distance(query, neighbor);
            candidates.push(Reverse(Candidate { idx: neighbor, distance: dist }));
            visited.set(neighbor, true);
        }

        if self.config.extend_candidates {
            for &neighbor in nearest_neighbors {
                self.hgraph.neighbors_snapshot(layer, neighbor, inner_snapshot);
                for &neighbor_of_neighbor in inner_snapshot.iter() {
                    if !visited.put(neighbor_of_neighbor) {
                        let dist = self.distance(query, neighbor_of_neighbor);
                        candidates.push(Reverse(Candidate { idx: neighbor_of_neighbor, distance: dist }));
                    }
                }
                inner_snapshot.clear();
            }
        }

        while let Some(Reverse(candidate)) = candidates.pop() {
            if selected_neighbors.len() >= m && candidate.distance > threshold {
                break;
            }
            if selected_neighbors.is_empty()
                || candidate.distance < self.min_distance_with_many(candidate.idx, selected_neighbors)
            {
                selected_neighbors.push(candidate.idx);
            } else {
                discarded_candidates.push(Reverse(candidate));
            }
        }

        if self.config.keep_pruned_connections {
            while let Some(Reverse(candidate)) = discarded_candidates.pop() {
                if selected_neighbors.len() >= m && candidate.distance >= threshold {
                    break;
                }
                selected_neighbors.push(candidate.idx);
            }
        }

        candidates.clear();
        discarded_candidates.clear();
        visited.clear();
    }
}
