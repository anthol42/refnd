use std::collections::BTreeMap;

#[derive(Clone)]
pub enum  INWeightType{
    Similarity,
    Distance,
    SimilarityComplement,
    Unweighted,
}
#[derive(Clone)]
pub struct CsrGraph {
    pub n: usize,
    pub m: f32,           // total weight (each edge counted once)
    offsets: Vec<usize>,
    adj: Vec<(u32, f32)>, // (neighbor, weight) — u32 to halve cache pressure
}

impl CsrGraph {
    /// Builds an undirected CSR graph from a raw edge list.
    ///
    /// - `n`: number of nodes.
    /// - `edges`: `(src, dst, w)` triples; each is stored on both endpoints' adjacency
    ///   lists (self-loops occupy a single slot).
    /// - `inweight_type`: how to interpret the raw `w` value in `edges` and convert it
    ///   to a similarity-like edge weight used by Leiden:
    ///   - `Similarity`: `w` is already a similarity — used as-is.
    ///   - `Distance`: `w` is a distance, mapped to `1 / (1 + w)` so closer nodes get
    ///     higher weight.
    ///   - `SimilarityComplement`: `w` is `1 - similarity`, mapped back via `1 - w`.
    ///   - `Unweighted`: `w` is ignored and every edge weight is set to `1.0`.
    pub fn new(n: usize, edges: &[(u32, u32, f32)], inweight_type: INWeightType) -> Self {
        let m = edges.iter().map(|&(_, _, w)| w).sum();

        // Degree count — self-loops occupy one slot, not two
        let mut offsets = vec![0usize; n + 1];
        for &(src, dst, _) in edges {
            let (src, dst) = (src as usize, dst as usize);
            offsets[src + 1] += 1;
            if src != dst { offsets[dst + 1] += 1; }
        }
        for i in 1..=n { offsets[i] += offsets[i - 1]; }

        let mut adj = vec![(0u32, 0.0f32); offsets[n]];
        let mut cursor = offsets[..n].to_vec();

        for &(src, dst, mut w) in edges {
            let (src, dst) = (src as usize, dst as usize);
            w = match inweight_type {
                INWeightType::Similarity => {w}
                INWeightType::Distance => {1.0 / (1.0 + w)}
                INWeightType::SimilarityComplement => {1.0 - w}
                INWeightType::Unweighted => {1.0}
            };
            adj[cursor[src]] = (dst as u32, w);
            cursor[src] += 1;
            if src != dst {
                adj[cursor[dst]] = (src as u32, w);
                cursor[dst] += 1;
            }
        }

        Self { n, m, offsets, adj }
    }

    /// Adjacency list of `v` as (neighbor, weight) pairs.
    #[inline]
    pub fn neighbors(&self, v: usize) -> &[(u32, f32)] {
        &self.adj[self.offsets[v]..self.offsets[v + 1]]
    }

    /// Sum of edge weights incident to `v` (self-loops counted once).
    #[inline]
    pub fn strength(&self, v: usize) -> f32 {
        self.neighbors(v).iter().map(|&(_, w)| w).sum()
    }

    /// Induced subgraph on `nodes`. New ids are assigned in the order `nodes` is given.
    /// Returns the subgraph plus a map from old node id to new node id.
    pub fn subgraph(&self, nodes: &[usize]) -> (Self, BTreeMap<usize, usize>) {
        let old_to_new: BTreeMap<usize, usize> = nodes
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id))
            .collect();

        let mut edges = Vec::new();
        for (&old_src, &new_src) in &old_to_new {
            for &(old_dst, w) in self.neighbors(old_src) {
                let old_dst = old_dst as usize;
                if old_src > old_dst {
                    continue; // already added from the other endpoint
                }
                if let Some(&new_dst) = old_to_new.get(&old_dst) {
                    edges.push((new_src as u32, new_dst as u32, w));
                }
            }
        }

        (Self::new(old_to_new.len(), &edges, INWeightType::Similarity), old_to_new)
    }
}
