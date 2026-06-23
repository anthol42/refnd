#[derive(Clone)]
pub struct CsrGraph {
    pub n: usize,
    pub m: f32,           // total weight (each edge counted once)
    offsets: Vec<usize>,
    adj: Vec<(u32, f32)>, // (neighbor, weight) — u32 to halve cache pressure
}

impl CsrGraph {
    pub fn new(n: usize, edges: &[(u32, u32, f32)], use_weight: bool, is_weight_distance: bool) -> Self {
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
            if is_weight_distance {
                w = 1.0 / (1.0 + w)
            }
            if !use_weight {
                w = 1.0
            }
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
}
