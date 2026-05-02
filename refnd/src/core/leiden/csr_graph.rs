#[derive(Clone)]
pub struct CsrGraph {
    pub n: usize,
    pub m: f32,           // total weight (each edge counted once)
    offsets: Vec<usize>,
    adj: Vec<(usize, f32)>, // (neighbor, weight)
}

impl CsrGraph {
    pub fn new(n: usize, edges: &[(usize, usize, f32)], use_weight: bool, is_weight_distance: bool) -> Self {
        let m = edges.iter().map(|&(_, _, w)| w).sum();

        // Degree count — self-loops occupy one slot, not two
        let mut offsets = vec![0usize; n + 1];
        for &(src, dst, _) in edges {
            offsets[src + 1] += 1;
            if src != dst { offsets[dst + 1] += 1; }
        }
        for i in 1..=n { offsets[i] += offsets[i - 1]; }

        let mut adj = vec![(0usize, 0.0f32); offsets[n]];
        let mut cursor = offsets[..n].to_vec();

        for &(src, dst, mut w) in edges {
            if is_weight_distance { 
                w = 1.0 / (1.0 + w) 
            }
            if !use_weight {
                w = 1.0
            }
            adj[cursor[src]] = (dst, w);
            cursor[src] += 1;
            if src != dst {
                adj[cursor[dst]] = (src, w);
                cursor[dst] += 1;
            }
        }

        Self { n, m, offsets, adj }
    }

    // pub fn save(&self, fs: &mut File) -> Result<(), String> {
    //     let data = bincode::encode_to_vec(self, config::standard()).map_err(|e| e.to_string())?;
    //     fs.write_all(&data).map_err(|e| e.to_string())?;
    //     Ok(())
    // }
    // 
    // pub fn from_file(fs: &mut File) -> Result<Self, String> {
    //     let mut bytes = Vec::new();
    //     fs.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    //     let (obj, _) = bincode::decode_from_slice(&bytes, config::standard()).map_err(|e| e.to_string())?;
    //     Ok(obj)
    // }

    /// Adjacency list of `v` as (neighbor, weight) pairs.
    #[inline]
    pub fn neighbors(&self, v: usize) -> &[(usize, f32)] {
        &self.adj[self.offsets[v]..self.offsets[v + 1]]
    }

    /// Sum of edge weights incident to `v` (self-loops counted once).
    #[inline]
    pub fn strength(&self, v: usize) -> f32 {
        self.neighbors(v).iter().map(|&(_, w)| w).sum()
    }
}
