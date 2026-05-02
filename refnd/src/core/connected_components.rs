use petgraph::unionfind::UnionFind;
use rustc_hash::FxHashMap;
use crate::core::leiden::CsrGraph;

/// Returns `(cluster_idx, size)` for the largest cluster in `clusters`.
/// Returns `(0, 0)` if the slice is empty.
pub fn largest_cluster(clusters: &[usize]) -> (usize, usize) {
    let mut sizes: FxHashMap<usize, usize> = FxHashMap::default();
    for &c in clusters {
        *sizes.entry(c).or_insert(0) += 1;
    }
    sizes.into_iter().max_by_key(|&(_, s)| s).unwrap_or((0, 0))
}

/// Assigns each node a contiguous cluster id [0, k) based on connected components.
pub fn find_connected_components(graph: &CsrGraph) -> Vec<usize> {
    let mut uf: UnionFind<usize> = UnionFind::new(graph.n);
    for v in 0..graph.n {
        for &(u, _) in graph.neighbors(v) {
            uf.union(v, u);
        }
    }
    let mut label_map: FxHashMap<usize, usize> = FxHashMap::default();
    let mut next = 0usize;
    (0..graph.n)
        .map(|v| {
            let root = uf.find(v);
            *label_map.entry(root).or_insert_with(|| {
                let l = next;
                next += 1;
                l
            })
        })
        .collect()
}
