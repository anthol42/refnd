use std::fmt;
use super::HNSWConfig;

/// Serializable snapshot of an [`super::HNSWState`].
///
/// Contains everything needed to reconstruct the index, minus the data
/// and the distance function (both supplied by the caller on load).
/// Concurrency wrappers (`RwLock`, `Mutex`, `DashMap`) are stripped:
/// the graph is stored as plain `Vec`s and proximity edges as a flat list.
#[derive(bincode::Encode, bincode::Decode)]
pub struct HNSWIndex {
    /// Number of data points the index was built on.
    /// Checked against the dataset length on load to catch mismatches early.
    pub dataset_size: usize,
    /// `layers[layer][node]` → neighbor list (no `RwLock`).
    pub layers: Vec<Vec<Vec<usize>>>,
    /// Global entry point as `(node, layer)`, or `None` if the index is empty.
    pub entry_point: Option<(usize, usize)>,
    pub config: HNSWConfig,
    pub max_layers: usize,
    /// All pairs `(i, j)` (with `i ≤ j`) whose distance is below `config.proximity_threshold`.
    pub proximity_edges: Vec<((usize, usize), f32)>,
}

impl fmt::Display for HNSWIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_empty_layers = self.layers.iter()
            .filter(|layer| layer.iter().any(|n| !n.is_empty()))
            .count();
        write!(
            f,
            "HNSWIndex(dataset_size={}, non_empty_layers={}, n_edges={})",
            self.dataset_size, non_empty_layers, self.proximity_edges.len(),
        )
    }
}

impl fmt::Debug for HNSWIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layers: Vec<String> = self.layers.iter()
            .map(|layer| layer.iter().filter(|n| !n.is_empty()).count())
            .filter(|&c| c > 0)
            .map(|c| c.to_string())
            .collect();
        let entry_point = match self.entry_point {
            Some((node, layer)) => format!("({}, {})", node, layer),
            None => "None".to_string(),
        };
        write!(
            f,
            "HNSWIndex(\n\
            \x20 dataset_size={},\n\
            \x20 entry_point={},\n\
            \x20 max_layers={},\n\
            \x20 n_edges={},\n\
            \x20 layers=[{}],\n\
            \x20 config={}\n\
            )",
            self.dataset_size,
            entry_point,
            self.max_layers,
            self.proximity_edges.len(),
            layers.join(", "),
            self.config,
        )
    }
}

impl HNSWIndex {
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::encode_to_vec(self, bincode::config::standard())?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let (index, _) = bincode::decode_from_slice(&bytes, bincode::config::standard())?;
        Ok(index)
    }
}
