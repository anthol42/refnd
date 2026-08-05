use std::fmt;
use super::HNSWConfig;

/// One layer's neighbor lists. Layer 0 holds every node (dense); layers above hold an
/// exponentially shrinking fraction (sparse) -- see `LayerStorage` in `mod.rs` for why.
/// Keeping that distinction in the *saved* format too matters: naively densifying every
/// sparse layer back into an `n_nodes`-long `Vec` just to serialize it would recreate the
/// same multi-GB-per-layer waste the in-memory sparse representation exists to avoid, right
/// at the point (end of a long build) where peak memory is already at its highest.
#[derive(bincode::Encode, bincode::Decode, Clone)]
pub enum LayerData {
    /// Every node's neighbor list, in node-id order.
    Dense(Vec<Vec<u32>>),
    /// Only the `(node, neighbors)` pairs that are actually non-empty.
    Sparse(Vec<(u32, Vec<u32>)>),
}

impl LayerData {
    fn n_non_empty(&self) -> usize {
        match self {
            LayerData::Dense(v) => v.iter().filter(|n| !n.is_empty()).count(),
            LayerData::Sparse(pairs) => pairs.len(),
        }
    }
}

/// Serializable snapshot of an [`super::HNSWState`].
///
/// Contains everything needed to reconstruct the index, minus the data
/// and the distance function (both supplied by the caller on load).
/// Concurrency wrappers (`RwLock`, `Mutex`) are stripped: the graph is stored
/// as plain `Vec`s and proximity edges as a flat list.
#[derive(bincode::Encode, bincode::Decode)]
pub struct HNSWIndex {
    /// `refnd`'s `(major, minor, patch)` version at save time. The crate is
    /// pre-1.0 and the on-disk format is not guaranteed stable across
    /// versions — checked against the running crate's version on load so a
    /// mismatch fails loudly instead of decoding into garbage.
    pub crate_version: (u16, u16, u16),
    /// Number of data points the index was built on.
    /// Checked against the dataset length on load to catch mismatches early.
    pub dataset_size: usize,
    /// `layers[layer]` → that layer's neighbor lists, dense or sparse (see `LayerData`).
    pub layers: Vec<LayerData>,
    /// Global entry point as `(node, layer)`, or `None` if the index is empty.
    pub entry_point: Option<(u32, usize)>,
    pub config: HNSWConfig,
    pub max_layers: usize,
    /// All pairs `(i, j)` (with `i ≤ j`) whose distance is below `config.proximity_threshold`.
    pub proximity_edges: Vec<((u32, u32), f32)>,
    pub has_been_built: bool,
}

impl fmt::Display for HNSWIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let non_empty_layers = self.layers.iter()
            .filter(|layer| layer.n_non_empty() > 0)
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
            .map(|layer| layer.n_non_empty())
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
            \x20 has_been_built={},\n\
            \x20 entry_point={},\n\
            \x20 max_layers={},\n\
            \x20 n_edges={},\n\
            \x20 layers=[{}],\n\
            \x20 config={}\n\
            )",
            self.dataset_size,
            self.has_been_built,
            entry_point,
            self.max_layers,
            self.proximity_edges.len(),
            layers.join(", "),
            self.config,
        )
    }
}

/// Parses the crate's own `CARGO_PKG_VERSION` (always `major.minor.patch`), so this
/// never sees arbitrary input and unwrapping is safe.
pub(crate) fn current_crate_version() -> (u16, u16, u16) {
    let mut parts = env!("CARGO_PKG_VERSION").split('.');
    let mut next = || parts.next().unwrap().parse().unwrap();
    (next(), next(), next())
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
