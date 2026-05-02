use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use crate::cli::display;
use std::fmt::Display;
use std::str::FromStr;
use std::cmp::PartialOrd;
use refnd::core::{
    Distance, EdgeStore,
    hnsw::{HNSWConfig, HNSWState},
    exact::{exact_edges, exact_edges_total},
    leiden::{CsrGraph, find_communities},
    connected_components::{find_connected_components, largest_cluster},
};
use refnd::kernels::proteins::parasail::ProteinKernel;
use super::parameters::LeidenArgs;

/// Check if the path exists, check if it's a file. If not, prints an error using display and
/// exit with code 1.
pub fn check_file_exists(path: &PathBuf){
    if !path.exists() {
        display::error(&format!("input file '{}' does not exist", path.display()));
        std::process::exit(1);
    }
    if !path.is_file() {
        display::error(&format!("'{}' is not a file", path.display()));
        std::process::exit(1);
    }
}

/// Check if the path exists and check if it's a directory. If not, prints an error using display
/// and exit with code 1.
pub fn check_dir_exists(path: &PathBuf){
    if !path.exists() {
        if let Err(e) = fs::create_dir(&path) {
            display::error(&format!("cannot create output directory '{}': {e}", path.display()));
            std::process::exit(1);
        }
    } else if !path.is_dir() {
        display::error(&format!("output path '{}' exists but is not a directory", path.display()));
        std::process::exit(1);
    }
}

/// Creates a clap value parser that validates a parsed integer lies within optional bounds.
///
/// # Type Parameters
/// - `T`: Any numeric type that can be parsed from a string, compared, and displayed.
///
/// # Arguments
/// - `min_value`: Optional lower bound (inclusive). `None` means no lower bound.
/// - `max_value`: Optional upper bound (inclusive). `None` means no upper bound.
///
/// # Returns
/// A closure compatible with clap's `value_parser`, which returns `Ok(T)` if the
/// value is within bounds, or an `Err(String)` with a descriptive message otherwise.
///
/// # Examples
/// ```
/// // Accept only strictly positive integers (i.e. >= 1)
/// #[arg(long, value_parser = bounded_integer::<i32>(Some(1), None))]
/// pub count: i32,
///
/// // Accept integers between 0 and 100 inclusive
/// #[arg(long, value_parser = bounded_integer::<u32>(Some(0), Some(100)))]
/// pub percentage: u32,
///
/// // No bounds at all
/// #[arg(long, value_parser = bounded_integer::<i64>(None, None))]
/// pub offset: i64,
/// ```
pub fn bounded_integer<T>(min_value: Option<T>, max_value: Option<T>) -> impl Fn(&str) -> Result<T, String> + Clone + Send + Sync + 'static
where
    T: FromStr + PartialOrd + Display + Copy + Send + Sync + 'static,
    T::Err: Display,
{
    move |s: &str| {
        let n: T = s.parse().map_err(|e| format!("Could not parse `{s}`: {e}"))?;

        if let Some(min) = min_value {
            if n < min {
                return Err(format!("Value `{n}` is below the minimum allowed value of `{min}`"));
            }
        }

        if let Some(max) = max_value {
            if n > max {
                return Err(format!("Value `{n}` is above the maximum allowed value of `{max}`"));
            }
        }

        Ok(n)
    }
}


// ── Shared edge-building helpers ──────────────────────────────────────────────

/// Build HNSW or exact edges from `data`. No index/edgestore caching.
pub fn build_edges<D, K>(
    data: Vec<D>,
    kernel: K,
    config: HNSWConfig,
    exact: bool,
    threads: usize,
) -> Vec<(usize, usize, f32)>
where
    D: Clone + Send + Sync + 'static,
    K: Distance<D> + Clone + Send + Sync + 'static,
{
    let proximity_threshold = config.proximity_threshold;
    if exact {
        display::section("Computing exact edges");
        let pb = display::linear_progress_bar(exact_edges_total(data.len()), "Computing pairwise distances");
        let edges = exact_edges(&data, &kernel, proximity_threshold, threads, Some(&pb));
        pb.finish();
        display::success(&format!("Found {} proximity edges", display::fmt_num(edges.len())));
        edges
    } else {
        display::section("Building HNSW");
        let pb = display::logfacto_progress_bar(data.len(), "Building HNSW");
        let hnsw = HNSWState::new(data, kernel, config);
        hnsw.build(Some(&pb));
        pb.finish();
        let sp = display::spinner("Extracting proximity edges…");
        let edges = hnsw.edges();
        display::finish_success(&sp, format!("Found {} proximity edges", display::fmt_num(edges.len())));
        edges
    }
}

/// Like `build_edges` but wraps with edgestore load/save.
pub fn get_edges<D, K>(
    data: Vec<D>,
    kernel: K,
    config: HNSWConfig,
    exact: bool,
    threads: usize,
    edgestore: Option<&Path>,
) -> Vec<(usize, usize, f32)>
where
    D: Clone + Send + Sync + 'static,
    K: Distance<D> + Clone + Send + Sync + 'static,
{
    let n = data.len();
    if let Some(path) = edgestore.filter(|p| p.exists()) {
        display::section("Loading edge store");
        let sp = display::spinner("Loading edge store…");
        let store = EdgeStore::load(path).unwrap_or_else(|e| {
            display::finish_error(&sp, format!("Failed to load edge store: {e}"));
        });
        if store.node_count != n {
            display::finish_error(&sp, format!(
                "Edge store node_count mismatch: got {}, expected {}",
                display::fmt_num(store.node_count), display::fmt_num(n)
            ));
        }
        let edges = store.edges();
        display::finish_success(&sp, format!("Loaded {} proximity edges", display::fmt_num(edges.len())));
        return edges;
    }
    let edges = build_edges(data, kernel, config, exact, threads);
    if let Some(path) = edgestore.filter(|p| !p.exists()) {
        display::section("Saving edge store");
        let sp = display::spinner("Saving edge store…");
        let store = EdgeStore::new(n, edges.clone());
        match store.save(path) {
            Ok(_)  => display::finish_success(&sp, "Edge store saved"),
            Err(e) => display::finish_error(&sp, format!("Failed to save edge store: {e}")),
        }
    }
    edges
}

/// Like `get_edges` but also handles HNSW index save/load (protein-specific).
pub fn protein_get_edges(
    data: Vec<String>,
    kernel: ProteinKernel,
    config: HNSWConfig,
    exact: bool,
    threads: usize,
    index: Option<&Path>,
    edgestore: Option<&Path>,
) -> Vec<(usize, usize, f32)> {
    let n = data.len();
    if let Some(path) = edgestore.filter(|p| p.exists()) {
        display::section("Loading edge store");
        let sp = display::spinner("Loading edge store…");
        let store = EdgeStore::load(path).unwrap_or_else(|e| {
            display::finish_error(&sp, format!("Failed to load edge store: {e}"));
        });
        if store.node_count != n {
            display::finish_error(&sp, format!(
                "Edge store node_count mismatch: got {}, expected {}",
                display::fmt_num(store.node_count), display::fmt_num(n)
            ));
        }
        let edges = store.edges();
        display::finish_success(&sp, format!("Loaded {} proximity edges", display::fmt_num(edges.len())));
        return edges;
    }
    let edges = if exact {
        display::section("Computing exact edges");
        let pb = display::linear_progress_bar(exact_edges_total(n), "Computing pairwise distances");
        let edges = exact_edges(&data, &kernel, config.proximity_threshold, threads, Some(&pb));
        pb.finish();
        display::success(&format!("Found {} proximity edges", display::fmt_num(edges.len())));
        edges
    } else {
        display::section("Building HNSW");
        let hnsw = match index.filter(|p| p.exists()) {
            Some(path) => {
                let sp = display::spinner("Loading HNSW index…");
                match HNSWState::load(path, data, Some(config), kernel) {
                    Ok(h)  => { display::finish_success(&sp, "Index loaded"); h }
                    Err(e) => display::finish_error(&sp, format!("Failed to load index: {e}")),
                }
            }
            _ => {
                let pb = display::logfacto_progress_bar(n, "Building HNSW");
                let h = HNSWState::new(data, kernel, config);
                h.build(Some(&pb));
                pb.finish();
                h
            }
        };
        if let Some(path) = index.filter(|p| !p.exists()) {
            let sp = display::spinner("Saving HNSW index…");
            match hnsw.save(path) {
                Ok(_)  => display::finish_success(&sp, "Index saved"),
                Err(e) => display::finish_error(&sp, format!("Failed to save index: {e}")),
            }
        }
        let sp = display::spinner("Extracting proximity edges…");
        let edges = hnsw.edges();
        display::finish_success(&sp, format!("Found {} proximity edges", display::fmt_num(edges.len())));
        edges
    };
    if let Some(path) = edgestore.filter(|p| !p.exists()) {
        display::section("Saving edge store");
        let sp = display::spinner("Saving edge store…");
        let store = EdgeStore::new(n, edges.clone());
        match store.save(path) {
            Ok(_)  => display::finish_success(&sp, "Edge store saved"),
            Err(e) => display::finish_error(&sp, format!("Failed to save edge store: {e}")),
        }
    }
    edges
}

/// Detect clusters from an edge list. Returns a cluster assignment per node.
pub fn detect_clusters(graph: &CsrGraph, leiden: &LeidenArgs) -> Vec<usize> {
    if leiden.leiden {
        let sp = display::spinner("Searching clusters (leiden)…");
        let clusters = find_communities(
            graph.clone(), leiden.resolution, leiden.beta,
            leiden.leiden_iteration, leiden.leiden_objective.clone(),
        );
        let n = clusters.iter().max().map(|&m| m + 1)
            .unwrap_or_else(|| display::finish_error(&sp, "Dataset is empty — no clusters found"));
        display::finish_success(&sp, format!("Found {} clusters", display::fmt_num(n)));
        clusters
    } else {
        let sp = display::spinner("Searching clusters (connected components)…");
        let clusters = find_connected_components(graph);
        let n = clusters.iter().max().map(|&m| m + 1)
            .unwrap_or_else(|| display::finish_error(&sp, "Dataset is empty — no clusters found"));
        display::finish_success(&sp, format!("Found {} clusters", display::fmt_num(n)));
        let (largest_idx, largest_size) = largest_cluster(&clusters);
        if largest_size * 10 > clusters.len() {
            display::warn(&format!(
                "Cluster {} contains {}/{} samples ([cyan]{:.1}%[/] of the dataset). \
                 Consider using --leiden to find finer-grained communities.",
                largest_idx, display::fmt_num(largest_size), display::fmt_num(clusters.len()),
                100.0 * largest_size as f32 / clusters.len() as f32,
            ));
        }
        clusters
    }
}

/// Write KNN results as a TSV.
pub fn write_knn_results(writer: &mut impl Write, results: &[Vec<(usize, f32)>]) {
    for (i, neighbors) in results.iter().enumerate() {
        let row: String = neighbors.iter()
            .flat_map(|&(j, dist)| [j.to_string(), format!("{dist:.6}")])
            .collect::<Vec<_>>()
            .join("\t");
        writeln!(writer, "{i}\t{row}").unwrap();
    }
}

// ── Build HNSW for KNN search (no edge extraction, returns the state) ─────────

/// Build HNSW from `ref_data` for knn search. Handles index save/load for proteins.
pub fn build_hnsw<T: Sync, D: Distance<T>>(
    data: Vec<T>,
    kernel: D,
    config: HNSWConfig,
    index: Option<&Path>,
) -> HNSWState<T, D> {
    let n = data.len();
    let hnsw = match index.filter(|p| p.exists()) {
        Some(path) => {
            let sp = display::spinner("Loading HNSW index…");
            match HNSWState::load(path, data, Some(config), kernel) {
                Ok(h)  => { display::finish_success(&sp, "Index loaded"); h }
                Err(e) => display::finish_error(&sp, format!("Failed to load index: {e}")),
            }
        }
        _ => {
            let pb = display::logfacto_progress_bar(n, "Building HNSW");
            let h = HNSWState::new(data, kernel, config);
            h.build(Some(&pb));
            pb.finish();
            h
        }
    };
    if let Some(path) = index.filter(|p| !p.exists()) {
        let sp = display::spinner("Saving HNSW index…");
        match hnsw.save(path) {
            Ok(_)  => display::finish_success(&sp, "Index saved"),
            Err(e) => display::finish_error(&sp, format!("Failed to save index: {e}")),
        }
    }
    hnsw
}

// ── fields! macro ─────────────────────────────────────────────────────────────

/// Build a `BTreeMap<String, String>` from struct fields with minimal boilerplate.
///
/// Syntax per entry (comma-separated):
///   - `field`                      → key = "field",   val = field.to_string()
///   - `field as "alias"`           → key = "alias",   val = field.to_string()
///   - `field => "{:?}"`            → key = "field",   val = format!("{:?}", field)
///   - `field as "alias" => "{:?}"` → key = "alias",   val = format!("{:?}", field)
///   - `field as "alias" => !`      → key = "alias",   val = (!field).to_string()
///
/// # Example
/// ```ignore
/// let map = fields!(self.config;
///     threshold,                              // key = "threshold"
///     n_threads,                              // key = "n_threads"
///     objective as "obj" => "{:?}",           // key = "obj", Debug-formatted
///     leiden_iteration as "iterations",       // key = "iterations"
/// );
/// ```
#[macro_export] macro_rules! fields {
    // Entry point
    ($src:expr; $($input:tt)*) => {{
        let mut _map = ::std::collections::BTreeMap::new();
        fields!(@parse _map, $src, $($input)*);
        _map
    }};
    // Base cases: nothing left (empty or trailing comma)
    (@parse $map:ident, $src:expr) => {};
    (@parse $map:ident, $src:expr,) => {};
    // field as "alias" => "fmt"  (with and without trailing comma)
    (@parse $map:ident, $src:expr, $field:ident as $alias:literal => $fmt:literal, $($rest:tt)*) => {
        $map.insert($alias.to_string(), format!($fmt, $src.$field));
        fields!(@parse $map, $src, $($rest)*);
    };
    (@parse $map:ident, $src:expr, $field:ident as $alias:literal => $fmt:literal) => {
        $map.insert($alias.to_string(), format!($fmt, $src.$field));
    };
    // field as "alias"  (with and without trailing comma)
    (@parse $map:ident, $src:expr, $field:ident as $alias:literal, $($rest:tt)*) => {
        $map.insert($alias.to_string(), $src.$field.to_string());
        fields!(@parse $map, $src, $($rest)*);
    };
    (@parse $map:ident, $src:expr, $field:ident as $alias:literal) => {
        $map.insert($alias.to_string(), $src.$field.to_string());
    };
    // field => "fmt"  (with and without trailing comma)
    (@parse $map:ident, $src:expr, $field:ident => $fmt:literal, $($rest:tt)*) => {
        $map.insert(stringify!($field).to_string(), format!($fmt, $src.$field));
        fields!(@parse $map, $src, $($rest)*);
    };
    (@parse $map:ident, $src:expr, $field:ident => $fmt:literal) => {
        $map.insert(stringify!($field).to_string(), format!($fmt, $src.$field));
    };
    // field  (with and without trailing comma)
    (@parse $map:ident, $src:expr, $field:ident, $($rest:tt)*) => {
        $map.insert(stringify!($field).to_string(), $src.$field.to_string());
        fields!(@parse $map, $src, $($rest)*);
    };
    (@parse $map:ident, $src:expr, $field:ident) => {
        $map.insert(stringify!($field).to_string(), $src.$field.to_string());
    };
    // field as "alias" => !  (with and without trailing comma)
    (@parse $map:ident, $src:expr, $field:ident as $alias:literal => !, $($rest:tt)*) => {
        let _flipped = !$src.$field;
        $map.insert($alias.to_string(), _flipped.to_string());
        fields!(@parse $map, $src, $($rest)*);
    };
    (@parse $map:ident, $src:expr, $field:ident as $alias:literal => !) => {
        let _flipped = !$src.$field;
        $map.insert($alias.to_string(), _flipped.to_string());
    };
}