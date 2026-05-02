pub mod autocomplete;
pub mod display;
pub mod split;
pub mod custer;
pub mod index;
pub mod edge;
mod utils;
pub mod parameters;
pub mod kernel_parameters;
pub mod renderer;
pub mod knn;

use clap::{Parser, Subcommand};
use clap_complete::Shell;

// ── Top-level CLI ─────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(
    name    = "refnd",
    about   = "RGP dataset toolkit — bias-free train/test splitting",
    version,
    propagate_version = true,
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

// ── Subcommands ───────────────────────────────────────────────────────────────

#[derive(Subcommand)]
pub enum Command {
    /// Approximate dataset split via HNSW
    Split(split::SplitArgs),
    /// Detect clusters and save a TSV with node_idx and cluster_idx
    Cluster(custer::ClusterArgs),
    /// Compute k-nearest neighbors for every sequence in an input file
    Knn(knn::KnnArgs),
    /// Build and save an HNSW index from an input file
    Index(index::IndexArgs),
    /// Compute and save the proximity edge list from an input file or HNSW index file
    Edge(edge::EdgeArgs),
    /// Install shell tab-completion for refnd
    InstallAutocomplete {
        /// Shell to target (default: detected from $SHELL)
        #[arg(value_enum, value_name = "SHELL")]
        shell: Option<Shell>,
    },
}
