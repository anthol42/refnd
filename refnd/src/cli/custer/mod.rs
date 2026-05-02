use std::{fs::File, io::{BufWriter, Write}, path::PathBuf};
use clap::{Args, Subcommand, ValueHint};
use refnd::core::leiden::CsrGraph;
use super::{
    display,
    kernel_parameters::{KernelDispatch, ProteinKernelArgs, MoleculeKernelArgs},
    parameters::{build_hnsw_config, hnsw_params, leiden_params, HnswArgs, LeidenArgs},
    utils::{check_file_exists, get_edges, protein_get_edges, detect_clusters},
};
use crate::fields;

// ── Command ───────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct ClusterArgs {
    /// Input file (FASTA for protein, molecule file (sdf, smi, csv with smiles col))
    #[clap(value_name = "INPUT", value_hint = ValueHint::FilePath)]
    pub input: PathBuf,

    /// Output TSV file (node_idx, cluster_idx)
    #[clap(value_name = "OUTPUT", value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Distance threshold for proximity edges
    #[arg(long, short = 'p', default_value_t = 0.5, value_name = "FLOAT")]
    pub proximity_threshold: f32,

    /// Use exact O(n²) pairwise distance computation instead of HNSW
    #[clap(long)]
    pub exact: bool,

    /// Path to save/load edge store (.edgelist or .edgestr)
    #[clap(long, value_name = "PATH")]
    pub edgestore: Option<PathBuf>,

    /// Path to save/load the HNSW index (protein only)
    #[clap(long, value_name = "PATH")]
    pub index: Option<PathBuf>,

    /// Number of rayon threads (0 = all available cores)
    #[clap(long, short, value_name = "INT", default_value_t = 0)]
    pub threads: usize,

    #[command(flatten)]
    pub hnsw: HnswArgs,

    #[command(flatten)]
    pub leiden: LeidenArgs,

    #[command(subcommand)]
    pub kernel: ClusterKernel,
}

#[derive(Subcommand)]
pub enum ClusterKernel {
    /// Protein sequences from a FASTA file
    Protein(ClusterProteinArgs),
    /// Small molecules from a SDF file or directory
    Molecule(ClusterMoleculeArgs),
}

#[derive(Args)]
pub struct ClusterProteinArgs {
    #[command(flatten)]
    pub kernel: ProteinKernelArgs,
}

#[derive(Args)]
pub struct ClusterMoleculeArgs {
    #[command(flatten)]
    pub kernel: MoleculeKernelArgs,
}

// ── Run ───────────────────────────────────────────────────────────────────────

impl ClusterArgs {
    pub fn run(self) {
        check_file_exists(&self.input);

        display::section("Parameters");
        let params = fields!(self; input => "{:?}", output => "{:?}", exact, proximity_threshold,
            edgestore => "{:?}", index => "{:?}", threads);
        display::parameter_panel("Cluster Configuration", &params);
        if !self.exact { display::parameter_panel("HNSW Configuration", &hnsw_params(&self.hnsw)); }
        if self.leiden.leiden { display::parameter_panel("Leiden Configuration", &leiden_params(&self.leiden)); }

        let config = build_hnsw_config(&self.hnsw, self.proximity_threshold, self.threads);

        display::section("Loading data");
        let (n, edges) = match self.kernel {
            ClusterKernel::Protein(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                let sp = display::spinner("Reading FASTA…");
                let dataset = refnd::utils::read_fasta(&self.input).unwrap_or_else(|e| {
                    display::error(&e); std::process::exit(1)
                });
                let sequences: Vec<String> = dataset.into_iter().map(|(_, s)| s).collect();
                display::finish_success(&sp, format!("Loaded {} sequences", display::fmt_num(sequences.len())));
                let n = sequences.len();
                let edges = protein_get_edges(sequences, a.kernel.build_kernel(), config, self.exact,
                    self.threads, self.index.as_deref(), self.edgestore.as_deref());
                (n, edges)
            }
            ClusterKernel::Molecule(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                let sp = display::spinner("Reading file…");
                let (_, data) = a.kernel.load(&self.input);
                display::finish_success(&sp, format!("Loaded {} molecules", display::fmt_num(data.len())));
                let n = data.len();
                let edges = get_edges(data, a.kernel.build_kernel(), config, self.exact,
                    self.threads, self.edgestore.as_deref());
                (n, edges)
            }
        };

        let graph = CsrGraph::new(n, &edges, !self.leiden.weighted, true);
        display::section("Detecting clusters");
        let clusters = detect_clusters(&graph, &self.leiden);
        display::section("Saving files");
        write_clusters(&self.output, &clusters);
        display::section("Done");
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

fn write_clusters(path: &PathBuf, clusters: &[usize]) {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let sp = display::spinner(format!("Saving {name}…"));
    let mut f = match File::create(path) {
        Ok(f) => BufWriter::new(f),
        Err(e) => display::finish_error(&sp, format!("Cannot create {name}: {e}")),
    };
    if let Err(e) = writeln!(f, "node_idx\tcluster_idx") {
        display::finish_error(&sp, format!("Write error ({name}): {e}"));
    }
    for (node_idx, &cluster_idx) in clusters.iter().enumerate() {
        if let Err(e) = writeln!(f, "{node_idx}\t{cluster_idx}") {
            display::finish_error(&sp, format!("Write error ({name}): {e}"));
        }
    }
    display::finish_success(&sp, format!("{name}"));
}
