use std::{path::PathBuf, process::exit};
use clap::{Args, Subcommand, ValueHint};
use refnd::core::{hnsw::HNSWIndex, EdgeStore};
use super::{
    display,
    kernel_parameters::{KernelDispatch, ProteinKernelArgs, MoleculeKernelArgs},
    parameters::{build_hnsw_config, hnsw_params, HnswArgs},
    utils::{check_file_exists, build_edges, get_edges},
};
use crate::fields;

// ── Command ───────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct EdgeArgs {
    /// Input: FASTA (.fasta/.fa), HNSW index (.hnsw), or molecule file (smi sdf or csv with smile col)
    #[clap(value_name = "INPUT", value_hint = ValueHint::FilePath)]
    pub input: PathBuf,

    /// Output file: .edgelist (text) or .edgestr (binary)
    #[clap(value_name = "OUTPUT", value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Distance threshold for proximity edges
    #[arg(long, short = 'p', default_value_t = 0.5, value_name = "FLOAT")]
    pub proximity_threshold: f32,

    /// Use exact O(n²) pairwise distance computation instead of HNSW
    #[clap(long)]
    pub exact: bool,

    /// Number of rayon threads (0 = all available cores)
    #[clap(long, short, value_name = "INT", default_value_t = 0)]
    pub threads: usize,

    #[command(flatten)]
    pub hnsw: HnswArgs,

    #[command(subcommand)]
    pub kernel: EdgeKernel,
}

#[derive(Subcommand)]
pub enum EdgeKernel {
    /// Protein sequences from FASTA or an HNSW index file
    Protein(EdgeProteinArgs),
    /// Small molecules from a SDF file or directory, or an HNSW index file
    Molecule(EdgeMoleculeArgs),
}

#[derive(Args)]
pub struct EdgeProteinArgs {
    #[command(flatten)]
    pub kernel: ProteinKernelArgs,
}

#[derive(Args)]
pub struct EdgeMoleculeArgs {
    #[command(flatten)]
    pub kernel: MoleculeKernelArgs,
}

// ── Run ───────────────────────────────────────────────────────────────────────

impl EdgeArgs {
    pub fn run(self) {
        check_file_exists(&self.input);

        let is_hnsw = self.input.extension().and_then(|e| e.to_str())
            .map(|e| e.eq_ignore_ascii_case("hnsw")).unwrap_or(false);

        if self.exact && is_hnsw {
            display::error("--exact requires a data input, not an HNSW index");
            exit(1);
        }

        display::section("Parameters");
        let params = fields!(self; input => "{:?}", output => "{:?}", exact, proximity_threshold, threads);
        display::parameter_panel("Edge Configuration", &params);

        // Loading from a saved index is data-type agnostic — handle it before kernel dispatch.
        if is_hnsw {
            display::section("Loading HNSW index");
            let sp = display::spinner("Reading index…");
            let index = HNSWIndex::load(&self.input).unwrap_or_else(|e| {
                display::finish_error(&sp, format!("Invalid HNSW file: {e}"));
            });
            display::finish_success(&sp, format!(
                "Loaded index ({} nodes, {} proximity edges)",
                display::fmt_num(index.dataset_size), display::fmt_num(index.proximity_edges.len()),
            ));
            let edges = index.proximity_edges.into_iter().map(|((i, j), d)| (i, j, d)).collect();
            display::section("Saving edges");
            save_edges(&self.output, index.dataset_size, edges);
            return;
        }

        let config = build_hnsw_config(&self.hnsw, self.proximity_threshold, self.threads);
        if !self.exact { display::parameter_panel("HNSW Configuration", &hnsw_params(&self.hnsw)); }

        let (n, edges) = match self.kernel {
            EdgeKernel::Protein(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                display::section("Loading FASTA");
                let sp = display::spinner("Reading FASTA…");
                let dataset = refnd::utils::read_fasta(&self.input).unwrap_or_else(|e| {
                    display::error(&e); exit(1)
                });
                let sequences: Vec<String> = dataset.into_iter().map(|(_, s)| s).collect();
                let n = sequences.len();
                display::finish_success(&sp, format!("Loaded {} sequences", display::fmt_num(n)));
                let edges = build_edges(sequences, a.kernel.build_kernel(), config, self.exact, self.threads);
                (n, edges)
            }
            EdgeKernel::Molecule(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                display::section("Loading file");
                let sp = display::spinner("Reading file…");
                let (_, data) = a.kernel.load(&self.input);
                let n = data.len();
                display::finish_success(&sp, format!("Loaded {} molecules", display::fmt_num(n)));
                let edges = get_edges(data, a.kernel.build_kernel(), config, self.exact, self.threads, None);
                (n, edges)
            }
        };

        display::section("Saving edges");
        save_edges(&self.output, n, edges);
    }
}

// ── Output ────────────────────────────────────────────────────────────────────

enum OutputFormat { EdgeList, EdgeStr }

fn infer_output_format(path: &PathBuf) -> OutputFormat {
    match path.extension().and_then(|e| e.to_str()).map(str::to_lowercase).as_deref() {
        Some("edgelist") => OutputFormat::EdgeList,
        Some("edgestr")  => OutputFormat::EdgeStr,
        _ => {
            display::error(&format!(
                "unrecognized output extension '{}' — expected .edgelist or .edgestr",
                path.display()
            ));
            exit(1)
        }
    }
}

fn save_edges(path: &PathBuf, n: usize, edges: Vec<(usize, usize, f32)>) {
    let fmt = infer_output_format(path);
    let sp = display::spinner(format!("Saving to {}…", path.display()));
    let store = EdgeStore::new(n, edges);
    let result = match fmt {
        OutputFormat::EdgeStr  => store.save_binary(path),
        OutputFormat::EdgeList => store.save_text(path),
    };
    match result {
        Ok(_)  => display::finish_success(&sp, "Edges saved"),
        Err(e) => display::finish_error(&sp, format!("Failed to save edges: {e}")),
    }
}
