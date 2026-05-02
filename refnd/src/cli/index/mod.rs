use std::path::PathBuf;
use clap::{Args, Subcommand, ValueHint};
use refnd::core::hnsw::HNSWState;
use crate::fields;
use super::{
    display, utils::check_file_exists,
    parameters::{build_hnsw_config, hnsw_params, HnswArgs},
    kernel_parameters::{KernelDispatch, ProteinKernelArgs, MoleculeKernelArgs},
};

// ── Command ───────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct IndexArgs {
    /// Input file: FASTA (.fasta/.fa) or molecule file (sdf, smi, csv with smiles column)
    #[clap(value_name = "INPUT", value_hint = ValueHint::FilePath)]
    pub input: PathBuf,

    /// Output HNSW index file (.hnsw)
    #[clap(value_name = "OUTPUT", value_hint = ValueHint::FilePath)]
    pub out_index: PathBuf,

    /// Distance threshold for proximity edges
    #[arg(long, short = 'p', default_value_t = 0.5, value_name = "FLOAT")]
    pub proximity_threshold: f32,

    /// Number of rayon threads (0 = all available cores)
    #[clap(long, short, value_name = "INT", default_value_t = 0)]
    pub threads: usize,

    #[command(flatten)]
    pub hnsw: HnswArgs,

    #[command(subcommand)]
    pub kernel: IndexKernel,
}

#[derive(Subcommand)]
pub enum IndexKernel {
    /// Protein sequences from a FASTA file
    Protein(IndexProteinArgs),
    /// Small molecules from a molecule file
    Molecule(IndexMoleculeArgs),
}

#[derive(Args)]
pub struct IndexProteinArgs {
    #[command(flatten)]
    pub kernel: ProteinKernelArgs,
}

#[derive(Args)]
pub struct IndexMoleculeArgs {
    #[command(flatten)]
    pub kernel: MoleculeKernelArgs,
}

// ── Run ───────────────────────────────────────────────────────────────────────

impl IndexArgs {
    pub fn run(self) {
        check_file_exists(&self.input);

        display::section("Parameters");
        let params = fields!(self; input => "{:?}", out_index => "{:?}", proximity_threshold, threads);
        display::parameter_panel("Index Configuration", &params);
        display::parameter_panel("HNSW Configuration", &hnsw_params(&self.hnsw));

        let config = build_hnsw_config(&self.hnsw, self.proximity_threshold, self.threads);

        let save_result = match self.kernel {
            IndexKernel::Protein(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                display::section("Loading data");
                let sp = display::spinner("Reading FASTA…");
                let dataset = refnd::utils::read_fasta(&self.input).unwrap_or_else(|e| {
                    display::error(&e); std::process::exit(1)
                });
                let sequences: Vec<String> = dataset.into_iter().map(|(_, s)| s).collect();
                display::finish_success(&sp, format!("Loaded {} sequences", display::fmt_num(sequences.len())));
                display::section("Building HNSW index");
                build_and_save(sequences, a.kernel.build_kernel(), config, &self.out_index)
            }
            IndexKernel::Molecule(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                display::section("Loading data");
                let sp = display::spinner("Reading file…");
                let (_, data) = a.kernel.load(&self.input);
                display::finish_success(&sp, format!("Loaded {} molecules", display::fmt_num(data.len())));
                display::section("Building HNSW index");
                build_and_save(data, a.kernel.build_kernel(), config, &self.out_index)
            }
        };

        display::section("Saving");
        match save_result {
            Ok(_)  => display::success(&format!("Index saved to {}", self.out_index.display())),
            Err(e) => display::error(&format!("Failed to save index: {e}")),
        }
        display::section("Done");
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_and_save<D, K>(
    data: Vec<D>,
    kernel: K,
    config: refnd::core::hnsw::HNSWConfig,
    out: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>>
where
    D: Clone + Send + Sync + 'static,
    K: refnd::core::Distance<D> + Clone + Send + Sync + 'static,
{
    let pb = display::logfacto_progress_bar(data.len(), "Building HNSW");
    let hnsw = HNSWState::new(data, kernel, config);
    hnsw.build(Some(&pb));
    pb.finish();
    hnsw.save(out)
}
