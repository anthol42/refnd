use std::{fs::File, io::{BufWriter, Write}, path::PathBuf, process::exit};
use clap::{Args, Subcommand, ValueHint};
use refnd::core::{leiden::CsrGraph, partition_dataset};
use super::{
    display,
    kernel_parameters::{KernelDispatch, ProteinKernelArgs, MoleculeKernelArgs},
    parameters::{build_hnsw_config, hnsw_params, leiden_params, HnswArgs, LeidenArgs},
    utils::{check_dir_exists, check_file_exists, get_edges, protein_get_edges, detect_clusters},
};
use crate::fields;

// ── Command ───────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct SplitArgs {
    /// Input file (FASTA for protein or molecule file (smi, sdf, csv))
    #[clap(value_name = "INPUT", value_hint = ValueHint::FilePath)]
    pub input: PathBuf,

    /// Output directory for train/test files
    #[clap(value_name = "OUTPUT", value_hint = ValueHint::DirPath)]
    pub output_path: PathBuf,

    /// Test size ratio; [0.0 - 1.0]
    #[clap(long, default_value_t = 0.2, value_name = "FLOAT")]
    pub test_ratio: f32,

    /// Remove train samples within proximity threshold of any test sample
    #[clap(long)]
    pub post_filtering: bool,

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
    pub kernel: SplitKernel,
}

#[derive(Subcommand)]
pub enum SplitKernel {
    /// Protein sequences from a FASTA file
    Protein(SplitProteinArgs),
    /// Small molecules from a SDF file or directory
    Molecule(SplitMoleculeArgs),
}

#[derive(Args)]
pub struct SplitProteinArgs {
    #[command(flatten)]
    pub kernel: ProteinKernelArgs,
}

#[derive(Args)]
pub struct SplitMoleculeArgs {
    #[command(flatten)]
    pub kernel: MoleculeKernelArgs,
}

// ── Run ───────────────────────────────────────────────────────────────────────

impl SplitArgs {
    pub fn run(self) {
        check_file_exists(&self.input);
        check_dir_exists(&self.output_path);

        display::section("Parameters");
        let params = fields!(self; input => "{:?}", output_path => "{:?}", test_ratio, post_filtering,
            exact, proximity_threshold, edgestore => "{:?}", index => "{:?}", threads);
        display::parameter_panel("Split Configuration", &params);
        if !self.exact { display::parameter_panel("HNSW Configuration", &hnsw_params(&self.hnsw)); }
        if self.leiden.leiden { display::parameter_panel("Leiden Configuration", &leiden_params(&self.leiden)); }

        let config = build_hnsw_config(&self.hnsw, self.proximity_threshold, self.threads);

        display::section("Loading data");
        let (labels, train_ids, test_ids) = match self.kernel {
            SplitKernel::Protein(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                let sp = display::spinner("Reading FASTA…");
                let dataset = refnd::utils::read_fasta(&self.input).unwrap_or_else(|e| {
                    display::error(&e); exit(1)
                });
                let sequences: Vec<String> = dataset.iter().map(|(_, s)| s.clone()).collect();
                display::finish_success(&sp, format!("Loaded {} sequences", display::fmt_num(sequences.len())));

                let edges = protein_get_edges(sequences, a.kernel.build_kernel(), config, self.exact,
                    self.threads, self.index.as_deref(), self.edgestore.as_deref());
                let (train_ids, test_ids) = run_partition(dataset.len(), edges, &self.leiden,
                    self.test_ratio, self.post_filtering);

                let labels: Vec<String> = dataset.iter().map(|(h, _)| h.clone()).collect();
                (labels, train_ids, test_ids)
            }
            SplitKernel::Molecule(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                let sp = display::spinner("Reading file…");
                let (labels, data) = a.kernel.load(&self.input);
                display::finish_success(&sp, format!("Loaded {} molecules", display::fmt_num(data.len())));

                let edges = get_edges(data, a.kernel.build_kernel(), config, self.exact,
                    self.threads, self.edgestore.as_deref());
                let (train_ids, test_ids) = run_partition(labels.len(), edges, &self.leiden,
                    self.test_ratio, self.post_filtering);
                (labels, train_ids, test_ids)
            }
        };
        display::section("Saving files");
        write_lines(&self.output_path.join("train.txt"), &train_ids, &labels);
        write_lines(&self.output_path.join("test.txt"),  &test_ids,  &labels);
        display::section("Done");
    }
}

// ── Shared logic ──────────────────────────────────────────────────────────────

fn run_partition(
    n: usize,
    edges: Vec<(usize, usize, f32)>,
    leiden: &LeidenArgs,
    test_ratio: f32,
    post_filtering: bool,
) -> (Vec<usize>, Vec<usize>) {
    let graph = CsrGraph::new(n, &edges, !leiden.weighted, true);
    display::section("Detecting clusters");
    let clusters = detect_clusters(&graph, leiden);
    display::section("Partitioning");
    let sp = display::spinner("Partitioning…");
    let (train_ids, test_ids) = partition_dataset(clusters, &graph, test_ratio, None, post_filtering)
        .unwrap_or_else(|e| { display::error(&e); exit(1) });
    display::finish_success(&sp, format!(
        "Partitioning done — train: {}, test: {} samples",
        display::fmt_num(train_ids.len()), display::fmt_num(test_ids.len())
    ));
    (train_ids, test_ids)
}

fn write_lines(path: &PathBuf, ids: &[usize], labels: &[String]) {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let sp = display::spinner(format!("Saving {name}…"));
    let mut f = match File::create(path) {
        Ok(f) => BufWriter::new(f),
        Err(e) => display::finish_error(&sp, format!("Cannot create {name}: {e}")),
    };
    for &v in ids {
        if let Err(e) = writeln!(f, "{}", labels[v]) {
            display::finish_error(&sp, format!("Write error ({name}): {e}"));
        }
    }
    display::finish_success(&sp, format!("{name}"));
}
