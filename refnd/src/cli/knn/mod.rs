use std::{fs::File, io::BufWriter, path::PathBuf, process::exit};
use clap::{Args, Subcommand, ValueHint};
use indicatif::ProgressBar;
use refnd::core::{exact::{exact_nearest_neighbors, exact_nearest_neighbors_total}, Distance};
use refnd::core::hnsw::HNSWConfig;
use super::{
    display,
    display::fmt_num,
    kernel_parameters::{KernelDispatch, ProteinKernelArgs, MoleculeKernelArgs},
    parameters::{build_hnsw_config, hnsw_params, HnswArgs},
    utils::{check_file_exists, write_knn_results, build_hnsw},
};
use crate::fields;

// ── Command ───────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct KnnArgs {
    /// Reference input file (FASTA for protein, SDF file/directory for molecule)
    #[clap(value_name = "REF", value_hint = ValueHint::FilePath)]
    pub ref_input: PathBuf,

    /// Query input file (FASTA for protein, SDF file/directory for molecule)
    #[clap(value_name = "QUERY", value_hint = ValueHint::FilePath)]
    pub query_input: PathBuf,

    /// Output TSV file
    #[clap(value_name = "OUTPUT", value_hint = ValueHint::FilePath)]
    pub output: PathBuf,

    /// Number of nearest neighbors per query
    #[clap(short = 'k', long, default_value_t = 1, value_name = "INT")]
    pub k: usize,

    /// Expansion factor for HNSW search (higher = more accurate, slower)
    #[clap(long, default_value_t = 64, value_name = "INT")]
    pub ef: usize,

    /// Use exact O(N×M) pairwise computation instead of HNSW
    #[clap(long)]
    pub exact: bool,

    /// Distance threshold for proximity edges
    #[arg(long, short = 'p', default_value_t = 0.5, value_name = "FLOAT")]
    pub proximity_threshold: f32,

    /// Path to save/load the HNSW index (protein only)
    #[clap(long, value_name = "PATH")]
    pub index: Option<PathBuf>,

    /// Number of rayon threads (0 = all available cores)
    #[clap(long, short, value_name = "INT", default_value_t = 0)]
    pub threads: usize,

    #[command(flatten)]
    pub hnsw: HnswArgs,

    #[command(subcommand)]
    pub kernel: KnnKernel,
}

#[derive(Subcommand)]
pub enum KnnKernel {
    /// Protein sequences from FASTA files
    Protein(KnnProteinArgs),
    /// Small molecules from SDF files or directories
    Molecule(KnnMoleculeArgs),
}

#[derive(Args)]
pub struct KnnProteinArgs {
    #[command(flatten)]
    pub kernel: ProteinKernelArgs,
}

#[derive(Args)]
pub struct KnnMoleculeArgs {
    #[command(flatten)]
    pub kernel: MoleculeKernelArgs,
}

// ── Run ───────────────────────────────────────────────────────────────────────

impl KnnArgs {
    pub fn run(&self) {
        check_file_exists(&self.ref_input);
        check_file_exists(&self.query_input);

        display::section("Parameters");
        let params = fields!(self; ref_input => "{:?}", query_input => "{:?}", output => "{:?}",
            k, ef, exact, proximity_threshold, index => "{:?}", threads);
        display::parameter_panel("KNN Configuration", &params);
        if !self.exact { display::parameter_panel("HNSW Configuration", &hnsw_params(&self.hnsw)); }

        display::section("Loading data");
        let mut writer = open_knn_writer(&self.output, self.k);

        let config = build_hnsw_config(&self.hnsw, self.proximity_threshold, self.threads);

        let results = match &self.kernel {
            KnnKernel::Protein(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                let sp = display::spinner("Loading data...");
                let ref_ds = refnd::utils::read_fasta(&self.ref_input).unwrap_or_else(|e| {
                    display::finish_error(&sp, e);
                });
                let query_ds = refnd::utils::read_fasta(&self.query_input).unwrap_or_else(|e| {
                    display::finish_error(&sp, e);
                });
                let ref_seqs: Vec<String>   = ref_ds.into_iter().map(|(_, s)| s).collect();
                let query_seqs: Vec<String> = query_ds.into_iter().map(|(_, s)| s).collect();
                find_knn(&self, &config, ref_seqs, query_seqs, a.kernel.build_kernel(), &sp)
            }
            KnnKernel::Molecule(a) => {
                display::parameter_panel("Kernel Configuration", &a.kernel.kernel_params().to_map());
                let sp = display::spinner("Reading reference file…");
                let (_, ref_data) = a.kernel.load(&self.ref_input);
                display::finish_success(&sp, format!("Loaded {} reference molecules", fmt_num(ref_data.len())));

                let sp = display::spinner("Reading query file…");
                let (_, query_data) = a.kernel.load(&self.query_input);
                find_knn(&self, &config, ref_data, query_data, a.kernel.build_kernel(), &sp)
            }
        };

        display::section("Saving results");
        let sp = display::spinner("Writing results…");
        write_knn_results(&mut writer, &results);
        display::finish_success(&sp, format!("Saved to {}", self.output.display()));
        display::section("Done");
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
fn find_knn<T: Sync, D: Distance<T>>(args: &KnnArgs, hnsw_config: &HNSWConfig, queries: Vec<T>, refs: Vec<T>, kernel: D, sp: &ProgressBar) -> Vec<Vec<(usize, f32)>>{
    display::finish_success(sp, format!("Data loaded successfully with {} ref samples and {} query samples",
    fmt_num(refs.len()), fmt_num(queries.len())));

    if args.exact {
        display::section("Computing KNN (exact)");
        let total = exact_nearest_neighbors_total(queries.len(), refs.len());
        let pb = display::linear_progress_bar(total, "Computing distances");
        let r = exact_nearest_neighbors(&queries, &refs, &kernel, args.k, args.threads, Some(&pb));
        pb.finish();
        r
    } else {
        display::section("Building HNSW");
        let hnsw = build_hnsw(refs, kernel, hnsw_config.clone(), args.index.as_deref());
        display::section("Searching");
        let pb = display::linear_progress_bar(queries.len() as u64, "Searching");
        let r = hnsw.parallel_search(&queries, args.k, args.ef, args.threads, Some(&pb))
            .unwrap_or_else(|e| {
                display::error(&format!("Index is not built: {e}"));
                exit(1);
            });
        pb.finish();
        r
    }
}

fn open_knn_writer(output: &PathBuf, k: usize) -> BufWriter<File> {
    use std::io::Write;
    let mut w = BufWriter::new(File::create(output).unwrap_or_else(|e| {
        display::error(&format!("Cannot create '{}': {e}", output.display()));
        exit(1);
    }));
    let header: String = (1..=k)
        .flat_map(|i| [format!("neighbor[{i}]"), format!("distance[{i}]")])
        .collect::<Vec<_>>()
        .join("\t");
    writeln!(w, "query[idx]\t{header}").unwrap();
    w
}
