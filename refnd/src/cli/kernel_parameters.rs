use std::collections::BTreeMap;
use std::path::Path;
use clap::Args;
use refnd::kernels::proteins::parasail::{
    BundledMatrix, GlobalIdentityMode, LocalIdentityMode, CoverageMode, VectorizationStrategy,
    GlobalAlignerBuilder, LocalAlignerBuilder, AlignerMatrix, AlignerConfigTrait, ProteinKernel,
};
use refnd::kernels::molecules::tanimoto::Tanimoto;
use refnd::utils::{BitFingerprint, read_molecule_file, read_fasta, FingerprintType};
use refnd::core::Distance;
use super::utils::bounded_integer;
use crate::cli::display;

#[derive(Args)]
#[command(next_help_heading = "Kernel Options")]
pub struct ProteinKernelArgs {
    /// Use global (Needleman-Wunsch) alignment; default is local (Smith-Waterman)
    #[arg(long)]
    pub global: bool,

    /// Substitution matrix
    #[arg(long, default_value = "blosum62", value_name = "MODE")]
    pub matrix: BundledMatrix,

    /// Gap open penalty (strictly positive)
    #[arg(long, default_value_t = 11, value_name = "INT", value_parser = bounded_integer(Some(0), None))]
    pub gap_open: i32,

    /// Gap extension penalty (strictly positive)
    #[arg(long, default_value_t = 1, value_name = "INT", value_parser = bounded_integer(Some(0), None))]
    pub gap_extend: i32,

    /// SIMD vectorization strategy. Striped is usually faster with local alignment whereas scan is
    /// faster with global alignments.
    #[arg(long, default_value = "striped", value_name = "MODE")]
    pub vectorization: VectorizationStrategy,

    /// [Global] Identity normalization: how the match count is divided to produce identity
    #[arg(long, default_value = "max-length", value_name = "MODE")]
    pub global_identity_mode: GlobalIdentityMode,

    /// [Local] Minimum fraction of the sequence that must be covered by the alignment
    #[arg(long, default_value_t = 0.8, value_name = "FLOAT")]
    pub min_coverage: f32,

    /// [Local] Coverage mode (mirrors mmseqs2 --cov-mode)
    #[arg(long, default_value = "both-query-target", value_name = "MODE")]
    pub cov_mode: CoverageMode,

    /// [Local] Identity normalisation: how the match count is divided to produce identity
    #[arg(long, default_value = "alignment-length", value_name = "MODE")]
    pub local_identity_mode: LocalIdentityMode,
}

#[derive(Args)]
#[command(next_help_heading = "Kernel Options")]
pub struct MoleculeKernelArgs {
    /// Fingerprint type (Morgan, Rdk, or Pattern)
    #[arg(long, default_value = "morgan", value_name = "TYPE")]
    pub fingerprint: FingerprintType,
}

pub enum KernelParams {
    Protein { global: bool, matrix: String, gap_open: i32, gap_extend: i32,
              vectorization: String, identity_mode: String,
              min_coverage: Option<f32>, cov_mode: Option<String> },
    Molecule { fingerprint: String },
}

impl KernelParams {
    pub fn to_map(&self) -> BTreeMap<String, String> {
        match self {
            KernelParams::Protein { global, matrix, gap_open, gap_extend, vectorization, identity_mode, min_coverage, cov_mode } => {
                let mut map = BTreeMap::new();
                map.insert("global".to_string(), global.to_string());
                map.insert("matrix".to_string(), matrix.clone());
                map.insert("gap_open".to_string(), gap_open.to_string());
                map.insert("gap_extend".to_string(), gap_extend.to_string());
                map.insert("vectorization".to_string(), vectorization.clone());
                if *global {
                    map.insert("identity_mode".to_string(), identity_mode.clone());
                } else {
                    if let Some(cov) = min_coverage {
                        map.insert("min_coverage".to_string(), cov.to_string());
                    }
                    if let Some(mode) = cov_mode {
                        map.insert("cov_mode".to_string(), mode.clone());
                    }
                    map.insert("identity_mode".to_string(), identity_mode.clone());
                }
                map
            }
            KernelParams::Molecule { fingerprint } => {
                let mut map = BTreeMap::new();
                map.insert("fingerprint".to_string(), fingerprint.clone());
                map
            }
        }
    }
}

pub trait KernelDispatch {
    type Data: Clone + Send + Sync + 'static;
    type Kernel: Distance<Self::Data> + Send + Sync + 'static;

    fn kernel_params(&self) -> KernelParams;
    fn build_kernel(&self) -> Self::Kernel;
    fn load(&self, input: &Path) -> (Vec<String>, Vec<Self::Data>);
}

impl KernelDispatch for ProteinKernelArgs {
    type Data = String;
    type Kernel = ProteinKernel;

    fn kernel_params(&self) -> KernelParams {
        let identity_mode = if self.global {
            format!("{:?}", self.global_identity_mode)
        } else {
            format!("{:?}", self.local_identity_mode)
        };
        KernelParams::Protein {
            global: self.global,
            matrix: format!("{:?}", self.matrix),
            gap_open: self.gap_open,
            gap_extend: self.gap_extend,
            vectorization: format!("{:?}", self.vectorization),
            identity_mode,
            min_coverage: if self.global { None } else { Some(self.min_coverage) },
            cov_mode: if self.global { None } else { Some(format!("{:?}", self.cov_mode)) },
        }
    }

    fn build_kernel(&self) -> ProteinKernel {
        let matrix = AlignerMatrix::Bundled(self.matrix.clone());
        if self.global {
            let mut builder = GlobalAlignerBuilder::new();
            builder
                .set_matrix(matrix)
                .set_gap_open(self.gap_open)
                .set_gap_extend(self.gap_extend)
                .set_vectorization(self.vectorization.clone())
                .identity_mode(self.global_identity_mode);
            ProteinKernel::Global(builder.build())
        } else {
            let mut builder = LocalAlignerBuilder::new();
            builder
                .set_matrix(matrix)
                .set_gap_open(self.gap_open)
                .set_gap_extend(self.gap_extend)
                .set_vectorization(self.vectorization.clone())
                .identity_mode(self.local_identity_mode)
                .min_coverage(self.min_coverage)
                .cov_mode(self.cov_mode);
            ProteinKernel::Local(builder.build())
        }
    }

    fn load(&self, input: &Path) -> (Vec<String>, Vec<String>) {
        let dataset = read_fasta(input).unwrap_or_else(|e| {
            display::error(&e);
            std::process::exit(1)
        });
        let headers: Vec<String> = dataset.iter().map(|(h, _)| h.clone()).collect();
        let sequences: Vec<String> = dataset.into_iter().map(|(_, s)| s).collect();
        (headers, sequences)
    }
}

impl KernelDispatch for MoleculeKernelArgs {
    type Data = BitFingerprint;
    type Kernel = Tanimoto;

    fn kernel_params(&self) -> KernelParams {
        KernelParams::Molecule {
            fingerprint: format!("{:?}", self.fingerprint),
        }
    }

    fn build_kernel(&self) -> Tanimoto {
        Tanimoto
    }

    fn load(&self, input: &Path) -> (Vec<String>, Vec<BitFingerprint>) {
        let entries = match read_molecule_file(input, &self.fingerprint) {
            Ok(e) => e,
            Err(e) => { display::error(&e); std::process::exit(1); }
        };
        if entries.is_empty() {
            display::error("No molecules loaded");
            std::process::exit(1);
        }
        entries.into_iter().unzip()
    }
}
