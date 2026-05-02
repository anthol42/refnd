use super::matrix::BundledMatrix;
use super::aligner_config::{AlignerMatrix, AlignerConfigTrait, AlignMode, VectorizationStrategy,
                            DatatypeWidth, AlignerConfig};
use crate::core::Distance;
use parasail_rs::prelude::Aligner;

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum LocalIdentityMode {
    AlignmentLength,
    MinSeqLength,
}

/// Mirrors mmseqs2 --cov-mode semantics.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum CoverageMode {
    /// 0: alignment_length / query_len >= min_cov AND alignment_length / target_len >= min_cov
    BothQueryTarget,
    /// 1: alignment_length / target_len >= min_cov
    Target,
    /// 2: alignment_length / query_len >= min_cov
    Query,
    /// 3: min(query_len, target_len) / max(query_len, target_len) >= min_cov
    LengthRatio,
    /// 4: alignment_length / min(query_len, target_len) >= min_cov
    ShorterSeq,
}

pub struct LocalAlignerBuilder {
    aligner_cfg: AlignerConfig,
    identity_mode: LocalIdentityMode,
    min_coverage: f32,
    cov_mode: CoverageMode,
}

impl AlignerConfigTrait for LocalAlignerBuilder {
    fn aligner_cfg(&mut self) -> &mut AlignerConfig {
        &mut self.aligner_cfg
    }
    fn aligner_cfg_ref(&self) -> &AlignerConfig {
        &self.aligner_cfg
    }
    fn set_mode(&mut self, _mode: AlignMode) -> &mut Self {
        panic!("Alignment mode cannot be changed on LocalAlignerBuilder");
    }
}

impl Default for LocalAlignerBuilder {
    fn default() -> Self {
        LocalAlignerBuilder {
            aligner_cfg: AlignerConfig {
                mode: AlignMode::Local,
                width: DatatypeWidth::Sat,
                matrix: AlignerMatrix::Bundled(BundledMatrix::Blosum62),
                vectorization: VectorizationStrategy::Striped,
                gap_open: 11,
                gap_extend: 1,
            },
            identity_mode: LocalIdentityMode::AlignmentLength,
            min_coverage: 0.8,
            cov_mode: CoverageMode::BothQueryTarget,
        }
    }
}

impl LocalAlignerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn identity_mode(&mut self, mode: LocalIdentityMode) -> &mut Self {
        self.identity_mode = mode;
        self
    }

    pub fn min_coverage(&mut self, min_coverage: f32) -> &mut Self {
        assert!((0.0..=1.0).contains(&min_coverage), "min_coverage must be between 0 and 1");
        self.min_coverage = min_coverage;
        self
    }

    pub fn cov_mode(&mut self, mode: CoverageMode) -> &mut Self {
        self.cov_mode = mode;
        self
    }

    pub fn build(&self) -> LocalAligner {
        LocalAligner {
            aligner: self.build_aligner(),
            identity_mode: self.identity_mode,
            min_coverage: self.min_coverage,
            cov_mode: self.cov_mode,
        }
    }
}
#[derive(Clone)]
pub struct LocalAligner {
    aligner: Aligner,
    identity_mode: LocalIdentityMode,
    min_coverage: f32,
    cov_mode: CoverageMode,
}

impl LocalAligner {
    fn satisfies_coverage(&self, align_length: i32, query_len: usize, ref_len: usize) -> bool {
        let al = align_length as f32;
        let ql = query_len as f32;
        let rl = ref_len as f32;
        let cov = self.min_coverage;

        match self.cov_mode {
            CoverageMode::BothQueryTarget => {
                al / ql >= cov && al / rl >= cov
            }
            CoverageMode::Target => {
                al / rl >= cov
            }
            CoverageMode::Query => {
                al / ql >= cov
            }
            CoverageMode::LengthRatio => {
                let shorter = ql.min(rl);
                let longer = ql.max(rl);
                shorter / longer >= cov
            }
            CoverageMode::ShorterSeq => {
                let shorter = ql.min(rl);
                al / shorter >= cov
            }
        }
    }
}

impl Distance<str> for LocalAligner {
    fn call(&self, ref_sample: &str, query: &str) -> f32 {
        let stats = self.aligner
            .align(Some(query.as_bytes()), ref_sample.as_bytes())
            .expect("Failed to align, there must be a configuration problem");

        let matches = stats.get_matches().unwrap();
        let align_length = stats.get_length().unwrap();

        if !self.satisfies_coverage(align_length, query.len(), ref_sample.len()) {
            return 0.0;
        }

        match self.identity_mode {
            LocalIdentityMode::AlignmentLength => {
                1.0 - (matches as f32) / align_length as f32
            }
            LocalIdentityMode::MinSeqLength => {
                let shorter = query.len().min(ref_sample.len());
                1.0 - (matches as f32) / shorter as f32
            }
        }
    }
}

impl Distance<String> for LocalAligner {
    fn call(&self, ref_sample: &String, query: &String) -> f32 {
        self.call(ref_sample.as_str(), query.as_str())
    }
}
