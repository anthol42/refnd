use super::matrix::BundledMatrix;
use super::aligner_config::{AlignerMatrix, AlignerConfigTrait, AlignMode, VectorizationStrategy,
                            DatatypeWidth, AlignerConfig};
use crate::core::Distance;
use std::cmp::{max, min};
use parasail_rs::prelude::Aligner;

pub struct GlobalAlignerBuilder {
    aligner_cfg: AlignerConfig,
    identity_mode: GlobalIdentityMode,
}

impl AlignerConfigTrait for GlobalAlignerBuilder {
    fn aligner_cfg(&mut self) -> &mut AlignerConfig {
        &mut self.aligner_cfg
    }
    fn aligner_cfg_ref(&self) -> &AlignerConfig {
        &self.aligner_cfg
    }
    fn set_mode(&mut self, _mode: AlignMode) -> &mut Self {
        panic!("Alignment mode cannot be changed on GlobalAlignerBuilder");
    }
}

impl Default for GlobalAlignerBuilder {
    fn default() -> Self {
        GlobalAlignerBuilder{
            aligner_cfg: AlignerConfig{
                mode: AlignMode::Global,
                width: DatatypeWidth::Sat,
                matrix: AlignerMatrix::Bundled(BundledMatrix::Blosum62),
                vectorization: VectorizationStrategy::Scan,
                gap_open: 11,
                gap_extend: 1,
            },
            identity_mode: GlobalIdentityMode::MaxLength
        }
    }
}

impl GlobalAlignerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn identity_mode(&mut self, mode: GlobalIdentityMode) -> &mut Self {
        self.identity_mode = mode;
        self
    }

    pub fn build(&self) -> GlobalAligner {
        GlobalAligner {
            aligner: self.build_aligner(),
            identity_mode: self.identity_mode,
        }
    }
}
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum GlobalIdentityMode {
    AlignmentLength,
    MaxSeqLength,
    MinSeqLength,
    // max(AlignmentLength, MaxSeqLength)
    MaxLength
}

#[derive(Clone)]
pub struct GlobalAligner {
    aligner: Aligner,
    identity_mode: GlobalIdentityMode,
}

impl Distance<str> for GlobalAligner {
    fn call(&self, ref_sample: &str, query: &str) -> f32 {
        let stats = self.aligner.align(Some(query.as_bytes()), ref_sample.as_bytes()).expect("Failed to align, there must be a configuration problem");
        let matches = stats.get_matches().unwrap();
        let align_length = stats.get_length().unwrap();
        match self.identity_mode {
            GlobalIdentityMode::AlignmentLength => {
                1.0 - (matches as f32) / align_length as f32
            }
            GlobalIdentityMode::MaxSeqLength => {
                1.0 - (matches as f32) / max(ref_sample.len(), query.len()) as f32
            }
            GlobalIdentityMode::MinSeqLength => {
                1.0 - (matches as f32) / min(ref_sample.len(), query.len()) as f32
            }
            GlobalIdentityMode::MaxLength => {
                1.0 - (matches as f32) / max(max(ref_sample.len(), query.len()) as i32, align_length) as f32
            }
        }
    }
}

impl Distance<String> for GlobalAligner {
    fn call(&self, ref_sample: &String, query: &String) -> f32 {
        self.call(ref_sample.as_str(), query.as_str())
    }
}
