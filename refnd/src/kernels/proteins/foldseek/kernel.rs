use crate::core::Distance;
use crate::kernels::proteins::parasail::{
    AlignerConfigTrait, AlignerMatrix, LocalAlignerBuilder, LocalIdentityMode, CoverageMode,
};
use super::encoder::StructureData;
use super::matrix::mat3di;
use parasail_rs::prelude::Aligner;

/// Compares two protein structures by aligning their 3Di sequences with the
/// 3Di20a substitution matrix (local Smith-Waterman via parasail).
#[derive(Clone)]
pub struct FoldseekKernel {
    aligner: Aligner,
    min_coverage: f32,
    cov_mode: CoverageMode,
    identity_mode: LocalIdentityMode,
}

impl FoldseekKernel {
    pub fn new(min_coverage: f32, cov_mode: CoverageMode, identity_mode: LocalIdentityMode) -> Self {
        let mut builder = LocalAlignerBuilder::new();
        builder
            .set_matrix(AlignerMatrix::Custom(mat3di()))
            .set_gap_open(11)
            .set_gap_extend(1);
        Self {
            aligner: builder.build_aligner(),
            min_coverage,
            cov_mode,
            identity_mode,
        }
    }

    fn satisfies_coverage(&self, align_len: i32, q_len: usize, r_len: usize) -> bool {
        let al = align_len as f32;
        let ql = q_len as f32;
        let rl = r_len as f32;
        let cov = self.min_coverage;
        match self.cov_mode {
            CoverageMode::BothQueryTarget => al / ql >= cov && al / rl >= cov,
            CoverageMode::Target          => al / rl >= cov,
            CoverageMode::Query           => al / ql >= cov,
            CoverageMode::LengthRatio     => ql.min(rl) / ql.max(rl) >= cov,
            CoverageMode::ShorterSeq      => al / ql.min(rl) >= cov,
        }
    }
}

impl Distance<StructureData> for FoldseekKernel {
    fn call(&self, a: &StructureData, b: &StructureData) -> f32 {
        let stats = self.aligner
            .align(Some(b.tdi_seq.as_bytes()), a.tdi_seq.as_bytes())
            .expect("3Di alignment failed");
        let matches    = stats.get_matches().unwrap();
        let align_len  = stats.get_length().unwrap();

        if !self.satisfies_coverage(align_len, b.tdi_seq.len(), a.tdi_seq.len()) {
            return 1.0; // no valid alignment → fully dissimilar
        }

        match self.identity_mode {
            LocalIdentityMode::AlignmentLength => {
                1.0 - matches as f32 / align_len as f32
            }
            LocalIdentityMode::MinSeqLength => {
                let shorter = a.tdi_seq.len().min(b.tdi_seq.len()) as f32;
                1.0 - matches as f32 / shorter
            }
        }
    }
}
