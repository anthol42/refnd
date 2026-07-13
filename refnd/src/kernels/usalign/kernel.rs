use std::ffi::CString;
use crate::core::Distance;
use super::config::NormMode;
use super::pdb_structure::PdbStructure;

/// Protein structure kernel using USalign TM-score.
///
/// Returns `1.0 - tm_score` so that identical structures → distance 0.
#[derive(Clone, Debug, Default)]
pub struct USAlignKernel {
    pub norm: NormMode,
    /// Use USalign's fast alignment mode (~3-5x faster, slight accuracy loss).
    /// Recommended for HNSW index construction where approximate distances suffice.
    pub fast: bool,
}

impl USAlignKernel {
    pub fn new(norm: NormMode, fast: bool) -> Self {
        Self { norm, fast }
    }
}

impl Distance<String> for USAlignKernel {
    fn call(&self, a: &String, b: &String) -> f32 {
        let Ok(ca) = CString::new(a.as_str()) else { return 1.0 };
        let Ok(cb) = CString::new(b.as_str()) else { return 1.0 };

        let mut tm1: f32 = 0.0;
        let mut tm2: f32 = 0.0;
        let mut rmsd: f32 = 0.0;
        let mut n_aligned: i32 = 0;

        let rc = unsafe {
            libusalign_sys::usalign_align(
                ca.as_ptr(), cb.as_ptr(),
                &mut tm1, &mut tm2, &mut rmsd, &mut n_aligned,
                self.fast as i32,
            )
        };

        if rc != 0 { return 1.0; }

        let tm = match self.norm {
            NormMode::Min    => tm1.min(tm2),
            NormMode::Query  => tm1,
            NormMode::Target => tm2,
        };
        1.0 - tm
    }
}

impl Distance<PdbStructure> for USAlignKernel {
    fn call(&self, a: &PdbStructure, b: &PdbStructure) -> f32 {
        let mut tm1: f32 = 0.0;
        let mut tm2: f32 = 0.0;
        let mut rmsd: f32 = 0.0;
        let mut n_aligned: i32 = 0;

        let rc = unsafe {
            libusalign_sys::usalign_align_structures(
                a.as_ptr(), b.as_ptr(),
                &mut tm1, &mut tm2, &mut rmsd, &mut n_aligned,
                self.fast as i32,
            )
        };

        if rc != 0 { return 1.0; }

        let tm = match self.norm {
            NormMode::Min    => tm1.min(tm2),
            NormMode::Query  => tm1,
            NormMode::Target => tm2,
        };
        1.0 - tm
    }
}
