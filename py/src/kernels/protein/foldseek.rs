use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods, gen_stub_pyfunction};
use refnd_core::kernels::proteins::foldseek::{
    FoldseekKernel as CoreKernel, StructureData as CoreData,
};
use refnd_core::utils::read_structure_dir;
use crate::kernels::protein::sequence::{CoverageMode, LocalIdentityMode};
use std::path::Path;

// ── StructureData ─────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(get_all, from_py_object, module = "refnd.kernels.protein.foldseek")]
#[derive(Clone)]
pub struct StructureData {
    pub aa_seq: String,
    pub tdi_seq: String,
}

// ── FoldseekKernel ────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.protein.foldseek")]
#[derive(Clone)]
pub struct FoldseekKernel {
    pub inner: CoreKernel,
}

#[gen_stub_pymethods]
#[pymethods]
impl FoldseekKernel {
    #[new]
    #[pyo3(signature = (
        min_coverage = 0.5,
        cov_mode = CoverageMode::BothQueryTarget,
        identity_mode = LocalIdentityMode::AlignmentLength
    ))]
    fn new(min_coverage: f32, cov_mode: CoverageMode, identity_mode: LocalIdentityMode) -> Self {
        Self { inner: CoreKernel::new(min_coverage, cov_mode.into(), identity_mode.into()) }
    }

    /// Compute the 3Di-alignment distance between two structures.
    /// Returns a value in ``[0, 1]`` where 0 means identical and 1 means fully dissimilar.
    fn call(&self, a: &StructureData, b: &StructureData) -> f32 {
        use refnd_core::core::Distance;
        let core_a = CoreData { aa_seq: a.aa_seq.clone(), tdi_seq: a.tdi_seq.clone() };
        let core_b = CoreData { aa_seq: b.aa_seq.clone(), tdi_seq: b.tdi_seq.clone() };
        self.inner.call(&core_a, &core_b)
    }
}

// ── load_structures ───────────────────────────────────────────────────────────

/// Load all PDB/mmCIF files from *directory*, returning ``[(label, StructureData)]``.
///
/// Labels are the file stems, sorted alphabetically.
/// Files that fail to parse are silently skipped with a warning on stderr.
#[gen_stub_pyfunction(module = "refnd.kernels.protein.foldseek")]
#[pyfunction]
pub fn load_structures(directory: &str) -> Vec<(String, StructureData)> {
    read_structure_dir(Path::new(directory))
        .into_iter()
        .map(|(label, data)| {
            (label, StructureData { aa_seq: data.aa_seq, tdi_seq: data.tdi_seq })
        })
        .collect()
}
