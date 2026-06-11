use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use refnd_core::core::Distance;
use refnd_core::kernels::alignments::usalign::{
    USAlignKernel as CoreUSAlignKernel,
    NormMode as CoreNormMode,
    PdbStructure as CorePdbStructure,
};

// ── Distance<PdbStructure> bridge ─────────────────────────────────────────────

impl Distance<PdbStructure> for CoreUSAlignKernel {
    #[inline(always)]
    fn call(&self, a: &PdbStructure, b: &PdbStructure) -> f32 {
        Distance::<CorePdbStructure>::call(self, &a.inner, &b.inner)
    }
}

// ── NormMode ──────────────────────────────────────────────────────────────────

/// TM-score normalization strategy.
///
/// USalign returns two scores per alignment:
///
/// - ``TM1``: normalized by the length of the first structure (query).
/// - ``TM2``: normalized by the length of the second structure (target).
///
/// - ``Min`` (default): ``min(TM1, TM2)`` — conservative, symmetric.
/// - ``Query``: use ``TM1`` only.
/// - ``Target``: use ``TM2`` only.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.structures")]
#[derive(Clone, Copy, PartialEq)]
pub enum NormMode {
    Min,
    Query,
    Target,
}

impl From<NormMode> for CoreNormMode {
    fn from(m: NormMode) -> Self {
        match m {
            NormMode::Min    => CoreNormMode::Min,
            NormMode::Query  => CoreNormMode::Query,
            NormMode::Target => CoreNormMode::Target,
        }
    }
}

// ── PdbStructure ──────────────────────────────────────────────────────────────

/// A protein structure loaded from a PDB file and held in memory.
///
/// The structure is parsed once on construction and the coordinate arrays are
/// kept alive for the lifetime of the object. Cloning is cheap (reference-counted).
///
/// Example::
///
///     from refnd.kernels.structures import PdbStructure
///     s = PdbStructure("1abc.pdb")
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.structures", from_py_object)]
#[derive(Clone)]
pub struct PdbStructure {
    pub inner: CorePdbStructure,
}

#[gen_stub_pymethods]
#[pymethods]
impl PdbStructure {
    /// Load a protein structure from a PDB file.
    ///
    /// Args:
    ///     path: Path to the PDB file (standard format, not gzipped).
    ///
    /// Raises:
    ///     RuntimeError: If the file cannot be opened or contains fewer than 3 residues.
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        CorePdbStructure::load(path)
            .map(|inner| Self { inner })
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}

// ── USAlignKernel ─────────────────────────────────────────────────────────────

/// Protein structure comparison kernel using USalign TM-score.
///
/// Operates on pre-loaded ``PdbStructure`` objects.
/// Returns ``1.0 - TM-score`` so that identical structures have distance 0
/// and unrelated structures have distance approaching 1.
///
/// Example::
///
///     from refnd.kernels.structures import PdbStructure, USAlignKernel, NormMode
///
///     s1 = PdbStructure("1abc.pdb")
///     s2 = PdbStructure("1xyz.pdb")
///     k  = USAlignKernel()                  # default: NormMode.Min
///     d  = k(s1, s2)                         # float in [0, 1]
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.structures")]
pub struct USAlignKernel {
    pub inner: CoreUSAlignKernel,
}

#[gen_stub_pymethods]
#[pymethods]
impl USAlignKernel {
    /// Create a USAlignKernel.
    ///
    /// Args:
    ///     norm_mode: Which TM-score normalization to use. Defaults to ``NormMode.Min``.
    ///     fast: Use USalign's fast alignment mode (~3-5x faster, slight accuracy loss).
    ///         Recommended when building HNSW indexes where approximate distances suffice.
    ///         Defaults to ``False``.
    #[new]
    #[pyo3(signature = (norm_mode = NormMode::Min, fast = false))]
    pub fn new(norm_mode: NormMode, fast: bool) -> Self {
        Self { inner: CoreUSAlignKernel::new(norm_mode.into(), fast) }
    }

    /// Compute the structure distance between two ``PdbStructure`` objects.
    ///
    /// Args:
    ///     a: First structure.
    ///     b: Second structure.
    ///
    /// Returns:
    ///     Distance in ``[0.0, 1.0]``. ``0.0`` means structurally identical,
    ///     values near ``1.0`` indicate unrelated folds.
    pub fn call(&self, a: &PdbStructure, b: &PdbStructure) -> f32 {
        Distance::<PdbStructure>::call(&self.inner, a, b)
    }

    /// Alias for ``call`` — enables ``kernel(a, b)`` syntax.
    pub fn __call__(&self, a: &PdbStructure, b: &PdbStructure) -> f32 {
        self.call(a, b)
    }
}
