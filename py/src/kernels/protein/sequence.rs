use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use refnd_core::core::Distance;
use refnd_core::kernels::proteins::parasail::{
    AlignerConfigTrait, AlignerMatrix, BundledMatrix, GlobalAlignerBuilder, GlobalAligner as CoreGlobalAligner,
    GlobalIdentityMode as CoreGlobalIdentityMode, LocalAlignerBuilder, LocalAligner as CoreLocalAligner,
    LocalIdentityMode as CoreLocalIdentityMode, CoverageMode as CoreCoverageMode,
    VectorizationStrategy as CoreVectorizationStrategy,
    DatatypeWidth as CoreDatatypeWidth,
};

// ── Config enums ─────────────────────────────────────────────────────────────

/// Denominator used to normalise a global-alignment identity score.
///
/// After counting identical aligned positions the raw count is divided by:
///
/// - ``AlignmentLength``: the total length of the alignment (including gaps).
/// - ``MaxSeqLength``: the length of the longer of the two sequences.
/// - ``MinSeqLength``: the length of the shorter of the two sequences.
/// - ``MaxLength`` (default): same as ``MaxSeqLength`` — recommended for RGP datasets.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.protein.sequence")]
#[derive(Clone, Copy, PartialEq)]
pub enum GlobalIdentityMode {
    AlignmentLength,
    MaxSeqLength,
    MinSeqLength,
    MaxLength,
}

impl From<GlobalIdentityMode> for CoreGlobalIdentityMode {
    fn from(m: GlobalIdentityMode) -> Self {
        match m {
            GlobalIdentityMode::AlignmentLength => CoreGlobalIdentityMode::AlignmentLength,
            GlobalIdentityMode::MaxSeqLength    => CoreGlobalIdentityMode::MaxSeqLength,
            GlobalIdentityMode::MinSeqLength    => CoreGlobalIdentityMode::MinSeqLength,
            GlobalIdentityMode::MaxLength       => CoreGlobalIdentityMode::MaxLength,
        }
    }
}

/// Denominator used to normalise a local-alignment identity score.
///
/// - ``AlignmentLength`` (default): divide by the length of the local alignment.
/// - ``MinSeqLength``: divide by the shorter sequence length.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.protein.sequence")]
#[derive(Clone, Copy, PartialEq)]
pub enum LocalIdentityMode {
    AlignmentLength,
    MinSeqLength,
}

impl From<LocalIdentityMode> for CoreLocalIdentityMode {
    fn from(m: LocalIdentityMode) -> Self {
        match m {
            LocalIdentityMode::AlignmentLength => CoreLocalIdentityMode::AlignmentLength,
            LocalIdentityMode::MinSeqLength    => CoreLocalIdentityMode::MinSeqLength,
        }
    }
}

/// Coverage filter applied before accepting a local alignment as valid.
///
/// A pair is scored only when the alignment covers enough of the sequences
/// as specified by the mode and ``min_coverage`` threshold:
///
/// - ``BothQueryTarget`` (default): both query and target must meet ``min_coverage``.
/// - ``Target``: only the target must meet ``min_coverage``.
/// - ``Query``: only the query must meet ``min_coverage``.
/// - ``LengthRatio``: the shorter / longer length ratio must meet ``min_coverage``.
/// - ``ShorterSeq``: coverage computed relative to the shorter sequence.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.protein.sequence")]
#[derive(Clone, Copy, PartialEq)]
pub enum CoverageMode {
    BothQueryTarget,
    Target,
    Query,
    LengthRatio,
    ShorterSeq,
}

impl From<CoverageMode> for CoreCoverageMode {
    fn from(m: CoverageMode) -> Self {
        match m {
            CoverageMode::BothQueryTarget => CoreCoverageMode::BothQueryTarget,
            CoverageMode::Target          => CoreCoverageMode::Target,
            CoverageMode::Query           => CoreCoverageMode::Query,
            CoverageMode::LengthRatio     => CoreCoverageMode::LengthRatio,
            CoverageMode::ShorterSeq      => CoreCoverageMode::ShorterSeq,
        }
    }
}

/// SIMD vectorization layout used by the parasail alignment engine.
///
/// - ``Striped`` (default for local): interleaved layout, best for short sequences.
/// - ``Scan``: sequential scan layout, often faster for long sequences or global alignment.
/// - ``Diag``: diagonal layout; niche use-case, rarely needed.
///
/// In practice the default per-aligner is a good choice; change only if profiling
/// shows a bottleneck.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.protein.sequence")]
#[derive(Clone, Copy, PartialEq)]
pub enum VectorizationStrategy {
    Striped,
    Scan,
    Diag,
}

impl From<VectorizationStrategy> for CoreVectorizationStrategy {
    fn from(s: VectorizationStrategy) -> Self {
        match s {
            VectorizationStrategy::Striped => CoreVectorizationStrategy::Striped,
            VectorizationStrategy::Scan    => CoreVectorizationStrategy::Scan,
            VectorizationStrategy::Diag    => CoreVectorizationStrategy::Diag,
        }
    }
}

#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.protein.sequence")]
#[derive(Clone, Copy, PartialEq)]
pub enum ScoringMatrix {
    Blosum30, Blosum35, Blosum40, Blosum45, Blosum50,
    Blosum55, Blosum60, Blosum62, Blosum65, Blosum70,
    Blosum75, Blosum80, Blosum85, Blosum90, Blosum95, Blosum100,
    Pam10,  Pam20,  Pam30,  Pam40,  Pam50,
    Pam60,  Pam70,  Pam80,  Pam90,  Pam100,
    Pam110, Pam120, Pam130, Pam140, Pam150,
    Pam160, Pam170, Pam180, Pam190, Pam200,
    Pam210, Pam220, Pam230, Pam240, Pam250,
    Pam260, Pam270, Pam280, Pam290, Pam300,
    Pam310, Pam320, Pam330, Pam340, Pam350,
    Pam360, Pam370, Pam380, Pam390, Pam400,
    Pam410, Pam420, Pam430, Pam440, Pam450,
    Pam460, Pam470, Pam480, Pam490, Pam500,
}

impl From<ScoringMatrix> for BundledMatrix {
    fn from(m: ScoringMatrix) -> Self {
        match m {
            ScoringMatrix::Blosum30  => BundledMatrix::Blosum30,
            ScoringMatrix::Blosum35  => BundledMatrix::Blosum35,
            ScoringMatrix::Blosum40  => BundledMatrix::Blosum40,
            ScoringMatrix::Blosum45  => BundledMatrix::Blosum45,
            ScoringMatrix::Blosum50  => BundledMatrix::Blosum50,
            ScoringMatrix::Blosum55  => BundledMatrix::Blosum55,
            ScoringMatrix::Blosum60  => BundledMatrix::Blosum60,
            ScoringMatrix::Blosum62  => BundledMatrix::Blosum62,
            ScoringMatrix::Blosum65  => BundledMatrix::Blosum65,
            ScoringMatrix::Blosum70  => BundledMatrix::Blosum70,
            ScoringMatrix::Blosum75  => BundledMatrix::Blosum75,
            ScoringMatrix::Blosum80  => BundledMatrix::Blosum80,
            ScoringMatrix::Blosum85  => BundledMatrix::Blosum85,
            ScoringMatrix::Blosum90  => BundledMatrix::Blosum90,
            ScoringMatrix::Blosum95  => BundledMatrix::Blosum95,
            ScoringMatrix::Blosum100 => BundledMatrix::Blosum100,
            ScoringMatrix::Pam10  => BundledMatrix::Pam10,
            ScoringMatrix::Pam20  => BundledMatrix::Pam20,
            ScoringMatrix::Pam30  => BundledMatrix::Pam30,
            ScoringMatrix::Pam40  => BundledMatrix::Pam40,
            ScoringMatrix::Pam50  => BundledMatrix::Pam50,
            ScoringMatrix::Pam60  => BundledMatrix::Pam60,
            ScoringMatrix::Pam70  => BundledMatrix::Pam70,
            ScoringMatrix::Pam80  => BundledMatrix::Pam80,
            ScoringMatrix::Pam90  => BundledMatrix::Pam90,
            ScoringMatrix::Pam100 => BundledMatrix::Pam100,
            ScoringMatrix::Pam110 => BundledMatrix::Pam110,
            ScoringMatrix::Pam120 => BundledMatrix::Pam120,
            ScoringMatrix::Pam130 => BundledMatrix::Pam130,
            ScoringMatrix::Pam140 => BundledMatrix::Pam140,
            ScoringMatrix::Pam150 => BundledMatrix::Pam150,
            ScoringMatrix::Pam160 => BundledMatrix::Pam160,
            ScoringMatrix::Pam170 => BundledMatrix::Pam170,
            ScoringMatrix::Pam180 => BundledMatrix::Pam180,
            ScoringMatrix::Pam190 => BundledMatrix::Pam190,
            ScoringMatrix::Pam200 => BundledMatrix::Pam200,
            ScoringMatrix::Pam210 => BundledMatrix::Pam210,
            ScoringMatrix::Pam220 => BundledMatrix::Pam220,
            ScoringMatrix::Pam230 => BundledMatrix::Pam230,
            ScoringMatrix::Pam240 => BundledMatrix::Pam240,
            ScoringMatrix::Pam250 => BundledMatrix::Pam250,
            ScoringMatrix::Pam260 => BundledMatrix::Pam260,
            ScoringMatrix::Pam270 => BundledMatrix::Pam270,
            ScoringMatrix::Pam280 => BundledMatrix::Pam280,
            ScoringMatrix::Pam290 => BundledMatrix::Pam290,
            ScoringMatrix::Pam300 => BundledMatrix::Pam300,
            ScoringMatrix::Pam310 => BundledMatrix::Pam310,
            ScoringMatrix::Pam320 => BundledMatrix::Pam320,
            ScoringMatrix::Pam330 => BundledMatrix::Pam330,
            ScoringMatrix::Pam340 => BundledMatrix::Pam340,
            ScoringMatrix::Pam350 => BundledMatrix::Pam350,
            ScoringMatrix::Pam360 => BundledMatrix::Pam360,
            ScoringMatrix::Pam370 => BundledMatrix::Pam370,
            ScoringMatrix::Pam380 => BundledMatrix::Pam380,
            ScoringMatrix::Pam390 => BundledMatrix::Pam390,
            ScoringMatrix::Pam400 => BundledMatrix::Pam400,
            ScoringMatrix::Pam410 => BundledMatrix::Pam410,
            ScoringMatrix::Pam420 => BundledMatrix::Pam420,
            ScoringMatrix::Pam430 => BundledMatrix::Pam430,
            ScoringMatrix::Pam440 => BundledMatrix::Pam440,
            ScoringMatrix::Pam450 => BundledMatrix::Pam450,
            ScoringMatrix::Pam460 => BundledMatrix::Pam460,
            ScoringMatrix::Pam470 => BundledMatrix::Pam470,
            ScoringMatrix::Pam480 => BundledMatrix::Pam480,
            ScoringMatrix::Pam490 => BundledMatrix::Pam490,
            ScoringMatrix::Pam500 => BundledMatrix::Pam500,
        }
    }
}
/// Integer precision used for alignment score accumulation.
///
/// - ``Short`` (8-bit), ``Half`` (16-bit), ``Full`` (32-bit), ``Long`` (64-bit):
///   fixed-width integers — lower width is faster but can overflow on long sequences.
/// - ``Sat`` (default): 8-bit saturating arithmetic; If it saturates, silently restart with 16-bit.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.protein.sequence")]
#[derive(Clone, Copy, PartialEq)]
pub enum DatatypeWidth {
    Short = 8,
    Half = 16,
    Full = 32,
    Long = 64,
    Sat
}

impl From<DatatypeWidth> for CoreDatatypeWidth {
    fn from(datatype: DatatypeWidth) -> Self {
        match datatype {
            DatatypeWidth::Short => CoreDatatypeWidth::Short,
            DatatypeWidth::Half => CoreDatatypeWidth::Half,
            DatatypeWidth::Full => CoreDatatypeWidth::Full,
            DatatypeWidth::Long => CoreDatatypeWidth::Long,
            DatatypeWidth::Sat => CoreDatatypeWidth::Sat,
        }
    }
}

// Kernel Binding

/// Needleman–Wunsch global sequence aligner returning a normalised identity score.
///
/// Wraps the parasail SIMD alignment library. The identity score is computed as
/// the number of identical aligned positions divided by the denominator selected
/// by ``identity_mode``.
///
/// ``GlobalAligner`` is used as a kernel in ``HNSWState``, ``exact_edges``, and
/// ``exact_nearest_neighbors`` via ``KernelVariant.ProteinGlobal``.
///
/// Args:
///     gap_open: Affine gap-open penalty (positive integer, subtracted). Default ``11``.
///     gap_extend: Affine gap-extend penalty (positive integer, subtracted). Default ``1``.
///     matrix: Amino-acid substitution matrix. Default ``ScoringMatrix.Blosum62``.
///     identity_mode: Normalisation denominator. Default ``GlobalIdentityMode.MaxLength``.
///     vectorization: SIMD layout. Default ``VectorizationStrategy.Scan``.
///     width: Integer precision. Default ``DatatypeWidth.Sat``.
///
/// Example::
///
///     from refnd.kernels.protein.sequence import GlobalAligner
///
///     aligner = GlobalAligner(gap_open=11, gap_extend=1)
///     score = aligner.call("MKTAYIAK", "MKTAYIAKQR")
///     score = aligner("MKTAYIAK", "MKTAYIAKQR") # Alternative
///     # score in [0.0, 1.0]
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.protein.sequence")]
pub struct GlobalAligner {
    pub inner: CoreGlobalAligner
}
#[gen_stub_pymethods]
#[pymethods]
impl GlobalAligner {
    #[new]
    #[pyo3(signature = (
        gap_open = 11,
        gap_extend = 1,
        matrix = ScoringMatrix::Blosum62,
        identity_mode = GlobalIdentityMode::MaxLength,
        vectorization = VectorizationStrategy::Scan,
        width = DatatypeWidth::Sat,
    ))]
    fn new(
        gap_open: i32,
        gap_extend: i32,
        matrix: ScoringMatrix,
        identity_mode: GlobalIdentityMode,
        vectorization: VectorizationStrategy,
        width: DatatypeWidth,
    ) -> Self {
        let inner = GlobalAlignerBuilder::new()
            .set_gap_open(gap_open)
            .set_gap_extend(gap_extend)
            .set_matrix(AlignerMatrix::Bundled(matrix.into()))
            .identity_mode(identity_mode.into())
            .set_vectorization(vectorization.into())
            .set_width(width.into())
        .build();

        Self {
            inner
        }
    }
    /// Compute the global alignment identity between two sequences.
    ///
    /// Args:
    ///     ref_sample: Reference protein sequence (single-letter amino acid codes).
    ///     query: Query protein sequence.
    ///
    /// Returns:
    ///     Identity score in ``[0.0, 1.0]``.
    #[pyo3(signature = (ref_sample, query))]
    fn call(&self, ref_sample: &str, query: &str) -> f32 {
        self.inner.call(ref_sample, query)
    }

    #[pyo3(signature = (ref_sample, query))]
    fn __call__(&self, ref_sample: &str, query: &str) -> f32 {
        self.inner.call(ref_sample, query)
    }
}

/// Smith–Waterman local sequence aligner returning a normalised identity score.
///
/// Like ``GlobalAligner`` but aligns only the most similar sub-region of each
/// sequence. Pairs that do not meet the ``min_coverage`` criterion after alignment
/// receive a score of ``0.0``.
///
/// ``LocalAligner`` is used as a kernel via ``KernelVariant.ProteinLocal``.
///
/// Args:
///     gap_open: Affine gap-open penalty. Default ``11``.
///     gap_extend: Affine gap-extend penalty. Default ``1``.
///     min_coverage: Minimum fraction of sequence covered by the local alignment
///                   (per ``cov_mode``) for the pair to be accepted. Default ``0.8``.
///     cov_mode: Which sequence(s) must meet ``min_coverage``. Default
///               ``CoverageMode.BothQueryTarget``.
///     matrix: Substitution matrix. Default ``ScoringMatrix.Blosum62``.
///     identity_mode: Normalisation denominator. Default ``LocalIdentityMode.AlignmentLength``.
///     vectorization: SIMD layout. Default ``VectorizationStrategy.Striped``.
///     width: Integer precision. Default ``DatatypeWidth.Sat``.
///
/// Example::
///
///     from refnd.kernels.protein.sequence import LocalAligner, CoverageMode
///
///     aligner = LocalAligner(min_coverage=0.5, cov_mode=CoverageMode.Query)
///     score = aligner.call("ACDEFGHIKLM", "CDEFGHI")
///     score = aligner("ACDEFGHIKLM", "CDEFGHI") # Alternative
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.protein.sequence")]
pub struct LocalAligner {
    pub inner: CoreLocalAligner
}

#[gen_stub_pymethods]
#[pymethods]
impl LocalAligner {
    #[new]
    #[pyo3(signature = (
        gap_open = 11,
        gap_extend = 1,
        min_coverage = 0.8,
        cov_mode = CoverageMode::BothQueryTarget,
        matrix = ScoringMatrix::Blosum62,
        identity_mode = LocalIdentityMode::AlignmentLength,
        vectorization = VectorizationStrategy::Striped,
        width = DatatypeWidth::Sat,
    ))]
    fn new(
        gap_open: i32,
        gap_extend: i32,
        min_coverage: f32,
        cov_mode: CoverageMode,
        matrix: ScoringMatrix,
        identity_mode: LocalIdentityMode,
        vectorization: VectorizationStrategy,
        width: DatatypeWidth,
    ) -> Self {
        let inner = LocalAlignerBuilder::new()
            .set_gap_open(gap_open)
            .set_gap_extend(gap_extend)
            .min_coverage(min_coverage)
            .cov_mode(cov_mode.into())
            .set_matrix(AlignerMatrix::Bundled(matrix.into()))
            .identity_mode(identity_mode.into())
            .set_vectorization(vectorization.into())
            .set_width(width.into())
            .build();

        Self {
            inner
        }
    }
    /// Compute the local alignment identity between two sequences.
    ///
    /// Args:
    ///     ref_sample: Reference protein sequence (single-letter amino acid codes).
    ///     query: Query protein sequence.
    ///
    /// Returns:
    ///     Identity score in ``[0.0, 1.0]``, or ``0.0`` if the coverage filter fails.
    #[pyo3(signature = (ref_sample, query))]
    fn call(&self, ref_sample: &str, query: &str) -> f32 {
        self.inner.call(ref_sample, query)
    }

    #[pyo3(signature = (ref_sample, query))]
    fn __call__(&self, ref_sample: &str, query: &str) -> f32 {
        self.inner.call(ref_sample, query)
    }
}
