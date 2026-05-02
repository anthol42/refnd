use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use refnd_core::core::Distance;
use refnd_core::kernels::molecules::tanimoto::Tanimoto;
use crate::utils::{BitFingerprint, RealFingerprint};

impl Distance<BitFingerprint> for Tanimoto {
    #[inline(always)]
    fn call(&self, a: &BitFingerprint, b: &BitFingerprint) -> f32 {
        Distance::<refnd_core::utils::BitFingerprint>::call(self, &a.inner, &b.inner)
    }
}

impl Distance<RealFingerprint> for Tanimoto {
    #[inline(always)]
    fn call(&self, a: &RealFingerprint, b: &RealFingerprint) -> f32 {
        Distance::<refnd_core::utils::RealFingerprint>::call(self, &a.inner, &b.inner)
    }
}

// ── TanimotoBit ───────────────────────────────────────────────────────────────

/// Tanimoto distance kernel for binary (bit) molecular fingerprints.
///
/// Measures structural dissimilarity between two molecules as the complement of
/// the Jaccard index over their feature sets. A distance of 0 means the two
/// fingerprints are identical; 1 means they share no features at all.
///
/// Formula: ``1 - |A ∩ B| / |A ∪ B|``
///
/// This kernel is the standard choice when working with ``ExplicitBitVect``
/// fingerprints such as Morgan, RDKit, or MACCS keys.
///
/// Example::
///
///     from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
///     from refnd.kernels.molecules import BitFingerprint, TanimotoBit
///
///     mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
///     benzene    = BitFingerprint(mfpgen.GetFingerprint(MolFromSmiles("c1ccccc1")))
///     naphthalene = BitFingerprint(mfpgen.GetFingerprint(MolFromSmiles("c1ccc2ccccc2c1")))
///     acetic_acid = BitFingerprint(mfpgen.GetFingerprint(MolFromSmiles("CC(=O)O")))
///
///     k = TanimotoBit()
///     print(k(benzene, naphthalene))  # low  — structurally similar
///     print(k(benzene, acetic_acid))  # high — structurally dissimilar
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.molecules")]
pub struct TanimotoBit {
    pub inner: Tanimoto,
}

#[gen_stub_pymethods]
#[pymethods]
impl TanimotoBit {
    #[new]
    pub fn new() -> Self { Self { inner: Tanimoto } }

    /// Compute the Tanimoto distance between two ``BitFingerprint`` objects.
    ///
    /// Args:
    ///     a: First fingerprint.
    ///     b: Second fingerprint.
    ///
    /// Returns:
    ///     Distance in ``[0.0, 1.0]``. ``0.0`` means identical feature sets,
    ///     ``1.0`` means fully disjoint.
    ///
    /// Example::
    ///
    ///     fp1 = BitFingerprint.from_list([True, False, True, True])
    ///     fp2 = BitFingerprint.from_list([True, True,  True, False])
    ///     k = TanimotoBit()
    ///     assert k.call(fp1, fp2) == k(fp1, fp2)  # both forms are equivalent
    ///     # intersection={0,2}=2, union={0,1,2,3}=4 → distance = 0.5
    pub fn call(&self, a: &BitFingerprint, b: &BitFingerprint) -> f32 {
        self.inner.call(&a.inner, &b.inner)
    }

    /// Alias for ``call`` — enables ``kernel(a, b)`` syntax.
    pub fn __call__(&self, a: &BitFingerprint, b: &BitFingerprint) -> f32 {
        self.call(a, b)
    }
}

// ── TanimotoReal ──────────────────────────────────────────────────────────────

/// Tanimoto distance kernel for real-valued (count) molecular fingerprints.
///
/// Generalises the binary Tanimoto to continuous feature vectors using the
/// dot-product formulation. A distance of 0 means the two fingerprints are
/// proportional; values approach 1 as the vectors become orthogonal.
///
/// Formula: ``1 - dot(a, b) / (||a||² + ||b||² - dot(a, b))``
///
/// This kernel is the standard choice when working with count fingerprints
/// such as those returned by ``GetCountFingerprint`` (Morgan counts, etc.).
///
/// Example::
///
///     from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
///     from refnd.kernels.molecules import RealFingerprint, TanimotoReal
///
///     mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
///     benzene     = RealFingerprint(mfpgen.GetCountFingerprint(MolFromSmiles("c1ccccc1")))
///     naphthalene = RealFingerprint(mfpgen.GetCountFingerprint(MolFromSmiles("c1ccc2ccccc2c1")))
///     acetic_acid = RealFingerprint(mfpgen.GetCountFingerprint(MolFromSmiles("CC(=O)O")))
///
///     k = TanimotoReal()
///     print(k(benzene, naphthalene))  # low  — structurally similar
///     print(k(benzene, acetic_acid))  # high — structurally dissimilar
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.molecules")]
pub struct TanimotoReal {
    pub inner: Tanimoto,
}

#[gen_stub_pymethods]
#[pymethods]
impl TanimotoReal {
    #[new]
    pub fn new() -> Self { Self { inner: Tanimoto } }

    /// Compute the Tanimoto distance between two ``RealFingerprint`` objects.
    ///
    /// Args:
    ///     a: First fingerprint.
    ///     b: Second fingerprint.
    ///
    /// Returns:
    ///     Distance in ``[0.0, 1.0]``. ``0.0`` means identical (proportional)
    ///     feature vectors, ``1.0`` means fully orthogonal.
    ///
    /// Example::
    ///
    ///     fp1 = RealFingerprint.from_list([1.0, 0.0, 1.0])
    ///     fp2 = RealFingerprint.from_list([0.0, 1.0, 1.0])
    ///     k = TanimotoReal()
    ///     assert k.call(fp1, fp2) == k(fp1, fp2)  # both forms are equivalent
    ///     # dot=1, ||fp1||²=2, ||fp2||²=2 → distance = 1 - 1/(2+2-1) ≈ 0.667
    pub fn call(&self, a: &RealFingerprint, b: &RealFingerprint) -> f32 {
        self.inner.call(&a.inner, &b.inner)
    }

    /// Alias for ``call`` — enables ``kernel(a, b)`` syntax.
    pub fn __call__(&self, a: &RealFingerprint, b: &RealFingerprint) -> f32 {
        self.call(a, b)
    }
}
