use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use pyo3_stub_gen::{PyStubType, TypeInfo};
use refnd_core::utils::read_fasta as core_read_fasta;
use std::path::Path;
use numpy::{IntoPyArray, PyArray1};
use refnd_core::utils::{BitFingerprint as CoreBitFP, InlineBitSet, RealFingerprint as CoreRealFP};
use refnd_core::kernels::usalign::PdbStructure as CorePdbStructure;
use std::collections::{HashMap, HashSet};
use refnd_core::core::largest_cluster as largest_cluster_core;
use refnd_core::utils::{SWPattern as CoreSWPattern, SWPatternSet as CoreSWPatternSet, SWSequence as CoreSWSequence, SWWord as CoreSWWord};
use std::str::FromStr;

/// Newtype so `PyArray1<bool>` gets a stub type — `pyo3_stub_gen` doesn't implement
/// `NumPyScalar` for `bool`, so we bypass it with a local wrapper.
pub struct BoolArray<'py>(Bound<'py, PyArray1<bool>>);

impl PyStubType for BoolArray<'_> {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: "numpy.typing.NDArray[numpy.bool_]".into(),
            source_module: None,
            import: HashSet::from(["numpy".into(), "numpy.typing".into()]),
            type_refs: HashMap::new(),
        }
    }
}

impl<'py> IntoPyObject<'py> for BoolArray<'py> {
    type Target = PyArray1<bool>;
    type Output = Bound<'py, PyArray1<bool>>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, _py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0)
    }
}

// ── BitFingerprint ────────────────────────────────────────────────────────────

/// Dense binary fingerprint backed by a dense bitset.
///
/// Each bit represents the presence or absence of a structural feature.
/// The primary source is an RDKit ``ExplicitBitVect`` (from ``GetFingerprint``),
/// but plain Python lists and numpy arrays are also accepted.
///
/// ``count`` caches the popcount so Tanimoto computation avoids re-counting.
///
/// Example::
///
///     from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
///     from refnd.utils import BitFingerprint
///
///     mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
///     mol = MolFromSmiles("c1ccccc1")
///     fp = BitFingerprint(mfpgen.GetFingerprint(mol))
///     print(fp.count(), len(fp))   # set bits, total bits
#[gen_stub_pyclass]
#[pyclass(module = "refnd.utils", skip_from_py_object)]
#[derive(Clone)]
pub struct BitFingerprint {
    pub inner: CoreBitFP,
}

#[gen_stub_pymethods]
#[pymethods]
impl BitFingerprint {
    /// Construct from an RDKit ``ExplicitBitVect`` (e.g. from ``GetFingerprint``).
    ///
    /// Raises ``TypeError`` if given a ``SparseBitVect`` from ``GetSparseFingerprint``.
    /// That type uses a hash-based sparse representation incompatible with the dense
    /// bit layout required here. Use ``GetFingerprint`` to get an ``ExplicitBitVect``.
    #[new]
    pub fn from_rdkit(fp: &Bound<'_, PyAny>) -> PyResult<Self> {
        let class_name: String = fp.get_type().name()?.extract()?;
        if class_name == "SparseBitVect" {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "SparseBitVect (from GetSparseFingerprint) is not supported. \
                 Use GetFingerprint, which returns an ExplicitBitVect.",
            ));
        }
        let n_bits: usize = fp.call_method0("GetNumBits")?.extract()?;
        let on_bits: Vec<usize> = fp.call_method0("GetOnBits")?.extract()?;
        let mut bits = InlineBitSet::with_capacity(n_bits);
        for b in on_bits {
            bits.insert(b);
        }
        Ok(Self { inner: CoreBitFP::new(bits) })
    }

    /// Construct from a list of booleans (or 0/1 ints).
    #[staticmethod]
    pub fn from_list(values: Vec<bool>) -> Self {
        let mut bits = InlineBitSet::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            if *v { bits.insert(i); }
        }
        Self { inner: CoreBitFP::new(bits) }
    }

    /// Construct from a numpy boolean or uint8 array.
    ///
    /// For `uint8`/`bool` dtypes, reads the array's raw buffer directly -- a zero-copy
    /// view into memory numpy already owns -- and sets bits from it in one pass, with no
    /// per-element Python object boxing. Any other dtype falls back to `.tolist()` +
    /// `from_list`, which does box each element as an individual Python object along the
    /// way; this covers arbitrary array-likes at the cost of that boxing.
    #[staticmethod]
    pub fn from_np(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(a) = arr.extract::<numpy::PyReadonlyArray1<u8>>() {
            let slice = a.as_slice()?;
            let mut bits = InlineBitSet::with_capacity(slice.len());
            for (i, &v) in slice.iter().enumerate() {
                if v != 0 {
                    bits.insert(i);
                }
            }
            return Ok(Self { inner: CoreBitFP::new(bits) });
        }
        if let Ok(a) = arr.extract::<numpy::PyReadonlyArray1<bool>>() {
            let slice = a.as_slice()?;
            let mut bits = InlineBitSet::with_capacity(slice.len());
            for (i, &v) in slice.iter().enumerate() {
                if v {
                    bits.insert(i);
                }
            }
            return Ok(Self { inner: CoreBitFP::new(bits) });
        }
        let raw: Vec<u8> = arr.call_method0("tolist")?.extract()?;
        Ok(Self::from_list(raw.iter().map(|&v| v != 0).collect()))
    }

    /// Export as an RDKit ``ExplicitBitVect``.
    pub fn to_rdkit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ds = py.import("rdkit.DataStructs")?;
        let bv = ds.call_method1("ExplicitBitVect", (self.inner.bits.len(),))?;
        for idx in self.inner.bits.ones() {
            bv.call_method1("SetBit", (idx,))?;
        }
        Ok(bv)
    }

    /// Export as a list of booleans.
    pub fn to_list(&self) -> Vec<bool> {
        (0..self.inner.bits.len())
            .map(|i| self.inner.bits.contains(i))
            .collect()
    }

    /// Export as a numpy bool array.
    pub fn to_np<'py>(&self, py: Python<'py>) -> BoolArray<'py> {
        let data: Vec<bool> = (0..self.inner.bits.len())
            .map(|i| self.inner.bits.contains(i))
            .collect();
        BoolArray(data.into_pyarray(py))
    }

    /// Create a fingerprint of ``len`` bits with exactly ``count`` bits set at random.
    ///
    /// Useful for randomized testing.
    ///
    /// Args:
    ///     len: Total number of bits.
    ///     count: Number of bits to turn on. Must be ``<= len``.
    ///
    /// Raises:
    ///     ValueError: If ``count > len``.
    #[staticmethod]
    pub fn random(len: usize, count: usize) -> PyResult<Self> {
        if count > len {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "count ({count}) must be <= len ({len})"
            )));
        }
        Ok(Self { inner: CoreBitFP::random(len, count) })
    }

    pub fn __len__(&self) -> usize { self.inner.bits.len() }
    /// Number of on bits (popcount).
    pub fn count(&self) -> u32 { self.inner.count }
}

impl<'a, 'py> FromPyObject<'a, 'py> for BitFingerprint {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        // already a BitFingerprint wrapper
        if let Ok(fp) = ob.cast::<BitFingerprint>() {
            let inner = fp.borrow().inner.clone();
            return Ok(Self { inner });
        }
        // RDKit ExplicitBitVect — has GetNumBits / GetOnBits
        if ob.hasattr("GetNumBits")? {
            return Self::from_rdkit(&*ob);
        }
        // numpy ndarray — has dtype
        if ob.hasattr("dtype")? {
            return Self::from_np(&*ob);
        }
        // plain list / sequence of booleans
        let values: Vec<bool> = ob.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "cannot convert to BitFingerprint: expected ExplicitBitVect, \
                 numpy array, or list of booleans",
            )
        })?;
        Ok(Self::from_list(values))
    }
}

// ── RealFingerprint ───────────────────────────────────────────────────────────

/// Dense real-valued fingerprint backed by a ``Vec<f32>``.
///
/// Each element represents a feature count or continuous value. The primary
/// source is an RDKit ``UIntSparseIntVect`` (from ``GetCountFingerprint``),
/// but plain Python lists and numpy arrays are also accepted.
///
/// ``norm_sq`` caches ``||x||²`` so Tanimoto computation avoids recomputing it.
///
/// Example::
///
///     from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
///     from refnd.utils import RealFingerprint
///     from refnd.kernels.molecules import TanimotoReal
///
///     mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
///     mol = MolFromSmiles("c1ccccc1")
///     fp = RealFingerprint(mfpgen.GetCountFingerprint(mol))
///     print(fp.norm_sq(), len(fp))
#[gen_stub_pyclass]
#[pyclass(module = "refnd.utils", skip_from_py_object)]
#[derive(Clone)]
pub struct RealFingerprint {
    pub inner: CoreRealFP,
}

#[gen_stub_pymethods]
#[pymethods]
impl RealFingerprint {
    /// Construct from an RDKit ``UIntSparseIntVect`` (e.g. from ``GetCountFingerprint``).
    ///
    /// Raises ``TypeError`` if given a ``ULongSparseIntVect`` from
    /// ``GetSparseCountFingerprint``. That variant uses the full 64-bit hash space
    /// as its length, which cannot be stored as a dense vector. Use
    /// ``GetCountFingerprint`` instead, which folds indices modulo ``fpSize`` and
    /// returns a ``UIntSparseIntVect`` with a manageable length.
    #[new]
    pub fn from_rdkit(fp: &Bound<'_, PyAny>) -> PyResult<Self> {
        let length: u64 = fp.call_method0("GetLength")?.extract()?;
        const MAX_DENSE: u64 = 1 << 24; // 16 M — proxy for ULongSparseIntVect
        if length > MAX_DENSE {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Fingerprint length {length} looks like a ULongSparseIntVect from \
                 GetSparseCountFingerprint, which uses the full 64-bit hash space and cannot \
                 be stored as a dense vector. Use GetCountFingerprint so the length equals fpSize.",
            )));
        }
        let length = length as usize;
        let nonzero: HashMap<usize, i64> =
            fp.call_method0("GetNonzeroElements")?.extract()?;
        let mut data = vec![0.0f32; length];
        for (idx, count) in nonzero {
            if idx < length {
                data[idx] = count as f32;
            }
        }
        Ok(Self { inner: CoreRealFP::new(data) })
    }

    /// Construct from a list of floats.
    #[staticmethod]
    pub fn from_list(values: Vec<f32>) -> Self {
        Self { inner: CoreRealFP::new(values) }
    }

    /// Construct from a numpy float32 or float64 array.
    #[staticmethod]
    pub fn from_np(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        let values: Vec<f32> = arr.call_method0("tolist")?.extract()?;
        Ok(Self { inner: CoreRealFP::new(values) })
    }

    /// Export as an RDKit ``UIntSparseIntVect``, compatible with ``GetCountFingerprint``.
    /// Only non-zero elements are stored; the length equals ``len(self)``.
    pub fn to_rdkit<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ds = py.import("rdkit.DataStructs")?;
        let sv = ds.call_method1("UIntSparseIntVect", (self.inner.data.len(),))?;
        for (idx, &val) in self.inner.data.iter().enumerate() {
            if val != 0.0 {
                sv.call_method1("__setitem__", (idx, val as u32))?;
            }
        }
        Ok(sv)
    }

    /// Export as a list of floats.
    pub fn to_list(&self) -> Vec<f32> {
        self.inner.data.clone()
    }

    /// Export as a numpy float32 array.
    pub fn to_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.data.clone().into_pyarray(py)
    }

    pub fn __len__(&self) -> usize { self.inner.data.len() }
    /// Squared Euclidean norm of the feature vector (``||x||²``).
    pub fn norm_sq(&self) -> f32 { self.inner.norm_sq }
}

impl<'a, 'py> FromPyObject<'a, 'py> for RealFingerprint {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        // already a RealFingerprint wrapper
        if let Ok(fp) = ob.cast::<RealFingerprint>() {
            let inner = fp.borrow().inner.clone();
            return Ok(Self { inner });
        }
        // RDKit UIntSparseIntVect — has GetLength / GetNonzeroElements
        if ob.hasattr("GetNonzeroElements")? {
            return Self::from_rdkit(&*ob);
        }
        // numpy ndarray — has dtype
        if ob.hasattr("dtype")? {
            return Self::from_np(&*ob);
        }
        // plain list of floats
        let values: Vec<f32> = ob.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "cannot convert to RealFingerprint: expected UIntSparseIntVect, \
                 numpy array, or list of floats",
            )
        })?;
        Ok(Self::from_list(values))
    }
}

// ── Vector ────────────────────────────────────────────────────────────────────

/// A dense ``f32`` vector, used by the ``refnd.kernels.vectors`` kernels (``Cosine``,
/// ``EluDot``, ``L1``, ``L2``). Backed by a plain ``Vec<f32>`` -- unlike
/// ``RealFingerprint``, there's no precomputed cache.
///
/// Example::
///
///     import numpy as np
///     from refnd.utils import Vector
///
///     v = Vector(np.array([1.0, 2.0, 3.0], dtype=np.float32))
///     print(len(v))
#[gen_stub_pyclass]
#[pyclass(module = "refnd.utils", skip_from_py_object)]
#[derive(Clone)]
pub struct Vector {
    pub inner: Vec<f32>,
}

#[gen_stub_pymethods]
#[pymethods]
impl Vector {
    /// Construct from a numpy array or a list of floats.
    #[new]
    pub fn new(values: &Bound<'_, PyAny>) -> PyResult<Self> {
        values.extract()
    }

    /// Construct from a list of floats.
    #[staticmethod]
    pub fn from_list(values: Vec<f32>) -> Self {
        Self { inner: values }
    }

    /// Construct from a numpy float32 array, reading its buffer directly. Falls back
    /// to ``.tolist()`` for other dtypes.
    #[staticmethod]
    pub fn from_np(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(a) = arr.extract::<numpy::PyReadonlyArray1<f32>>() {
            return Ok(Self { inner: a.as_slice()?.to_vec() });
        }
        let values: Vec<f32> = arr.call_method0("tolist")?.extract()?;
        Ok(Self { inner: values })
    }

    /// Export as a list of floats.
    pub fn to_list(&self) -> Vec<f32> {
        self.inner.clone()
    }

    /// Export as a numpy float32 array.
    pub fn to_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.clone().into_pyarray(py)
    }

    pub fn __len__(&self) -> usize { self.inner.len() }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Vector {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        // already a Vector wrapper
        if let Ok(v) = ob.cast::<Vector>() {
            let inner = v.borrow().inner.clone();
            return Ok(Self { inner });
        }
        // numpy ndarray — has dtype
        if ob.hasattr("dtype")? {
            return Self::from_np(&*ob);
        }
        // plain list of floats
        let values: Vec<f32> = ob.extract().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(
                "cannot convert to Vector: expected numpy array or list of floats",
            )
        })?;
        Ok(Self::from_list(values))
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
///     from refnd.utils import PdbStructure
///     s = PdbStructure("1abc.pdb")
#[gen_stub_pyclass]
#[pyclass(module = "refnd.utils", from_py_object)]
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

/// Parse a FASTA file and return all records as a list of ``(header, sequence)`` pairs.
///
/// The header string is the full description line without the leading ``>``.
/// The sequence is the concatenation of all continuation lines for that record,
/// with whitespace stripped.
///
/// Args:
///     path: Path to the FASTA file.
///
/// Returns:
///     A list of ``(header, sequence)`` tuples, one per FASTA record.
///
/// Raises:
///     IOError: If the file cannot be opened or is not valid UTF-8.
///
/// Example::
///
///     from refnd.utils import read_fasta
///
///     records = read_fasta("proteins.fasta")
///     header, seq = records[0]
///     print(header)  # "sp|P12345|MYPR_HUMAN ..."
///     print(seq)     # "MKTAYIAKQRQISFVKSHFSRQ..."
#[gen_stub_pyfunction(module = "refnd.utils")]
#[pyfunction]
pub fn read_fasta(path: &str) -> PyResult<Vec<(String, String)>> {
    core_read_fasta(Path::new(path)).map_err(|e| PyIOError::new_err(e))
}

/// Return the ID and size of the largest cluster.
///
/// Convenience helper — iterates over a cluster-label vector and finds the
/// most populous label.
///
/// Args:
///     clusters: A list of cluster IDs (e.g. from ``connected_components`` or
///               ``find_communities``).
///
/// Returns:
///     A tuple ``(cluster_id, size)`` for the largest cluster.
#[gen_stub_pyfunction(module = "refnd.utils")]
#[pyfunction]
pub fn largest_cluster(clusters: Vec<usize>) -> (usize, usize) {
    largest_cluster_core(&clusters)
}

// ── SWPattern ─────────────────────────────────────────────────────────────────

/// A spaced-word pattern: a binary mask over `length` positions where a match position
/// ("1") contributes a residue to the spaced word's key and a don't-care position ("0")
/// is skipped when hashing but still compared for mismatches. Position 0 and the last
/// position are always match positions.
///
/// Example::
///
///     from refnd.utils import SWPattern
///
///     pat = SWPattern(6, 20)
///     assert len(pat) == 26
///     assert pat.weight() == 6
///     assert pat.dontcare() == 20
///     assert pat.is_match(0)
///     assert pat.is_match(25)
///
///     # Parse/render as a string of '1's (match) and '0's (don't-care)
///     pat2 = SWPattern.parse("10100101")
///     assert pat2.weight() == 4
///     assert str(pat2) == "10100101"
#[gen_stub_pyclass]
#[pyclass(module = "refnd.utils", from_py_object)]
#[derive(Clone)]
pub struct SWPattern {
    pub inner: CoreSWPattern,
}

#[gen_stub_pymethods]
#[pymethods]
impl SWPattern {
    /// Build a pattern with ``weight`` match positions (including the fixed first
    /// and last) and ``dont_care`` don't-care positions, total length
    /// ``weight + dont_care``. Positions in-between are randomly sampled.
    ///
    /// Args:
    ///     weight: Number of match positions, including both endpoints. Must be ``>= 2``.
    ///     dont_care: Number of don't-care positions.
    ///
    /// Raises:
    ///     ValueError: If ``weight < 2`` (there would be no way to place both required endpoints).
    #[new]
    pub fn new(weight: usize, dont_care: usize) -> PyResult<Self> {
        if weight < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!("pattern weight must be >= 2, got {weight}")));
        }
        Ok(Self { inner: CoreSWPattern::random(weight, dont_care) })
    }

    /// Parse a pattern from a string of ``'1'``s (match) and ``'0'``s (don't-care),
    /// the same format ``str()`` produces.
    ///
    /// Raises:
    ///     ValueError: If the string contains a character other than ``'0'``/``'1'``,
    ///         or doesn't start and end with ``'1'``.
    #[staticmethod]
    pub fn parse(s: &str) -> PyResult<Self> {
        CoreSWPattern::from_str(s).map(|inner| Self { inner }).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Number of match ("1") positions.
    pub fn weight(&self) -> usize {
        self.inner.weight()
    }

    /// Number of don't-care ("0") positions.
    pub fn dontcare(&self) -> usize {
        self.inner.dontcare()
    }

    /// Ascending indices of match ("1") positions.
    pub fn match_positions(&self) -> Vec<usize> {
        self.inner.match_positions().to_vec()
    }

    /// Whether ``pos`` is a match position. Never raises: an out-of-range ``pos``
    /// simply reads as ``False``.
    pub fn is_match(&self, pos: usize) -> bool {
        self.inner.is_match(pos)
    }

    /// Total number of positions (``weight() + dontcare()``).
    pub fn __len__(&self) -> usize {
        self.inner.length()
    }

    /// Render as a string of the same length: ``'1'`` for match positions, ``'0'``
    /// for don't-care positions -- the inverse of ``SWPattern.parse``.
    pub fn __str__(&self) -> String {
        self.inner.to_string()
    }

    pub fn __repr__(&self) -> String {
        format!("SWPattern('{}')", self.inner)
    }
}

// ── SWPatternSet ──────────────────────────────────────────────────────────────

/// A set of ``SWPattern``s used together to compute spaced words for a sequence.
///
/// The default constructor builds a RasBhari-optimized set (recommended for real
/// use); use ``SWPatternSet.random`` for a cheap, unoptimized set, or
/// ``SWPatternSet.from_patterns`` to build one from hand-picked patterns.
///
/// Example::
///
///     from refnd.utils import SWPatternSet
///
///     # RasBhari-optimized (recommended)
///     patterns = SWPatternSet(5, 6, 20)
///     assert len(patterns) == 5
///
///     # Cheap, unoptimized baseline, refined by hand
///     random_patterns = SWPatternSet.random(5, 6, 20)
///     score_before = random_patterns.optimize(0)    # limit=0: score only, no changes
///     score_after = random_patterns.optimize(2000)
///     assert score_after <= score_before
#[gen_stub_pyclass]
#[pyclass(module = "refnd.utils", from_py_object)]
#[derive(Clone)]
pub struct SWPatternSet {
    pub inner: CoreSWPatternSet,
}

#[gen_stub_pymethods]
#[pymethods]
impl SWPatternSet {
    /// Build a RasBhari-optimized pattern set: ``n`` distinct random patterns of the
    /// given ``weight``/``dont_care``, refined by hill climbing for ProtSpaM's
    /// default step budget (25,000).
    ///
    /// Args:
    ///     n: Number of distinct patterns to build.
    ///     weight: Number of match positions per pattern. ProtSpaM's own default is ``6``.
    ///     dont_care: Number of don't-care positions per pattern. ProtSpaM's own
    ///         default is ``40``.
    ///
    /// Warning:
    ///     Same as ``SWPatternSet.random``: hangs if ``n`` isn't well below the
    ///     number of distinct patterns possible for ``weight``/``dont_care``.
    #[new]
    pub fn new(n: usize, weight: usize, dont_care: usize) -> Self {
        Self { inner: CoreSWPatternSet::new(n, weight, dont_care) }
    }

    /// Build ``n`` distinct unoptimized random patterns of the given
    /// ``weight``/``dont_care``. Useful as a cheap baseline, or as the unoptimized
    /// starting point ``optimize`` refines.
    ///
    /// Warning:
    ///     Patterns are generated by rejection sampling on uniqueness, so this hangs
    ///     (never returns) if ``n`` isn't well below the number of distinct patterns
    ///     possible for ``weight``/``dont_care`` (``C(weight + dont_care - 2, weight - 2)``).
    ///     This is sharpest at ``weight == 2``: there are no interior positions to
    ///     vary at all, so every generated pattern is identical and any ``n > 1``
    ///     hangs immediately.
    #[staticmethod]
    pub fn random(n: usize, weight: usize, dont_care: usize) -> Self {
        Self { inner: CoreSWPatternSet::random(n, weight, dont_care) }
    }

    /// Same as the default constructor, but with an explicit hill-climbing step
    /// budget instead of ProtSpaM's default of 25,000.
    #[staticmethod]
    pub fn with_limit(n: usize, weight: usize, dont_care: usize, limit: usize) -> Self {
        Self { inner: CoreSWPatternSet::with_limit(n, weight, dont_care, limit) }
    }

    /// Build a set from already-constructed patterns. Unlike ``random``, there's no uniqueness check -- duplicate
    /// patterns are allowed, though they add nothing (two identical patterns always
    /// find exactly the same matches).
    #[staticmethod]
    pub fn from_patterns(patterns: Vec<SWPattern>) -> Self {
        Self { inner: CoreSWPatternSet::from_patterns(patterns.into_iter().map(|p| p.inner).collect()) }
    }

    /// Optimize this pattern set in place with RasBhari's overlap-complexity hill
    /// climbing: repeatedly picks a pattern round-robin, swaps one of its interior
    /// match positions for a don't-care position, and keeps the change only if it
    /// strictly lowers the set's total pairwise overlap-complexity score.
    ///
    /// Args:
    ///     limit: Number of hill-climbing steps to run. Pass ``0`` to just compute
    ///         and return the current score without changing anything.
    ///
    /// Returns:
    ///     The achieved overlap-complexity score after optimizing (lower is better).
    pub fn optimize(&mut self, limit: usize) -> f64 {
        self.inner.optimize(limit)
    }

    /// The patterns in this set, in construction order.
    pub fn patterns(&self) -> Vec<SWPattern> {
        self.inner.patterns().iter().map(|p| SWPattern { inner: p.clone() }).collect()
    }

    /// Number of patterns in this set.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Serialize this set to ``path``. Inverse of ``load``.
    ///
    /// Raises:
    ///     IOError: If ``path`` can't be written.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner.save(path).map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Deserialize a set previously written by ``save``.
    ///
    /// Raises:
    ///     IOError: If ``path`` can't be read, or its contents aren't a valid pattern set.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        CoreSWPatternSet::load(path).map(|inner| Self { inner }).map_err(|e| PyIOError::new_err(e.to_string()))
    }
}

// ── SWWord ────────────────────────────────────────────────────────────────────

/// One complete spaced word: the residues at a pattern's match positions.
///
/// Example::
///
///     from refnd.utils import SWPatternSet, SWSequence
///
///     patterns = SWPatternSet.random(1, 2, 1)  # single pattern, weight 2, dc 1 -> "101"
///     seq = SWSequence("ACDE", patterns)
///     words = seq.sorted_words(0)
///     assert all(words[i].key() <= words[i + 1].key() for i in range(len(words) - 1))
#[gen_stub_pyclass]
#[pyclass(eq, ord, module = "refnd.utils", from_py_object)]
#[derive(Clone, Copy, PartialEq, PartialOrd)]
pub struct SWWord {
    pub inner: CoreSWWord,
}

#[gen_stub_pymethods]
#[pymethods]
impl SWWord {
    /// The packed key: the pattern's match-position residues, 5 bits each,
    /// most-significant residue first.
    pub fn key(&self) -> u64 {
        self.inner.key()
    }

    /// Start position of this word's window in the sequence it came from.
    pub fn pos(&self) -> usize {
        self.inner.pos()
    }

    pub fn __repr__(&self) -> String {
        format!("SWWord(key={}, pos={})", self.inner.key(), self.inner.pos())
    }
}

// ── SWSequence ────────────────────────────────────────────────────────────────

/// A sequence prepared for spaced-word comparison (ProtSpaM-style). Its residues are
/// pre-encoded as 5-bit amino-acid codes, and it holds the sorted spaced words for
/// every pattern in the ``SWPatternSet`` it was built with.
///
/// Two ``SWSequence``s must be built from the same ``SWPatternSet`` (or two equal
/// copies of it) to be compared meaningfully -- see
/// ``refnd.kernels.protspam.ProtSpamKernel``, which is what actually compares two of
/// these.
///
/// Picklable: ``pickle.dumps``/``pickle.loads`` round-trip an ``SWSequence`` without
/// needing the original pattern set again (its encoded residues and sorted spaced
/// words are serialized directly).
///
/// Example::
///
///     import pickle
///     from refnd.utils import SWPatternSet, SWSequence
///
///     patterns = SWPatternSet(5, 6, 20)
///     seq = SWSequence("MKTAYIAKQRQISFVKSHFSRQ", patterns)
///     assert len(seq) == 22
///
///     restored = pickle.loads(pickle.dumps(seq))
///     assert restored.seq() == seq.seq()
#[gen_stub_pyclass]
#[pyclass(module = "refnd.utils", from_py_object)]
#[derive(Clone)]
pub struct SWSequence {
    pub inner: CoreSWSequence,
}

#[gen_stub_pymethods]
#[pymethods]
impl SWSequence {
    /// Encode ``seq`` and compute its sorted spaced words for every pattern in
    /// ``patterns``.
    ///
    /// Args:
    ///     seq: The amino-acid sequence.
    ///     patterns: The pattern set to compute spaced words for. Must be the same
    ///         set (or an equal copy) used for every other ``SWSequence`` this one
    ///         will be compared against, and for the kernel doing the comparing.
    ///
    /// Raises:
    ///     ValueError: If ``seq`` contains a character outside ProtSpaM's amino-acid
    ///         alphabet (the 20 standard amino acids, ambiguity codes B/Z/X, stop
    ///         ``*``, and J; case-insensitive).
    #[new]
    pub fn new(seq: String, patterns: &SWPatternSet) -> PyResult<Self> {
        CoreSWSequence::new(&seq, &patterns.inner).map(|inner| Self { inner }).map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Residues as 5-bit amino-acid codes, not raw ASCII -- e.g. ``'A'`` reads back
    /// as ``0``, not ``65``.
    pub fn seq(&self) -> Vec<u8> {
        self.inner.seq().to_vec()
    }

    /// Sequence length in residues.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Sorted spaced words for ``patterns.patterns()[pattern_idx]``, where
    /// ``patterns`` is the set this sequence was built with. Empty if this sequence
    /// is shorter than that pattern (no window fits).
    ///
    /// Raises:
    ///     IndexError: If ``pattern_idx`` is out of range for the pattern set this
    ///         sequence was built with.
    pub fn sorted_words(&self, pattern_idx: usize) -> PyResult<Vec<SWWord>> {
        if pattern_idx >= self.inner.pattern_count() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "pattern_idx {pattern_idx} out of range for {} patterns",
                self.inner.pattern_count()
            )));
        }
        Ok(self.inner.sorted_words(pattern_idx).iter().map(|w| SWWord { inner: *w }).collect())
    }

    /// Number of patterns this sequence has spaced words for -- the ``len()`` of the
    /// ``SWPatternSet`` it was built with. Valid indices for ``sorted_words`` are
    /// ``0 .. pattern_count()``.
    pub fn pattern_count(&self) -> usize {
        self.inner.pattern_count()
    }

    /// Pickle support.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (Vec<u8>,))> {
        let state = bincode::encode_to_vec(&self.inner, bincode::config::standard())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let ctor = py.get_type::<Self>().getattr("_from_bytes")?;
        Ok((ctor, (state,)))
    }

    #[staticmethod]
    fn _from_bytes(data: Vec<u8>) -> PyResult<Self> {
        let (inner, _): (CoreSWSequence, usize) = bincode::decode_from_slice(&data, bincode::config::standard())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}