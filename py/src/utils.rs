use pyo3::prelude::*;
use pyo3::exceptions::PyIOError;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use refnd_core::utils::read_fasta as core_read_fasta;
use std::path::Path;
use fixedbitset::FixedBitSet;
use numpy::{IntoPyArray, PyArray1};
use refnd_core::utils::{BitFingerprint as CoreBitFP, RealFingerprint as CoreRealFP};

// ── BitFingerprint ────────────────────────────────────────────────────────────

/// Dense binary fingerprint backed by a packed bitset.
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
///     from refnd.kernels.molecules import BitFingerprint, TanimotoBit
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
        let mut bits = FixedBitSet::with_capacity(n_bits);
        for b in on_bits {
            bits.insert(b);
        }
        Ok(Self { inner: CoreBitFP::new(bits) })
    }

    /// Construct from a list of booleans (or 0/1 ints).
    #[staticmethod]
    pub fn from_list(values: Vec<bool>) -> Self {
        let mut bits = FixedBitSet::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            if *v { bits.insert(i); }
        }
        Self { inner: CoreBitFP::new(bits) }
    }

    /// Construct from a numpy boolean or uint8 array.
    #[staticmethod]
    pub fn from_np(arr: &Bound<'_, PyAny>) -> PyResult<Self> {
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

    /// Export as a numpy uint8 array (0 or 1 per bit).
    pub fn to_np<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let data: Vec<u8> = (0..self.inner.bits.len())
            .map(|i| self.inner.bits.contains(i) as u8)
            .collect();
        data.into_pyarray(py)
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
///     from refnd.kernels.molecules import RealFingerprint, TanimotoReal
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
        let nonzero: std::collections::HashMap<usize, i64> =
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
