import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from refnd.kernels.molecules import TanimotoBit, TanimotoReal
from refnd.utils import BitFingerprint, RealFingerprint


@pytest.fixture(scope="module")
def mfpgen():
    return rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)


@pytest.fixture(scope="module")
def mols():
    return (
        Chem.MolFromSmiles("c1ccccc1"),        # benzene
        Chem.MolFromSmiles("c1ccc2ccccc2c1"),  # naphthalene — similar to benzene
        Chem.MolFromSmiles("CC(=O)O"),         # acetic acid — different
    )


# ── BitFingerprint constructors ───────────────────────────────────────────────

class TestBitFingerprintConstructors:
    def test_from_rdkit(self, mfpgen, mols):
        rdkit_fp = mfpgen.GetFingerprint(mols[0])
        fp = BitFingerprint(rdkit_fp)
        assert len(fp) == 1024
        assert fp.count() > 0

    def test_from_list(self):
        fp = BitFingerprint.from_list([True, False, True, True, False])
        assert len(fp) == 5
        assert fp.count() == 3

    def test_from_np(self):
        arr = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        fp = BitFingerprint.from_np(arr)
        assert len(fp) == 5
        assert fp.count() == 3

    def test_from_list_matches_from_np(self):
        values = [True, False, True, False, True, True]
        fp_list = BitFingerprint.from_list(values)
        fp_np = BitFingerprint.from_np(np.array(values))
        assert fp_list.count() == fp_np.count()
        assert len(fp_list) == len(fp_np)

    def test_from_rdkit_sparse_raises(self, mfpgen, mols):
        sparse_fp = mfpgen.GetSparseFingerprint(mols[0])
        with pytest.raises(TypeError, match="SparseBitVect"):
            BitFingerprint(sparse_fp)


# ── BitFingerprint export ─────────────────────────────────────────────────────

class TestBitFingerprintExport:
    def test_to_list_roundtrip(self):
        values = [True, False, True, True, False]
        fp = BitFingerprint.from_list(values)
        assert fp.to_list() == values

    def test_to_np_roundtrip(self):
        arr = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
        fp = BitFingerprint.from_np(arr)
        np.testing.assert_array_equal(fp.to_np(), arr)

    def test_to_rdkit_roundtrip(self, mfpgen, mols):
        rdkit_fp = mfpgen.GetFingerprint(mols[0])
        fp = BitFingerprint(rdkit_fp)
        exported = fp.to_rdkit()
        assert exported.GetNumBits() == len(fp)
        assert list(exported.GetOnBits()) == list(rdkit_fp.GetOnBits())

    def test_to_rdkit_class(self, mfpgen, mols):
        from rdkit.DataStructs.cDataStructs import ExplicitBitVect
        fp = BitFingerprint(mfpgen.GetFingerprint(mols[0]))
        assert isinstance(fp.to_rdkit(), ExplicitBitVect)


# ── RealFingerprint constructors ──────────────────────────────────────────────

class TestRealFingerprintConstructors:
    def test_from_rdkit(self, mfpgen, mols):
        rdkit_fp = mfpgen.GetCountFingerprint(mols[0])
        fp = RealFingerprint(rdkit_fp)
        assert len(fp) == rdkit_fp.GetLength()
        assert fp.norm_sq() > 0.0

    def test_from_list(self):
        fp = RealFingerprint.from_list([1.0, 0.0, 2.0])
        assert len(fp) == 3
        assert abs(fp.norm_sq() - 5.0) < 1e-6

    def test_from_np(self):
        arr = np.array([1.0, 2.0, 0.0], dtype=np.float32)
        fp = RealFingerprint.from_np(arr)
        assert len(fp) == 3
        assert abs(fp.norm_sq() - 5.0) < 1e-6

    def test_from_rdkit_sparse_raises(self, mfpgen, mols):
        sparse_fp = mfpgen.GetSparseCountFingerprint(mols[0])
        with pytest.raises(TypeError, match="ULongSparseIntVect"):
            RealFingerprint(sparse_fp)

    def test_from_list_matches_from_np(self):
        values = [1.0, 0.5, 0.0, 2.0]
        fp_list = RealFingerprint.from_list(values)
        fp_np = RealFingerprint.from_np(np.array(values, dtype=np.float32))
        assert abs(fp_list.norm_sq() - fp_np.norm_sq()) < 1e-5


# ── RealFingerprint export ────────────────────────────────────────────────────

class TestRealFingerprintExport:
    def test_to_list_roundtrip(self):
        values = [1.0, 0.0, 2.5, 0.0]
        fp = RealFingerprint.from_list(values)
        assert fp.to_list() == pytest.approx(values)

    def test_to_np_roundtrip(self):
        arr = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float32)
        fp = RealFingerprint.from_np(arr)
        np.testing.assert_allclose(fp.to_np(), arr)

    def test_to_rdkit_roundtrip(self, mfpgen, mols):
        rdkit_fp = mfpgen.GetCountFingerprint(mols[0])
        fp = RealFingerprint(rdkit_fp)
        exported = fp.to_rdkit()
        assert exported.GetLength() == len(fp)
        assert exported.GetNonzeroElements() == {
            k: int(v) for k, v in rdkit_fp.GetNonzeroElements().items()
        }

    def test_to_rdkit_class(self, mfpgen, mols):
        from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
        fp = RealFingerprint(mfpgen.GetCountFingerprint(mols[0]))
        assert isinstance(fp.to_rdkit(), UIntSparseIntVect)

