use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = refnd::stub_info()?;
    stub.generate()?;
    std::fs::write(
        "python/refnd/__init__.py",
        "from .refnd import *\nfrom .refnd import core\nfrom .refnd import kernels\nfrom .refnd import utils\n",
    )?;
    patch_utils_stubs();
    Ok(())
}

fn patch_utils_stubs() {
    let path = "python/refnd/utils/__init__.pyi";
    let src = std::fs::read_to_string(path).expect("utils stub not found");

    let src = src
        // ── imports ──────────────────────────────────────────────────────────
        .replacen(
            "import typing\n",
            "import typing\nfrom rdkit.DataStructs.cDataStructs import ExplicitBitVect, UIntSparseIntVect\n",
            1,
        )
        // ── BitFingerprint constructor ────────────────────────────────────────
        .replace(
            "def __new__(cls, fp: typing.Any) -> BitFingerprint:",
            "def __new__(cls, fp: ExplicitBitVect) -> BitFingerprint:",
        )
        // ── BitFingerprint.to_rdkit ───────────────────────────────────────────
        .replace(
            "def to_rdkit(self) -> typing.Any:\n        r\"\"\"\n        Export as an RDKit ``ExplicitBitVect``.",
            "def to_rdkit(self) -> ExplicitBitVect:\n        r\"\"\"\n        Export as an RDKit ``ExplicitBitVect``.",
        )
        // ── RealFingerprint constructor ───────────────────────────────────────
        .replace(
            "def __new__(cls, fp: typing.Any) -> RealFingerprint:",
            "def __new__(cls, fp: UIntSparseIntVect) -> RealFingerprint:",
        )
        // ── RealFingerprint.to_rdkit ──────────────────────────────────────────
        .replace(
            "def to_rdkit(self) -> typing.Any:\n        r\"\"\"\n        Export as an RDKit ``UIntSparseIntVect``,",
            "def to_rdkit(self) -> UIntSparseIntVect:\n        r\"\"\"\n        Export as an RDKit ``UIntSparseIntVect``,",
        )
        // ── BitFingerprint.from_np: Any → NDArray[bool_] ─────────────────────
        .replace(
            "def from_np(arr: typing.Any) -> BitFingerprint:",
            "def from_np(arr: numpy.typing.NDArray[numpy.bool_]) -> BitFingerprint:",
        )
        // ── RealFingerprint.from_np: Any → NDArray[float32] ──────────────────
        .replace(
            "def from_np(arr: typing.Any) -> RealFingerprint:",
            "def from_np(arr: numpy.typing.NDArray[numpy.float32]) -> RealFingerprint:",
        );

    std::fs::write(path, src).expect("failed to write patched utils stub");
}
