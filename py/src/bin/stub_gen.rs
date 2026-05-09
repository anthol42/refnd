use pyo3_stub_gen::Result;
use std::path::Path;

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn main() -> Result<()> {
    let stub = refnd::stub_info()
        .map_err(|e| anyhow::anyhow!("stub_info() failed (ROOT={}): {}", ROOT, e))?;
    stub.generate()
        .map_err(|e| anyhow::anyhow!("stub.generate() failed: {}", e))?;
    let init_py = Path::new(ROOT).join("python/refnd/__init__.py");
    std::fs::write(&init_py, "from .refnd import *\n\
    from .refnd import core\n\
    from .refnd import kernels\n\
    from .refnd import utils\n\
    from refnd.kernels import KernelVariant\n\
    from refnd.core import (HNSWState, LeidenObjective, find_communities, find_components, \n\
            partition, exact_edges, exact_nearest_neighbors)\n")
        .map_err(|e| anyhow::anyhow!("failed to write {}: {}", init_py.display(), e))?;

    let init_pyi = Path::new(ROOT).join("python/refnd/__init__.pyi");
    let existing = std::fs::read_to_string(&init_pyi)
        .map_err(|e| anyhow::anyhow!("failed to read {}: {}", init_pyi.display(), e))?;
    let reexports = "\
from .kernels import KernelVariant as KernelVariant\n\
from .core import HNSWState as HNSWState\n\
from .core import LeidenObjective as LeidenObjective\n\
from .core import find_communities as find_communities\n\
from .core import find_components as find_components\n\
from .core import partition as partition\n\
from .core import exact_edges as exact_edges\n\
from .core import exact_nearest_neighbors as exact_nearest_neighbors\n";
    std::fs::write(&init_pyi, format!("{existing}{reexports}"))
        .map_err(|e| anyhow::anyhow!("failed to write {}: {}", init_pyi.display(), e))?;
    patch_utils_stubs();
    Ok(())
}

fn patch_utils_stubs() {
    let path = Path::new(ROOT).join("python/refnd/utils/__init__.pyi");
    let src = std::fs::read_to_string(&path).expect("utils stub not found");

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

    std::fs::write(&path, src).expect("failed to write patched utils stub");
}
