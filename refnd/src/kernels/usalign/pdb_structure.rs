use std::sync::Arc;
use std::ffi::CString;

struct PdbStructureInner {
    ptr: *mut libusalign_sys::PdbStructure,
}

// Safe: USalign only reads from the coordinate arrays; no mutation after construction.
unsafe impl Send for PdbStructureInner {}
unsafe impl Sync for PdbStructureInner {}

impl Drop for PdbStructureInner {
    fn drop(&mut self) {
        unsafe { libusalign_sys::usalign_free_pdb(self.ptr) };
    }
}

/// A protein structure loaded from a PDB file and held in memory.
///
/// Cheap to clone (Arc). Thread-safe (Sync). The underlying coordinate arrays
/// are freed when the last clone is dropped.
#[derive(Clone)]
pub struct PdbStructure(Arc<PdbStructureInner>);

impl PdbStructure {
    /// Parse `path` into memory. Fails if the file is absent or has fewer than 3 residues.
    pub fn load(path: &str) -> Result<Self, String> {
        let cpath = CString::new(path)
            .map_err(|_| format!("path contains null byte: {path}"))?;
        let ptr = unsafe { libusalign_sys::usalign_load_pdb(cpath.as_ptr()) };
        if ptr.is_null() {
            Err(format!("failed to load PDB: {path}"))
        } else {
            Ok(Self(Arc::new(PdbStructureInner { ptr })))
        }
    }

    #[inline]
    pub(crate) fn as_ptr(&self) -> *const libusalign_sys::PdbStructure {
        self.0.ptr
    }
}
