use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass_enum;

pub mod alignments;
pub mod molecules;
pub mod structures;
pub mod zip_kernel;

#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels")]
#[derive(Clone, Copy, PartialEq)]
pub enum KernelVariant {
    AlignmentGlobal,
    AlignmentLocal,
    TanimotoBit,
    TanimotoReal,
    Structure,
}
