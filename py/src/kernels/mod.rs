use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass_enum;

pub mod protein;
pub mod molecules;

#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels")]
#[derive(Clone, Copy, PartialEq)]
pub enum KernelVariant {
    ProteinGlobal,
    ProteinLocal,
    TanimotoBit,
    TanimotoReal
}
