use super::{GlobalAligner, LocalAligner};
use crate::core::Distance;

#[derive(Clone)]
pub enum ProteinKernel {
    Global(GlobalAligner),
    Local(LocalAligner),
}

impl Distance<str> for ProteinKernel {
    fn call(&self, ref_sample: &str, query: &str) -> f32 {
        match self {
            ProteinKernel::Global(a) => a.call(ref_sample, query),
            ProteinKernel::Local(a) => a.call(ref_sample, query),
        }
    }
}

impl Distance<String> for ProteinKernel {
    fn call(&self, ref_sample: &String, query: &String) -> f32 {
        Distance::<str>::call(self, ref_sample.as_str(), query.as_str())
    }
}
