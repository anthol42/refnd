mod matrix;
mod aligner_config;
mod global;
mod local;
mod meta;

pub use aligner_config::{DatatypeWidth, AlignerMatrix, VectorizationStrategy, AlignerConfigTrait};
pub use matrix::BundledMatrix;
pub use global::{GlobalAlignerBuilder, GlobalAligner, GlobalIdentityMode};
pub use local::{LocalAlignerBuilder, LocalAligner, LocalIdentityMode, CoverageMode};
pub use meta::ProteinKernel;
