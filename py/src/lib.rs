use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
define_stub_info_gatherer!(stub_info);

mod kernels;
pub mod core;
pub mod utils;

#[pymodule]
mod refnd {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let sys = m.py().import("sys")?;
        let modules = sys.getattr("modules")?;
        let core = m.getattr("core")?;
        modules.set_item("refnd.core", &core)?;
        let utils = m.getattr("utils")?;
        modules.set_item("refnd.utils", &utils)?;
        let kernels = m.getattr("kernels")?;
        let protein = kernels.getattr("protein")?;
        let sequence = protein.getattr("sequence")?;
        modules.set_item("refnd.kernels", &kernels)?;
        modules.set_item("refnd.kernels.protein", &protein)?;
        modules.set_item("refnd.kernels.protein.sequence", &sequence)?;
        let molecules = kernels.getattr("molecules")?;
        modules.set_item("refnd.kernels.molecules", &molecules)?;
        Ok(())
    }

    #[pymodule]
    mod core {
        #[pymodule_export]
        use crate::core::hnsw::HNSWState;
        #[pymodule_export]
        use crate::core::hnsw::HNSWConfig;
        #[pymodule_export]
        use crate::core::hnsw::HNSWIndex;
        #[pymodule_export]
        use crate::core::exact::exact_edges;
        #[pymodule_export]
        use crate::core::exact::exact_nearest_neighbors;
        #[pymodule_export]
        use crate::core::leiden::CsrGraph;
        #[pymodule_export]
        use crate::core::leiden::LeidenObjective;
        #[pymodule_export]
        use crate::core::leiden::find_communities;
        #[pymodule_export]
        use crate::core::edge_store::EdgeStore;
        #[pymodule_export]
        use crate::core::edge_store::EdgeStoreIter;
        #[pymodule_export]
        use crate::core::functional::partition;
        #[pymodule_export]
        use crate::core::functional::find_components;
    }

    #[pymodule]
    mod utils {
        #[pymodule_export]
        use crate::utils::{BitFingerprint, RealFingerprint, read_fasta, largest_cluster};
    }

    #[pymodule]
    mod kernels {
        use pyo3::prelude::*;
        #[pymodule_export]
        use crate::kernels::KernelVariant;
        
        #[pymodule]
        mod molecules {
            #[pymodule_export]
            use crate::kernels::molecules::{TanimotoBit, TanimotoReal};
        }

        #[pymodule]
        mod protein {
            use pyo3::prelude::*;

            #[pymodule]
            mod sequence {

                #[pymodule_export]
                use crate::kernels::protein::sequence::{
                    GlobalAligner, LocalAligner, ScoringMatrix, GlobalIdentityMode,
                    VectorizationStrategy, DatatypeWidth, CoverageMode, LocalIdentityMode
                };
            }
        }
    }
}
