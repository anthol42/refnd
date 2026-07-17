use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3_stub_gen::derive::gen_stub_pyfunction;
use rayon::prelude::*;
use indicatif::ProgressBar;
use refnd_core::core::Distance;
use crate::kernels::{
    KernelVariant,
    alignments::{GlobalAligner, LocalAligner},
    molecules::{TanimotoBit, TanimotoReal},
    structures::USAlignKernel,
};
use crate::utils::{BitFingerprint, RealFingerprint, PdbStructure};
use crate::core::_utils::linear_progress_bar;

fn compute<T, K>(data1: Vec<T>, data2: Vec<T>, kernel: K, n_threads: usize, pb: Option<&ProgressBar>) -> Vec<f32>
where
    T: Sync,
    K: Distance<T> + Sync,
{
    let num_threads = if n_threads == 0 { rayon::current_num_threads() } else { n_threads };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build rayon thread pool");
    pool.install(|| {
        data1.par_iter()
            .zip(data2.par_iter())
            .map(|(a, b)| {
                let score = kernel.call(a, b);
                if let Some(pb) = pb { pb.inc(1); }
                score
            })
            .collect()
    })
}

macro_rules! dispatch {
    ($py:expr, $T:ty, $pykernel:ty, $args:expr, $kwargs:expr;
     $data1:expr, $data2:expr; $n_threads:expr, $pb:expr) => {{
        let obj = $py.get_type::<$pykernel>().call($args, $kwargs)?;
        let r: ::pyo3::PyRef<$pykernel> = obj.extract()?;
        let kernel = r.inner.clone();
        let d1 = $data1.extract::<Vec<$T>>($py)?;
        let d2 = $data2.extract::<Vec<$T>>($py)?;
        compute(d1, d2, kernel, $n_threads, $pb)
    }};
}

/// Compute pairwise kernel scores between two equal-length sequences of data items in parallel.
///
/// Evaluates ``kernel(data1[i], data2[i])`` for each index ``i`` and returns the scores
/// as a flat ``list[float]`` of length ``len(data1)``.
///
/// Extra positional and keyword arguments are forwarded to the kernel constructor.
///
/// Args:
///     variant: Which kernel to use.
///     data1: First sequence of items.
///     data2: Second sequence of items. Must have the same length as ``data1``.
///     n_threads: Number of parallel threads. ``0`` uses all available cores.
///     progress: Show a progress bar. Defaults to ``True``.
///
/// Returns:
///     A ``list[float]`` of length ``len(data1)`` where ``result[i] = kernel(data1[i], data2[i])``.
///
/// Raises:
///     ValueError: If ``data1`` and ``data2`` have different lengths.
///
/// Example::
///
///     from refnd import KernelVariant
///     from refnd.kernels import zip_kernel
///
///     seqs1 = ["MKTAYIAK", "ACDEFGHIKLM"]
///     seqs2 = ["MKTAYIAKQR", "ACDEF"]
///     scores = zip_kernel(KernelVariant.AlignmentGlobal, seqs1, seqs2)
///     # scores[i] == kernel(seqs1[i], seqs2[i])
#[gen_stub_pyfunction(module = "refnd.kernels")]
#[pyfunction]
#[pyo3(signature = (variant, data1, data2, n_threads = 0, progress = true, *args, **kwargs))]
pub fn zip_kernel(
    py: Python,
    variant: KernelVariant,
    data1: Py<PyAny>,
    data2: Py<PyAny>,
    n_threads: usize,
    progress: bool,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Vec<f32>> {
    let n1 = data1.bind(py).len()?;
    let n2 = data2.bind(py).len()?;
    if n1 != n2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "data1 and data2 must have the same length, got {} and {}", n1, n2
        )));
    }
    let pb = if progress {
        let pb = linear_progress_bar(n1, "Computing kernel");
        pb.set_length(n1 as u64);
        Some(pb)
    } else {
        None
    };

    let result = match variant {
        KernelVariant::AlignmentGlobal => dispatch!(
            py, String, GlobalAligner, args, kwargs; data1, data2; n_threads, pb.as_ref()),
        KernelVariant::AlignmentLocal => dispatch!(
            py, String, LocalAligner, args, kwargs; data1, data2; n_threads, pb.as_ref()),
        KernelVariant::TanimotoBit => dispatch!(
            py, BitFingerprint, TanimotoBit, args, kwargs; data1, data2; n_threads, pb.as_ref()),
        KernelVariant::TanimotoReal => dispatch!(
            py, RealFingerprint, TanimotoReal, args, kwargs; data1, data2; n_threads, pb.as_ref()),
        KernelVariant::Structure => dispatch!(
            py, PdbStructure, USAlignKernel, args, kwargs; data1, data2; n_threads, pb.as_ref()),
    };

    if let Some(ref pb) = pb { pb.finish(); }
    Ok(result)
}
