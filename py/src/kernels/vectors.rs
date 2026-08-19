use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use numpy::PyReadonlyArray1;
use refnd_core::core::Distance;
use refnd_core::kernels::vectors::{Cosine as CoreCosine, L1 as CoreL1, L2 as CoreL2};
use crate::utils::Vector;

impl Distance<Vector> for CoreCosine {
    #[inline(always)]
    fn call(&self, a: &Vector, b: &Vector) -> f32 {
        Distance::<[f32]>::call(self, &a.inner, &b.inner)
    }
}

impl Distance<Vector> for CoreL1 {
    #[inline(always)]
    fn call(&self, a: &Vector, b: &Vector) -> f32 {
        Distance::<[f32]>::call(self, &a.inner, &b.inner)
    }
}

impl Distance<Vector> for CoreL2 {
    #[inline(always)]
    fn call(&self, a: &Vector, b: &Vector) -> f32 {
        Distance::<[f32]>::call(self, &a.inner, &b.inner)
    }
}


// ── Cosine ────────────────────────────────────────────────────────────────────

/// Cosine distance: ``1 - dot(a, b) / (||a|| * ||b||)``. ``0.0`` if either vector
/// is the zero vector.
///
/// Example::
///
///     import numpy as np
///     from refnd.kernels.vectors import Cosine
///
///     a = np.array([1.0, 0.0], dtype=np.float32)
///     b = np.array([0.0, 1.0], dtype=np.float32)
///     k = Cosine()
///     assert k.call(a, b) == k(a, b) == 1.0
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.vectors")]
pub struct Cosine {
    pub inner: CoreCosine,
}

#[gen_stub_pymethods]
#[pymethods]
impl Cosine {
    #[new]
    pub fn new() -> Self { Self { inner: CoreCosine } }

    /// Compute the cosine distance between two 1D ``float32`` numpy arrays.
    ///
    /// Reads directly from each array's buffer -- no copy.
    ///
    /// Args:
    ///     a: First vector.
    ///     b: Second vector.
    ///
    /// Returns:
    ///     Distance in ``[0.0, 2.0]``. ``0.0`` means the same direction, ``2.0``
    ///     means exactly opposite directions.
    pub fn call(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
        Ok(self.inner.call(a.as_slice()?, b.as_slice()?))
    }

    /// Alias for ``call`` — enables ``kernel(a, b)`` syntax.
    pub fn __call__(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
        self.call(a, b)
    }
}

// ── L1 ────────────────────────────────────────────────────────────────────────

/// Manhattan (L1) distance: ``sum(|a_i - b_i|)``.
///
/// Example::
///
///     import numpy as np
///     from refnd.kernels.vectors import L1
///
///     a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
///     b = np.array([4.0, 0.0, 3.0], dtype=np.float32)
///     k = L1()
///     assert k.call(a, b) == k(a, b) == 5.0
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.vectors")]
pub struct L1 {
    pub inner: CoreL1,
}

#[gen_stub_pymethods]
#[pymethods]
impl L1 {
    #[new]
    pub fn new() -> Self { Self { inner: CoreL1 } }

    /// Compute the L1 distance between two 1D ``float32`` numpy arrays.
    ///
    /// Reads directly from each array's buffer -- no copy.
    ///
    /// Args:
    ///     a: First vector.
    ///     b: Second vector.
    ///
    /// Returns:
    ///     Distance in ``[0.0, inf]``. ``0.0`` means identical vectors.
    pub fn call(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
        Ok(self.inner.call(a.as_slice()?, b.as_slice()?))
    }

    /// Alias for ``call`` — enables ``kernel(a, b)`` syntax.
    pub fn __call__(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
        self.call(a, b)
    }
}

// ── L2 ────────────────────────────────────────────────────────────────────────

/// Euclidean (L2) distance: ``sqrt(sum((a_i - b_i)²))``.
///
/// Example::
///
///     import numpy as np
///     from refnd.kernels.vectors import L2
///
///     a = np.array([0.0, 0.0], dtype=np.float32)
///     b = np.array([3.0, 4.0], dtype=np.float32)
///     k = L2()
///     assert k.call(a, b) == k(a, b) == 5.0
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.vectors")]
pub struct L2 {
    pub inner: CoreL2,
}

#[gen_stub_pymethods]
#[pymethods]
impl L2 {
    #[new]
    pub fn new() -> Self { Self { inner: CoreL2 } }

    /// Compute the L2 distance between two 1D ``float32`` numpy arrays.
    ///
    /// Reads directly from each array's buffer -- no copy.
    ///
    /// Args:
    ///     a: First vector.
    ///     b: Second vector.
    ///
    /// Returns:
    ///     Distance in ``[0.0, inf]``. ``0.0`` means identical vectors.
    pub fn call(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
        Ok(self.inner.call(a.as_slice()?, b.as_slice()?))
    }

    /// Alias for ``call`` — enables ``kernel(a, b)`` syntax.
    pub fn __call__(&self, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
        self.call(a, b)
    }
}
