use pyo3::prelude::*;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pymethods};
use refnd_core::core::Distance;
use refnd_core::kernels::protspam::{ProtSpamDistance as CoreProtSpamDistance, ProtSpamKernel as CoreProtSpamKernel};
use crate::utils::{SWPatternSet, SWSequence};

impl Distance<SWSequence> for CoreProtSpamKernel {
    #[inline(always)]
    fn call(&self, a: &SWSequence, b: &SWSequence) -> f32 {
        Distance::<refnd_core::utils::SWSequence>::call(self, &a.inner, &b.inner)
    }
}

/// Which distance ``ProtSpamKernel.call`` reports.
///
/// - ``MismatchRate``: the raw mismatch rate itself (fraction of don't-care
///   positions that mismatch, pooled across every pattern's accepted spaced-word
///   matches) -- always in ``[0, 1]``. ``1.0`` if no spaced word matched at all, or
///   if every accepted match's don't-care positions mismatched outright.
/// - ``Evolutionary``: the mismatch rate corrected into a distance via the Kimura
///   two-parameter formula -- in ``[0, inf]``. ``inf`` if the mismatch rate is
///   undefined (no match at all) or high enough (``> ~0.8541``) that the
///   correction's log argument is non-positive.
#[gen_stub_pyclass_enum]
#[pyclass(eq, eq_int, from_py_object, module = "refnd.kernels.protspam")]
#[derive(Clone, Copy, PartialEq)]
pub enum ProtSpamDistance {
    MismatchRate,
    Evolutionary,
}

impl From<ProtSpamDistance> for CoreProtSpamDistance {
    fn from(d: ProtSpamDistance) -> Self {
        match d {
            ProtSpamDistance::MismatchRate => CoreProtSpamDistance::MismatchRate,
            ProtSpamDistance::Evolutionary => CoreProtSpamDistance::Evolutionary,
        }
    }
}

impl From<CoreProtSpamDistance> for ProtSpamDistance {
    fn from(d: CoreProtSpamDistance) -> Self {
        match d {
            CoreProtSpamDistance::MismatchRate => ProtSpamDistance::MismatchRate,
            CoreProtSpamDistance::Evolutionary => ProtSpamDistance::Evolutionary,
        }
    }
}

/// Distance between two ``SWSequence``s, ProtSpaM-style: for each pattern in a
/// shared ``SWPatternSet``, match spaced words by key (grouping ties into "blocks"),
/// score the best-aligned spaced word pair within each matching block against
/// BLOSUM62, and pool mismatches at don't-care positions into a mismatch rate.
/// Reports either that raw rate or a Kimura-corrected evolutionary distance, per
/// ``distance``.
///
/// **Every ``SWSequence`` passed to ``call`` must have been built from the same
/// ``SWPatternSet`` as this kernel's ``patterns()``** (or an equal copy of it).
///
/// Example::
///
///     from refnd.kernels.protspam import ProtSpamKernel, ProtSpamDistance
///     from refnd.utils import SWPatternSet, SWSequence
///
///     sequences = [
///         "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
///         "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEA",
///     ]
///
///     # A RasBhari-optimized pattern set, shared by every SWSequence and the kernel.
///     patterns = SWPatternSet(5, 6, 20)
///
///     swseqs = [SWSequence(s, patterns) for s in sequences]
///
///     kernel = ProtSpamKernel(patterns, distance=ProtSpamDistance.Evolutionary)
///     distance = kernel.call(swseqs[0], swseqs[1])
///     distance = kernel(swseqs[0], swseqs[1])  # equivalent -- kernel(a, b) syntax
///     assert distance >= 0.0
#[gen_stub_pyclass]
#[pyclass(module = "refnd.kernels.protspam")]
pub struct ProtSpamKernel {
    pub inner: CoreProtSpamKernel,
}

#[gen_stub_pymethods]
#[pymethods]
impl ProtSpamKernel {
    /// Build a kernel over ``patterns`` -- also the pattern set every ``SWSequence``
    /// passed to ``call`` must have been built from.
    ///
    /// Args:
    ///     patterns: The shared pattern set.
    ///     significance_threshold: Minimum BLOSUM62 score (over don't-care
    ///         positions) for a spaced-word match to be considered homologous.
    ///         ProtSpaM's own default is ``0``.
    ///     distance: Which value ``call`` reports. Default ``ProtSpamDistance.Evolutionary``.
    #[new]
    #[pyo3(signature = (patterns, significance_threshold = 0, distance = ProtSpamDistance::Evolutionary))]
    pub fn new(patterns: &SWPatternSet, significance_threshold: i32, distance: ProtSpamDistance) -> Self {
        Self { inner: CoreProtSpamKernel::new(patterns.inner.clone(), significance_threshold, distance.into()) }
    }

    /// Compute the distance between ``ref_sample`` and ``query``, per this kernel's
    /// ``distance()`` mode.
    ///
    /// Args:
    ///     ref_sample: Reference sequence. Must have been built from the same
    ///         ``SWPatternSet`` as this kernel's ``patterns()``.
    ///     query: Query sequence. Same requirement.
    ///
    /// Returns:
    ///     The distance -- see ``ProtSpamDistance``.
    ///
    /// Raises:
    ///     IndexError: If ``ref_sample``/``query`` weren't built from the same
    ///         ``SWPatternSet`` as this kernel's ``patterns()`` and the mismatched
    ///         set has fewer patterns -- or, if the mismatched set merely has a
    ///         different pattern at the same index, this can silently compare the
    ///         wrong patterns' spaced words against each other instead, with no
    ///         error at all.
    pub fn call(&self, ref_sample: &SWSequence, query: &SWSequence) -> f32 {
        self.inner.call(&ref_sample.inner, &query.inner)
    }

    /// Alias for ``call`` -- enables ``kernel(a, b)`` syntax.
    pub fn __call__(&self, ref_sample: &SWSequence, query: &SWSequence) -> f32 {
        self.call(ref_sample, query)
    }

    /// The shared pattern set spaced words are matched against.
    pub fn patterns(&self) -> SWPatternSet {
        SWPatternSet { inner: self.inner.patterns.clone() }
    }

    /// Minimum BLOSUM62 score for a spaced-word match to be considered homologous.
    pub fn significance_threshold(&self) -> i32 {
        self.inner.significance_threshold
    }

    /// Which value ``call`` reports.
    pub fn distance(&self) -> ProtSpamDistance {
        self.inner.distance.into()
    }
}
