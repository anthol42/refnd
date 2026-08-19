# refnd.kernels

## Kernel variant selector

### *class* refnd.kernels.KernelVariant

Bases: `object`

#### AlignmentGlobal *= KernelVariant.AlignmentGlobal*

#### AlignmentLocal *= KernelVariant.AlignmentLocal*

#### Cosine *= KernelVariant.Cosine*

#### L1 *= KernelVariant.L1*

#### L2 *= KernelVariant.L2*

#### ProtSpam *= KernelVariant.ProtSpam*

#### Structure *= KernelVariant.Structure*

#### TanimotoBit *= KernelVariant.TanimotoBit*

#### TanimotoReal *= KernelVariant.TanimotoReal*

### refnd.kernels.zip_kernel(variant: [KernelVariant](#refnd.kernels.KernelVariant), data1: Any, data2: Any, n_threads: int = 0, progress: bool = True, \*args: Any, \*\*kwargs: Any) → list[float]

Compute pairwise kernel scores between two equal-length sequences of data items in parallel.

Evaluates `kernel(data1[i], data2[i])` for each index `i` and returns the scores
as a flat `list[float]` of length `len(data1)`.

Extra positional and keyword arguments are forwarded to the kernel constructor.

* **Parameters:**
  * **variant** – Which kernel to use.
  * **data1** – First sequence of items.
  * **data2** – Second sequence of items. Must have the same length as `data1`.
  * **n_threads** – Number of parallel threads. `0` uses all available cores.
  * **progress** – Show a progress bar. Defaults to `True`.
* **Returns:**
  A `list[float]` of length `len(data1)` where `result[i] = kernel(data1[i], data2[i])`.
* **Raises:**
  **ValueError** – If `data1` and `data2` have different lengths.

Example:

```default
from refnd import KernelVariant
from refnd.kernels import zip_kernel

seqs1 = ["MKTAYIAK", "ACDEFGHIKLM"]
seqs2 = ["MKTAYIAKQR", "ACDEF"]
scores = zip_kernel(KernelVariant.AlignmentGlobal, seqs1, seqs2)
# scores[i] == kernel(seqs1[i], seqs2[i])
```

## refnd.kernels.alignments

Protein or Nucleotide sequence aligners and their configuration enums.

<a id="module-refnd.kernels.alignments"></a>

### *class* refnd.kernels.alignments.GlobalAligner(gap_open: int = 11, gap_extend: int = 1, matrix: [ScoringMatrix](#refnd.kernels.alignments.ScoringMatrix) = ScoringMatrix.Blosum62, identity_mode: [GlobalIdentityMode](#refnd.kernels.alignments.GlobalIdentityMode) = GlobalIdentityMode.MaxLength, vectorization: [VectorizationStrategy](#refnd.kernels.alignments.VectorizationStrategy) = VectorizationStrategy.Scan, width: [DatatypeWidth](#refnd.kernels.alignments.DatatypeWidth) = DatatypeWidth.Sat)

Bases: `object`

Needleman–Wunsch global sequence aligner returning a normalised identity score.

Wraps the parasail SIMD alignment library. The identity score is computed as
the number of identical aligned positions divided by the denominator selected
by `identity_mode`.

`GlobalAligner` is used as a kernel in `HNSWState`, `exact_edges`, and
`exact_nearest_neighbors` via `KernelVariant.AlignmentGlobal`.

* **Parameters:**
  * **gap_open** – Affine gap-open penalty (positive integer, subtracted). Default `11`.
  * **gap_extend** – Affine gap-extend penalty (positive integer, subtracted). Default `1`.
  * **matrix** – Amino-acid substitution matrix. Default `ScoringMatrix.Blosum62`.
  * **identity_mode** – Normalisation denominator. Default `GlobalIdentityMode.MaxLength`.
  * **vectorization** – SIMD layout. Default `VectorizationStrategy.Scan`.
  * **width** – Integer precision. Default `DatatypeWidth.Sat`.

Example:

```default
from refnd.kernels.alignments import GlobalAligner

aligner = GlobalAligner(gap_open=11, gap_extend=1)
score = aligner.call("MKTAYIAK", "MKTAYIAKQR")
score = aligner("MKTAYIAK", "MKTAYIAKQR") # Alternative
# score in [0.0, 1.0]
```

#### call(ref_sample: str, query: str) → float

Compute the global alignment identity between two sequences.

* **Parameters:**
  * **ref_sample** – Reference sequence (single-letter amino acid codes or nucleotides).
  * **query** – Query sequence.
* **Returns:**
  Identity score in `[0.0, 1.0]`.

### *class* refnd.kernels.alignments.LocalAligner(gap_open: int = 11, gap_extend: int = 1, min_coverage: float = 0.800000011920929, cov_mode: [CoverageMode](#refnd.kernels.alignments.CoverageMode) = CoverageMode.BothQueryTarget, matrix: [ScoringMatrix](#refnd.kernels.alignments.ScoringMatrix) = ScoringMatrix.Blosum62, identity_mode: [LocalIdentityMode](#refnd.kernels.alignments.LocalIdentityMode) = LocalIdentityMode.AlignmentLength, vectorization: [VectorizationStrategy](#refnd.kernels.alignments.VectorizationStrategy) = VectorizationStrategy.Striped, width: [DatatypeWidth](#refnd.kernels.alignments.DatatypeWidth) = DatatypeWidth.Sat)

Bases: `object`

Smith–Waterman local sequence aligner returning a normalised identity score.

Like `GlobalAligner` but aligns only the most similar sub-region of each
sequence. Pairs that do not meet the `min_coverage` criterion after alignment
receive a score of `0.0`.

`LocalAligner` is used as a kernel via `KernelVariant.AlignmentLocal`.

* **Parameters:**
  * **gap_open** – Affine gap-open penalty. Default `11`.
  * **gap_extend** – Affine gap-extend penalty. Default `1`.
  * **min_coverage** – Minimum fraction of sequence covered by the local alignment
    (per `cov_mode`) for the pair to be accepted. Default `0.8`.
  * **cov_mode** – Which sequence(s) must meet `min_coverage`. Default
    `CoverageMode.BothQueryTarget`.
  * **matrix** – Substitution matrix. Default `ScoringMatrix.Blosum62`.
  * **identity_mode** – Normalisation denominator. Default `LocalIdentityMode.AlignmentLength`.
  * **vectorization** – SIMD layout. Default `VectorizationStrategy.Striped`.
  * **width** – Integer precision. Default `DatatypeWidth.Sat`.

Example:

```default
from refnd.kernels.alignments import LocalAligner, CoverageMode

aligner = LocalAligner(min_coverage=0.5, cov_mode=CoverageMode.Query)
score = aligner.call("ACDEFGHIKLM", "CDEFGHI")
score = aligner("ACDEFGHIKLM", "CDEFGHI") # Alternative
```

#### call(ref_sample: str, query: str) → float

Compute the local alignment identity between two sequences.

* **Parameters:**
  * **ref_sample** – Reference alignments sequence (single-letter amino acid codes).
  * **query** – Query alignments sequence.
* **Returns:**
  Identity score in `[0.0, 1.0]`, or `1.0` if the coverage filter fails.

### *class* refnd.kernels.alignments.ScoringMatrix

Bases: `object`

#### Blosum100 *= ScoringMatrix.Blosum100*

#### Blosum30 *= ScoringMatrix.Blosum30*

#### Blosum35 *= ScoringMatrix.Blosum35*

#### Blosum40 *= ScoringMatrix.Blosum40*

#### Blosum45 *= ScoringMatrix.Blosum45*

#### Blosum50 *= ScoringMatrix.Blosum50*

#### Blosum55 *= ScoringMatrix.Blosum55*

#### Blosum60 *= ScoringMatrix.Blosum60*

#### Blosum62 *= ScoringMatrix.Blosum62*

#### Blosum65 *= ScoringMatrix.Blosum65*

#### Blosum70 *= ScoringMatrix.Blosum70*

#### Blosum75 *= ScoringMatrix.Blosum75*

#### Blosum80 *= ScoringMatrix.Blosum80*

#### Blosum85 *= ScoringMatrix.Blosum85*

#### Blosum90 *= ScoringMatrix.Blosum90*

#### Blosum95 *= ScoringMatrix.Blosum95*

#### Dnafull *= ScoringMatrix.Dnafull*

#### Identity *= ScoringMatrix.Identity*

#### Nuc44 *= ScoringMatrix.Nuc44*

#### Pam10 *= ScoringMatrix.Pam10*

#### Pam100 *= ScoringMatrix.Pam100*

#### Pam110 *= ScoringMatrix.Pam110*

#### Pam120 *= ScoringMatrix.Pam120*

#### Pam130 *= ScoringMatrix.Pam130*

#### Pam140 *= ScoringMatrix.Pam140*

#### Pam150 *= ScoringMatrix.Pam150*

#### Pam160 *= ScoringMatrix.Pam160*

#### Pam170 *= ScoringMatrix.Pam170*

#### Pam180 *= ScoringMatrix.Pam180*

#### Pam190 *= ScoringMatrix.Pam190*

#### Pam20 *= ScoringMatrix.Pam20*

#### Pam200 *= ScoringMatrix.Pam200*

#### Pam210 *= ScoringMatrix.Pam210*

#### Pam220 *= ScoringMatrix.Pam220*

#### Pam230 *= ScoringMatrix.Pam230*

#### Pam240 *= ScoringMatrix.Pam240*

#### Pam250 *= ScoringMatrix.Pam250*

#### Pam260 *= ScoringMatrix.Pam260*

#### Pam270 *= ScoringMatrix.Pam270*

#### Pam280 *= ScoringMatrix.Pam280*

#### Pam290 *= ScoringMatrix.Pam290*

#### Pam30 *= ScoringMatrix.Pam30*

#### Pam300 *= ScoringMatrix.Pam300*

#### Pam310 *= ScoringMatrix.Pam310*

#### Pam320 *= ScoringMatrix.Pam320*

#### Pam330 *= ScoringMatrix.Pam330*

#### Pam340 *= ScoringMatrix.Pam340*

#### Pam350 *= ScoringMatrix.Pam350*

#### Pam360 *= ScoringMatrix.Pam360*

#### Pam370 *= ScoringMatrix.Pam370*

#### Pam380 *= ScoringMatrix.Pam380*

#### Pam390 *= ScoringMatrix.Pam390*

#### Pam40 *= ScoringMatrix.Pam40*

#### Pam400 *= ScoringMatrix.Pam400*

#### Pam410 *= ScoringMatrix.Pam410*

#### Pam420 *= ScoringMatrix.Pam420*

#### Pam430 *= ScoringMatrix.Pam430*

#### Pam440 *= ScoringMatrix.Pam440*

#### Pam450 *= ScoringMatrix.Pam450*

#### Pam460 *= ScoringMatrix.Pam460*

#### Pam470 *= ScoringMatrix.Pam470*

#### Pam480 *= ScoringMatrix.Pam480*

#### Pam490 *= ScoringMatrix.Pam490*

#### Pam50 *= ScoringMatrix.Pam50*

#### Pam500 *= ScoringMatrix.Pam500*

#### Pam60 *= ScoringMatrix.Pam60*

#### Pam70 *= ScoringMatrix.Pam70*

#### Pam80 *= ScoringMatrix.Pam80*

#### Pam90 *= ScoringMatrix.Pam90*

### *class* refnd.kernels.alignments.GlobalIdentityMode

Bases: `object`

Denominator used to normalise a global-alignments identity score.

After counting identical aligned positions the raw count is divided by:

- `AlignmentLength`: the total length of the alignment (including gaps).
- `MaxSeqLength`: the length of the longer of the two sequences.
- `MinSeqLength`: the length of the shorter of the two sequences.
- `MaxLength` (default): same as `MaxSeqLength` — recommended for RGP datasets.

#### AlignmentLength *= GlobalIdentityMode.AlignmentLength*

#### MaxLength *= GlobalIdentityMode.MaxLength*

#### MaxSeqLength *= GlobalIdentityMode.MaxSeqLength*

#### MinSeqLength *= GlobalIdentityMode.MinSeqLength*

### *class* refnd.kernels.alignments.VectorizationStrategy

Bases: `object`

SIMD vectorization layout used by the parasail alignment engine.

- `Striped` (default for local): interleaved layout, best for short sequences.
- `Scan`: sequential scan layout, often faster for long sequences or global alignment.
- `Diag`: diagonal layout; niche use-case, rarely needed.

In practice the default per-aligner is a good choice; change only if profiling
shows a bottleneck.

#### Diag *= VectorizationStrategy.Diag*

#### Scan *= VectorizationStrategy.Scan*

#### Striped *= VectorizationStrategy.Striped*

### *class* refnd.kernels.alignments.DatatypeWidth

Bases: `object`

Integer precision used for alignment score accumulation.

- `Short` (8-bit), `Half` (16-bit), `Full` (32-bit), `Long` (64-bit):
  fixed-width integers — lower width is faster but can overflow on long sequences.
- `Sat` (default): 8-bit saturating arithmetic; If it saturates, silently restart with 16-bit.

#### Full *= DatatypeWidth.Full*

#### Half *= DatatypeWidth.Half*

#### Long *= DatatypeWidth.Long*

#### Sat *= DatatypeWidth.Sat*

#### Short *= DatatypeWidth.Short*

### *class* refnd.kernels.alignments.CoverageMode

Bases: `object`

Coverage filter applied before accepting a local alignment as valid.

A pair is scored only when the alignment covers enough of the sequences
as specified by the mode and `min_coverage` threshold:

- `BothQueryTarget` (default): both query and target must meet `min_coverage`.
- `Target`: only the target must meet `min_coverage`.
- `Query`: only the query must meet `min_coverage`.
- `LengthRatio`: the shorter / longer length ratio must meet `min_coverage`.
- `ShorterSeq`: coverage computed relative to the shorter sequence.

#### BothQueryTarget *= CoverageMode.BothQueryTarget*

#### LengthRatio *= CoverageMode.LengthRatio*

#### Query *= CoverageMode.Query*

#### ShorterSeq *= CoverageMode.ShorterSeq*

#### Target *= CoverageMode.Target*

### *class* refnd.kernels.alignments.LocalIdentityMode

Bases: `object`

Denominator used to normalise a local-alignment identity score.

- `AlignmentLength` (default): divide by the length of the local alignment.
- `MinSeqLength`: divide by the shorter sequence length.

#### AlignmentLength *= LocalIdentityMode.AlignmentLength*

#### MinSeqLength *= LocalIdentityMode.MinSeqLength*

## refnd.kernels.molecules

Molecules Tanimoto kernels.

<a id="module-refnd.kernels.molecules"></a>

### *class* refnd.kernels.molecules.TanimotoBit

Bases: `object`

Tanimoto distance kernel for binary (bit) molecular fingerprints.

Measures structural dissimilarity between two molecules as the complement of
the Jaccard index over their feature sets. A distance of 0 means the two
fingerprints are identical; 1 means they share no features at all.

Formula: `1 - |A ∩ B| / |A ∪ B|`

This kernel is the standard choice when working with `ExplicitBitVect`
fingerprints such as Morgan, RDKit, or MACCS keys.

Example:

```default
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from refnd.utils import BitFingerprint
from refnd.kernels.molecules import TanimotoBit

mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
benzene    = BitFingerprint(mfpgen.GetFingerprint(MolFromSmiles("c1ccccc1")))
naphthalene = BitFingerprint(mfpgen.GetFingerprint(MolFromSmiles("c1ccc2ccccc2c1")))
acetic_acid = BitFingerprint(mfpgen.GetFingerprint(MolFromSmiles("CC(=O)O")))

k = TanimotoBit()
print(k(benzene, naphthalene))  # low  — structurally similar
print(k(benzene, acetic_acid))  # high — structurally dissimilar
```

#### call(a: [utils.BitFingerprint](utils.md#refnd.utils.BitFingerprint), b: [utils.BitFingerprint](utils.md#refnd.utils.BitFingerprint)) → float

Compute the Tanimoto distance between two `BitFingerprint` objects.

* **Parameters:**
  * **a** – First fingerprint.
  * **b** – Second fingerprint.
* **Returns:**
  Distance in `[0.0, 1.0]`. `0.0` means identical feature sets,
  `1.0` means fully disjoint.

Example:

```default
from refnd.utils import BitFingerprint
from refnd.kernels.molecules import TanimotoBit

fp1 = BitFingerprint.from_list([True, False, True, True])
fp2 = BitFingerprint.from_list([True, True,  True, False])
k = TanimotoBit()
assert k.call(fp1, fp2) == k(fp1, fp2)  # both forms are equivalent
# intersection={0,2}=2, union={0,1,2,3}=4 → distance = 0.5
```

### *class* refnd.kernels.molecules.TanimotoReal

Bases: `object`

Tanimoto distance kernel for real-valued (count) molecular fingerprints.

Generalises the binary Tanimoto to continuous feature vectors using the
dot-product formulation. A distance of 0 means the two fingerprints are
proportional; values approach 1 as the vectors become orthogonal.

Formula: `1 - dot(a, b) / (||a||² + ||b||² - dot(a, b))`

This kernel is the standard choice when working with count fingerprints
such as those returned by `GetCountFingerprint` (Morgan counts, etc.).

Example:

```default
from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator
from refnd.utils import RealFingerprint
from refnd.kernels.molecules import TanimotoReal

mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
benzene     = RealFingerprint(mfpgen.GetCountFingerprint(MolFromSmiles("c1ccccc1")))
naphthalene = RealFingerprint(mfpgen.GetCountFingerprint(MolFromSmiles("c1ccc2ccccc2c1")))
acetic_acid = RealFingerprint(mfpgen.GetCountFingerprint(MolFromSmiles("CC(=O)O")))

k = TanimotoReal()
print(k(benzene, naphthalene))  # low  — structurally similar
print(k(benzene, acetic_acid))  # high — structurally dissimilar
```

#### call(a: [utils.RealFingerprint](utils.md#refnd.utils.RealFingerprint), b: [utils.RealFingerprint](utils.md#refnd.utils.RealFingerprint)) → float

Compute the Tanimoto distance between two `RealFingerprint` objects.

* **Parameters:**
  * **a** – First fingerprint.
  * **b** – Second fingerprint.
* **Returns:**
  Distance in `[0.0, 1.0]`. `0.0` means identical (proportional)
  feature vectors, `1.0` means fully orthogonal.

Example:

```default
from refnd.utils import RealFingerprint
from refnd.kernels.molecules import TanimotoReal

fp1 = RealFingerprint.from_list([1.0, 0.0, 1.0])
fp2 = RealFingerprint.from_list([0.0, 1.0, 1.0])
k = TanimotoReal()
assert k.call(fp1, fp2) == k(fp1, fp2)  # both forms are equivalent
# dot=1, ||fp1||²=2, ||fp2||²=2 → distance = 1 - 1/(2+2-1) ≈ 0.667
```

## refnd.kernels.structures

Protein structure kernels (USalign TM-score).

<a id="module-refnd.kernels.structures"></a>

### *class* refnd.kernels.structures.USAlignKernel(norm_mode: [NormMode](#refnd.kernels.structures.NormMode) = NormMode.Min, fast: bool = True)

Bases: `object`

Protein structure comparison kernel using USalign TM-score.

Operates on pre-loaded `PdbStructure` objects (from `refnd.utils`).
Returns `1.0 - TM-score` so that identical structures have distance 0
and unrelated structures have distance approaching 1.

Example:

```default
from refnd.utils import PdbStructure
from refnd.kernels.structures import USAlignKernel, NormMode

s1 = PdbStructure("1abc.pdb")
s2 = PdbStructure("1xyz.pdb")
k  = USAlignKernel()                  # default: NormMode.Min
d  = k(s1, s2)                         # float in [0, 1]
```

#### call(a: [utils.PdbStructure](utils.md#refnd.utils.PdbStructure), b: [utils.PdbStructure](utils.md#refnd.utils.PdbStructure)) → float

Compute the structure distance between two `PdbStructure` objects.

* **Parameters:**
  * **a** – First structure.
  * **b** – Second structure.
* **Returns:**
  Distance in `[0.0, 1.0]`. `0.0` means structurally identical,
  values near `1.0` indicate unrelated folds.

### *class* refnd.kernels.structures.NormMode

Bases: `object`

TM-score normalization strategy.

USalign returns two scores per alignment:

- `TM1`: normalized by the length of the first structure (query).
- `TM2`: normalized by the length of the second structure (target).
- `Min` (default): `min(TM1, TM2)` — conservative, symmetric.
- `Query`: use `TM1` only.
- `Target`: use `TM2` only.

#### Min *= NormMode.Min*

#### Query *= NormMode.Query*

#### Target *= NormMode.Target*

## refnd.kernels.protspam

ProtSpaM-style spaced-word kernel for protein sequences.

<a id="module-refnd.kernels.protspam"></a>

### *class* refnd.kernels.protspam.ProtSpamKernel(patterns: [utils.SWPatternSet](utils.md#refnd.utils.SWPatternSet), significance_threshold: int = 0, distance: [ProtSpamDistance](#refnd.kernels.protspam.ProtSpamDistance) = ProtSpamDistance.Evolutionary)

Bases: `object`

Distance between two `SWSequence`, ProtSpaM-style: for each pattern in a
shared `SWPatternSet`, match spaced words by key (grouping ties into “blocks”),
score the best-aligned spaced word pair within each matching block against
BLOSUM62, and pool mismatches at don’t-care positions into a mismatch rate.
Reports either that raw rate or a Kimura-corrected evolutionary distance, per
`distance`.

Every `SWSequence` passed to `call` must have been built from the same
`SWPatternSet` as this kernel’s `patterns()` (or an equal copy of it).

Example:

```default
from refnd.kernels.protspam import ProtSpamKernel, ProtSpamDistance
from refnd.utils import SWPatternSet, SWSequence

sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEA",
]

# A RasBhari-optimized pattern set, shared by every SWSequence and the kernel.
patterns = SWPatternSet(5, 6, 20)

swseqs = [SWSequence(s, patterns) for s in sequences]

kernel = ProtSpamKernel(patterns, distance=ProtSpamDistance.Evolutionary)
distance = kernel.call(swseqs[0], swseqs[1])
distance = kernel(swseqs[0], swseqs[1])  # equivalent -- kernel(a, b) syntax
assert distance >= 0.0
```

#### call(ref_sample: [utils.SWSequence](utils.md#refnd.utils.SWSequence), query: [utils.SWSequence](utils.md#refnd.utils.SWSequence)) → float

Compute the distance between `ref_sample` and `query`, per this kernel’s
`distance()` mode.

* **Parameters:**
  * **ref_sample** – Reference sequence. Must have been built from the same
    `SWPatternSet` as this kernel’s `patterns()`.
  * **query** – Query sequence. Same requirement.
* **Returns:**
  The distance – see `ProtSpamDistance`.

Panics:
: `ref_sample`/`query` must have been built from the same
  `SWPatternSet` as this kernel’s `patterns()`. If they weren’t, this
  can panic (index out of range, if the mismatched set has fewer patterns)
  or – more insidiously – silently compare the wrong patterns’ spaced
  words against each other with no panic at all, if the mismatched set
  merely has a different pattern at the same index.

#### distance() → [ProtSpamDistance](#refnd.kernels.protspam.ProtSpamDistance)

Which value `call` reports.

#### patterns() → [utils.SWPatternSet](utils.md#refnd.utils.SWPatternSet)

The shared pattern set spaced words are matched against.

#### significance_threshold() → int

Minimum BLOSUM62 score for a spaced-word match to be considered homologous.

### *class* refnd.kernels.protspam.ProtSpamDistance

Bases: `object`

Which distance `ProtSpamKernel.call` reports.

- `MismatchRate`: the raw mismatch rate itself (fraction of don’t-care
  positions that mismatch, pooled across every pattern’s accepted spaced-word
  matches) – always in `[0, 1]`. `1.0` if no spaced word matched at all, or
  if every accepted match’s don’t-care positions mismatched outright.
- `Evolutionary`: the mismatch rate corrected into a distance via the Kimura
  two-parameter formula – in `[0, inf]`. `inf` if the mismatch rate is
  undefined (no match at all) or high enough (`> ~0.8541`) that the
  correction’s log argument is non-positive.

#### Evolutionary *= ProtSpamDistance.Evolutionary*

#### MismatchRate *= ProtSpamDistance.MismatchRate*

## refnd.kernels.vectors

Vector distance kernels.

<a id="module-refnd.kernels.vectors"></a>

### *class* refnd.kernels.vectors.Cosine

Bases: `object`

Cosine distance: `1 - dot(a, b) / (||a|| * ||b||)`. `0.0` if either vector
is the zero vector.

Example:

```default
import numpy as np
from refnd.kernels.vectors import Cosine

a = np.array([1.0, 0.0], dtype=np.float32)
b = np.array([0.0, 1.0], dtype=np.float32)
k = Cosine()
assert k.call(a, b) == k(a, b) == 1.0
```

#### call(a: numpy.typing.NDArray[numpy.float32], b: numpy.typing.NDArray[numpy.float32]) → float

Compute the cosine distance between two 1D `float32` numpy arrays.

Reads directly from each array’s buffer – no copy.

* **Parameters:**
  * **a** – First vector.
  * **b** – Second vector.
* **Returns:**
  Distance in `[0.0, 2.0]`. `0.0` means the same direction, `2.0`
  means exactly opposite directions.

### *class* refnd.kernels.vectors.L1

Bases: `object`

Manhattan (L1) distance: `sum(|a_i - b_i|)`.

Example:

```default
import numpy as np
from refnd.kernels.vectors import L1

a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
b = np.array([4.0, 0.0, 3.0], dtype=np.float32)
k = L1()
assert k.call(a, b) == k(a, b) == 5.0
```

#### call(a: numpy.typing.NDArray[numpy.float32], b: numpy.typing.NDArray[numpy.float32]) → float

Compute the L1 distance between two 1D `float32` numpy arrays.

Reads directly from each array’s buffer – no copy.

* **Parameters:**
  * **a** – First vector.
  * **b** – Second vector.
* **Returns:**
  Distance in `[0.0, inf]`. `0.0` means identical vectors.

### *class* refnd.kernels.vectors.L2

Bases: `object`

Euclidean (L2) distance: `sqrt(sum((a_i - b_i)²))`.

Example:

```default
import numpy as np
from refnd.kernels.vectors import L2

a = np.array([0.0, 0.0], dtype=np.float32)
b = np.array([3.0, 4.0], dtype=np.float32)
k = L2()
assert k.call(a, b) == k(a, b) == 5.0
```

#### call(a: numpy.typing.NDArray[numpy.float32], b: numpy.typing.NDArray[numpy.float32]) → float

Compute the L2 distance between two 1D `float32` numpy arrays.

Reads directly from each array’s buffer – no copy.

* **Parameters:**
  * **a** – First vector.
  * **b** – Second vector.
* **Returns:**
  Distance in `[0.0, inf]`. `0.0` means identical vectors.
