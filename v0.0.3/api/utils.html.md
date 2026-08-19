# refnd.utils

### *class* refnd.utils.BitFingerprint(fp: ExplicitBitVect)

Bases: `object`

Dense binary fingerprint backed by a dense bitset.

Each bit represents the presence or absence of a structural feature.
The primary source is an RDKit `ExplicitBitVect` (from `GetFingerprint`),
but plain Python lists and numpy arrays are also accepted.

`count` caches the popcount so Tanimoto computation avoids re-counting.

Example:

```default
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
from refnd.utils import BitFingerprint

mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
mol = MolFromSmiles("c1ccccc1")
fp = BitFingerprint(mfpgen.GetFingerprint(mol))
print(fp.count(), len(fp))   # set bits, total bits
```

#### count() → int

Number of on bits (popcount).

#### *static* from_list(values: Sequence[bool]) → [BitFingerprint](#refnd.utils.BitFingerprint)

Construct from a list of booleans (or 0/1 ints).

#### *static* from_np(arr: numpy.typing.NDArray[numpy.bool_]) → [BitFingerprint](#refnd.utils.BitFingerprint)

Construct from a numpy boolean or uint8 array.

For uint8/bool dtypes, reads the array’s raw buffer directly – a zero-copy
view into memory numpy already owns – and sets bits from it in one pass, with no
per-element Python object boxing. Any other dtype falls back to .tolist() +
from_list, which does box each element as an individual Python object along the
way; this covers arbitrary array-likes at the cost of that boxing.

#### *static* random(len: int, count: int) → [BitFingerprint](#refnd.utils.BitFingerprint)

Create a fingerprint of `len` bits with exactly `count` bits set at random.

Useful for randomized testing.

* **Parameters:**
  * **len** – Total number of bits.
  * **count** – Number of bits to turn on. Must be `<= len`.
* **Raises:**
  **ValueError** – If `count > len`.

#### to_list() → list[bool]

Export as a list of booleans.

#### to_np() → numpy.typing.NDArray[numpy.bool_]

Export as a numpy bool array.

#### to_rdkit() → ExplicitBitVect

Export as an RDKit `ExplicitBitVect`.

#### fp *: ExplicitBitVect*

#### return *: [BitFingerprint](#refnd.utils.BitFingerprint)*

### *class* refnd.utils.RealFingerprint(fp: UIntSparseIntVect)

Bases: `object`

Dense real-valued fingerprint backed by a `Vec<f32>`.

Each element represents a feature count or continuous value. The primary
source is an RDKit `UIntSparseIntVect` (from `GetCountFingerprint`),
but plain Python lists and numpy arrays are also accepted.

`norm_sq` caches `||x||²` so Tanimoto computation avoids recomputing it.

Example:

```default
from rdkit.Chem import rdFingerprintGenerator, MolFromSmiles
from refnd.utils import RealFingerprint
from refnd.kernels.molecules import TanimotoReal

mfpgen = rdFingerprintGenerator.GetMorganGenerator(fpSize=1024, radius=2)
mol = MolFromSmiles("c1ccccc1")
fp = RealFingerprint(mfpgen.GetCountFingerprint(mol))
print(fp.norm_sq(), len(fp))
```

#### *static* from_list(values: Sequence[float]) → [RealFingerprint](#refnd.utils.RealFingerprint)

Construct from a list of floats.

#### *static* from_np(arr: numpy.typing.NDArray[numpy.float32]) → [RealFingerprint](#refnd.utils.RealFingerprint)

Construct from a numpy float32 or float64 array.

#### norm_sq() → float

Squared Euclidean norm of the feature vector (`||x||²`).

#### to_list() → list[float]

Export as a list of floats.

#### to_np() → numpy.typing.NDArray[numpy.float32]

Export as a numpy float32 array.

#### to_rdkit() → UIntSparseIntVect

Export as an RDKit `UIntSparseIntVect`, compatible with `GetCountFingerprint`.
Only non-zero elements are stored; the length equals `len(self)`.

#### fp *: UIntSparseIntVect*

#### return *: [RealFingerprint](#refnd.utils.RealFingerprint)*

### *class* refnd.utils.Vector(values: Any)

Bases: `object`

A dense `f32` vector, used by the `refnd.kernels.vectors` kernels (`Cosine`,
`EluDot`, `L1`, `L2`). Backed by a plain `Vec<f32>` – unlike
`RealFingerprint`, there’s no precomputed cache.

Example:

```default
import numpy as np
from refnd.utils import Vector

v = Vector(np.array([1.0, 2.0, 3.0], dtype=np.float32))
print(len(v))
```

#### *static* from_list(values: Sequence[float]) → [Vector](#refnd.utils.Vector)

Construct from a list of floats.

#### *static* from_np(arr: Any) → [Vector](#refnd.utils.Vector)

Construct from a numpy float32 array, reading its buffer directly. Falls back
to `.tolist()` for other dtypes.

#### to_list() → list[float]

Export as a list of floats.

#### to_np() → numpy.typing.NDArray[numpy.float32]

Export as a numpy float32 array.

#### values *: Any*

#### return *: [Vector](#refnd.utils.Vector)*

### *class* refnd.utils.PdbStructure(path: str)

Bases: `object`

A protein structure loaded from a PDB file and held in memory.

The structure is parsed once on construction and the coordinate arrays are
kept alive for the lifetime of the object. Cloning is cheap (reference-counted).

Example:

```default
from refnd.utils import PdbStructure
s = PdbStructure("1abc.pdb")
```

#### path *: str*

#### return *: [PdbStructure](#refnd.utils.PdbStructure)*

### refnd.utils.read_fasta(path: str) → list[tuple[str, str]]

Parse a FASTA file and return all records as a list of `(header, sequence)` pairs.

The header string is the full description line without the leading `>`.
The sequence is the concatenation of all continuation lines for that record,
with whitespace stripped.

* **Parameters:**
  **path** – Path to the FASTA file.
* **Returns:**
  A list of `(header, sequence)` tuples, one per FASTA record.
* **Raises:**
  **IOError** – If the file cannot be opened or is not valid UTF-8.

Example:

```default
from refnd.utils import read_fasta

records = read_fasta("proteins.fasta")
header, seq = records[0]
print(header)  # "sp|P12345|MYPR_HUMAN ..."
print(seq)     # "MKTAYIAKQRQISFVKSHFSRQ..."
```

### refnd.utils.largest_cluster(clusters: Sequence[int]) → tuple[int, int]

Return the ID and size of the largest cluster.

Convenience helper — iterates over a cluster-label vector and finds the
most populous label.

* **Parameters:**
  **clusters** – A list of cluster IDs (e.g. from `connected_components` or
  `find_communities`).
* **Returns:**
  A tuple `(cluster_id, size)` for the largest cluster.

### *class* refnd.utils.SWPattern(weight: int, dont_care: int)

Bases: `object`

A spaced-word pattern: a binary mask over length positions where a match position
(“1”) contributes a residue to the spaced word’s key and a don’t-care position (“0”)
is skipped when hashing but still compared for mismatches. Position 0 and the last
position are always match positions.

Example:

```default
from refnd.utils import SWPattern

pat = SWPattern(6, 20)
assert len(pat) == 26
assert pat.weight() == 6
assert pat.dontcare() == 20
assert pat.is_match(0)
assert pat.is_match(25)

# Parse/render as a string of '1's (match) and '0's (don't-care)
pat2 = SWPattern.parse("10100101")
assert pat2.weight() == 4
assert str(pat2) == "10100101"
```

#### dontcare() → int

Number of don’t-care (“0”) positions.

#### is_match(pos: int) → bool

Whether `pos` is a match position. Never raises: an out-of-range `pos`
simply reads as `False`.

#### match_positions() → list[int]

Ascending indices of match (“1”) positions.

#### *static* parse(s: str) → [SWPattern](#refnd.utils.SWPattern)

Parse a pattern from a string of `'1'``s (match) and ``'0'``s (don't-care),
the same format ``str()` produces.

* **Raises:**
  **ValueError** – If the string contains a character other than `'0'`/`'1'`,
      or doesn’t start and end with `'1'`.

#### weight() → int

Number of match (“1”) positions.

#### dont_care *: int*

#### return *: [SWPattern](#refnd.utils.SWPattern)*

### *class* refnd.utils.SWPatternSet(n: int, weight: int, dont_care: int)

Bases: `object`

A set of 

```
``
```

SWPattern\`\`s used together to compute spaced words for a sequence.

The default constructor builds a RasBhari-optimized set (recommended for real
use); use `SWPatternSet.random` for a cheap, unoptimized set, or
`SWPatternSet.from_patterns` to build one from hand-picked patterns.

Example:

```default
from refnd.utils import SWPatternSet

# RasBhari-optimized (recommended)
patterns = SWPatternSet(5, 6, 20)
assert len(patterns) == 5

# Cheap, unoptimized baseline, refined by hand
random_patterns = SWPatternSet.random(5, 6, 20)
score_before = random_patterns.optimize(0)    # limit=0: score only, no changes
score_after = random_patterns.optimize(2000)
assert score_after <= score_before
```

#### *static* from_patterns(patterns: Sequence[[SWPattern](#refnd.utils.SWPattern)]) → [SWPatternSet](#refnd.utils.SWPatternSet)

Build a set from already-constructed patterns. Unlike `random`, there’s no uniqueness check – duplicate
patterns are allowed, though they add nothing (two identical patterns always
find exactly the same matches).

#### *static* load(path: str) → [SWPatternSet](#refnd.utils.SWPatternSet)

Deserialize a set previously written by `save`.

* **Raises:**
  **IOError** – If `path` can’t be read, or its contents aren’t a valid pattern set.

#### optimize(limit: int) → float

Optimize this pattern set in place with RasBhari’s overlap-complexity hill
climbing: repeatedly picks a pattern round-robin, swaps one of its interior
match positions for a don’t-care position, and keeps the change only if it
strictly lowers the set’s total pairwise overlap-complexity score.

* **Parameters:**
  **limit** – Number of hill-climbing steps to run. Pass `0` to just compute
  and return the current score without changing anything.
* **Returns:**
  The achieved overlap-complexity score after optimizing (lower is better).

#### patterns() → list[[SWPattern](#refnd.utils.SWPattern)]

The patterns in this set, in construction order.

#### *static* random(n: int, weight: int, dont_care: int) → [SWPatternSet](#refnd.utils.SWPatternSet)

Build `n` distinct unoptimized random patterns of the given
`weight`/`dont_care`. Useful as a cheap baseline, or as the unoptimized
starting point `optimize` refines.

#### WARNING
Patterns are generated by rejection sampling on uniqueness, so this hangs
(never returns) if `n` isn’t well below the number of distinct patterns
possible for `weight`/`dont_care` (`C(weight + dont_care - 2, weight - 2)`).
This is sharpest at `weight == 2`: there are no interior positions to
vary at all, so every generated pattern is identical and any `n > 1`
hangs immediately.

#### save(path: str) → None

Serialize this set to `path`. Inverse of `load`.

* **Raises:**
  **IOError** – If `path` can’t be written.

#### *static* with_limit(n: int, weight: int, dont_care: int, limit: int) → [SWPatternSet](#refnd.utils.SWPatternSet)

Same as the default constructor, but with an explicit hill-climbing step
budget instead of ProtSpaM’s default of 25,000.

#### n *: int*

#### weight *: int*

#### dont_care *: int*

#### return *: [SWPatternSet](#refnd.utils.SWPatternSet)*

### *class* refnd.utils.SWSequence(seq: str, patterns: [SWPatternSet](#refnd.utils.SWPatternSet))

Bases: `object`

A sequence prepared for spaced-word comparison (ProtSpaM-style). Its residues are
pre-encoded as 5-bit amino-acid codes, and it holds the sorted spaced words for
every pattern in the `SWPatternSet` it was built with.

Two `SWSequence``s must be built from the same ``SWPatternSet` (or two equal
copies of it) to be compared meaningfully – see
`refnd.kernels.protspam.ProtSpamKernel`, which is what actually compares two of
these.

Picklable: `pickle.dumps`/`pickle.loads` round-trip an `SWSequence` without
needing the original pattern set again (its encoded residues and sorted spaced
words are serialized directly).

Example:

```default
import pickle
from refnd.utils import SWPatternSet, SWSequence

patterns = SWPatternSet(5, 6, 20)
seq = SWSequence("MKTAYIAKQRQISFVKSHFSRQ", patterns)
assert len(seq) == 22

restored = pickle.loads(pickle.dumps(seq))
assert restored.seq() == seq.seq()
```

#### pattern_count() → int

Number of patterns this sequence has spaced words for – the `len()` of the
`SWPatternSet` it was built with. Valid indices for `sorted_words` are
`0 .. pattern_count()`.

#### seq() → list[int]

Residues as 5-bit amino-acid codes, not raw ASCII – e.g. `'A'` reads back
as `0`, not `65`.

#### sorted_words(pattern_idx: int) → list[[SWWord](#refnd.utils.SWWord)]

Sorted spaced words for `patterns.patterns()[pattern_idx]`, where
`patterns` is the set this sequence was built with. Empty if this sequence
is shorter than that pattern (no window fits).

* **Raises:**
  **IndexError** – If `pattern_idx` is out of range for the pattern set this
      sequence was built with.

#### patterns *: [SWPatternSet](#refnd.utils.SWPatternSet)*

#### return *: [SWSequence](#refnd.utils.SWSequence)*

### *class* refnd.utils.SWWord

Bases: `object`

One complete spaced word: the residues at a pattern’s match positions.

Example:

```default
from refnd.utils import SWPatternSet, SWSequence

patterns = SWPatternSet.random(1, 2, 1)  # single pattern, weight 2, dc 1 -> "101"
seq = SWSequence("ACDE", patterns)
words = seq.sorted_words(0)
assert all(words[i].key() <= words[i + 1].key() for i in range(len(words) - 1))
```

#### key() → int

The packed key: the pattern’s match-position residues, 5 bits each,
most-significant residue first.

#### pos() → int

Start position of this word’s window in the sequence it came from.
