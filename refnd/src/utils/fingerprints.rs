use ndarray::Array1;
use rand::seq::index::sample as rand_sample;
use smallvec::{smallvec, SmallVec};

type Block = u64;
const BITS_PER_BLOCK: usize = Block::BITS as usize; // 64
/// 32 x 64-bit words = 2048 bits held inline, no heap allocation -- the size of a
/// standard Morgan/RDKit fingerprint. Anything larger transparently spills to the heap.
const INLINE_BLOCKS: usize = 32;

/// A bitset whose storage lives inline (alongside its length, no separate heap
/// allocation) for bit lengths up to `INLINE_BLOCKS * 64`, and spills to the heap
/// transparently above that -- so callers never need to think about which case they're
/// in, only `BitFingerprint`'s memory profile at scale changes.
#[derive(Clone)]
pub struct InlineBitSet {
    blocks: SmallVec<[Block; INLINE_BLOCKS]>,
    len: usize,
}

impl InlineBitSet {
    pub fn with_capacity(bits: usize) -> Self {
        let n_blocks = bits.div_ceil(BITS_PER_BLOCK);
        InlineBitSet { blocks: smallvec![0; n_blocks], len: bits }
    }

    /// Builds a `bits`-long set directly from already-packed words (e.g. read back from
    /// a file of packed fingerprints), skipping the `with_capacity` + per-bit `insert`
    /// path entirely.
    pub fn with_capacity_and_blocks(bits: usize, blocks: impl IntoIterator<Item = Block>) -> Self {
        let mut blocks: SmallVec<[Block; INLINE_BLOCKS]> = blocks.into_iter().collect();
        blocks.resize(bits.div_ceil(BITS_PER_BLOCK), 0);
        InlineBitSet { blocks, len: bits }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn insert(&mut self, bit: usize) {
        debug_assert!(bit < self.len, "bit index {bit} out of bounds for length {}", self.len);
        self.blocks[bit / BITS_PER_BLOCK] |= 1 << (bit % BITS_PER_BLOCK);
    }

    #[inline]
    pub fn set(&mut self, bit: usize, value: bool) {
        debug_assert!(bit < self.len, "bit index {bit} out of bounds for length {}", self.len);
        if value {
            self.blocks[bit / BITS_PER_BLOCK] |= 1 << (bit % BITS_PER_BLOCK);
        } else {
            self.blocks[bit / BITS_PER_BLOCK] &= !(1 << (bit % BITS_PER_BLOCK));
        }
    }

    #[inline]
    pub fn contains(&self, bit: usize) -> bool {
        if bit >= self.len {
            return false;
        }
        (self.blocks[bit / BITS_PER_BLOCK] >> (bit % BITS_PER_BLOCK)) & 1 != 0
    }

    pub fn count_ones(&self) -> usize {
        self.blocks.iter().map(|b| b.count_ones() as usize).sum()
    }

    /// Whether storage has spilled to the heap (only happens above `INLINE_BLOCKS * 64`
    /// bits). Not read anywhere currently; kept for ad hoc memory-profiling.
    #[allow(dead_code)]
    pub fn spilled(&self) -> bool {
        self.blocks.spilled()
    }

    /// Indices of all set bits, ascending.
    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        let len = self.len;
        self.blocks.iter().enumerate().flat_map(move |(word_idx, &word)| {
            (0..BITS_PER_BLOCK).filter_map(move |bit_idx| {
                let global = word_idx * BITS_PER_BLOCK + bit_idx;
                (global < len && (word >> bit_idx) & 1 != 0).then_some(global)
            })
        })
    }

    /// Raw underlying words, used by `Tanimoto`'s per-block AND+popcount hot path.
    #[inline]
    pub fn as_slice(&self) -> &[Block] {
        &self.blocks
    }
}

#[derive(Clone)]
pub struct BitFingerprint {
    /// Internal bitset
    pub bits: InlineBitSet,
    /// Number of set bits
    pub count: u32,
}

impl BitFingerprint {
    pub fn new(bits: InlineBitSet) -> Self {
        let count = bits.count_ones() as u32;
        Self { bits, count }
    }

    /// Create a fingerprint of `len` bits with exactly `count` bits set at random.
    ///
    /// Panics if `count > len`.
    pub fn random(len: usize, count: usize) -> Self {
        assert!(count <= len, "count ({count}) must be <= len ({len})");
        let indices = rand_sample(&mut rand::rng(), len, count);
        let mut bits = InlineBitSet::with_capacity(len);
        for i in indices.iter() {
            bits.insert(i);
        }
        Self::new(bits)
    }
}

#[derive(Clone)]
pub struct RealFingerprint {
    /// Vector data
    pub data: Vec<f32>,
    /// Norm squared
    pub norm_sq: f32,
}

impl RealFingerprint {
    pub fn new(data: Vec<f32>) -> Self {
        let norm_sq = data.iter().map(|x| x * x).sum();
        Self { data, norm_sq }
    }

    /// Zero-copy: consumes the Array1 and moves its buffer into the Vec.
    /// Requires the array to be standard (C) layout and contiguous.
    pub fn from_array(arr: Array1<f32>) -> Self {
        let (data, _offset) = arr.into_raw_vec_and_offset();
        Self::new(data)
    }
    /// Copy the RealFingerprints internal to a ndarray vector
    pub fn to_array(&self) -> Array1<f32> {
        Array1::from_vec(self.data.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_and_contains_roundtrip() {
        let mut bs = InlineBitSet::with_capacity(100);
        let set_bits = [0, 1, 33, 63, 64, 65, 99];
        for &b in &set_bits {
            bs.insert(b);
        }
        for i in 0..100 {
            assert_eq!(bs.contains(i), set_bits.contains(&i), "bit {i}");
        }
        assert_eq!(bs.count_ones(), set_bits.len());
    }

    #[test]
    fn set_false_clears_bit() {
        let mut bs = InlineBitSet::with_capacity(64);
        bs.insert(10);
        bs.insert(20);
        assert!(bs.contains(10));
        bs.set(10, false);
        assert!(!bs.contains(10));
        assert!(bs.contains(20));
        assert_eq!(bs.count_ones(), 1);
    }

    #[test]
    fn ones_returns_ascending_set_indices() {
        let mut bs = InlineBitSet::with_capacity(200);
        let set_bits = [5, 4, 130, 63, 64, 199, 0];
        for &b in &set_bits {
            bs.insert(b);
        }
        let mut expected: Vec<usize> = set_bits.to_vec();
        expected.sort_unstable();
        assert_eq!(bs.ones().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn contains_out_of_bounds_is_false_not_a_panic() {
        let bs = InlineBitSet::with_capacity(10);
        assert!(!bs.contains(10));
        assert!(!bs.contains(1000));
    }

    #[test]
    fn stays_inline_at_2048_bits_spills_above() {
        let at_capacity = InlineBitSet::with_capacity(2048);
        assert!(!at_capacity.spilled(), "2048 bits should fit in the 32 inline u64 words");

        let over_capacity = InlineBitSet::with_capacity(2049);
        assert!(over_capacity.spilled(), "2049 bits should need a 33rd word, spilling to the heap");
    }

    #[test]
    fn with_capacity_and_blocks_matches_bit_layout() {
        // bit i is word (i / 64), shift (i % 64) -- verify a fingerprint built directly
        // from words agrees with one built bit-by-bit for the same pattern.
        let mut expected = InlineBitSet::with_capacity(128);
        for b in [0, 5, 63, 64, 100, 127] {
            expected.insert(b);
        }

        let words = [expected.as_slice()[0], expected.as_slice()[1]];
        let from_blocks = InlineBitSet::with_capacity_and_blocks(128, words);

        assert_eq!(from_blocks.as_slice(), expected.as_slice());
        for i in 0..128 {
            assert_eq!(from_blocks.contains(i), expected.contains(i), "bit {i}");
        }
    }

    #[test]
    fn padding_bits_in_final_partial_block_dont_leak_into_count() {
        // 10 bits fit in one 64-bit block; the other 54 bits in that block are padding
        // and must never read as set.
        let mut bs = InlineBitSet::with_capacity(10);
        for i in 0..10 {
            bs.insert(i);
        }
        assert_eq!(bs.count_ones(), 10);
        assert_eq!(bs.ones().collect::<Vec<_>>(), (0..10).collect::<Vec<_>>());
    }
}
