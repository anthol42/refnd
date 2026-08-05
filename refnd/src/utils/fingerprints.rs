use ndarray::Array1;
use rand::seq::index::sample as rand_sample;
use smallvec::{smallvec, SmallVec};

type Block = u64;
const BITS_PER_BLOCK: usize = Block::BITS as usize; // 64

/// A bitset backed by `SmallVec<[Block; 32]>` instead of a separately heap-allocated
/// buffer (as `fixedbitset::FixedBitSet` uses internally, via its own inner `Vec`).
///
/// Fingerprints at or under 2048 bits -- i.e. virtually every real Morgan/RDKit
/// fingerprint -- are stored inline, with no separate heap allocation per fingerprint;
/// anything larger transparently falls back to a heap allocation via `SmallVec`'s spill
/// mechanism, so no bit-length flexibility is lost for unusual sizes.
///
/// This matters when millions of `BitFingerprint`s are held in one `Vec<BitFingerprint>`
/// (e.g. building an HNSW index over a large molecule library): with a heap-allocated
/// inner bitset, that's one separate allocation per fingerprint -- allocator bookkeeping
/// and fragmentation on top of the raw bit data, measured at ~603 bytes/fingerprint for a
/// 2048-bit (256-byte) fingerprint, more than 2x the theoretical minimum. With this type,
/// the whole outer `Vec<BitFingerprint>` is one contiguous allocation once fingerprints
/// fit inline.
#[derive(Clone)]
pub struct InlineBitSet {
    blocks: SmallVec<[Block; 32]>,
    len: usize,
}

impl InlineBitSet {
    pub fn with_capacity(bits: usize) -> Self {
        let n_blocks = bits.div_ceil(BITS_PER_BLOCK);
        InlineBitSet { blocks: smallvec![0; n_blocks], len: bits }
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

    /// Whether the bit data has spilled to a heap allocation (only happens above 2048 bits).
    pub fn spilled(&self) -> bool {
        self.blocks.spilled()
    }

    /// Indices of all set bits, in ascending order.
    pub fn ones(&self) -> impl Iterator<Item = usize> + '_ {
        let len = self.len;
        self.blocks.iter().enumerate().flat_map(move |(word_idx, &word)| {
            (0..BITS_PER_BLOCK).filter_map(move |bit_idx| {
                let global = word_idx * BITS_PER_BLOCK + bit_idx;
                (global < len && (word >> bit_idx) & 1 != 0).then_some(global)
            })
        })
    }

    /// Raw underlying words -- used by `Tanimoto`'s per-block AND+popcount hot path.
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
