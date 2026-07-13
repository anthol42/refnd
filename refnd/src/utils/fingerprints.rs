use fixedbitset::FixedBitSet;
use ndarray::Array1;
use rand::seq::index::sample as rand_sample;

#[derive(Clone)]
pub struct BitFingerprint {
    /// Internal bitset
    pub bits: FixedBitSet,
    /// Number of set bits
    pub count: u32,
}

impl BitFingerprint {
    pub fn new(bits: FixedBitSet) -> Self {
        let count = bits.count_ones(..) as u32;
        Self { bits, count }
    }

    /// Create a fingerprint of `len` bits with exactly `count` bits set at random.
    ///
    /// Panics if `count > len`.
    pub fn random(len: usize, count: usize) -> Self {
        assert!(count <= len, "count ({count}) must be <= len ({len})");
        let indices = rand_sample(&mut rand::rng(), len, count);
        let mut bits = FixedBitSet::with_capacity(len);
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