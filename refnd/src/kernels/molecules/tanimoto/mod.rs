use crate::utils::{BitFingerprint, RealFingerprint};
use crate::core::Distance;

#[derive(Clone)]
pub struct Tanimoto;

impl Distance<BitFingerprint> for Tanimoto {
    #[inline(always)]
    fn call(&self, a: &BitFingerprint, b: &BitFingerprint) -> f32 {
        let intersection: u32 = a.bits.as_slice().iter()
            .zip(b.bits.as_slice().iter())
            .map(|(x, y)| (x & y).count_ones())
            .sum();
        let union = a.count + b.count - intersection;
        if union == 0 { return 0.0; }
        1.0 - intersection as f32 / union as f32
    }
}

impl Distance<RealFingerprint> for Tanimoto {
    #[inline(always)]
    fn call(&self, a: &RealFingerprint, b: &RealFingerprint) -> f32 {
        let dot: f32 = a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).sum();
        let denom = a.norm_sq + b.norm_sq - dot;
        if denom == 0.0 { return 0.0; }
        1.0 - dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::InlineBitSet;

    #[test]
    fn bit_distance_matches_hand_computed_value() {
        // a = {0,1,2}, b = {1,2,3} over 8 bits -> intersection=2, union=3+3-2=4, distance=0.5
        let mut a_bits = InlineBitSet::with_capacity(8);
        for i in [0, 1, 2] { a_bits.insert(i); }
        let a = BitFingerprint::new(a_bits);

        let mut b_bits = InlineBitSet::with_capacity(8);
        for i in [1, 2, 3] { b_bits.insert(i); }
        let b = BitFingerprint::new(b_bits);

        let d = Tanimoto.call(&a, &b);
        assert!((d - 0.5).abs() < 1e-6, "expected 0.5, got {d}");
    }

    #[test]
    fn bit_distance_zero_for_identical_fingerprints() {
        let mut bits = InlineBitSet::with_capacity(2048);
        for i in [0, 100, 500, 2000] { bits.insert(i); }
        let a = BitFingerprint::new(bits.clone());
        let b = BitFingerprint::new(bits);
        assert_eq!(Tanimoto.call(&a, &b), 0.0);
    }

    /// Fingerprint pair whose only set bits sit either side of a 64-bit word boundary
    /// (bit 63 vs bit 64), to catch an off-by-one in InlineBitSet's word/shift math.
    #[test]
    fn bit_distance_across_word_boundary() {
        let mut a_bits = InlineBitSet::with_capacity(128);
        a_bits.insert(63);
        let a = BitFingerprint::new(a_bits);

        let mut b_bits = InlineBitSet::with_capacity(128);
        b_bits.insert(64);
        let b = BitFingerprint::new(b_bits);

        // Disjoint single-bit fingerprints: intersection=0, union=1+1-0=2, distance=1.0
        assert_eq!(Tanimoto.call(&a, &b), 1.0);
    }
}