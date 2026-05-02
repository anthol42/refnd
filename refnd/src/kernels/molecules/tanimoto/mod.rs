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