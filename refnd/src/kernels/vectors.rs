use crate::core::Distance;

/// Cosine distance: `1 - dot(a, b) / (||a|| * ||b||)`. `1.0` if either vector is the zero
/// vector.
#[derive(Clone)]
pub struct Cosine;

impl Distance<[f32]> for Cosine {
    #[inline(always)]
    fn call(&self, a: &[f32], b: &[f32]) -> f32 {
        let norm_a_sq: f32 = a.iter().map(|x| x * x).sum();
        let norm_b_sq: f32 = b.iter().map(|x| x * x).sum();
        if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
            return 1.0;
        }
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        1.0 - dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt())
    }
}

/// Manhattan (L1) distance: `sum(|a_i - b_i|)`.
#[derive(Clone)]
pub struct L1;

impl Distance<[f32]> for L1 {
    #[inline(always)]
    fn call(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }
}

/// Euclidean (L2) distance: `sqrt(sum((a_i - b_i)²))`.
#[derive(Clone)]
pub struct L2;

impl Distance<[f32]> for L2 {
    #[inline(always)]
    fn call(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn cosine_is_zero_for_parallel_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        assert!((Cosine.call(&a, &b) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_is_one_for_orthogonal_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((Cosine.call(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_is_zero_when_a_vector_is_all_zero() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(Cosine.call(&a, &b), 0.0);
    }

    #[test]
    fn l1_sums_absolute_differences() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 0.0, 3.0];
        assert_eq!(L1.call(&a, &b), 5.0);
    }

    #[test]
    fn l2_matches_hand_computed_euclidean_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_eq!(L2.call(&a, &b), 5.0);
    }

    #[test]
    fn l2_is_zero_for_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(L2.call(&a, &b), 0.0);
    }
}
