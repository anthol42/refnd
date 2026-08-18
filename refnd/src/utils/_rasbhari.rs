//! RasBhari pattern-set optimization (overlap-complexity `Oc` variant, ProtSpaM's
//! default): hill-climbs a [`SWPatternSet`] to minimize redundancy between its
//! patterns' matches.
//!
//! Reference: Hahn, Leimeister, Morgenstern, "rasbhari: Optimizing Spaced Seeds for
//! Database Searching, Read Mapping and Alignment-Free Sequence Comparison", PLOS
//! Computational Biology (2016).

use super::sw_pattern::{SWPattern, SWPatternSet};
use rand::RngExt;
use std::collections::HashMap;

impl SWPattern {
    /// Swaps one interior match position (never position 0 or the last) for a random
    /// don't-care position, keeping weight and don't-care count unchanged. No-op if
    /// there's no don't-care position to swap in, or fewer than 3 match positions
    /// (nothing interior to give up).
    pub fn random_swap(&mut self) {
        if self.dontcare() == 0 || self.weight() < 3 {
            return;
        }
        let mut rng = rand::rng();
        let interior_idx = rng.random_range(1..self.weight() - 1);
        let dc_positions: Vec<usize> = (0..self.length).filter(|&p| !self.is_match(p)).collect();
        let dc_pos = dc_positions[rng.random_range(0..dc_positions.len())];

        self.match_positions[interior_idx] = dc_pos;
        self.match_positions.sort_unstable();
    }
}

/// Overlap-complexity coefficient between two patterns (RasBhari's `Oc` measure): for
/// every relative shift `s` of `b` against `a` (from `-(b.length()-1)` to `a.length()-1`),
/// sums `2^overlap(s)`, where `overlap(s)` counts match positions that coincide at that
/// shift. Lower is better -- it quantifies how redundant two patterns' matches are.
///
/// Implemented via a histogram of `pa - pb` over all match-position pairs `(pa, pb)`,
/// which groups exactly by shift and avoids a per-shift O(length) scan.
fn pair_coef_oc(a: &SWPattern, b: &SWPattern) -> f64 {
    let mut shift_overlaps: HashMap<i64, u32> = HashMap::new();
    for &pa in a.match_positions() {
        for &pb in b.match_positions() {
            *shift_overlaps.entry(pa as i64 - pb as i64).or_insert(0) += 1;
        }
    }
    let total_shifts = (a.length() + b.length() - 1) as f64;
    let zero_overlap_shifts = total_shifts - shift_overlaps.len() as f64;
    zero_overlap_shifts + shift_overlaps.values().map(|&k| 2f64.powi(k as i32)).sum::<f64>()
}

/// Pairwise overlap-complexity matrix for a pattern set, plus the running total score
/// (sum of the upper triangle including the diagonal). Mirrors RasBhari's incremental
/// `update()`: after one pattern changes, only its row/column need recomputing.
struct RasbhariState {
    coef_mat: Vec<Vec<f64>>,
    total_score: f64,
}

impl RasbhariState {
    fn new(patterns: &[SWPattern]) -> Self {
        let n = patterns.len();
        let coef_mat: Vec<Vec<f64>> =
            (0..n).map(|i| (0..n).map(|j| pair_coef_oc(&patterns[i], &patterns[j])).collect()).collect();
        let total_score = (0..n).map(|i| coef_mat[i][i..].iter().sum::<f64>()).sum();
        Self { coef_mat, total_score }
    }

    fn update(&mut self, patterns: &[SWPattern], idx: usize) {
        for i in 0..patterns.len() {
            self.total_score -= self.coef_mat[i][idx];
            let new_coef = pair_coef_oc(&patterns[i], &patterns[idx]);
            self.coef_mat[i][idx] = new_coef;
            self.coef_mat[idx][i] = new_coef;
            self.total_score += new_coef;
        }
    }
}

/// Mutates `patterns[idx]` into a random-swap variant that's distinct from every other
/// pattern in the slice, retrying from the original up to `(n + 10)^2` times. Leaves
/// `patterns[idx]` unchanged and returns `false` if no unique variant was found.
fn try_unique_swap(patterns: &mut [SWPattern], idx: usize) -> bool {
    let original = patterns[idx].clone();
    let attempts = (patterns.len() + 10) * (patterns.len() + 10);
    let mut candidate = original.clone();
    for _ in 0..attempts {
        candidate.random_swap();
        let is_unique = !patterns.iter().enumerate().any(|(j, p)| j != idx && *p == candidate);
        if is_unique {
            patterns[idx] = candidate;
            return true;
        }
        candidate = original.clone();
    }
    false
}

/// Number of hill-climbing steps RasBhari's `-r` option defaults to, and what ProtSpaM
/// itself uses when building its default pattern set.
const RASBHARI_DEFAULT_LIMIT: usize = 25_000;

impl SWPatternSet {
    /// Optimizes this pattern set in place with RasBhari's overlap-complexity hill
    /// climbing (the `Oc` variant, ProtSpaM's default): repeatedly picks a pattern
    /// round-robin, swaps one of its interior match positions for a don't-care position
    /// (via [`SWPattern::random_swap`], rejecting swaps that collide with another
    /// pattern in the set), and keeps the change only if it strictly lowers the set's
    /// total pairwise overlap-complexity score. Runs for `limit` steps and returns the
    /// achieved score (lower is better).
    pub fn optimize(&mut self, limit: usize) -> f64 {
        let n = self.patterns.len();
        let mut state = RasbhariState::new(&self.patterns);
        if n < 2 {
            return state.total_score;
        }

        let mut pat_no = 0usize;
        for _ in 0..limit {
            let idx = pat_no % n;
            let before = self.patterns[idx].clone();

            if try_unique_swap(&mut self.patterns, idx) {
                let before_score = state.total_score;
                state.update(&self.patterns, idx);
                if state.total_score < before_score {
                    pat_no = 0;
                    continue;
                }
                self.patterns[idx] = before;
                state.update(&self.patterns, idx);
            }
            pat_no += 1;
        }
        state.total_score
    }

    /// Builds a RasBhari-optimized pattern set: `n` distinct random patterns of the
    /// given `weight`/`dont_care` (see [`SWPatternSet::random`]), refined by
    /// [`Self::optimize`] for `limit` steps.
    pub fn with_limit(n: usize, weight: usize, dont_care: usize, limit: usize) -> Self {
        let mut set = Self::random(n, weight, dont_care);
        set.optimize(limit);
        set
    }

    /// The default way to build a pattern set: [`Self::with_limit`] with ProtSpaM's
    /// default step budget (25_000). Use [`SWPatternSet::random`] directly if you want
    /// an unoptimized set instead (e.g. as a cheap baseline).
    pub fn new(n: usize, weight: usize, dont_care: usize) -> Self {
        Self::with_limit(n, weight, dont_care, RASBHARI_DEFAULT_LIMIT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_swap_preserves_weight_dontcare_and_edges() {
        let mut pat = SWPattern::random(6, 20);
        let before = pat.clone();
        pat.random_swap();
        assert_eq!(pat.weight(), before.weight());
        assert_eq!(pat.dontcare(), before.dontcare());
        assert!(pat.is_match(0));
        assert!(pat.is_match(pat.length() - 1));
    }

    #[test]
    fn random_swap_is_noop_without_interior_or_dontcare() {
        let mut pat = SWPattern::random(2, 0); // "11": no interior match, no don't-care
        let before = pat.clone();
        pat.random_swap();
        assert_eq!(pat, before);
    }

    /// Reimplements the RasBhari `Oc` definition directly (brute-force over every
    /// shift, clipping at pattern boundaries) as an independent check on the
    /// histogram-based `pair_coef_oc`.
    #[test]
    fn pair_coef_oc_matches_naive_shift_definition() {
        let a: SWPattern = "1011001".parse().unwrap();
        let b: SWPattern = "1001101".parse().unwrap();
        let matches_at = |pat: &SWPattern, pos: i64| pos >= 0 && (pos as usize) < pat.length() && pat.is_match(pos as usize);

        let mut expected = 0.0;
        for s in -(b.length() as i64 - 1)..(a.length() as i64) {
            let overlap = (0..a.length() as i64).filter(|&i| matches_at(&a, i) && matches_at(&b, i - s)).count();
            expected += 2f64.powi(overlap as i32);
        }
        assert_eq!(pair_coef_oc(&a, &b), expected);
    }

    /// Cross-validated against the real ProtSpaM C++ implementation: these 6 patterns
    /// (weight 6, dont_care 20) were fed into `rasbhari_compute::pair_coef_oc` and
    /// `rasbhari::calculate()` from the actual `include/rasbcomp.hpp`/`rasbhari.hpp`
    /// sources (compiled and linked against ProtSpaM's own `.o` files, see
    /// `~/Downloads/ProtSpaM/xval_oc.cpp`), and every pairwise coefficient plus the
    /// triangle-sum total matched this Rust implementation's output exactly (2374.0,
    /// full f64 precision, no epsilon needed).
    #[test]
    fn pair_coef_oc_matches_protspam_reference_values() {
        let patterns: Vec<SWPattern> = [
            "10010000100010010000000001",
            "10100111000000000000000001",
            "10010011000000100000000001",
            "10001001010001000000000001",
            "10010001000000000000110001",
            "11000010001000000000000101",
        ]
        .iter()
        .map(|s| s.parse().unwrap())
        .collect();

        #[rustfmt::skip]
        let expected_upper_triangle: [[f64; 6]; 6] = [
            [148.0,  94.0,  98.0, 101.0,  99.0, 101.0],
            [  0.0, 150.0, 104.0, 101.0, 100.0,  99.0],
            [  0.0,   0.0, 150.0,  97.0, 104.0,  94.0],
            [  0.0,   0.0,   0.0, 148.0, 101.0,  97.0],
            [  0.0,   0.0,   0.0,   0.0, 148.0,  96.0],
            [  0.0,   0.0,   0.0,   0.0,   0.0, 144.0],
        ];

        let mut total = 0.0;
        for i in 0..6 {
            for j in i..6 {
                let coef = pair_coef_oc(&patterns[i], &patterns[j]);
                assert_eq!(coef, expected_upper_triangle[i][j], "coef[{i}][{j}]");
                total += coef;
            }
        }
        assert_eq!(total, 2374.0);
        assert_eq!(RasbhariState::new(&patterns).total_score, 2374.0);
    }

    #[test]
    fn optimize_never_increases_overlap_score() {
        let mut set = SWPatternSet::random(6, 6, 20);
        let baseline = RasbhariState::new(set.patterns()).total_score;
        let optimized_score = set.optimize(3000);
        assert!(optimized_score <= baseline, "optimized {optimized_score} > baseline {baseline}");
    }

    #[test]
    fn optimize_preserves_shape_and_uniqueness() {
        let mut set = SWPatternSet::random(6, 6, 20);
        set.optimize(2000);
        assert_eq!(set.len(), 6);
        for pat in set.patterns() {
            assert_eq!(pat.weight(), 6);
            assert_eq!(pat.dontcare(), 20);
        }
        for i in 0..set.len() {
            for j in (i + 1)..set.len() {
                assert_ne!(set.patterns()[i], set.patterns()[j]);
            }
        }
    }

    #[test]
    fn with_limit_builds_valid_optimized_set() {
        let set = SWPatternSet::with_limit(5, 6, 20, 2000);
        assert_eq!(set.len(), 5);
        for pat in set.patterns() {
            assert_eq!(pat.weight(), 6);
            assert_eq!(pat.dontcare(), 20);
        }
    }

    #[test]
    fn new_builds_valid_set() {
        let set = SWPatternSet::new(5, 6, 20);
        assert_eq!(set.len(), 5);
        for pat in set.patterns() {
            assert_eq!(pat.weight(), 6);
            assert_eq!(pat.dontcare(), 20);
        }
    }
}
