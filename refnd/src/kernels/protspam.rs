use crate::core::Distance;
use crate::utils::{SWPattern, SWPatternSet, SWSequence, SWWord};

/// BLOSUM62 substitution matrix, indexed by ProtSpaM's 25-symbol amino-acid encoding
/// (`utils::sw_sequence::encode_residue`)
#[rustfmt::skip]
const BLOSUM62: [[i32; 24]; 24] = [
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4],
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4],
    [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4],
    [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
    [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4],
    [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1],
];

/// Best-scoring pair found in one spaced-word block-pair cross product.
struct BlockBest {
    score: i32,
    mismatches: u32,
}

/// Which distance [`ProtSpamKernel::call`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtSpamDistance {
    /// The raw mismatch rate itself (fraction of don't-care positions that mismatch,
    /// pooled across every pattern's accepted spaced-word matches) -- always in
    /// `[0, 1]`. `1.0` if no spaced word matched at all or all don't care position mismatch.
    MismatchRate,
    /// The mismatch rate corrected into a distance via the Kimura two-parameter
    /// formula -- always in `[0, inf]`.
    Evolutionary,
}

/// Distance between two [`SWSequence`]s, ProtSpaM-style: for each pattern in a shared
/// [`SWPatternSet`], match spaced words by key (grouping ties into "blocks"), score the
/// best-aligned pair within each matching block against BLOSUM62, and pool mismatches
/// at don't-care positions across all accepted matches into a mismatch rate. Reports
/// either that raw rate or a Kimura-corrected evolutionary distance, per
/// [`ProtSpamDistance`].
pub struct ProtSpamKernel {
    pub patterns: SWPatternSet,
    /// Minimum BLOSUM62 score (over don't-care positions) for a spaced-word match to be
    /// considered homologous. ProtSpaM's own default is 0.
    pub threshold: i32,
    pub distance: ProtSpamDistance,
}

impl ProtSpamKernel {
    pub fn new(patterns: SWPatternSet, threshold: i32, distance: ProtSpamDistance) -> Self {
        Self { patterns, threshold, distance }
    }

    /// Number of consecutive words starting at `start` that share the same key
    /// (spaced words are pre-sorted by key, so ties are always adjacent).
    fn count_consecutive(words: &[SWWord], start: usize) -> usize {
        let key = words[start].key();
        let mut len = 1;
        while start + len < words.len() && words[start + len].key() == key {
            len += 1;
        }
        len
    }

    /// BLOSUM62 score for one residue pair, routing any code outside BLOSUM62's 24x24
    /// table to `X`'s ("unknown")
    fn blosum62_score(a: u8, b: u8) -> i32 {
        const X: u8 = 22;
        let idx = |c: u8| if c > 23 { X } else { c } as usize; // mirrors parasail: anything outside the 24x24 table falls back to X
        BLOSUM62[idx(a)][idx(b)]
    }

    /// BLOSUM62 score and mismatch count over `pattern`'s don't-care positions, for one
    /// candidate alignment of two windows starting at `pos1`/`pos2`.
    fn score_pair(seq1: &[u8], seq2: &[u8], pos1: usize, pos2: usize, pattern: &SWPattern) -> (i32, u32) {
        let mut score = 0;
        let mut mismatches = 0;
        for (offset, is_match) in pattern.iter().enumerate() {
            if !is_match {
                let (a, b) = (seq1[pos1 + offset], seq2[pos2 + offset]);
                score += Self::blosum62_score(a, b);
                mismatches += (a != b) as u32;
            }
        }
        (score, mismatches)
    }

    /// Best-scoring pair over the full cross product of two matching spaced-word blocks
    /// (words `[i, i+bl1)` in `words1`, `[j, j+bl2)` in `words2`, all sharing the same key),
    /// or a below-threshold placeholder if nothing in the cross product reaches
    /// `self.threshold`.
    #[allow(clippy::too_many_arguments)]
    fn best_over_block_pair(
        &self,
        seq1: &[u8],
        seq2: &[u8],
        words1: &[SWWord],
        pos1: usize,
        block_length1: usize,
        words2: &[SWWord],
        pos2: usize,
        block_length2: usize,
        pattern: &SWPattern,
    ) -> BlockBest {
        let mut best = BlockBest { score: self.threshold - 1, mismatches: 0 };
        for a in pos1..pos1 + block_length1 {
            for b in pos2..pos2 + block_length2 {
                let (score, mismatches) = Self::score_pair(seq1, seq2, words1[a].pos(), words2[b].pos(), pattern);
                if score >= self.threshold && score > best.score {
                    best = BlockBest { score, mismatches };
                }
            }
        }
        best
    }

    /// Total (mismatches, don't-care positions) contributed by one pattern: walks both
    /// sorted spaced-word lists in lockstep, matching equal keys block by block.
    fn pattern_mismatches(&self, seq1: &[u8], seq2: &[u8], words1: &[SWWord], words2: &[SWWord], pattern: &SWPattern) -> (u64, u64) {
        let dc = pattern.dontcare() as u64;
        let mut total_mismatches = 0u64;
        let mut total_dc = 0u64; // total don't care position
        let mut i = 0usize;
        let mut j = 0usize;
        while i < words1.len() {
            let block_length1 = Self::count_consecutive(words1, i);
            while j < words2.len() {
                let block_length2 = Self::count_consecutive(words2, j);
                // Fast-forward in sequence 1 to catch up sequence 2
                if words1[i].key() < words2[j].key() {
                    break;
                }
                // Fast-forward in sequence 2 to catch up sequence 1
                if words1[i].key() > words2[j].key() {
                    j += block_length2;
                    continue;
                }
                // Here, SW1 == SW2
                let best = self.best_over_block_pair(seq1, seq2, words1, i, block_length1, words2, j, block_length2, pattern);
                if best.score >= self.threshold {
                    total_mismatches += best.mismatches as u64;
                    total_dc += dc;
                }
                j += block_length2;
                break;
            }
            i += block_length1;
        }
        (total_mismatches, total_dc)
    }

    /// Fraction of don't-care positions that mismatch, pooled across every pattern's
    /// accepted spaced-word matches. NaN if no match was accepted anywhere (0/0).
    fn mismatch_rate(&self, s1: &SWSequence, s2: &SWSequence) -> f64 {
        let (mismatches, dc) = self.patterns.patterns().iter().enumerate()
            .fold((0u64, 0u64),
                  |(m, d), (idx, pattern)| {
                        let (pm, pd) = self.pattern_mismatches(
                            s1.seq(), s2.seq(),
                            s1.sorted_words(idx), s2.sorted_words(idx),
                            pattern
                        );
                        (m + pm, d + pd)
        });
        mismatches as f64 / dc as f64
    }
}

impl Distance<SWSequence> for ProtSpamKernel {
    fn call(&self, ref_sample: &SWSequence, query: &SWSequence) -> f32 {
        let mmr = self.mismatch_rate(ref_sample, query);
        match self.distance {
            // mismatch_rate is a 0/0 NaN exactly when no spaced word matched at all;
            // report that as maximally dissimilar (1.0) rather than propagating NaN,
            // for the same reason Evolutionary avoids it below.
            ProtSpamDistance::MismatchRate => {
                if mmr.is_nan() { 1.0 } else { mmr as f32 }
            }
            ProtSpamDistance::Evolutionary => {
                // Kimura two-parameter correction. Its log argument is non-positive (or
                // NaN, from that same 0/0 mismatch rate) once mmr exceeds ~0.8541.
                // Rather than propagate NaN into distance comparisons downstream --
                // which silently break HNSW's ordering, since NaN compares false
                // against everything -- report these pairs as infinitely distant.
                // `!(arg > 0.0)` (not `arg <= 0.0`) so NaN is caught too.
                let arg = 1.0 - mmr - 0.2 * mmr * mmr;
                if !(arg > 0.0) {
                    return f32::INFINITY;
                }
                -arg.ln() as f32
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern() -> SWPattern {
        "10100101".parse().unwrap() // weight 4, dc 4, length 8
    }

    #[test]
    fn multi_match_counts_consecutive_equal_keys() {
        let patterns = SWPatternSet::from_patterns(vec![pattern()]);
        let seq = SWSequence::new("AAAAAAAAAAAA".to_string(), &patterns).unwrap();
        let words = seq.sorted_words(0);
        // every window is all-'A' -> one giant block sharing the same key
        assert_eq!(ProtSpamKernel::count_consecutive(words, 0), words.len());
    }

    #[test]
    fn blosum62_score_diagonal_and_out_of_range_maps_to_x() {
        assert_eq!(ProtSpamKernel::blosum62_score(0, 0), 4); // A vs A
        assert_eq!(ProtSpamKernel::blosum62_score(17, 17), 11); // W vs W
        assert_eq!(ProtSpamKernel::blosum62_score(24, 24), ProtSpamKernel::blosum62_score(22, 22)); // J vs J == X vs X
        assert_eq!(ProtSpamKernel::blosum62_score(24, 0), ProtSpamKernel::blosum62_score(22, 0)); // J vs A == X vs A
        assert_eq!(ProtSpamKernel::blosum62_score(30, 0), ProtSpamKernel::blosum62_score(22, 0)); // any code > 23, not just J, falls back to X
    }

    #[test]
    fn score_pair_only_scores_dontcare_positions() {
        let pat = pattern(); // matches at 0,2,5,7; don't-care at 1,3,4,6
        let seq1 = [0u8; 8]; // all 'A'
        let mut seq2 = [0u8; 8];
        seq2[0] = 19; // differs at a MATCH position -> must not affect score/mismatches
        seq2[1] = 4; // differs at a don't-care position ('C') -> must be counted
        let (score, mismatches) = ProtSpamKernel::score_pair(&seq1, &seq2, 0, 0, &pat);
        assert_eq!(mismatches, 1);
        // 3 dc positions still A/A (score 4 each) + 1 dc position A/C (score 0)
        assert_eq!(score, 4 + 4 + 4 + 0);
    }

    #[test]
    fn call_is_zero_for_identical_sequences() {
        let patterns = SWPatternSet::from_patterns(vec![pattern()]);
        let s1 = SWSequence::new("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK".to_string(), &patterns).unwrap();
        let s2 = SWSequence::new("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK".to_string(), &patterns).unwrap();
        let k = ProtSpamKernel::new(patterns, 0, ProtSpamDistance::Evolutionary);
        assert_eq!(k.mismatch_rate(&s1, &s2), 0.0);
        assert_eq!(k.call(&s1, &s2), 0.0);
    }

    #[test]
    fn call_is_infinite_when_no_spaced_word_matches() {
        let patterns = SWPatternSet::from_patterns(vec![pattern()]);
        let s1 = SWSequence::new("A".repeat(52), &patterns).unwrap();
        let s2 = SWSequence::new("W".repeat(52), &patterns).unwrap();
        let k = ProtSpamKernel::new(patterns, 0, ProtSpamDistance::Evolutionary);
        assert!(k.mismatch_rate(&s1, &s2).is_nan());
        assert_eq!(k.call(&s1, &s2), f32::INFINITY);
    }

    #[test]
    fn mismatch_rate_mode_returns_raw_rate() {
        let s1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK";
        let mut s2_bytes = s1.as_bytes().to_vec();
        s2_bytes[20] = b'A';
        let s2 = String::from_utf8(s2_bytes).unwrap();

        let patterns = SWPatternSet::from_patterns(vec![pattern()]);
        let seq1 = SWSequence::new(s1.to_string(), &patterns).unwrap();
        let seq2 = SWSequence::new(s2, &patterns).unwrap();
        let k = ProtSpamKernel::new(patterns, 0, ProtSpamDistance::MismatchRate);

        assert_eq!(k.call(&seq1, &seq2), (1.0 / 42.0) as f32);
    }

    #[test]
    fn mismatch_rate_mode_is_one_when_no_spaced_word_matches() {
        let patterns = SWPatternSet::from_patterns(vec![pattern()]);
        let s1 = SWSequence::new("A".repeat(52), &patterns).unwrap();
        let s2 = SWSequence::new("W".repeat(52), &patterns).unwrap();
        let k = ProtSpamKernel::new(patterns, 0, ProtSpamDistance::MismatchRate);
        assert_eq!(k.call(&s1, &s2), 1.0);
    }

    /// Cross-validated against the real ProtSpaM C++ implementation: `Species` built
    /// directly (bypassing file parsing), `spacedWords`/`calc_matches` from the actual
    /// `include/misc.h`/`calc_matches.h` sources, compiled and linked against ProtSpaM's
    /// own `.o` files (see `~/Downloads/ProtSpaM/xval_kernel.cpp`), pattern
    /// "10100101" (weight 4, dc 4), threshold 0.
    #[test]
    fn matches_protspam_reference_on_a_single_point_mutation() {
        let s1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK";
        let mut s2_bytes = s1.as_bytes().to_vec();
        s2_bytes[20] = b'A'; // mutate a middle residue, matching the C++ harness exactly
        let s2 = String::from_utf8(s2_bytes).unwrap();

        let patterns = SWPatternSet::from_patterns(vec![pattern()]);
        let seq1 = SWSequence::new(s1.to_string(), &patterns).unwrap();
        let seq2 = SWSequence::new(s2, &patterns).unwrap();
        let k = ProtSpamKernel::new(patterns, 0, ProtSpamDistance::Evolutionary);

        assert_eq!(k.mismatch_rate(&seq1, &seq2), 1.0 / 42.0);
        assert!((k.call(&seq1, &seq2) - 0.024213702342882334_f32).abs() < 1e-6);
    }
}
