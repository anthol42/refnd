use super::sw_pattern::{SWPattern, SWPatternSet};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SWWordError {
    /// A match was added after the word already had all `weight` of them.
    AlreadyComplete,
    /// A residue code didn't fit in 5 bits (must be `< 32`).
    ResidueTooLarge(u8),
    /// The word was finished (via the builder's `word()`) before it had all `weight`
    /// matches added.
    Incomplete { added: u8, weight: u8 },
}

impl std::fmt::Display for SWWordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SWWordError::AlreadyComplete => write!(f, "add_match called on an already-complete SWPartialWord"),
            SWWordError::ResidueTooLarge(r) => write!(f, "residue code {r} doesn't fit in 5 bits"),
            SWWordError::Incomplete { added, weight } => {
                write!(f, "SWPartialWord::word() called with only {added}/{weight} matches added")
            }
        }
    }
}

impl std::error::Error for SWWordError {}

/// A spaced word under construction: call [`Self::add_match`] once per match position
/// (in ascending order), then [`Self::word`] to get the finished, always-valid
/// [`SWWord`].
///
/// Residues pack 5 bits each into a `u64` key, which caps pattern weight at 12 (`12 * 5 = 60` bits).
struct SWPartialWord {
    key: u64,
    pos: usize,
    matches_added: u8,
    weight: u8,
}

impl SWPartialWord {
    /// Starts a word at `pos`, expecting `weight` calls to `add_match`. Panics if
    /// `weight` is 0 or greater than 12
    pub fn new(pos: usize, weight: usize) -> Self {
        assert!((1..=12).contains(&weight), "SWPartialWord weight must be between 1 and 12 to fit a u64 key, got {weight}");
        Self { key: 0, pos, matches_added: 0, weight: weight as u8 }
    }

    /// Packs one more match-position residue (a 5-bit amino-acid code, see
    /// [`encode_residue`]) into the key, most-significant first.
    pub fn add_match(&mut self, residue: u8) -> Result<(), SWWordError> {
        if self.matches_added >= self.weight {
            return Err(SWWordError::AlreadyComplete);
        }
        if residue >= 32 {
            return Err(SWWordError::ResidueTooLarge(residue));
        }
        self.key = (self.key << 5) | residue as u64;
        self.matches_added += 1;
        Ok(())
    }

    /// Finishes the word.
    pub fn word(self) -> Result<SWWord, SWWordError> {
        if self.matches_added != self.weight {
            return Err(SWWordError::Incomplete { added: self.matches_added, weight: self.weight });
        }
        Ok(SWWord { key: self.key, pos: self.pos })
    }
}

/// A Spaced Word: the residues at a pattern's match positions
#[derive(Clone, Copy, Debug, bincode::Encode, bincode::Decode)]
pub struct SWWord {
    key: u64,
    pos: usize,
}

impl SWWord {
    /// The packed key: the pattern's match-position residues, 5 bits each,
    /// most-significant residue first. Two words with equal keys have identical
    /// residues at every match position.
    pub fn key(&self) -> u64 {
        self.key
    }

    /// Start position of this word's window in the sequence it came from.
    pub fn pos(&self) -> usize {
        self.pos
    }
}

/// Equality/ordering is on `key` alone.
impl PartialEq for SWWord {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for SWWord {}

impl PartialOrd for SWWord {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SWWord {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// ProtSpaM's 25-symbol amino-acid alphabet, each mapped to a 5-bit code (0..=24): the
/// 20 standard amino acids, ambiguity codes B/Z/X, stop `*`, and J (Leu/Ile ambiguity).
/// Case-insensitive.
fn encode_residue(c: u8) -> Option<u8> {
    Some(match c.to_ascii_uppercase() {
        b'A' => 0,
        b'R' => 1,
        b'N' => 2,
        b'D' => 3,
        b'C' => 4,
        b'Q' => 5,
        b'E' => 6,
        b'G' => 7,
        b'H' => 8,
        b'I' => 9,
        b'L' => 10,
        b'K' => 11,
        b'M' => 12,
        b'F' => 13,
        b'P' => 14,
        b'S' => 15,
        b'T' => 16,
        b'W' => 17,
        b'Y' => 18,
        b'V' => 19,
        b'B' => 20,
        b'Z' => 21,
        b'X' => 22,
        b'*' => 23,
        b'J' => 24,
        _ => return None,
    })
}

/// A sequence prepared for spaced-word comparison (ProtSpaM-style). Its residues
/// are pre-encoded as 5-bit amino-acid codes. It also contains
/// the sorted Spaced Words per patterns.
///
/// Two `SWSequence`s must be built from the same [`SWPatternSet`] (or two equal clones
/// of it) to be compared meaningfully: [`Self::sorted_words`] is indexed positionally
/// by pattern index, not by pattern identity, so a mismatched pattern set won't
/// necessarily panic -- it can silently compare the wrong patterns' words against each
/// other. `refnd::kernels::protspam::ProtSpamKernel`, which is what actually compares
/// two of these, relies on this invariant.
#[derive(Clone, Debug, bincode::Encode, bincode::Decode)]
pub struct SWSequence {
    seq: Vec<u8>,
    /// `sorted_words[i]` holds the spaced words for `patterns.patterns()[i]`, sorted.
    sorted_words: Vec<Vec<SWWord>>,
}

impl SWSequence {
    /// Encodes `seq` and computes its sorted spaced words for every pattern in
    /// `patterns`.
    ///
    /// # Parameters
    /// - `seq`: the amino-acid sequence
    /// - `patterns`: the pattern set to compute spaced words for. Must be the same set
    ///   (or an equal clone) used for every other `SWSequence` this one will be
    ///   compared against, and for the kernel doing the comparing.
    ///
    /// # Errors
    /// Returns `Err` if `seq` contains a character outside ProtSpaM's amino-acid
    /// alphabet (see [`encode_residue`]: the 20 standard amino acids, ambiguity codes
    /// B/Z/X, stop `*`, and J; case-insensitive).
    ///
    /// # Examples
    /// ```
    /// use refnd::utils::{SWPatternSet, SWSequence};
    ///
    /// let patterns = SWPatternSet::random(3, 6, 10);
    /// let seq = SWSequence::new(&"MKTAYIAKQRQISFVKSHFSRQ".to_string(), &patterns).unwrap();
    /// assert_eq!(seq.len(), 22);
    ///
    /// assert!(SWSequence::new(&"MK?AY".to_string(), &patterns).is_err()); // '?' isn't a residue
    /// ```
    pub fn new(seq: &String, patterns: &SWPatternSet) -> Result<Self, String> {
        let seq: Vec<u8> = seq
            .bytes()
            .map(|c| encode_residue(c).ok_or_else(|| format!("invalid amino acid character '{}'", c as char)))
            .collect::<Result<_, _>>()?;
        let sorted_words = patterns.patterns().iter().map(|pattern| spaced_words(&seq, pattern)).collect();
        Ok(Self { seq, sorted_words })
    }

    /// Residues as 5-bit amino-acid codes (see [`encode_residue`]), not raw ASCII --
    /// e.g. `'A'` reads back as `0`, not `65`.
    pub fn seq(&self) -> &[u8] {
        &self.seq
    }

    /// Sequence length in residues.
    pub fn len(&self) -> usize {
        self.seq.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }

    /// Sorted spaced words for `patterns.patterns()[pattern_idx]`, where `patterns` is
    /// the set this sequence was built with (see the type-level docs on the
    /// same-pattern-set requirement). Empty if this sequence is shorter than that
    /// pattern (no window fits).
    ///
    /// # Panics
    /// Panics if `pattern_idx` is out of range for the pattern set this sequence was
    /// built with (i.e. `pattern_idx >= ` that set's `len()`, see [`Self::pattern_count`]).
    pub fn sorted_words(&self, pattern_idx: usize) -> &[SWWord] {
        &self.sorted_words[pattern_idx]
    }

    /// Number of patterns this sequence has spaced words for, i.e. the `len()` of the
    /// [`SWPatternSet`] it was built with. Valid indices for [`Self::sorted_words`]
    /// are `0..pattern_count()`.
    pub fn pattern_count(&self) -> usize {
        self.sorted_words.len()
    }
}

/// All spaced words a pattern produces sliding across an already-encoded `seq`, sorted
/// by key. Empty if `seq` is shorter than the pattern (no window fits).
fn spaced_words(seq: &[u8], pattern: &SWPattern) -> Vec<SWWord> {
    // `add_match`/`word` are infallible here by construction: `seq` was already validated
    // by `encode_residue` (every residue < 25 < 32), and the loop always supplies exactly
    // `pattern.weight()` matches -- so a `SWWordError` at this call site would mean
    // `spaced_words` itself is broken, not that the input was bad. This is why we can safely unwrap.
    if seq.len() < pattern.length() {
        return Vec::new();
    }
    let weight = pattern.weight();
    let mut words: Vec<SWWord> = (0..=seq.len() - pattern.length())
        .map(|pos| {
            let mut word = SWPartialWord::new(pos, weight);
            for &offset in pattern.match_positions() {
                word.add_match(seq[pos + offset]).expect("residue from an already-encoded sequence must fit 5 bits");
            }
            word.word().expect("loop adds exactly pattern.weight() matches")
        })
        .collect();
    words.sort_unstable();
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_word_add_match_then_word_packs_key() {
        let mut partial = SWPartialWord::new(0, 2);
        partial.add_match(0).unwrap(); // 'A'
        partial.add_match(3).unwrap(); // 'D'
        let word = partial.word().unwrap();
        assert_eq!(word.key(), 0b00000_00011); // (0 << 5) | 3
        assert_eq!(word.pos(), 0);
    }

    #[test]
    fn partial_word_word_errs_if_incomplete() {
        let mut partial = SWPartialWord::new(0, 2);
        partial.add_match(0).unwrap();
        assert_eq!(partial.word(), Err(SWWordError::Incomplete { added: 1, weight: 2 }));
    }

    #[test]
    fn partial_word_add_match_past_weight_errs() {
        let mut partial = SWPartialWord::new(0, 1);
        partial.add_match(0).unwrap();
        assert_eq!(partial.add_match(1), Err(SWWordError::AlreadyComplete));
    }

    #[test]
    fn partial_word_add_match_rejects_residue_above_five_bits() {
        let mut partial = SWPartialWord::new(0, 1);
        assert_eq!(partial.add_match(32), Err(SWWordError::ResidueTooLarge(32)));
    }

    #[test]
    #[should_panic(expected = "between 1 and 12")]
    fn partial_word_new_rejects_weight_above_twelve() {
        SWPartialWord::new(0, 13);
    }

    #[test]
    fn equality_and_ordering_ignore_pos() {
        let mut a = SWPartialWord::new(0, 1);
        a.add_match(5).unwrap();
        let mut b = SWPartialWord::new(99, 1);
        b.add_match(5).unwrap();
        let (a, b) = (a.word().unwrap(), b.word().unwrap());
        assert_eq!(a, b); // same key, different pos -> still equal
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn encode_residue_covers_the_25_symbol_alphabet_case_insensitively() {
        for c in b"ARNDCQEGHILKMFPSTWYVBZX*J" {
            let code = encode_residue(*c).unwrap();
            assert!(code < 25);
            assert_eq!(encode_residue(c.to_ascii_lowercase()), Some(code));
        }
        assert_eq!(encode_residue(b'?'), None);
    }

    #[test]
    fn new_rejects_invalid_characters() {
        let patterns = SWPatternSet::random(1, 2, 0);
        assert!(SWSequence::new(&"AC?E".to_string(), &patterns).is_err());
    }

    #[test]
    fn spaced_words_extract_correct_residues_and_positions() {
        let patterns = SWPatternSet::random(1, 2, 1); // single pattern, weight 2, dc 1 -> "101"
        let pattern = &patterns.patterns()[0];
        assert_eq!(pattern.length(), 3);

        let seq = SWSequence::new(&"ACDE".to_string(), &patterns).unwrap();
        let words = seq.sorted_words(0);
        assert_eq!(words.len(), 2); // windows at pos 0 and pos 1

        let match_offsets = pattern.match_positions();
        let expected_key = |pos: usize| match_offsets.iter().fold(0u64, |acc, &o| (acc << 5) | seq.seq()[pos + o] as u64);
        let mut expected: Vec<(u64, usize)> = (0..2).map(|pos| (expected_key(pos), pos)).collect();
        expected.sort_unstable();

        for (word, (key, pos)) in words.iter().zip(expected) {
            assert_eq!(word.key(), key);
            assert_eq!(word.pos(), pos);
        }
    }

    #[test]
    fn spaced_words_are_sorted_by_key() {
        let patterns = SWPatternSet::random(3, 6, 10);
        let seq = SWSequence::new(&"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK".to_string(), &patterns).unwrap();
        for i in 0..patterns.len() {
            let words = seq.sorted_words(i);
            assert!(words.windows(2).all(|w| w[0].key() <= w[1].key()));
        }
    }

    #[test]
    fn spaced_words_count_matches_window_count() {
        let patterns = SWPatternSet::random(2, 6, 10); // length 16
        let seq = SWSequence::new(&"A".repeat(40), &patterns).unwrap();
        for (i, pattern) in patterns.patterns().iter().enumerate() {
            assert_eq!(seq.sorted_words(i).len(), 40 - pattern.length() + 1);
        }
    }

    #[test]
    fn spaced_words_empty_when_sequence_shorter_than_pattern() {
        let patterns = SWPatternSet::random(1, 6, 10); // length 16
        let seq = SWSequence::new(&"SHRT".to_string(), &patterns).unwrap();
        assert!(seq.sorted_words(0).is_empty());
    }

    #[test]
    fn seq_accessors_expose_encoded_codes() {
        let patterns = SWPatternSet::random(1, 2, 0);
        let seq = SWSequence::new(&"ACDE".to_string(), &patterns).unwrap();
        assert_eq!(seq.seq(), &[0, 4, 3, 6]); // A, C, D, E
        assert_eq!(seq.len(), 4);
        assert!(!seq.is_empty());
    }
}
