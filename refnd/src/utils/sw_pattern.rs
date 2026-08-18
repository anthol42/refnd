use rand::seq::index::sample as rand_sample;
use std::path::Path;
use std::str::FromStr;

/// A spaced-word pattern: a binary mask over `length` positions where a match position
/// ("1") contributes a residue to the spaced word's key and a don't-care position ("0")
/// is skipped when hashing but still compared for mismatches. Position 0 and the last
/// position are always match positions.
#[derive(Clone, Debug, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct SWPattern {
    pub(super) length: usize,
    /// Ascending, always contains 0 and `length - 1`.
    pub(super) match_positions: Vec<usize>,
}

impl SWPattern {
    /// Builds a pattern with `weight` match positions (including the fixed first and
    /// last) and `dont_care` don't-care positions, total length `weight + dont_care`.
    /// Positions in-between are randomly sampled.
    ///
    /// # Parameters
    /// - `weight`: number of match positions, including both endpoints. Must be `>= 2`.
    /// - `dont_care`: number of don't-care positions.
    ///
    /// # Panics
    /// Panics if `weight < 2` (there would be no way to place both required endpoints).
    ///
    /// # Examples
    /// ```
    /// use refnd::utils::SWPattern;
    ///
    /// let pat = SWPattern::random(6, 20);
    /// assert_eq!(pat.length(), 26);
    /// assert_eq!(pat.weight(), 6);
    /// assert_eq!(pat.dontcare(), 20);
    /// assert!(pat.is_match(0));
    /// assert!(pat.is_match(25));
    /// ```
    pub fn random(weight: usize, dont_care: usize) -> Self {
        assert!(weight >= 2, "pattern weight must be >= 2, got {weight}");
        let length = weight + dont_care;
        let mut match_positions = vec![0, length - 1];
        if length > 2 {
            match_positions.extend(rand_sample(&mut rand::rng(), length - 2, weight - 2).iter().map(|i| i + 1));
        }
        match_positions.sort_unstable();
        Self { length, match_positions }
    }

    /// Total number of positions (`weight() + dontcare()`).
    pub fn length(&self) -> usize {
        self.length
    }

    /// Number of match ("1") positions.
    pub fn weight(&self) -> usize {
        self.match_positions.len()
    }

    /// Number of don't-care ("0") positions (`length() - weight()`).
    pub fn dontcare(&self) -> usize {
        self.length - self.match_positions.len()
    }

    /// Ascending indices of match ("1") positions.
    pub fn match_positions(&self) -> &[usize] {
        &self.match_positions
    }

    /// Whether `pos` is a match position. Never panics: a `pos >= length()` simply
    /// isn't in [`Self::match_positions`], so it reads as `false` rather than an
    /// out-of-range error.
    pub fn is_match(&self, pos: usize) -> bool {
        self.match_positions.binary_search(&pos).is_ok()
    }

    /// Iterates `is_match(0), is_match(1), ..., is_match(length() - 1)` in order.
    pub fn iter(&self) -> SWPatternIterator<'_> {
        SWPatternIterator::new(self)
    }
}

/// Renders the pattern as a string of the same length: `'1'` for match positions,
/// `'0'` for don't-care positions -- the inverse of [`FromStr`].
impl std::fmt::Display for SWPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for pos in 0..self.length {
            write!(f, "{}", if self.is_match(pos) { '1' } else { '0' })?;
        }
        Ok(())
    }
}

impl FromStr for SWPattern {
    type Err = String;

    /// Parses a pattern from a string of `'1'`s (match) and `'0'`s (don't-care), the
    /// same format [`Display`](std::fmt::Display) produces.
    ///
    /// # Errors
    /// Returns `Err` if `s` contains a character other than `'0'`/`'1'`, or if it
    /// doesn't start and end with `'1'`.
    ///
    /// # Examples
    /// ```
    /// use refnd::utils::SWPattern;
    ///
    /// let pat: SWPattern = "10100101".parse().unwrap();
    /// assert_eq!(pat.weight(), 4);
    /// assert_eq!(pat.to_string(), "10100101");
    ///
    /// assert!("01100101".parse::<SWPattern>().is_err()); // doesn't start with '1'
    /// assert!("1010010x".parse::<SWPattern>().is_err()); // invalid character
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let length = s.len();
        let mut match_positions = Vec::new();
        for (pos, c) in s.chars().enumerate() {
            match c {
                '1' => match_positions.push(pos),
                '0' => {}
                other => return Err(format!("invalid pattern character '{other}', expected '0' or '1'")),
            }
        }
        if match_positions.first() != Some(&0) || match_positions.last() != Some(&(length - 1)) {
            return Err("pattern must start and end with a match position ('1')".to_string());
        }
        Ok(Self { length, match_positions })
    }
}

/// Iterator over a [`SWPattern`]'s positions, yielding `true` for a match position and
/// `false` for a don't-care position, from position `0` to `length() - 1`. Built by
/// [`SWPattern::iter`].
pub struct SWPatternIterator<'a> {
    pos: usize,
    pattern: &'a SWPattern,
}

impl<'a> SWPatternIterator<'a> {
    fn new(pattern: &'a SWPattern) -> Self {
        Self { pos: 0, pattern }
    }
}

impl Iterator for SWPatternIterator<'_> {
    type Item = bool;
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.pattern.length {
            None
        } else {
            let out = self.pattern.is_match(self.pos);
            self.pos += 1;
            Some(out)
        }
    }
}

/// A set of [`SWPattern`]s used together to compute spaced words for a sequence.
///
/// Build one with [`Self::new`] (RasBhari-optimized, recommended for real use.
/// [`Self::random`] (cheap and unoptimized), or [`Self::from_patterns`] (from
/// hand-picked patterns).
#[derive(Clone, Debug, PartialEq, Eq, bincode::Encode, bincode::Decode)]
pub struct SWPatternSet {
    pub(super) patterns: Vec<SWPattern>,
}

impl SWPatternSet {
    /// Builds `n` distinct unoptimized random patterns, each of the given `weight` and `dont_care`
    /// (see [`SWPattern::random`]).
    /// Useful as a cheap baseline, or as the unoptimized starting point. For an optimized pattern set, use the 
    /// [`new()`] method, or [`SWPattern::random`] followed by [`SWPattern::optimize`]
    ///
    /// # Parameters
    /// - `n`: number of distinct patterns to build.
    /// - `weight`, `dont_care`: passed to [`SWPattern::random`] for each pattern.
    ///
    /// # Panics
    /// Patterns are generated by rejection sampling on uniqueness, so this hangs
    /// (never returns) if `n` isn't well below the number of distinct patterns possible
    /// for `weight`/`dont_care` (`C(weight + dont_care - 2, weight - 2)`). This is
    /// sharpest at `weight == 2`: there are no interior positions to vary at all, so
    /// every generated pattern is identical and any `n > 1` hangs immediately.
    ///
    /// # Examples
    /// ```
    /// use refnd::utils::SWPatternSet;
    ///
    /// let set = SWPatternSet::random(5, 6, 20);
    /// assert_eq!(set.len(), 5);
    /// for pattern in set.patterns() {
    ///     assert_eq!(pattern.weight(), 6);
    ///     assert_eq!(pattern.dontcare(), 20);
    /// }
    /// ```
    pub fn random(n: usize, weight: usize, dont_care: usize) -> Self {
        let mut patterns: Vec<SWPattern> = Vec::with_capacity(n);
        while patterns.len() < n {
            let pat = SWPattern::random(weight, dont_care);
            if !patterns.contains(&pat) {
                patterns.push(pat);
            }
        }
        Self { patterns }
    }

    /// Builds a set from already-constructed patterns. Unlike [`Self::random`],
    /// there's no uniqueness check -- duplicate patterns are allowed, though they add
    /// nothing (two identical patterns always find exactly the same matches).
    pub fn from_patterns(patterns: Vec<SWPattern>) -> Self {
        Self { patterns }
    }

    /// The patterns in this set, in construction order.
    pub fn patterns(&self) -> &[SWPattern] {
        &self.patterns
    }

    /// Number of patterns in this set.
    pub fn len(&self) -> usize {
        self.patterns.len()
    }
    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Serializes this set to `path` via `bincode`. Inverse of [`Self::load`].
    ///
    /// # Errors
    /// Returns `Err` if `path` can't be written (e.g. permissions, missing parent
    /// directory), or if serialization itself fails (shouldn't happen for this type in
    /// practice).
    ///
    /// # Examples
    /// ```
    /// use refnd::utils::SWPatternSet;
    ///
    /// let set = SWPatternSet::random(4, 6, 20);
    /// let path = std::env::temp_dir().join("refnd_doctest_patterns.bin");
    /// set.save(&path).unwrap();
    ///
    /// let loaded = SWPatternSet::load(&path).unwrap();
    /// assert_eq!(set, loaded);
    /// ```
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let bytes = bincode::encode_to_vec(self, bincode::config::standard())?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Deserializes a set previously written by [`Self::save`].
    ///
    /// # Errors
    /// Returns `Err` if `path` can't be read, or its contents aren't a valid
    /// `bincode`-encoded `SWPatternSet`.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let (set, _) = bincode::decode_from_slice(&bytes, bincode::config::standard())?;
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_pattern_has_expected_shape() {
        let pat = SWPattern::random(6, 20);
        assert_eq!(pat.length(), 26);
        assert_eq!(pat.weight(), 6);
        assert_eq!(pat.dontcare(), 20);
        assert!(pat.is_match(0));
        assert!(pat.is_match(25));
    }

    #[test]
    fn random_pattern_minimal_weight_no_dontcare() {
        let pat = SWPattern::random(2, 0);
        assert_eq!(pat.match_positions(), &[0, 1]);
    }

    #[test]
    fn display_and_parse_roundtrip() {
        let pat = SWPattern::random(6, 20);
        let s = pat.to_string();
        assert_eq!(s.len(), 26);
        let parsed: SWPattern = s.parse().unwrap();
        assert_eq!(pat, parsed);
    }

    #[test]
    fn parse_rejects_bad_edges_and_chars() {
        assert!("01110".parse::<SWPattern>().is_err(), "must start with 1");
        assert!("10110".parse::<SWPattern>().is_err(), "must end with 1");
        assert!("1x110".parse::<SWPattern>().is_err(), "invalid character");
    }

    #[test]
    fn pattern_set_random_is_unique_and_uniform_shape() {
        let set = SWPatternSet::random(5, 6, 20);
        assert_eq!(set.len(), 5);
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
    fn save_load_roundtrip() {
        let set = SWPatternSet::random(4, 6, 20);
        let path = std::env::temp_dir().join("test_sw_patternset.bin");
        set.save(&path).unwrap();
        let loaded = SWPatternSet::load(&path).unwrap();
        assert_eq!(set, loaded);
    }
}
