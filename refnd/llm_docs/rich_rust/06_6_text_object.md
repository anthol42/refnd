## 6. Text Object

> Source: `rich/text.py` (1361 lines)

### 6.1 Text Data Structure

```rust
/// Justify method for text alignment
enum JustifyMethod {
    Default,  // Use console default
    Left,
    Center,
    Right,
    Full,     // Justify to fill width
}

/// Overflow handling method
enum OverflowMethod {
    Fold,     // Fold onto next line (default)
    Crop,     // Crop at boundary
    Ellipsis, // Show "..." at truncation
    Ignore,   // No overflow handling
}

/// A span of styled text (indices are CHARACTER offsets, not byte offsets)
struct Span {
    start: usize,   // Start character index (inclusive)
    end: usize,     // End character index (exclusive)
    style: Style,   // Style to apply
}

/// Rich text with spans
struct Text {
    plain: String,           // Plain text content (String of text pieces joined)
    spans: Vec<Span>,        // List of style spans
    length: usize,           // Cached character length
    style: Style,            // Base style for entire text
    justify: JustifyMethod,
    overflow: OverflowMethod,
    no_wrap: bool,           // Disable wrapping
    end: String,             // String to append after text (default "\n")
    tab_size: usize,         // Tab expansion size (default 8)
}
```

### 6.2 Span Management

**Span Invariants:**
- `start <= end`
- Spans can overlap (later spans take precedence in rendering)
- Indices are character positions, NOT byte positions

**Key Methods:**

```rust
impl Span {
    /// Right-adjust span by offset
    fn move_right(&self, offset: usize, max: usize) -> Span {
        Span {
            start: (self.start + offset).min(max),
            end: (self.end + offset).min(max),
            style: self.style.clone(),
        }
    }

    /// Split span at position
    fn split(&self, offset: usize) -> (Span, Span) {
        (
            Span { start: self.start, end: self.start + offset, style: self.style.clone() },
            Span { start: self.start + offset, end: self.end, style: self.style.clone() },
        )
    }
}
```

### 6.3 Text Manipulation Methods

```rust
impl Text {
    /// Create from plain string
    fn new(text: &str) -> Self;

    /// Create from markup string (parses [tags])
    fn from_markup(markup: &str) -> Self;

    /// Append plain text
    fn append(&mut self, text: &str);

    /// Append another Text object (merges spans)
    fn append_text(&mut self, text: &Text);

    /// Apply style to range
    fn stylize(&mut self, start: usize, end: usize, style: Style);

    /// Highlight text matching regex with style
    fn highlight_regex(&mut self, pattern: &str, style: Style);

    /// Highlight text matching string with style
    fn highlight_words(&mut self, words: &[&str], style: Style, case_sensitive: bool);

    /// Truncate to max width, adding suffix if needed
    fn truncate(&mut self, max_width: usize, overflow: OverflowMethod, pad: bool);

    /// Pad text to width
    fn pad(&mut self, width: usize, align: JustifyMethod);

    /// Split into lines at newlines
    fn split_lines(&self, split_on_space: bool) -> Vec<Text>;

    /// Get substring as new Text (preserves styles)
    fn slice(&self, start: usize, end: usize) -> Text;
}
```

### 6.4 Text Division Algorithm (CRITICAL)

The `divide()` method splits Text at specified cut points while preserving spans.

```rust
/// Divide text into parts at specified character offsets
fn divide(&self, offsets: &[usize]) -> Vec<Text> {
    if offsets.is_empty() {
        return vec![self.clone()];
    }

    let text_length = self.length;
    let mut result = Vec::with_capacity(offsets.len());

    // For each span, distribute to appropriate output divisions
    for span in &self.spans {
        // Use binary search to find which divisions this span overlaps
        let lower = match offsets.binary_search(&span.start) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let upper = match offsets.binary_search(&span.end) {
            Ok(i) => i,
            Err(i) => i,
        };

        // Span may appear in multiple divisions
        for div_idx in lower..=upper {
            let div_start = if div_idx == 0 { 0 } else { offsets[div_idx - 1] };
            let div_end = offsets.get(div_idx).copied().unwrap_or(text_length);

            // Calculate span position relative to division
            let rel_start = span.start.saturating_sub(div_start);
            let rel_end = span.end.min(div_end).saturating_sub(div_start);

            if rel_start < rel_end {
                // Add adjusted span to this division
                result[div_idx].spans.push(Span {
                    start: rel_start,
                    end: rel_end,
                    style: span.style.clone(),
                });
            }
        }
    }

    result
}
```

### 6.5 Text Rendering to Segments

```rust
/// Render Text to iterator of Segments
fn render(&self, console: &Console, end: &str) -> Vec<Segment> {
    // Style combination cache for performance
    let mut style_cache: HashMap<usize, Style> = HashMap::new();

    let null_style = Style::null();
    let enumerated_spans: Vec<(usize, &Span)> = self.spans.iter().enumerate().collect();

    let mut result = Vec::new();

    // Build a map: character position -> list of (span_index, is_start)
    let mut events: BTreeMap<usize, Vec<(usize, bool)>> = BTreeMap::new();
    for (idx, span) in &enumerated_spans {
        events.entry(span.start).or_default().push((*idx, true));  // start
        events.entry(span.end).or_default().push((*idx, false));   // end
    }

    // Walk through text, tracking active spans via stack
    let mut active_spans: Vec<usize> = Vec::new();  // Stack of span indices
    let mut pos = 0;
    let chars: Vec<char> = self.plain.chars().collect();

    for (event_pos, span_events) in events {
        // Emit text before this event
        if event_pos > pos {
            let text: String = chars[pos..event_pos].iter().collect();
            let style = compute_combined_style(&active_spans, &enumerated_spans, &self.style, &mut style_cache);
            result.push(Segment { text, style: Some(style), control: None });
            pos = event_pos;
        }

        // Process events (ends before starts for correct nesting)
        for (span_idx, is_start) in span_events {
            if is_start {
                active_spans.push(span_idx);
            } else {
                active_spans.retain(|&x| x != span_idx);
            }
        }
    }

    // Emit remaining text
    if pos < chars.len() {
        let text: String = chars[pos..].iter().collect();
        let style = compute_combined_style(&active_spans, &enumerated_spans, &self.style, &mut style_cache);
        result.push(Segment { text, style: Some(style), control: None });
    }

    // Append end string
    if !end.is_empty() {
        result.push(Segment { text: end.to_string(), style: None, control: None });
    }

    result
}

/// Combine styles from active spans (stack-based, later spans override)
fn compute_combined_style(
    active_spans: &[usize],
    spans: &[(usize, &Span)],
    base_style: &Style,
    cache: &mut HashMap<usize, Style>,
) -> Style {
    // Create cache key from active span indices
    let cache_key = hash(active_spans);
    if let Some(cached) = cache.get(&cache_key) {
        return cached.clone();
    }

    let mut combined = base_style.clone();
    for &span_idx in active_spans {
        combined = combined + spans[span_idx].1.style.clone();
    }

    cache.insert(cache_key, combined.clone());
    combined
}
```

### 6.6 Text Wrapping

```rust
/// Wrap text to fit within width
fn wrap(
    &self,
    console: &Console,
    width: usize,
    justify: JustifyMethod,
    overflow: OverflowMethod,
    tab_size: usize,
    no_wrap: bool,
) -> Vec<Text> {
    // Expand tabs first
    let expanded = self.expand_tabs(tab_size);

    // If no_wrap or width is huge, return as single line
    if no_wrap || width >= expanded.cell_len() {
        return vec![expanded];
    }

    let mut lines = Vec::new();

    for line in expanded.split_lines(false) {
        if line.cell_len() <= width {
            lines.push(line);
        } else {
            // Need to wrap this line
            match overflow {
                OverflowMethod::Fold => {
                    lines.extend(wrap_fold(&line, width));
                }
                OverflowMethod::Crop => {
                    lines.push(line.slice(0, width));
                }
                OverflowMethod::Ellipsis => {
                    let mut truncated = line.slice(0, width.saturating_sub(1));
                    truncated.append("...");
                    lines.push(truncated);
                }
                OverflowMethod::Ignore => {
                    lines.push(line);
                }
            }
        }
    }

    // Apply justification
    for line in &mut lines {
        line.apply_justify(justify, width);
    }

    lines
}
```

---
