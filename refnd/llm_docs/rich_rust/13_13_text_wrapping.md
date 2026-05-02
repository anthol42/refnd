## 13. Text Wrapping

> Source: `rich/_wrap.py` (94 lines)

### 13.1 Word Tokenizer

```rust
/// Regex pattern for word extraction
/// Matches: optional leading whitespace + non-whitespace + optional trailing whitespace
const RE_WORD: &str = r"\s*\S+\s*";

/// Split text into words (preserving whitespace)
fn words(text: &str) -> Vec<&str> {
    let re = Regex::new(RE_WORD).unwrap();
    re.find_iter(text).map(|m| m.as_str()).collect()
}
```

### 13.2 Line Division Algorithm

```rust
/// Divide a single line of text at specified width
/// Returns: (line_content, remaining_text, has_more)
fn divide_line(text: &str, width: usize, fold: bool) -> Vec<(usize, usize)> {
    let mut breaks = Vec::new();
    let mut line_start = 0;
    let mut line_width = 0;

    for word in words(text) {
        let word_start = word.as_ptr() as usize - text.as_ptr() as usize;
        let word_width = cell_len(word.trim_end());  // Don't count trailing space

        if line_width > 0 && line_width + word_width > width {
            // Word doesn't fit, break here
            breaks.push((line_start, word_start));
            line_start = word_start;
            line_width = 0;
        }

        if fold && word_width > width {
            // Word itself is too wide, must fold within word
            let mut remaining = word;
            while cell_len(remaining) > width {
                let (chunk, rest) = chop_cells(remaining, width);
                let chunk_end = line_start + (chunk.as_ptr() as usize - text.as_ptr() as usize) + chunk.len();
                breaks.push((line_start, chunk_end));
                line_start = chunk_end;
                remaining = rest;
            }
            line_width = cell_len(remaining);
        } else {
            line_width += word_width;
        }
    }

    // Final segment
    if line_start < text.len() {
        breaks.push((line_start, text.len()));
    }

    breaks
}
```

### 13.3 Full Text Wrapping

```rust
/// Wrap text to fit within width
fn wrap_text(text: &str, width: usize, fold: bool) -> Vec<String> {
    let mut lines = Vec::new();

    // Process each existing line
    for line in text.split('\n') {
        if line.is_empty() {
            lines.push(String::new());
            continue;
        }

        let breaks = divide_line(line, width, fold);
        for (start, end) in breaks {
            let segment = &line[start..end];
            // Trim trailing whitespace from wrapped lines
            lines.push(segment.trim_end().to_string());
        }
    }

    lines
}
```

---
