## 12. Unicode Cell Width

> Source: `rich/cells.py` (175 lines), `rich/_cell_widths.py` (454 entries)

### 12.1 Cell Width Concept

Terminal cells have fixed width. Most characters occupy 1 cell, but some (CJK, emoji) occupy 2 cells. Rich must calculate cell width accurately for layout.

### 12.2 Cell Width Table

The `CELL_WIDTHS` table contains 454 entries of (start, end, width) tuples that define Unicode ranges with non-standard width:

```rust
/// Cell width lookup table
/// Each entry: (codepoint_start, codepoint_end, cell_width)
const CELL_WIDTHS: &[(u32, u32, usize)] = &[
    (0, 0, 0),           // NULL
    (1, 31, -1),         // C0 control (ignored)
    (127, 159, -1),      // C1 control (ignored)
    (768, 879, 0),       // Combining diacritical marks
    (1155, 1161, 0),     // Combining Cyrillic
    // ... 450+ more entries
    (4352, 4447, 2),     // Hangul Jamo
    (8986, 8987, 2),     // Watch, Hourglass
    (9193, 9203, 2),     // Various symbols
    (9725, 9726, 2),     // Medium squares
    // ... CJK ranges
    (12288, 12288, 2),   // Ideographic space
    (12289, 12350, 2),   // CJK punctuation
    (19968, 40956, 2),   // CJK Unified Ideographs
    // ... Emoji ranges
    (127744, 128591, 2), // Misc symbols/pictographs
    (128640, 128767, 2), // Transport/map symbols
    (129280, 129535, 2), // More emoji
];
```

### 12.3 Fast-Path Detection

For efficiency, single-cell ASCII is detected without table lookup:

```rust
/// Ranges known to be single-cell width
const SINGLE_CELL_RANGES: &[(u32, u32)] = &[
    (0x20, 0x7E),      // Basic ASCII printable
    (0xA0, 0x02FF),    // Latin Extended + IPA
    (0x0370, 0x0482),  // Greek
    // ... more known single-cell ranges
];

fn is_single_cell_fast(c: char) -> bool {
    let cp = c as u32;
    SINGLE_CELL_RANGES.iter().any(|(start, end)| cp >= *start && cp <= *end)
}
```

### 12.4 Cell Width Algorithm

```rust
/// Get cell width of a single character
fn get_character_cell_size(c: char) -> isize {
    let codepoint = c as u32;

    // Binary search in CELL_WIDTHS table
    let idx = CELL_WIDTHS.partition_point(|(start, _, _)| *start <= codepoint);

    if idx > 0 {
        let (start, end, width) = CELL_WIDTHS[idx - 1];
        if codepoint >= start && codepoint <= end {
            return width as isize;
        }
    }

    // Default: 1 cell
    1
}

/// Get total cell width of a string (cached)
fn cell_len(text: &str) -> usize {
    // Use thread-local cache
    CELL_LEN_CACHE.with(|cache| {
        if let Some(&cached) = cache.borrow().get(text) {
            return cached;
        }

        let width: usize = text.chars()
            .map(|c| get_character_cell_size(c).max(0) as usize)
            .sum();

        cache.borrow_mut().insert(text.to_string(), width);
        width
    })
}
```

### 12.5 Cell-Based String Operations

```rust
/// Truncate string to fit within cell width
fn set_cell_size(text: &str, total: usize) -> String {
    let current = cell_len(text);
    if current == total {
        return text.to_string();
    }
    if current < total {
        // Pad with spaces
        return format!("{}{}", text, " ".repeat(total - current));
    }

    // Need to truncate - use binary search
    let chars: Vec<char> = text.chars().collect();
    let mut pos = 0;
    let mut width = 0;

    // Find position where we exceed target
    while pos < chars.len() {
        let char_width = get_character_cell_size(chars[pos]).max(0) as usize;
        if width + char_width > total {
            break;
        }
        width += char_width;
        pos += 1;
    }

    let truncated: String = chars[..pos].iter().collect();

    // Pad if needed (due to wide character not fitting)
    if width < total {
        format!("{}{}", truncated, " ".repeat(total - width))
    } else {
        truncated
    }
}

/// Split string at cell position
fn chop_cells(text: &str, max_size: usize) -> (&str, &str) {
    let mut width = 0;
    let mut byte_pos = 0;

    for (i, c) in text.char_indices() {
        let char_width = get_character_cell_size(c).max(0) as usize;
        if width + char_width > max_size {
            break;
        }
        width += char_width;
        byte_pos = i + c.len_utf8();
    }

    (&text[..byte_pos], &text[byte_pos..])
}
```

---
