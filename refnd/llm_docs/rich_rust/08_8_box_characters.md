## 8. Box Characters

> Source: `rich/box.py` (474 lines)

### 8.1 Box Data Structure

Box characters are defined as an 8-line string, one character per position:

```rust
/// Box drawing definition
/// Format: 8 lines of 4 characters each
///   Line 0: top (left, middle, divider, right)
///   Line 1: head (left, center, vertical, right)
///   Line 2: head_row (left, middle, cross, right)
///   Line 3: mid (left, middle, cross, right)
///   Line 4: row (left, middle, cross, right)
///   Line 5: foot_row (left, middle, cross, right)
///   Line 6: foot (left, center, vertical, right)
///   Line 7: bottom (left, middle, divider, right)
struct Box {
    top: [char; 4],
    head: [char; 4],
    head_row: [char; 4],
    mid: [char; 4],
    row: [char; 4],
    foot_row: [char; 4],
    foot: [char; 4],
    bottom: [char; 4],
    ascii: bool,  // Whether this is ASCII-safe
}

impl Box {
    /// Parse from 8-line string format
    fn from_str(s: &str) -> Self;

    /// Get top row string for given widths
    fn get_top(&self, widths: &[usize]) -> String;

    /// Get bottom row string for given widths
    fn get_bottom(&self, widths: &[usize]) -> String;

    /// Get separator row string for given widths
    fn get_row(
        &self,
        widths: &[usize],
        level: RowLevel,  // Head, Mid, Foot, Row
        edge: bool,       // Include edge characters
    ) -> String;
}
```

### 8.2 Built-in Box Styles

```
ASCII:
+--+
| ||
|--+
|--+
|-+|
|--+
| ||
+--+

ASCII2:
+-++
| ||
+-++
+-++
+-++
+-++
| ||
+-++

ASCII_DOUBLE_HEAD:
+-++
| ||
+=++
|-+|
|-+|
|-+|
| ||
+-++

SQUARE:
+--+
| ||
+--+
+--+
+-++
+--+
| ||
+--+

SQUARE_DOUBLE_HEAD:
+--+
| ||
+==+
+--+
+-++
+--+
| ||
+--+

MINIMAL:
    (spaces)
| ||
+--+



| ||


MINIMAL_HEAVY_HEAD:

| ||
+==+



| ||


MINIMAL_DOUBLE_HEAD:

| ||
+==+



| ||


SIMPLE:


+--+


+--+



SIMPLE_HEAD:


+--+






SIMPLE_HEAVY:


+==+


+==+



HORIZONTALS:
+--+

+--+
+--+
+--+
+--+

+--+

ROUNDED:
(Uses Unicode rounded corners: ., ', etc.)
.--,
| ||
|--+
|--+
|-+|
|--+
| ||
`--'

HEAVY:
+==+
# ##
+=++
+=++
+=++
+=++
# ##
+==+

HEAVY_EDGE:
+==+
| ||
+--+
+--+
+-++
+--+
| ||
+==+

HEAVY_HEAD:
+--+
| ||
+==+
+--+
+-++
+--+
| ||
+--+

DOUBLE:
+==+
| ||
+=++
+=++
+=++
+=++
| ||
+==+

DOUBLE_EDGE:
+==+
| ||
+--+
+--+
+-++
+--+
| ||
+==+

MARKDOWN:

| ||
|-||



| ||

```

**Note:** The above uses ASCII placeholders. Actual Unicode characters:
- `+` variants: `+`, `+`, `+`, `+` (corners)
- `-` variants: `-`, `=`, `_` (horizontal)
- `|` variants: `|`, `||`, `#` (vertical)
- Rounded: `.-,/` corner variants

### 8.3 Box Substitution Maps

**LEGACY_WINDOWS_SUBSTITUTIONS:**
Maps Unicode box characters to ASCII equivalents for legacy Windows console:

```rust
const LEGACY_WINDOWS_SUBSTITUTIONS: &[(&str, &str)] = &[
    ("-", "-"),    // Heavy horizontal to light
    ("|", "|"),    // Heavy vertical to light
    // ... more mappings for double-line and rounded characters
];
```

**PLAIN_HEADED_SUBSTITUTIONS:**
Maps SQUARE boxes to SQUARE_DOUBLE_HEAD when header style is needed.

### 8.4 Row Generation Methods

```rust
impl Box {
    /// Generate a row with given column widths
    fn get_row(&self, widths: &[usize], level: RowLevel, edge: bool) -> String {
        let (left, mid, cross, right) = match level {
            RowLevel::Top => self.top,
            RowLevel::Head => self.head_row,
            RowLevel::Mid => self.mid,
            RowLevel::Row => self.row,
            RowLevel::Foot => self.foot_row,
            RowLevel::Bottom => self.bottom,
        };

        let mut result = String::new();

        if edge {
            result.push(left);
        }

        for (i, &width) in widths.iter().enumerate() {
            // Add horizontal chars to fill width
            for _ in 0..width {
                result.push(mid);
            }
            // Add cross or right edge
            if i < widths.len() - 1 {
                result.push(cross);
            }
        }

        if edge {
            result.push(right);
        }

        result
    }
}
```

---
