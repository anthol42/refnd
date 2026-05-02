## 3. Segment (Rendering Atom)

> Source: `rich/segment.py` (752 lines)

### 3.1 ControlType Enum

```rust
enum ControlType {
    BELL = 1,
    CARRIAGE_RETURN = 2,
    HOME = 3,
    CLEAR = 4,
    SHOW_CURSOR = 5,
    HIDE_CURSOR = 6,
    ENABLE_ALT_SCREEN = 7,
    DISABLE_ALT_SCREEN = 8,
    CURSOR_UP = 9,
    CURSOR_DOWN = 10,
    CURSOR_FORWARD = 11,
    CURSOR_BACKWARD = 12,
    CURSOR_MOVE_TO_COLUMN = 13,
    CURSOR_MOVE_TO = 14,
    ERASE_IN_LINE = 15,
    SET_WINDOW_TITLE = 16,
}
```

### 3.2 Segment Structure

```rust
struct Segment {
    text: String,
    style: Option<Style>,
    control: Option<Vec<ControlCode>>,
}

impl Segment {
    fn cell_length(&self) -> usize {
        if self.control.is_some() { 0 } else { cell_len(&self.text) }
    }

    fn is_control(&self) -> bool {
        self.control.is_some()
    }
}
```

### 3.3 Segment Operations

#### Line Creation
```rust
fn line() -> Segment { Segment { text: "\n".into(), style: None, control: None } }
```

#### Style Application
```rust
fn apply_style(segments: impl Iterator<Item=Segment>, style: Option<Style>, post_style: Option<Style>) -> impl Iterator<Item=Segment>
```
- If style provided: applies `style + segment.style`
- If post_style provided: applies `segment.style + post_style`

#### Line Splitting
```rust
fn split_lines(segments: impl Iterator<Item=Segment>) -> impl Iterator<Item=Vec<Segment>>
```
Splits at newline characters. Each yielded Vec is one line (excluding newline).

#### Line Length Adjustment
```rust
fn adjust_line_length(line: Vec<Segment>, length: usize, style: Option<Style>, pad: bool) -> Vec<Segment>
```
- If line shorter than length and pad=true: appends padding
- If line longer: truncates (may split segments)
- Control segments never truncated

#### Simplification
```rust
fn simplify(segments: impl Iterator<Item=Segment>) -> impl Iterator<Item=Segment>
```
Merges contiguous segments with identical styles.

#### Division
```rust
fn divide(segments: impl Iterator<Item=Segment>, cuts: impl Iterator<Item=usize>) -> impl Iterator<Item=Vec<Segment>>
```
Divides segments at specified cell positions.

#### Alignment Methods
```rust
fn align_top(lines: Vec<Vec<Segment>>, width: usize, height: usize, style: Style) -> Vec<Vec<Segment>>
fn align_bottom(lines: Vec<Vec<Segment>>, width: usize, height: usize, style: Style) -> Vec<Vec<Segment>>
fn align_middle(lines: Vec<Vec<Segment>>, width: usize, height: usize, style: Style) -> Vec<Vec<Segment>>
```

---
