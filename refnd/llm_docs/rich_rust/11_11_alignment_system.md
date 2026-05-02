## 11. Alignment System

> Source: `rich/align.py` (307 lines)

### 11.1 Alignment Types

```rust
/// Horizontal alignment methods
enum AlignMethod {
    Left,
    Center,
    Right,
}

/// Vertical alignment methods
enum VerticalAlignMethod {
    Top,
    Middle,
    Bottom,
}
```

### 11.2 Align Renderable

```rust
struct Align {
    renderable: Box<dyn Renderable>,
    align: AlignMethod,           // Horizontal alignment
    style: Style,                 // Background/fill style
    vertical: VerticalAlignMethod,
    pad: bool,                    // Pad lines to width
    width: Option<usize>,         // Override width
    height: Option<usize>,        // Override height
}

impl Align {
    fn left(renderable: impl Renderable) -> Self {
        Align { align: AlignMethod::Left, ..Self::new(renderable) }
    }

    fn center(renderable: impl Renderable) -> Self {
        Align { align: AlignMethod::Center, ..Self::new(renderable) }
    }

    fn right(renderable: impl Renderable) -> Self {
        Align { align: AlignMethod::Right, ..Self::new(renderable) }
    }
}
```

### 11.3 Alignment Rendering

```rust
impl Renderable for Align {
    fn rich_console(&self, console: &Console, options: &ConsoleOptions) -> Vec<RenderItem> {
        let width = self.width.unwrap_or(options.max_width);
        let height = self.height;

        // Render inner content
        let inner_options = options.update_width(width);
        let lines = console.render_lines(
            &*self.renderable,
            &inner_options,
            None,
            false,
            false,
        );

        let mut result_lines = Vec::new();

        for mut line in lines {
            let line_width: usize = line.iter().map(|s| s.cell_length()).sum();
            let excess = width.saturating_sub(line_width);

            match self.align {
                AlignMethod::Left => {
                    // Content on left, padding on right
                    if self.pad && excess > 0 {
                        line.push(Segment::new(&" ".repeat(excess), Some(self.style.clone())));
                    }
                }
                AlignMethod::Center => {
                    // Split padding between left and right
                    let left_pad = excess / 2;
                    let right_pad = excess - left_pad;
                    let mut new_line = Vec::new();
                    if left_pad > 0 {
                        new_line.push(Segment::new(&" ".repeat(left_pad), Some(self.style.clone())));
                    }
                    new_line.extend(line);
                    if self.pad && right_pad > 0 {
                        new_line.push(Segment::new(&" ".repeat(right_pad), Some(self.style.clone())));
                    }
                    line = new_line;
                }
                AlignMethod::Right => {
                    // Padding on left, content on right
                    let mut new_line = Vec::new();
                    if excess > 0 {
                        new_line.push(Segment::new(&" ".repeat(excess), Some(self.style.clone())));
                    }
                    new_line.extend(line);
                    line = new_line;
                }
            }

            result_lines.push(line);
        }

        // Apply vertical alignment if height specified
        if let Some(h) = height {
            if result_lines.len() < h {
                result_lines = match self.vertical {
                    VerticalAlignMethod::Top => {
                        Segment::align_top(result_lines, width, h, self.style.clone())
                    }
                    VerticalAlignMethod::Middle => {
                        Segment::align_middle(result_lines, width, h, self.style.clone())
                    }
                    VerticalAlignMethod::Bottom => {
                        Segment::align_bottom(result_lines, width, h, self.style.clone())
                    }
                };
            }
        }

        // Convert to segments with newlines
        let mut segments = Vec::new();
        for line in result_lines {
            segments.extend(line);
            segments.push(Segment::line());
        }

        segments.into_iter().map(RenderItem::Segment).collect()
    }
}
```

---
